"""Discard only MLP-wide saved tensors; replay their values for original VJPs."""
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
import torch


@dataclass
class _Invocation:
    index: int
    forward: object
    input_ticket: object
    requires_grad: bool
    autocast: bool
    autocast_dtype: object
    versions: list
    metadata: list = field(default_factory=list)
    values: dict = field(default_factory=dict)


class MLPWideRecompute:
    def __init__(self, layers, pack, unpack, offloader=None):
        self.layers, self.pack, self.unpack = layers, pack, unpack
        self.offloader = offloader
        self.owners = []
        self.invocations = []
        self.active = None
        self.stats = dict(dropped_tensors=0, dropped_bytes=0, replays=0,
                          replay_peak_bytes=0, invocations=0)

    @staticmethod
    def _wide(tensor, x, width):
        return (tensor.is_floating_point() and tensor.ndim >= 2
                and tensor.shape[-1] == width
                and tensor.numel() == x.numel() // x.shape[-1] * width)

    def __enter__(self):
        for index, layer in enumerate(self.layers):
            mlp = layer.mlp
            original = mlp.forward
            owned = mlp.__dict__.get('forward')
            self.owners.append((mlp, owned))
            def wrapped(x, *args, _index=index, _mlp=mlp, _forward=original, **kwargs):
                if args or kwargs:
                    raise RuntimeError('MLP replay requires a single tensor input')
                width = _mlp.down_proj.in_features
                inv = _Invocation(_index, _forward, self.pack(x), x.requires_grad,
                                  torch.is_autocast_enabled(x.device.type),
                                  torch.get_autocast_dtype(x.device.type),
                                  [(p, p._version) for p in _mlp.parameters()])
                self.invocations.append(inv)
                self.stats['invocations'] += 1
                def save(tensor):
                    if self._wide(tensor, x, width):
                        ordinal = len(inv.metadata)
                        inv.metadata.append((tuple(tensor.shape), tensor.dtype, tensor.device))
                        self.stats['dropped_tensors'] += 1
                        self.stats['dropped_bytes'] += tensor.numel() * tensor.element_size()
                        return ('wide', inv, ordinal)
                    return ('cache', self.pack(tensor))
                with torch.autograd.graph.saved_tensors_hooks(save, self._restore):
                    return _forward(x)
            mlp.forward = wrapped
        return self

    def _restore(self, ticket):
        if ticket[0] == 'cache':
            return self.unpack(ticket[1])
        _, inv, ordinal = ticket
        if ordinal not in inv.values:
            if any(p._version != version for p, version in inv.versions):
                raise RuntimeError('MLP parameters changed before replay')
            if self.active is not None:
                self.active.values.clear()
            self.active = inv
            x = self.unpack(inv.input_ticket).detach().requires_grad_(inv.requires_grad)
            width = inv.metadata[0][0][-1]
            captured = []
            def capture(tensor):
                if self._wide(tensor, x, width):
                    captured.append(tensor.detach())
                return None
            lease = self.offloader.recompute_group(inv.index) if self.offloader else nullcontext()
            with lease, torch.enable_grad(), torch.autocast(x.device.type, enabled=inv.autocast, dtype=inv.autocast_dtype):
                with torch.autograd.graph.saved_tensors_hooks(capture, lambda _: None):
                    output = inv.forward(x)
                del output
            metadata = [(tuple(t.shape), t.dtype, t.device) for t in captured]
            if metadata != inv.metadata:
                raise RuntimeError('MLP replay saved-tensor metadata changed')
            inv.values = dict(enumerate(captured))
            self.stats['replays'] += 1
            unique = {t.untyped_storage().data_ptr(): t.untyped_storage().nbytes() for t in captured}
            self.stats['replay_peak_bytes'] = max(self.stats['replay_peak_bytes'], sum(unique.values()))
        return inv.values.pop(ordinal)

    def __exit__(self, *_):
        for mlp, owned in reversed(self.owners):
            if owned is None:
                del mlp.forward
            else:
                mlp.forward = owned
        for inv in self.invocations:
            inv.values.clear()
        self.invocations.clear()
        self.active = None
        self.stats['live_invocations'] = 0
