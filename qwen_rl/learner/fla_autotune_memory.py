"""Release unused Torch cache before cold FLA L2Norm autotuning needs driver RAM."""
from functools import lru_cache, wraps
import importlib

import torch

MIN_DRIVER_FREE_BYTES = 1024**3


def guard_benchmark(benchmark, stats):
    @wraps(benchmark)
    def guarded(*args, **kwargs):
        free, _total = torch.cuda.mem_get_info()
        if free < MIN_DRIVER_FREE_BYTES:
            # Live tensors and autograd graphs remain intact. This releases only
            # allocator-owned free blocks for Triton's module/benchmark allocations.
            torch.cuda.empty_cache()
            stats['cache_releases'] += 1
        return benchmark(*args, **kwargs)
    return guarded


@lru_cache(maxsize=1)
def install_fla_autotune_memory_guard():
    module = importlib.import_module('fla.modules.l2norm')
    stats = {'kernels': 0, 'cache_releases': 0}
    for name in ('l2norm_fwd_kernel', 'l2norm_bwd_kernel',
                 'l2norm_fwd_kernel1', 'l2norm_bwd_kernel1'):
        kernel = getattr(module, name)
        benchmark = kernel.do_bench
        if not callable(benchmark):
            raise RuntimeError(f'pinned FLA kernel {name} has no benchmark callback')
        kernel.do_bench = guard_benchmark(benchmark, stats)
        stats['kernels'] += 1
    return stats
