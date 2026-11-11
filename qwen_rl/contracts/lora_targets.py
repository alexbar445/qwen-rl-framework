"""Exact PEFT projection inventory shared by T2 training and Native export."""
from __future__ import annotations

GDN_MODULES = ('in_proj_qkv', 'in_proj_z', 'in_proj_a', 'in_proj_b', 'out_proj')


def gdn_lora_enabled(identity: dict) -> bool:
    flag = identity['learner']['adapter'].get('gdn_projections', False)
    if type(flag) is not bool:
        raise ValueError('adapter.gdn_projections must be a boolean')
    return flag


def projection_specs(identity: dict) -> dict[str, dict]:
    geometry = identity['model']['geometry']
    hidden = geometry['hidden_size']
    intermediate = geometry['intermediate_size']
    q = geometry['num_attention_heads'] * geometry['head_dim']
    kv = geometry['num_key_value_heads'] * geometry['head_dim']
    include_gdn = gdn_lora_enabled(identity)
    result = {}
    def add(layer, branch, module, inp, out, logical):
        if type(inp) is not int or type(out) is not int or inp <= 0 or out <= 0:
            raise ValueError('LoRA projection dimensions must be positive integers')
        result[f'model.layers.{layer}.{branch}.{module}'] = {
            'input_width': inp, 'output_width': out,
            'logical_bindings': [f'text/layers/{layer}/{x}' for x in logical],
        }
    for layer, kind in enumerate(identity['model']['layer_types']):
        add(layer,'mlp','gate_proj',hidden,intermediate,['mlp/gate'])
        add(layer,'mlp','up_proj',hidden,intermediate,['mlp/up'])
        add(layer,'mlp','down_proj',intermediate,hidden,['mlp/down'])
        if kind == 'full_attention':
            add(layer,'self_attn','q_proj',hidden,2*q,['attention/query','attention/gate'])
            add(layer,'self_attn','k_proj',hidden,kv,['attention/key'])
            add(layer,'self_attn','v_proj',hidden,kv,['attention/value'])
            add(layer,'self_attn','o_proj',q,hidden,['attention/output'])
        elif kind == 'linear_attention' and include_gdn:
            key = geometry['linear_num_key_heads'] * geometry['linear_key_head_dim']
            value = geometry['linear_num_value_heads'] * geometry['linear_value_head_dim']
            add(layer,'linear_attn','in_proj_qkv',hidden,2*key+value,['gdn/query','gdn/key','gdn/value'])
            add(layer,'linear_attn','in_proj_z',hidden,value,['gdn/z'])
            add(layer,'linear_attn','in_proj_a',hidden,geometry['linear_num_value_heads'],['gdn/a_projection'])
            add(layer,'linear_attn','in_proj_b',hidden,geometry['linear_num_value_heads'],['gdn/b_projection'])
            add(layer,'linear_attn','out_proj',value,hidden,['gdn/output'])
        elif kind not in {'linear_attention','full_attention'}:
            raise ValueError('unsupported layer type in LoRA target inventory')
    return dict(sorted(result.items()))


def projection_input_bases(identity: dict) -> dict[str, str]:
    rotation = identity['rotation']
    raw_bases = rotation.get('input_basis_by_parameter', {})
    result = {}
    for name, spec in projection_specs(identity).items():
        if name.endswith(('.linear_attn.in_proj_a', '.linear_attn.in_proj_b')):
            basis = 'primal'
        elif rotation['input_basis'] == 'prism_rotated':
            basis = 'prism_rotated'
        else:
            values = {raw_bases[x] for x in spec['logical_bindings']}
            if len(values) != 1:
                raise ValueError(f'fused LoRA target has incompatible input bases: {name}')
            raw = values.pop()
            if raw not in {'hadamard1024_signed','primal'}:
                raise ValueError(f'unsupported LoRA input basis: {raw}')
            basis = 'prism_rotated' if raw == 'hadamard1024_signed' else 'primal'
        result[name] = basis
    return result
