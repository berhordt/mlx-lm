# Copyright © 2025 Apple Inc.

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import mlx.core as mx

from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .cache import CacheList, KVCache
from .deepseek_v32 import (
    DeepseekV32Attention,
    DeepseekV32DecoderLayer,
    DeepseekV32Model,
)
from .deepseek_v32 import Model as DSV32Model


# Diagnostic instrumentation for the GLM-5.2 DSA long-context token-0 loop.
# Instrument 4: bypass sparse top-k selection at decode (L == 1) and attend to
# all cached keys, to isolate whether the indexer selection or the cached KV /
# main attention is at fault. No-op unless GLM_DSA_FULL_ATTENTION is set.
_GLM_DSA_FULL_ATTENTION = os.environ.get("GLM_DSA_FULL_ATTENTION") is not None
_GLM_DSA_TRACE = os.environ.get("GLM_DSA_TRACE") is not None
_GLM_DSA_FP32_SDPA = os.environ.get("GLM_DSA_FP32_SDPA") is not None


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    vocab_size: int
    hidden_size: int
    index_head_dim: int
    index_n_heads: int
    index_topk: int
    intermediate_size: int
    moe_intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    n_shared_experts: Optional[int]
    n_routed_experts: Optional[int]
    routed_scaling_factor: float
    kv_lora_rank: int
    q_lora_rank: int
    qk_rope_head_dim: int
    v_head_dim: int
    qk_nope_head_dim: int
    topk_method: str
    scoring_func: str
    norm_topk_prob: bool
    n_group: int
    topk_group: int
    num_experts_per_tok: int
    moe_layer_freq: int
    first_k_dense_replace: int
    max_position_embeddings: int
    rms_norm_eps: float
    rope_parameters: Dict
    attention_bias: bool
    rope_scaling: Dict = None
    rope_theta: Optional[float] = None
    indexer_types: Optional[List[str]] = None
    index_topk_pattern: Optional[Any] = None
    index_topk_freq: int = 1
    index_skip_topk_offset: int = 2

    def __post_init__(self):
        self.rope_scaling = self.rope_parameters
        self.rope_theta = self.rope_parameters["rope_theta"]

        if self.indexer_types is None:
            if self.index_topk_pattern is not None:
                pattern = self.index_topk_pattern
                if isinstance(pattern, str):
                    self.indexer_types = [
                        {"F": "full", "S": "shared"}[c] for c in pattern
                    ]
                else:
                    self.indexer_types = list(pattern)
            else:
                freq = max(self.index_topk_freq, 1)
                offset = self.index_skip_topk_offset
                self.indexer_types = [
                    "full" if (max(i - offset + 1, 0) % freq) == 0 else "shared"
                    for i in range(self.num_hidden_layers)
                ]


class GlmMoeDsaAttention(DeepseekV32Attention):
    def __init__(self, config: ModelArgs, layer_idx: int):
        super().__init__(config)
        self.layer_idx = layer_idx
        self.skip_topk = config.indexer_types[layer_idx] == "shared"
        if self.skip_topk:
            self.indexer = None
        else:
            self.indexer.layer_idx = layer_idx

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        prev_topk_indices: Optional[mx.array] = None,
    ):
        B, L, D = x.shape

        qr = self.q_a_layernorm(self.q_a_proj(x))
        q = self.q_b_proj(qr)

        q = q.reshape(B, L, self.num_heads, self.q_head_dim).transpose(0, 2, 1, 3)
        q_nope, q_pe = mx.split(q, [self.qk_nope_head_dim], axis=-1)
        compressed_kv = self.kv_a_proj_with_mqa(x)
        compressed_kv, k_pe = mx.split(compressed_kv, [self.kv_lora_rank], axis=-1)
        k_pe = k_pe.reshape(B, L, 1, self.qk_rope_head_dim).transpose(0, 2, 1, 3)
        kv_latent = self.kv_a_layernorm(compressed_kv)

        offset = cache[0].offset if cache is not None else 0
        q_pe = self.rope(q_pe, offset)
        k_pe = self.rope(k_pe, offset)

        kv_latent = mx.expand_dims(kv_latent, axis=1)

        if cache is not None:
            kv_latent, k_pe = cache[0].update_and_fetch(kv_latent, k_pe)
        else:
            cache = [None] * 2

        if self.indexer is not None:
            topk_indices = self.indexer(x, qr, mask, cache=cache[1])
        else:
            topk_indices = prev_topk_indices

        if topk_indices is not None and not (_GLM_DSA_FULL_ATTENTION and L == 1):
            if L == 1:
                idx = topk_indices[:, :, 0, :, None]
                kv_latent = mx.take_along_axis(
                    kv_latent,
                    mx.broadcast_to(idx, idx.shape[:-1] + (kv_latent.shape[-1],)),
                    axis=2,
                )
                k_pe = mx.take_along_axis(
                    k_pe,
                    mx.broadcast_to(idx, idx.shape[:-1] + (k_pe.shape[-1],)),
                    axis=2,
                )
                if mask is not None:
                    mask = mx.take_along_axis(mask, topk_indices, axis=-1)
            else:
                shape = list(topk_indices.shape)
                shape[-1] = kv_latent.shape[2]
                sparse_mask = mx.zeros(shape, dtype=mx.bool_)
                sparse_mask = mx.put_along_axis(
                    sparse_mask, topk_indices, mx.array(True), axis=-1
                )
                if mask is not None:
                    sparse_mask = sparse_mask & mask
                mask = sparse_mask

        pe_scores = (q_pe * self.scale) @ k_pe.swapaxes(-1, -2)
        if mask is not None:
            pe_scores = mx.where(
                mask,
                pe_scores,
                mx.array(mx.finfo(pe_scores.dtype).min, pe_scores.dtype),
            )

        if _GLM_DSA_TRACE and self.layer_idx == 3:
            # Memory-light subset diagnostics (head 0, first 64 queries, last
            # 256 keys) so the trace adds ~tens of MB, not ~13 GB/rank.

            def _finite(name: str, arr, sl) -> None:
                if arr is not None:
                    try:
                        ok = bool(mx.isfinite(arr[sl]).all().item())
                    except Exception:  # pragma: no cover
                        ok = "ERR"
                    if ok is not True:
                        print(f"[GLM_DSA_TRACE] L={L} layer 3 {name}[{sl}]: finite={ok}")

            _finite("x", x, (slice(0, 1), slice(0, 64)))
            _finite("qr", qr, (slice(0, 1), slice(0, 64)))
            _finite("q_nope", q_nope, (0, 0, slice(0, 64)))
            _finite("q_pe", q_pe, (0, 0, slice(0, 64)))
            _finite("kv_latent", kv_latent, (0, 0, slice(-256, None)))
            _finite("k_pe", k_pe, (0, 0, slice(-256, None)))
            _finite("pe_scores", pe_scores, (0, 0, slice(0, 64)))

            if L > 1 and topk_indices is not None:
                ti = topk_indices[..., :64, :].reshape(-1).astype(mx.int32)
                print(
                    f"[GLM_DSA_TRACE] L={L} layer 3: topk min={int(mx.min(ti).item())} "
                    f"max={int(mx.max(ti).item())} keys={kv_latent.shape[2]}"
                )

        if L == 1:
            q_nope = self.embed_q(q_nope)
            k = v = kv_latent
        else:
            k = self.embed_q(kv_latent, transpose=False)
            v = self.unembed_out(kv_latent)

        if _GLM_DSA_TRACE and self.layer_idx == 3 and L > 1:
            # Replicate the fast-SDPA fallback on a subset (head 0, first 64
            # queries, all keys) to find the NaN step without OOM.
            qk = (q_nope[0, 0, :64] * self.scale) @ k[0, 0].swapaxes(-1, -2)
            ps = pe_scores[0, 0, :64]
            fs = qk + ps
            fs_ok = bool(mx.isfinite(fs).all().item())
            fsm = mx.softmax(fs, axis=-1, precise=True)
            fsm_ok = bool(mx.isfinite(fsm).all().item())
            fout = fsm @ v[0, 0]
            fout_ok = bool(mx.isfinite(fout).all().item())
            # Test split-K workaround for the 2^15 matmul NaN: reduce over the
            # key dim in two halves so no single matmul has K=32768.
            N2 = pe_scores.shape[-1] // 2
            fout_a = fsm[:, :N2] @ v[0, 0][:N2]
            fout_b = fsm[:, N2:] @ v[0, 0][N2:]
            fout_split = fout_a + fout_b
            split_ok = bool(mx.isfinite(fout_split).all().item())
            if not fout_ok and pe_scores.shape[-1] == 32768:
                # Dump the actual tensors for offline reproduction.
                try:
                    import numpy as np
                    np.save("/tmp/l3_fsm.npy", np.array(fsm.astype(mx.float16)))
                    np.save("/tmp/l3_v.npy", np.array(v[0, 0].astype(mx.float16)))
                    np.save("/tmp/l3_q.npy", np.array(q_nope[0, 0, :64].astype(mx.float16)))
                    np.save("/tmp/l3_k.npy", np.array(k[0, 0].astype(mx.float16)))
                    np.save("/tmp/l3_pe.npy", np.array(pe_scores[0, 0, :64].astype(mx.float16)))
                    print("[GLM_DSA_TRACE] dumped /tmp/l3_*.npy")
                except Exception as e:  # pragma: no cover
                    print(f"[GLM_DSA_TRACE] dump failed: {e}")
            print(
                f"[GLM_DSA_TRACE] L={L} layer 3 fallback(sub): scores_finite={fs_ok} "
                f"scores_max={float(mx.max(fs).item()):.3e} "
                f"scores_min={float(mx.min(fs).item()):.3e} "
                f"softmax_finite={fsm_ok} out_finite={fout_ok} "
                f"splitK_finite={split_ok}"
            )

        if _GLM_DSA_FP32_SDPA:
            # Test/fix: compute attention in fp32 to rule out bf16 precision
            # issues at long key counts (N=2^15).
            q_s = q_nope.astype(mx.float32)
            k_s = k.astype(mx.float32)
            v_s = v.astype(mx.float32)
            mask_s = pe_scores.astype(mx.float32)
        else:
            q_s, k_s, v_s, mask_s = q_nope, k, v, pe_scores

        output = scaled_dot_product_attention(
            q_s, k_s, v_s, cache=cache, scale=self.scale, mask=mask_s
        )
        if _GLM_DSA_FP32_SDPA:
            output = output.astype(mx.bfloat16)
        if _GLM_DSA_TRACE and self.layer_idx == 3:
            if not mx.isfinite(output).all().item():
                print(f"[GLM_DSA_TRACE] L={L} layer 3: SDPA output NOT finite")
        if L == 1:
            output = self.unembed_out(output)

        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        # Keep the indexer cache in the output graph (so it is evaluated with
        # the model output) WITHOUT rebinding cache[0].keys: mx.depends returns
        # a new array with private storage, and rebinding detaches the KVCache
        # buffer, corrupting writes at long-context growth boundaries (NaN KV
        # -> token-0 loop at ~2^15 tokens).
        if self.indexer is not None and cache is not None and cache[0] is not None:
            output = mx.depends(output, (cache[1].keys, cache[1].values))
        return self.o_proj(output), topk_indices


class GlmMoeDsaDecoderLayer(DeepseekV32DecoderLayer):
    def __init__(self, config: ModelArgs, layer_idx: int):
        super().__init__(config, layer_idx)
        self.layer_idx = layer_idx
        self.self_attn = GlmMoeDsaAttention(config, layer_idx)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        prev_topk_indices: Optional[mx.array] = None,
    ):
        r, topk_indices = self.self_attn(
            self.input_layernorm(x), mask, cache, prev_topk_indices
        )
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        return h + r, topk_indices


class GlmMoeDsaModel(DeepseekV32Model):
    def __init__(self, config: ModelArgs):
        super().__init__(config)
        self.layers = [
            GlmMoeDsaDecoderLayer(config, idx)
            for idx in range(config.num_hidden_layers)
        ]

    def __call__(
        self,
        x: mx.array,
        cache: Optional[Any] = None,
    ) -> mx.array:
        h = self.embed_tokens(x)

        pipeline_rank = self.pipeline_rank
        pipeline_size = self.pipeline_size

        if cache is None:
            cache = [None] * self.num_layers
        mask = create_attention_mask(
            h, cache[0][0] if cache[0] else None, return_array=True
        )

        # Receive from the previous process in the pipeline
        if pipeline_rank < pipeline_size - 1:
            h = mx.distributed.recv_like(h, (pipeline_rank + 1))

        prev_topk_indices = None
        for i in range(self.num_layers):
            h, prev_topk_indices = self.layers[self.start_idx + i](
                h, mask, cache[i], prev_topk_indices
            )

        # Send to the next process in the pipeline
        if pipeline_rank != 0:
            h = mx.distributed.send(h, (pipeline_rank - 1) % pipeline_size)
            # Force the send to complete without rebinding cache[-1] keys
            # (rebinding detaches the KVCache buffer -> NaN corruption).
            mx.eval(h)

        # Broadcast h while keeping it in the graph
        if pipeline_size > 1:
            h = mx.distributed.all_gather(h)[: h.shape[0]]

        return self.norm(h)


class Model(DSV32Model):
    def __init__(self, config: ModelArgs):
        super().__init__(config)
        self.model = GlmMoeDsaModel(config)

    def make_cache(self):
        # Shared layers run no indexer, so they get no indexer KVCache.
        caches = []
        for layer in self.layers:
            if getattr(layer.self_attn, "skip_topk", False):
                caches.append(CacheList(KVCache()))
            else:
                caches.append(CacheList(KVCache(), KVCache()))
        return caches
