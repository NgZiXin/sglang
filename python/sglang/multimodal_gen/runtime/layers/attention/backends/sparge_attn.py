# SPDX-License-Identifier: Apache-2.0

import torch
from spas_sage_attn import spas_sage2_attn_meansim_topk_cuda

from sglang.multimodal_gen.runtime.layers.attention.backends.attention_backend import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


class SpargeAttentionBackend(AttentionBackend):
    accept_output_buffer: bool = True

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [64, 128]

    @staticmethod
    def get_enum() -> AttentionBackendEnum:
        return AttentionBackendEnum.SPARGE_ATTN

    @staticmethod
    def get_impl_cls() -> type["SpargeAttentionImpl"]:
        return SpargeAttentionImpl


class SpargeAttentionImpl(AttentionImpl):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        causal: bool,
        softmax_scale: float,
        num_kv_heads: int | None = None,
        prefix: str = "",
        **extra_impl_args,
    ) -> None:
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.topk = extra_impl_args.get("topk", 0.5)
        self.prefix = prefix
        self.debug = True

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        if self.debug:
            logger.info(
                "Sparge start: layer=%s timestep=%s q=%s k=%s v=%s",
                self.prefix,
                getattr(attn_metadata, "current_timestep", None),
                tuple(query.shape),
                tuple(key.shape),
                tuple(value.shape),
            )
        output = spas_sage2_attn_meansim_topk_cuda(
            query.contiguous(),
            key.contiguous(),
            value.contiguous(),
            topk=self.topk,
            is_causal=self.causal,
            scale=self.softmax_scale,
            tensor_layout="NHD",
        )
        if self.debug:
            logger.info("Sparge finish: layer=%s", self.prefix)
        return output
