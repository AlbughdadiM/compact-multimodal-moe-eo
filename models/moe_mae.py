"""Lightweight multimodal MoE masked autoencoder for Earth observation."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_2d_positions(
    height: int,
    width: int,
    device: torch.device,
) -> torch.Tensor:
    """Return flattened patch coordinates as [1, height * width, 2]."""
    y, x = torch.meshgrid(
        torch.arange(height, device=device),
        torch.arange(width, device=device),
        indexing="ij",
    )
    return torch.stack([x.reshape(-1), y.reshape(-1)], dim=-1).unsqueeze(0)


def _rotate_pairs(x: torch.Tensor) -> torch.Tensor:
    """Rotate adjacent feature pairs by 90 degrees."""
    paired = x.reshape(*x.shape[:-1], -1, 2)
    first, second = paired.unbind(dim=-1)
    return torch.stack([-second, first], dim=-1).flatten(-2)


def apply_2d_rope(
    x: torch.Tensor,
    positions: Optional[torch.Tensor],
    theta: float = 10000.0,
) -> torch.Tensor:
    """Apply axial 2D RoPE to [B, H, N, D] attention queries or keys."""
    if positions is None:
        return x
    if positions.ndim == 2:
        positions = positions.unsqueeze(0)
    if positions.ndim != 3 or positions.shape[-1] != 2:
        raise ValueError("positions must have shape [N, 2] or [B, N, 2]")
    if positions.shape[1] != x.shape[-2]:
        raise ValueError("positions and attention tensors must have the same token count")
    if positions.shape[0] not in {1, x.shape[0]}:
        raise ValueError("positions batch dimension must be 1 or match the attention batch")

    rotary_dim = (x.shape[-1] // 4) * 4
    if rotary_dim == 0:
        return x
    axis_dim = rotary_dim // 2
    frequency = torch.arange(0, axis_dim, 2, device=x.device, dtype=torch.float32)
    frequency = theta ** (-frequency / axis_dim)
    coordinates = positions.to(device=x.device, dtype=torch.float32)

    rotated_axes = []
    for axis_index in range(2):
        axis = x[..., axis_index * axis_dim : (axis_index + 1) * axis_dim]
        angles = coordinates[..., axis_index].unsqueeze(-1) * frequency
        angles = angles.repeat_interleave(2, dim=-1).unsqueeze(1)
        cos = angles.cos().to(dtype=x.dtype)
        sin = angles.sin().to(dtype=x.dtype)
        rotated_axes.append(axis * cos + _rotate_pairs(axis) * sin)

    return torch.cat([*rotated_axes, x[..., rotary_dim:]], dim=-1)


class LowRankAdapter(nn.Module):
    """Small expert-specific residual adapter around a shared projection."""

    def __init__(self, in_dim: int, out_dim: int, rank: int):
        super().__init__()
        self.down = nn.Linear(in_dim, rank, bias=False)
        self.up = nn.Linear(rank, out_dim, bias=False)
        nn.init.kaiming_uniform_(self.down.weight, a=5**0.5)
        nn.init.zeros_(self.up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.down(x))


class SwiGLU(nn.Module):
    """SwiGLU expert with optional shared value and output projections."""

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        shared_value: Optional[nn.Linear] = None,
        shared_output: Optional[nn.Linear] = None,
        adapter_rank: int = 0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim)
        self.value_proj = shared_value or nn.Linear(dim, hidden_dim)
        self.output_proj = shared_output or nn.Linear(hidden_dim, dim)
        self.value_adapter = (
            LowRankAdapter(dim, hidden_dim, adapter_rank)
            if shared_value is not None and adapter_rank > 0
            else None
        )
        self.output_adapter = (
            LowRankAdapter(hidden_dim, dim, adapter_rank)
            if shared_output is not None and adapter_rank > 0
            else None
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value = self.value_proj(x)
        if self.value_adapter is not None:
            value = value + self.value_adapter(x)
        hidden = F.silu(self.gate_proj(x)) * value
        output = self.output_proj(hidden)
        if self.output_adapter is not None:
            output = output + self.output_adapter(hidden)
        return self.dropout(output)


class NoisyTopKGate(nn.Module):
    """Top-k router with learned training noise and deterministic evaluation."""

    def __init__(self, dim: int, num_experts: int, k: int = 2, eps: float = 1e-9):
        super().__init__()
        if not 1 <= k <= num_experts:
            raise ValueError("k must be between 1 and num_experts")
        self.logit_proj = nn.Linear(dim, num_experts)
        self.noise_proj = nn.Linear(dim, num_experts)
        self.num_experts = num_experts
        self.k = k
        self.eps = eps

    def forward(
        self,
        x: torch.Tensor,
        stochastic: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        clean_logits = self.logit_proj(x)
        noise_scale = F.softplus(self.noise_proj(x)) + self.eps
        use_noise = self.training if stochastic is None else stochastic
        routing_logits = clean_logits
        if use_noise:
            routing_logits = clean_logits + torch.randn_like(clean_logits) * noise_scale

        topk_values, topk_indices = torch.topk(routing_logits, self.k, dim=-1)
        topk_weights = F.softmax(topk_values, dim=-1)
        gates = x.new_zeros(routing_logits.shape)
        gates.scatter_(-1, topk_indices, topk_weights)
        return gates, routing_logits, topk_indices, noise_scale, clean_logits


class Attention(nn.Module):
    """Multi-head self-attention with optional grouped key/value heads."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        if num_heads % self.num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")

        self.head_dim = dim // num_heads
        self.q_proj = nn.Linear(dim, num_heads * self.head_dim)
        self.k_proj = nn.Linear(dim, self.num_kv_heads * self.head_dim)
        self.v_proj = nn.Linear(dim, self.num_kv_heads * self.head_dim)
        self.out_proj = nn.Linear(dim, dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, num_tokens, _ = x.shape
        q = self.q_proj(x).reshape(batch_size, num_tokens, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(batch_size, num_tokens, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).reshape(batch_size, num_tokens, self.num_kv_heads, self.head_dim)

        q = apply_2d_rope(q.transpose(1, 2), positions)
        k = apply_2d_rope(k.transpose(1, 2), positions)
        v = v.transpose(1, 2)

        repeat = self.num_heads // self.num_kv_heads
        if repeat > 1:
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)
        attention = (q @ k.transpose(-2, -1)) * (self.head_dim**-0.5)
        attention = self.attn_dropout(attention.softmax(dim=-1))
        output = (attention @ v).transpose(1, 2).reshape(batch_size, num_tokens, self.dim)
        return self.proj_dropout(self.out_proj(output))


class MoELayer(nn.Module):
    """Sparse experts with a single Switch-style load-balancing objective."""

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        num_experts: int = 3,
        k: int = 2,
        share_value_output: bool = True,
        expert_adapter_rank: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.k = k
        shared_value = nn.Linear(dim, hidden_dim) if share_value_output else None
        shared_output = nn.Linear(hidden_dim, dim) if share_value_output else None
        self.experts = nn.ModuleList(
            [
                SwiGLU(
                    dim,
                    hidden_dim,
                    shared_value=shared_value,
                    shared_output=shared_output,
                    adapter_rank=expert_adapter_rank,
                    dropout=dropout,
                )
                for _ in range(num_experts)
            ]
        )
        self.gate = NoisyTopKGate(dim, num_experts, k=k)

    def _balance_loss(self, clean_logits: torch.Tensor, gates: torch.Tensor) -> torch.Tensor:
        probabilities = clean_logits.float().softmax(dim=-1)
        selected = (gates > 0).float() / self.k
        selected_fraction = selected.mean(dim=0).detach()
        probability_fraction = probabilities.mean(dim=0)
        return self.num_experts * torch.sum(selected_fraction * probability_fraction)

    def forward(
        self,
        x: torch.Tensor,
        return_routing: bool = False,
        stochastic_routing: Optional[bool] = None,
    ):
        batch_size, num_tokens, dim = x.shape
        flat = x.reshape(batch_size * num_tokens, dim)
        gates, routing_logits, topk_indices, noise_scale, clean_logits = self.gate(
            flat, stochastic=stochastic_routing
        )
        balance_loss = self._balance_loss(clean_logits, gates)
        output = flat.new_zeros(flat.shape)

        for expert_index, expert in enumerate(self.experts):
            token_indices = torch.where(gates[:, expert_index] > 0)[0]
            if token_indices.numel() == 0:
                continue
            expert_output = expert(flat[token_indices])
            expert_weight = gates[token_indices, expert_index].unsqueeze(-1)
            output.index_add_(0, token_indices, expert_output * expert_weight)

        output = output.reshape(batch_size, num_tokens, dim)
        if not return_routing:
            return output, balance_loss

        routing = {
            "gates": gates.reshape(batch_size, num_tokens, self.num_experts),
            "topk_idx": topk_indices.reshape(batch_size, num_tokens, self.k),
            "clean_logits": clean_logits.reshape(batch_size, num_tokens, self.num_experts),
            "routing_logits": routing_logits.reshape(batch_size, num_tokens, self.num_experts),
            "noise_scale": noise_scale.reshape(batch_size, num_tokens, self.num_experts),
            "balance_loss": balance_loss,
        }
        return output, balance_loss, routing


class MoETransformerEncoderLayer(nn.Module):
    """Pre-normalized attention followed by a sparse MoE feed-forward block."""

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        num_experts: int,
        k: int,
        num_heads: int,
        num_kv_heads: Optional[int] = None,
        share_value_output: bool = True,
        expert_adapter_rank: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads, num_kv_heads=num_kv_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.moe = MoELayer(
            dim,
            hidden_dim,
            num_experts=num_experts,
            k=k,
            share_value_output=share_value_output,
            expert_adapter_rank=expert_adapter_rank,
            dropout=dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        return_routing: bool = False,
        stochastic_routing: Optional[bool] = None,
    ):
        x = x + self.attn(self.norm1(x), positions=positions)
        if return_routing:
            moe_output, balance_loss, routing = self.moe(
                self.norm2(x),
                return_routing=True,
                stochastic_routing=stochastic_routing,
            )
            return x + moe_output, balance_loss, routing
        moe_output, balance_loss = self.moe(
            self.norm2(x), stochastic_routing=stochastic_routing
        )
        return x + moe_output, balance_loss


class DenseTransformerLayer(nn.Module):
    """Pre-normalized attention and dense SwiGLU for the lightweight decoder."""

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        num_heads: int,
        num_kv_heads: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads, num_kv_heads=num_kv_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = SwiGLU(dim, hidden_dim, dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), positions=positions)
        return x + self.ffn(self.norm2(x))


class MOEEncoder(nn.Module):
    """Delayed-fusion multimodal encoder with a shared pre-fusion MoE layer."""

    def __init__(
        self,
        img_size: int = 64,
        patch_size: int = 4,
        in_chans: int = 3,
        input_adapters: Optional[Dict[str, int]] = None,
        input_band_names: Optional[Dict[str, Sequence[str]]] = None,
        primary_input_name: Optional[str] = None,
        metadata_dims: Optional[Dict[str, int]] = None,
        embed_dim: int = 108,
        depth: int = 9,
        first_hidden_dim: int = 81,
        last_hidden_dim: int = 27,
        num_heads: int = 6,
        num_kv_heads: Optional[int] = None,
        experts_per_stage: Tuple[int, int] = (3, 5),
        experts_config: Optional[Sequence[int]] = None,
        k: int = 2,
        share_value_output: bool = True,
        expert_adapter_rank: int = 8,
        moe_balance_weight: float = 1e-2,
        dropout: float = 0.1,
    ):
        super().__init__()
        if img_size % patch_size != 0:
            raise ValueError("img_size must be divisible by patch_size")
        if depth < 2:
            raise ValueError("depth must be at least 2 for delayed fusion")

        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.base_patch_height = img_size // patch_size
        self.base_patch_width = img_size // patch_size
        self.patch_grid = self.base_patch_width
        self.num_patches = self.base_patch_height * self.base_patch_width
        self.moe_balance_weight = float(moe_balance_weight)
        if self.moe_balance_weight < 0:
            raise ValueError("moe_balance_weight must be non-negative")

        if input_adapters is None:
            name = primary_input_name or "image"
            input_adapters = {name: in_chans}
        self.input_specs = dict(input_adapters)
        self.input_names = list(self.input_specs)
        if not self.input_names:
            raise ValueError("input_adapters must contain at least one modality")
        self.primary_input_name = primary_input_name or self.input_names[0]
        if self.primary_input_name not in self.input_specs:
            raise ValueError("primary_input_name must be present in input_adapters")
        self.in_chans = self.input_specs[self.primary_input_name]
        self.input_name_to_idx = {name: index for index, name in enumerate(self.input_names)}

        if input_band_names is None:
            input_band_names = {
                name: [f"band_{index}" for index in range(channels)]
                for name, channels in self.input_specs.items()
            }
        self.input_band_names = {
            name: list(input_band_names[name]) for name in self.input_names
        }
        for name, channels in self.input_specs.items():
            if len(self.input_band_names[name]) != channels:
                raise ValueError(f"input_band_names['{name}'] must contain {channels} names")
        self.input_band_name_to_idx = {
            name: {band: index for index, band in enumerate(bands)}
            for name, bands in self.input_band_names.items()
        }

        self.metadata_dims = dict(metadata_dims or {})
        self.meta_names = list(self.metadata_dims)
        self.num_meta_tokens = len(self.meta_names)

        self.patch_proj = nn.Conv2d(
            self.in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.patch_proj_extra = nn.ModuleDict(
            {
                name: nn.Conv2d(channels, embed_dim, kernel_size=patch_size, stride=patch_size)
                for name, channels in self.input_specs.items()
                if name != self.primary_input_name
            }
        )
        self.validity_proj = nn.Conv2d(1, embed_dim, kernel_size=patch_size, stride=patch_size, bias=False)
        self.validity_proj_extra = nn.ModuleDict(
            {
                name: nn.Conv2d(1, embed_dim, kernel_size=patch_size, stride=patch_size, bias=False)
                for name in self.input_names
                if name != self.primary_input_name
            }
        )
        self.band_proj = nn.Linear(self.in_chans, embed_dim)
        self.band_proj_extra = nn.ModuleDict(
            {
                name: nn.Linear(channels, embed_dim)
                for name, channels in self.input_specs.items()
                if name != self.primary_input_name
            }
        )
        self.metadata_proj = nn.ModuleDict(
            {name: nn.Linear(dim, embed_dim) for name, dim in self.metadata_dims.items()}
        )

        self.modality_token_embed = nn.Parameter(torch.zeros(1, len(self.input_names), embed_dim))
        self.meta_token_embed = nn.Parameter(torch.zeros(1, self.num_meta_tokens, embed_dim))
        self.meta_missing_embed = nn.Parameter(torch.zeros(1, self.num_meta_tokens, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.fusion_norm = nn.LayerNorm(embed_dim)
        self.fusion_score = nn.Linear(embed_dim, 1, bias=False)
        self.fusion_bias = nn.Parameter(torch.zeros(len(self.input_names)))

        hidden_dims = [
            int(((depth - 1 - index) / max(depth - 1, 1)) * (first_hidden_dim - last_hidden_dim))
            + last_hidden_dim
            for index in range(depth)
        ]
        if experts_config is None:
            stage_size = max(depth // 3, 1)
            expert_values = torch.linspace(
                experts_per_stage[0], experts_per_stage[1], steps=3
            ).round()
            experts_config = [
                int(expert_values[min(index // stage_size, 2)]) for index in range(depth)
            ]
        if len(experts_config) != depth:
            raise ValueError("experts_config must contain one value per encoder layer")

        kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        self.layers = nn.ModuleList(
            [
                MoETransformerEncoderLayer(
                    dim=embed_dim,
                    hidden_dim=hidden_dims[index],
                    num_experts=int(experts_config[index]),
                    k=k,
                    num_heads=num_heads,
                    num_kv_heads=kv_heads,
                    share_value_output=share_value_output,
                    expert_adapter_rank=expert_adapter_rank,
                    dropout=dropout,
                )
                for index in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self._initialize_parameters()

    def _initialize_parameters(self) -> None:
        nn.init.trunc_normal_(self.modality_token_embed, std=0.02)
        nn.init.trunc_normal_(self.meta_token_embed, std=0.02)
        nn.init.trunc_normal_(self.meta_missing_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.zeros_(self.fusion_score.weight)
        nn.init.zeros_(self.fusion_bias)
        nn.init.zeros_(self.validity_proj.weight)
        for projection in self.validity_proj_extra.values():
            nn.init.zeros_(projection.weight)

    def _get_patch_proj(self, name: str) -> nn.Module:
        return self.patch_proj if name == self.primary_input_name else self.patch_proj_extra[name]

    def _get_validity_proj(self, name: str) -> nn.Module:
        return self.validity_proj if name == self.primary_input_name else self.validity_proj_extra[name]

    def _get_band_proj(self, name: str) -> nn.Module:
        return self.band_proj if name == self.primary_input_name else self.band_proj_extra[name]

    def _resolve_raster_inputs(
        self,
        x: Optional[torch.Tensor] = None,
        raster_dict: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        if raster_dict is None and isinstance(x, dict):
            raster_dict = x
            x = None
        if raster_dict is not None:
            inputs = {name: tensor for name, tensor in raster_dict.items() if tensor is not None}
            unknown = set(inputs) - set(self.input_specs)
            if unknown:
                raise KeyError(f"Unknown raster modalities: {sorted(unknown)}")
            if not inputs:
                raise ValueError("raster_dict must contain at least one tensor")
            return inputs
        if x is None:
            raise ValueError("Either x or raster_dict must be provided")
        return {self.primary_input_name: x}

    def _runtime_layout_from_shape(self, height: int, width: int) -> Dict[str, int]:
        if height % self.patch_size != 0 or width % self.patch_size != 0:
            raise ValueError("Input height and width must be divisible by patch_size")
        fine_height = height // self.patch_size
        fine_width = width // self.patch_size
        return {
            "fine_height": fine_height,
            "fine_width": fine_width,
            "num_patches": fine_height * fine_width,
        }

    @staticmethod
    def _normalize_validity_mask(mask: torch.Tensor, tensor: torch.Tensor) -> torch.Tensor:
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
        if mask.ndim != 4 or mask.shape[0] != tensor.shape[0] or mask.shape[-2:] != tensor.shape[-2:]:
            raise ValueError("Raster validity masks must have shape [B, C, H, W] or [B, 1, H, W]")
        if mask.shape[1] == 1 and tensor.shape[1] > 1:
            mask = mask.expand(-1, tensor.shape[1], -1, -1)
        if mask.shape[1] != tensor.shape[1]:
            raise ValueError("Raster validity-mask channels must match the runtime tensor")
        return mask.to(device=tensor.device, dtype=tensor.dtype)

    def _prepare_modality_tensor(
        self,
        name: str,
        tensor: torch.Tensor,
        validity_mask: Optional[torch.Tensor] = None,
        band_names: Optional[Sequence[str]] = None,
        band_indices: Optional[Sequence[int]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if tensor.ndim != 4:
            raise ValueError(f"Raster modality '{name}' must be [B, C, H, W]")
        batch_size, channels, height, width = tensor.shape
        expected_channels = self.input_specs[name]
        if channels > expected_channels:
            raise ValueError(f"Raster modality '{name}' has too many channels")
        if band_names is not None and band_indices is not None:
            raise ValueError("Provide band_names or band_indices, not both")

        finite = torch.isfinite(tensor)
        if validity_mask is None:
            validity = finite.to(tensor.dtype)
        else:
            validity = self._normalize_validity_mask(validity_mask, tensor) * finite
        tensor = torch.where(finite, tensor, torch.zeros_like(tensor))

        if band_names is not None:
            if len(band_names) != channels:
                raise ValueError(f"Raster modality '{name}' has mismatched band names")
            try:
                target_indices = [self.input_band_name_to_idx[name][band] for band in band_names]
            except KeyError as exc:
                raise ValueError(f"Unknown band '{exc.args[0]}' for modality '{name}'") from exc
        elif band_indices is not None:
            if len(band_indices) != channels:
                raise ValueError(f"Raster modality '{name}' has mismatched band indices")
            target_indices = [int(index) for index in band_indices]
        else:
            target_indices = list(range(channels))

        if len(set(target_indices)) != len(target_indices):
            raise ValueError(f"Raster modality '{name}' contains duplicate bands")
        if any(index < 0 or index >= expected_channels for index in target_indices):
            raise ValueError(f"Raster modality '{name}' contains an out-of-range band")

        prepared = tensor.new_zeros(batch_size, expected_channels, height, width)
        prepared_validity = tensor.new_zeros(batch_size, expected_channels, height, width)
        prepared[:, target_indices] = tensor
        prepared_validity[:, target_indices] = validity
        band_mask = tensor.new_zeros(batch_size, expected_channels)
        band_mask[:, target_indices] = 1.0
        return prepared, band_mask, prepared_validity

    def _prepare_raster_inputs(
        self,
        x: Optional[torch.Tensor] = None,
        raster_dict: Optional[Dict[str, torch.Tensor]] = None,
        raster_valid_masks: Optional[Dict[str, torch.Tensor]] = None,
        raster_band_names: Optional[Dict[str, Sequence[str]]] = None,
        raster_band_indices: Optional[Dict[str, Sequence[int]]] = None,
    ):
        raster_inputs = self._resolve_raster_inputs(x=x, raster_dict=raster_dict)
        prepared_inputs: Dict[str, torch.Tensor] = {}
        band_masks: Dict[str, torch.Tensor] = {}
        validity_masks: Dict[str, torch.Tensor] = {}
        runtime_layout = None
        batch_size = height = width = None

        for name in self.input_names:
            if name not in raster_inputs:
                continue
            tensor = raster_inputs[name]
            if runtime_layout is None:
                batch_size = tensor.shape[0]
                height, width = tensor.shape[-2:]
                runtime_layout = self._runtime_layout_from_shape(height, width)
            elif tensor.shape[0] != batch_size or tensor.shape[-2:] != (height, width):
                raise ValueError("All raster modalities must share batch and spatial dimensions")

            validity = None if raster_valid_masks is None else raster_valid_masks.get(name)
            band_names = None if raster_band_names is None else raster_band_names.get(name)
            band_indices = None if raster_band_indices is None else raster_band_indices.get(name)
            prepared_inputs[name], band_masks[name], validity_masks[name] = self._prepare_modality_tensor(
                name,
                tensor,
                validity_mask=validity,
                band_names=band_names,
                band_indices=band_indices,
            )

        if runtime_layout is None:
            raise ValueError("No valid raster inputs were provided")
        return prepared_inputs, band_masks, validity_masks, runtime_layout, raster_inputs

    def _prepare_modality_tokens(
        self,
        x: Optional[torch.Tensor] = None,
        raster_dict: Optional[Dict[str, torch.Tensor]] = None,
        raster_valid_masks: Optional[Dict[str, torch.Tensor]] = None,
        raster_band_names: Optional[Dict[str, Sequence[str]]] = None,
        raster_band_indices: Optional[Dict[str, Sequence[int]]] = None,
    ):
        prepared, band_masks, validity_masks, runtime_layout, raster_inputs = self._prepare_raster_inputs(
            x=x,
            raster_dict=raster_dict,
            raster_valid_masks=raster_valid_masks,
            raster_band_names=raster_band_names,
            raster_band_indices=raster_band_indices,
        )
        tokens_by_modality: Dict[str, torch.Tensor] = {}
        patch_validity: Dict[str, torch.Tensor] = {}

        for name in self.input_names:
            if name not in prepared:
                continue
            values = prepared[name]
            validity = validity_masks[name]
            validity_fraction = validity.mean(dim=1, keepdim=True)
            tokens = self._get_patch_proj(name)(values)
            tokens = tokens + self._get_validity_proj(name)(validity_fraction)
            tokens = tokens.flatten(2).transpose(1, 2)
            modality_index = self.input_name_to_idx[name]
            modality_bias = self.modality_token_embed[:, modality_index : modality_index + 1]
            band_bias = self._get_band_proj(name)(band_masks[name]).unsqueeze(1)
            tokens_by_modality[name] = tokens + modality_bias + band_bias
            patch_validity[name] = F.avg_pool2d(
                validity_fraction,
                kernel_size=self.patch_size,
                stride=self.patch_size,
            ).flatten(2).transpose(1, 2)

        return tokens_by_modality, patch_validity, runtime_layout, raster_inputs

    def _build_metadata(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        meta_dict: Optional[Dict[str, torch.Tensor]] = None,
        meta_valid_masks: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        values = meta_dict or {}
        validity_values = meta_valid_masks or {}
        tokens = []
        for index, name in enumerate(self.meta_names):
            value = values.get(name)
            if value is None:
                value = torch.zeros(batch_size, self.metadata_dims[name], device=device, dtype=dtype)
                validity = torch.zeros_like(value)
            else:
                value = value.to(device=device, dtype=dtype).reshape(batch_size, -1)
                if value.shape[1] != self.metadata_dims[name]:
                    raise ValueError(f"Metadata '{name}' has the wrong dimension")
                finite = torch.isfinite(value)
                supplied_validity = validity_values.get(name)
                if supplied_validity is None:
                    validity = finite.to(dtype)
                else:
                    validity = supplied_validity.to(device=device, dtype=dtype).reshape_as(value) * finite
                value = torch.where(finite, value, torch.zeros_like(value))
            value = value * validity
            missing_fraction = 1.0 - validity.mean(dim=-1, keepdim=True)
            token = self.metadata_proj[name](value).unsqueeze(1)
            token = token + self.meta_token_embed[:, index : index + 1]
            token = token + missing_fraction.unsqueeze(-1) * self.meta_missing_embed[:, index : index + 1]
            tokens.append(token)
        if not tokens:
            return torch.empty(batch_size, 0, self.embed_dim, device=device, dtype=dtype)
        return torch.cat(tokens, dim=1)

    def _fuse_modalities(
        self,
        tokens_by_modality: Dict[str, torch.Tensor],
        patch_validity: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        names = [name for name in self.input_names if name in tokens_by_modality]
        token_stack = torch.stack([tokens_by_modality[name] for name in names], dim=2)
        validity_stack = torch.stack([patch_validity[name] for name in names], dim=2).squeeze(-1)
        scores = self.fusion_score(self.fusion_norm(token_stack)).squeeze(-1)
        modality_indices = torch.tensor(
            [self.input_name_to_idx[name] for name in names], device=scores.device
        )
        scores = scores + self.fusion_bias[modality_indices].view(1, 1, -1)
        scores = scores + validity_stack.clamp_min(1e-6).log()
        weights = scores.softmax(dim=2)
        fused = torch.sum(token_stack * weights.unsqueeze(-1), dim=2)
        return fused, weights, names

    def _token_layout(self, runtime_layout: Dict[str, int], num_fine_tokens: int) -> Dict[str, int]:
        return {
            "num_meta_tokens": self.num_meta_tokens,
            "num_fine_tokens": num_fine_tokens,
            "fine_height": runtime_layout["fine_height"],
            "fine_width": runtime_layout["fine_width"],
        }

    def _encode_modalities(
        self,
        tokens_by_modality: Dict[str, torch.Tensor],
        patch_validity: Dict[str, torch.Tensor],
        meta_tokens: torch.Tensor,
        runtime_layout: Dict[str, int],
        fine_positions: torch.Tensor,
        return_routing: bool = False,
        stochastic_routing: Optional[bool] = None,
    ):
        prefusion_tokens: Dict[str, torch.Tensor] = {}
        prefusion_losses = []
        prefusion_routing = {}
        for name, tokens in tokens_by_modality.items():
            if return_routing:
                encoded, loss, routing = self.layers[0](
                    tokens,
                    positions=fine_positions,
                    return_routing=True,
                    stochastic_routing=stochastic_routing,
                )
                prefusion_routing[name] = routing
            else:
                encoded, loss = self.layers[0](
                    tokens,
                    positions=fine_positions,
                    stochastic_routing=stochastic_routing,
                )
            prefusion_tokens[name] = encoded
            prefusion_losses.append(loss)

        fused, fusion_weights, fusion_modalities = self._fuse_modalities(
            prefusion_tokens, patch_validity
        )
        batch_size = fused.shape[0]
        cls = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([meta_tokens, cls, fused], dim=1)
        prefix_positions = fine_positions.new_zeros(
            batch_size, self.num_meta_tokens + 1, 2
        )
        token_positions = torch.cat([prefix_positions, fine_positions], dim=1)
        layer_losses = [torch.stack(prefusion_losses).mean()]
        routing_by_layer = []
        if return_routing:
            routing_by_layer.append(
                {"stage": "pre_fusion", "by_modality": prefusion_routing}
            )

        for layer in self.layers[1:]:
            if return_routing:
                tokens, loss, routing = layer(
                    tokens,
                    positions=token_positions,
                    return_routing=True,
                    stochastic_routing=stochastic_routing,
                )
                routing["stage"] = "fused"
                routing_by_layer.append(routing)
            else:
                tokens, loss = layer(
                    tokens,
                    positions=token_positions,
                    stochastic_routing=stochastic_routing,
                )
            layer_losses.append(loss)

        pre_norm = tokens
        post_norm = self.norm(tokens)
        moe_loss = self.moe_balance_weight * torch.stack(layer_losses).mean()
        token_layout = self._token_layout(runtime_layout, fused.shape[1])
        return (
            post_norm,
            pre_norm,
            moe_loss,
            token_layout,
            fusion_weights,
            fusion_modalities,
            routing_by_layer if return_routing else None,
        )

    def _split_tokens(self, tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        meta_cls_count = self.num_meta_tokens + 1
        return tokens[:, :meta_cls_count], tokens[:, meta_cls_count:]

    def forward_features(
        self,
        x: Optional[torch.Tensor] = None,
        raster_dict: Optional[Dict[str, torch.Tensor]] = None,
        raster_valid_masks: Optional[Dict[str, torch.Tensor]] = None,
        raster_band_names: Optional[Dict[str, Sequence[str]]] = None,
        raster_band_indices: Optional[Dict[str, Sequence[int]]] = None,
        meta_dict: Optional[Dict[str, torch.Tensor]] = None,
        meta_valid_masks: Optional[Dict[str, torch.Tensor]] = None,
        return_routing: bool = False,
        stochastic_routing: Optional[bool] = None,
    ) -> Dict[str, Any]:
        tokens_by_modality, patch_validity, runtime_layout, raster_inputs = (
            self._prepare_modality_tokens(
                x=x,
                raster_dict=raster_dict,
                raster_valid_masks=raster_valid_masks,
                raster_band_names=raster_band_names,
                raster_band_indices=raster_band_indices,
            )
        )
        first_tokens = next(iter(tokens_by_modality.values()))
        fine_positions = build_2d_positions(
            runtime_layout["fine_height"],
            runtime_layout["fine_width"],
            first_tokens.device,
        ).expand(first_tokens.shape[0], -1, -1)
        meta_tokens = self._build_metadata(
            first_tokens.shape[0],
            first_tokens.device,
            first_tokens.dtype,
            meta_dict=meta_dict,
            meta_valid_masks=meta_valid_masks,
        )
        encoded = self._encode_modalities(
            tokens_by_modality,
            patch_validity,
            meta_tokens,
            runtime_layout,
            fine_positions,
            return_routing=return_routing,
            stochastic_routing=stochastic_routing,
        )
        post_norm, pre_norm, moe_loss, token_layout, fusion_weights, fusion_modalities, routing = encoded
        meta_cls, fine = self._split_tokens(post_norm)
        pre_meta_cls, pre_fine = self._split_tokens(pre_norm)
        return {
            "meta_cls_tokens": meta_cls,
            "fine_tokens": fine,
            "pre_norm_meta_cls_tokens": pre_meta_cls,
            "pre_norm_fine_tokens": pre_fine,
            "token_layout": token_layout,
            "runtime_layout": runtime_layout,
            "moe_loss": moe_loss,
            "raster_inputs": raster_inputs,
            "fusion_weights": fusion_weights,
            "fusion_modalities": fusion_modalities,
            "routing": routing,
        }

    @staticmethod
    def _pool_feature_tokens(
        features: Dict[str, Any], token_source: str = "pre_norm", pooling: str = "mean_fine"
    ) -> torch.Tensor:
        if token_source == "pre_norm":
            meta_cls = features["pre_norm_meta_cls_tokens"]
            fine = features["pre_norm_fine_tokens"]
        elif token_source == "post_norm":
            meta_cls = features["meta_cls_tokens"]
            fine = features["fine_tokens"]
        else:
            raise ValueError("token_source must be 'pre_norm' or 'post_norm'")
        if pooling == "mean_fine":
            return fine.mean(dim=1)
        if pooling == "cls":
            return meta_cls[:, -1]
        if pooling == "mean_all":
            return torch.cat([meta_cls, fine], dim=1).mean(dim=1)
        raise ValueError("pooling must be one of {'cls', 'mean_all', 'mean_fine'}")

    def extract_embedding(self, token_source: str = "pre_norm", pooling: str = "mean_fine", **kwargs):
        features = self.forward_features(**kwargs)
        return self._pool_feature_tokens(features, token_source=token_source, pooling=pooling)

    def forward(self, **kwargs):
        features = self.forward_features(**kwargs)
        output = torch.cat([features["meta_cls_tokens"], features["fine_tokens"]], dim=1)
        pre_norm = torch.cat(
            [features["pre_norm_meta_cls_tokens"], features["pre_norm_fine_tokens"]], dim=1
        )
        return output, features["moe_loss"], pre_norm


class MOEMAE(nn.Module):
    """MAE wrapper around the delayed-fusion encoder."""

    def __init__(
        self,
        encoder: MOEEncoder,
        decoder_layers: int = 2,
        decoder_embed: int = 108,
        mask_ratio: float = 0.75,
    ):
        super().__init__()
        self.encoder = encoder
        self.mask_ratio = mask_ratio
        self.decoder_embed = nn.Linear(encoder.embed_dim, decoder_embed)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed))
        self.decoder_layers = nn.ModuleList(
            [
                DenseTransformerLayer(
                    dim=decoder_embed,
                    hidden_dim=decoder_embed // 2,
                    num_heads=6,
                    num_kv_heads=6,
                    dropout=0.1,
                )
                for _ in range(decoder_layers)
            ]
        )
        self.decoder_norm = nn.LayerNorm(decoder_embed)
        self.decoder_heads = nn.ModuleDict(
            {
                name: nn.Linear(
                    decoder_embed, encoder.patch_size * encoder.patch_size * channels
                )
                for name, channels in encoder.input_specs.items()
            }
        )
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def forward_features(self, **kwargs) -> Dict[str, Any]:
        return self.encoder.forward_features(**kwargs)

    def extract_embedding(self, **kwargs) -> torch.Tensor:
        return self.encoder.extract_embedding(**kwargs)

    def random_masking(
        self,
        batch_size: int,
        num_patches: int,
        device: torch.device,
        generator: Optional[torch.Generator] = None,
    ):
        keep_count = int(num_patches * (1.0 - self.mask_ratio))
        if generator is None:
            noise = torch.rand(batch_size, num_patches, device=device)
        else:
            noise = torch.rand(
                batch_size,
                num_patches,
                device=generator.device,
                generator=generator,
            ).to(device=device)
        shuffle = noise.argsort(dim=1)
        restore = shuffle.argsort(dim=1)
        keep = shuffle[:, :keep_count]
        mask = torch.ones(batch_size, num_patches, device=device)
        mask.scatter_(1, keep, 0.0)
        return keep, restore, mask

    def _restore_decoder_tokens(
        self, visible_tokens: torch.Tensor, ids_restore: torch.Tensor, total_patches: int
    ) -> torch.Tensor:
        batch_size, visible_count, dim = visible_tokens.shape
        masked_count = total_patches - visible_count
        mask_tokens = self.mask_token.to(dtype=visible_tokens.dtype).expand(
            batch_size, masked_count, dim
        )
        all_tokens = torch.cat([visible_tokens, mask_tokens], dim=1)
        return all_tokens.gather(1, ids_restore.unsqueeze(-1).expand(-1, -1, dim))

    def forward(
        self,
        imgs: Optional[torch.Tensor] = None,
        raster_dict: Optional[Dict[str, torch.Tensor]] = None,
        raster_valid_masks: Optional[Dict[str, torch.Tensor]] = None,
        raster_band_names: Optional[Dict[str, Sequence[str]]] = None,
        raster_band_indices: Optional[Dict[str, Sequence[int]]] = None,
        meta_dict: Optional[Dict[str, torch.Tensor]] = None,
        meta_valid_masks: Optional[Dict[str, torch.Tensor]] = None,
        mask_generator: Optional[torch.Generator] = None,
    ):
        tokens_by_modality, patch_validity, runtime_layout, _ = self.encoder._prepare_modality_tokens(
            x=imgs,
            raster_dict=raster_dict,
            raster_valid_masks=raster_valid_masks,
            raster_band_names=raster_band_names,
            raster_band_indices=raster_band_indices,
        )
        first_tokens = next(iter(tokens_by_modality.values()))
        batch_size, num_patches, dim = first_tokens.shape
        ids_keep, ids_restore, mask = self.random_masking(
            batch_size,
            num_patches,
            first_tokens.device,
            generator=mask_generator,
        )
        fine_positions = build_2d_positions(
            runtime_layout["fine_height"],
            runtime_layout["fine_width"],
            first_tokens.device,
        ).expand(batch_size, -1, -1)
        visible_positions = fine_positions.gather(
            1, ids_keep.unsqueeze(-1).expand(-1, -1, 2)
        )
        visible_tokens = {
            name: tokens.gather(1, ids_keep.unsqueeze(-1).expand(-1, -1, dim))
            for name, tokens in tokens_by_modality.items()
        }
        visible_validity = {
            name: validity.gather(1, ids_keep.unsqueeze(-1))
            for name, validity in patch_validity.items()
        }
        meta_tokens = self.encoder._build_metadata(
            batch_size,
            first_tokens.device,
            first_tokens.dtype,
            meta_dict=meta_dict,
            meta_valid_masks=meta_valid_masks,
        )
        encoded = self.encoder._encode_modalities(
            visible_tokens,
            visible_validity,
            meta_tokens,
            runtime_layout,
            visible_positions,
        )
        post_norm, _, moe_loss, _, _, _, _ = encoded
        encoded_meta_cls, encoded_visible = self.encoder._split_tokens(post_norm)

        meta_memory = self.decoder_embed(encoded_meta_cls)
        visible_memory = self.decoder_embed(encoded_visible)
        decoder_patches = self._restore_decoder_tokens(
            visible_memory, ids_restore, num_patches
        )
        decoder_tokens = torch.cat([meta_memory, decoder_patches], dim=1)
        prefix_positions = fine_positions.new_zeros(
            batch_size, self.encoder.num_meta_tokens + 1, 2
        )
        decoder_positions = torch.cat([prefix_positions, fine_positions], dim=1)
        for layer in self.decoder_layers:
            decoder_tokens = layer(decoder_tokens, positions=decoder_positions)
        decoder_tokens = self.decoder_norm(decoder_tokens)
        decoded_patches = decoder_tokens[:, self.encoder.num_meta_tokens + 1 :]
        predictions = {
            name: head(decoded_patches) for name, head in self.decoder_heads.items()
        }
        return predictions, mask, ids_restore, moe_loss


def build_model(
    size: str = "XXS",
    img_size: int = 64,
    patch_size: int = 4,
    in_chans: int = 3,
    input_adapters: Optional[Dict[str, int]] = None,
    input_band_names: Optional[Dict[str, Sequence[str]]] = None,
    primary_input_name: Optional[str] = None,
    metadata_dims: Optional[Dict[str, int]] = None,
    moe_balance_weight: float = 1e-2,
) -> MOEEncoder:
    configs = {
        "S": dict(embed_dim=144, depth=15, first_hidden_dim=144, last_hidden_dim=72, num_heads=8),
        "XS": dict(embed_dim=128, depth=12, first_hidden_dim=96, last_hidden_dim=32, num_heads=8),
        "XXS": dict(embed_dim=108, depth=9, first_hidden_dim=81, last_hidden_dim=27, num_heads=6),
    }
    if size not in configs:
        raise ValueError(f"Unknown model size '{size}'")
    return MOEEncoder(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        input_adapters=input_adapters,
        input_band_names=input_band_names,
        primary_input_name=primary_input_name,
        metadata_dims=metadata_dims,
        moe_balance_weight=moe_balance_weight,
        **configs[size],
    )


mmLiT = MOEMAE
