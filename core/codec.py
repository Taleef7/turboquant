"""Authoritative packed CPU reference codecs for fixed KV blocks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import torch

from core.packing import (
    pack_indices,
    pack_sign_bits,
    tensor_nbytes,
    unpack_indices,
    unpack_sign_bits,
)
from core.turboquant_simple import generate_rotation_matrix
from utils.math_utils import generate_qjl_matrix, get_codebook_tensors


_SUPPORTED_BITS = frozenset({2, 3, 4, 5, 6, 7, 8})
_FLOAT_DTYPES = frozenset({torch.float16, torch.bfloat16, torch.float32})


def _require_int(value: object, name: str, *, minimum: int = 1) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be a built-in int.")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")
    return value


def _validate_bits(bits: object) -> int:
    bits = _require_int(bits, "bits", minimum=2)
    if bits not in _SUPPORTED_BITS:
        raise ValueError(f"bits must be one of {sorted(_SUPPORTED_BITS)}.")
    return bits


def _validate_kv_tensor(x: torch.Tensor, head_dim: int, label: str) -> tuple[int, int]:
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"{label} must be a torch.Tensor.")
    if x.ndim != 4 or x.shape[0] != 1 or x.shape[-1] != head_dim:
        raise ValueError(
            f"{label} must have shape [1, num_kv_heads, tokens, {head_dim}], got {tuple(x.shape)}."
        )
    if x.shape[1] <= 0 or x.shape[2] <= 0:
        raise ValueError(f"{label} must contain at least one head and token.")
    if x.dtype not in _FLOAT_DTYPES:
        raise TypeError(f"{label} must use fp16, bf16, or fp32, got {x.dtype}.")
    return x.shape[1], x.shape[2]


@dataclass(frozen=True)
class KeyLayout:
    head_dim: int
    bits: int
    norm_dtype: torch.dtype
    qjl_enabled: bool = False

    def __post_init__(self) -> None:
        _require_int(self.head_dim, "head_dim", minimum=3)
        bits = _validate_bits(self.bits)
        if type(self.qjl_enabled) is not bool:
            raise TypeError("qjl_enabled must be a bool.")
        if self.qjl_enabled and bits < 3:
            raise ValueError("QJL requires at least 3 total bits.")
        if self.norm_dtype not in _FLOAT_DTYPES:
            raise TypeError("norm_dtype must be fp16, bf16, or fp32.")


@dataclass(frozen=True)
class ValueLayout:
    head_dim: int
    bits: int
    group_size: int
    metadata_dtype: torch.dtype = torch.float16

    def __post_init__(self) -> None:
        _require_int(self.head_dim, "head_dim", minimum=1)
        _validate_bits(self.bits)
        _require_int(self.group_size, "group_size", minimum=1)
        if self.head_dim % self.group_size != 0:
            raise ValueError("head_dim must be divisible by group_size.")
        if self.metadata_dtype not in _FLOAT_DTYPES:
            raise TypeError("metadata_dtype must be fp16, bf16, or fp32.")


@dataclass
class PackedKeyPayload:
    indices: torch.Tensor
    norms: torch.Tensor
    qjl_signs: torch.Tensor | None
    qjl_gamma: torch.Tensor | None
    logical_tokens: int
    _unpack: Callable[[], torch.Tensor] | None = field(
        default=None, repr=False, compare=False
    )

    def physical_nbytes(self) -> int:
        return sum(
            tensor_nbytes(tensor)
            for tensor in (self.indices, self.norms, self.qjl_signs, self.qjl_gamma)
            if tensor is not None
        )

    def unpack_reference(self) -> torch.Tensor:
        """Reconstruct just this payload for CPU reference tests and attention."""
        if self._unpack is None:
            raise RuntimeError("payload has no reference decoder.")
        return self._unpack()

    def unpack_reference_fp32(self) -> torch.Tensor:
        """Return the explicit fp32 representation used by reference attention."""
        return self.unpack_reference().float()


@dataclass
class PackedValuePayload:
    indices: torch.Tensor
    mins: torch.Tensor
    scales: torch.Tensor
    logical_tokens: int
    _unpack: Callable[[], torch.Tensor] | None = field(
        default=None, repr=False, compare=False
    )

    def physical_nbytes(self) -> int:
        return tensor_nbytes(self.indices) + tensor_nbytes(self.mins) + tensor_nbytes(
            self.scales
        )

    def unpack_reference(self) -> torch.Tensor:
        """Reconstruct just this payload for CPU reference tests and attention."""
        if self._unpack is None:
            raise RuntimeError("payload has no reference decoder.")
        return self._unpack()


class KeyCodec:
    """Lloyd-Max key codec with physical index packing and optional packed QJL."""

    def __init__(
        self,
        layout: KeyLayout,
        *,
        seed: int = 42,
        qjl_dim: int | None = None,
        qjl_seed: int = 12345,
    ) -> None:
        if not isinstance(layout, KeyLayout):
            raise TypeError("layout must be a KeyLayout.")
        _require_int(seed, "seed", minimum=0)
        if qjl_dim is not None:
            _require_int(qjl_dim, "qjl_dim", minimum=1)
        _require_int(qjl_seed, "qjl_seed", minimum=0)
        self.layout = layout
        self.mse_bits = layout.bits - 1 if layout.qjl_enabled else layout.bits
        self.qjl_dim = qjl_dim if qjl_dim is not None else layout.head_dim
        self._rotation = generate_rotation_matrix(layout.head_dim, seed=seed, device="cpu")
        self._centroids, self._boundaries = get_codebook_tensors(
            layout.head_dim, self.mse_bits, torch.device("cpu")
        )
        self._qjl_projection = (
            generate_qjl_matrix(layout.head_dim, self.qjl_dim, seed=qjl_seed)
            if layout.qjl_enabled
            else None
        )

    def _on_device(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        return self._rotation.to(device), self._centroids.to(device)

    def quantize_indices(self, keys: torch.Tensor) -> torch.Tensor:
        """Return unpacked discrete Lloyd-Max assignments for test comparison."""
        _validate_kv_tensor(keys, self.layout.head_dim, "keys")
        rotation = self._rotation.to(keys.device)
        boundaries = self._boundaries.to(keys.device)
        flat = keys.reshape(-1, self.layout.head_dim).float()
        norms = torch.linalg.vector_norm(flat, dim=-1, keepdim=True)
        rotated = (flat / (norms + 1e-8)) @ rotation.T
        return torch.searchsorted(boundaries[1:-1].contiguous(), rotated.contiguous()).to(
            torch.uint8
        )

    def rotate_reference_vectors(self, vectors: torch.Tensor) -> torch.Tensor:
        """Rotate fp16/bf16/fp32 query or decoded-key vectors into fp32 space."""
        if not isinstance(vectors, torch.Tensor):
            raise TypeError("vectors must be a torch.Tensor.")
        if (
            vectors.ndim != 4
            or vectors.shape[0] != 1
            or vectors.shape[-1] != self.layout.head_dim
        ):
            raise ValueError(
                "vectors must have shape [1, heads, tokens, "
                f"{self.layout.head_dim}], got {tuple(vectors.shape)}."
            )
        if vectors.dtype not in _FLOAT_DTYPES:
            raise TypeError("vectors must use fp16, bf16, or fp32.")
        return vectors.float() @ self._rotation.to(vectors.device, dtype=torch.float32).T

    def rotate_query_reference(self, query: torch.Tensor) -> torch.Tensor:
        """Compatibility wrapper for rotating a query into key-codec space."""
        return self.rotate_reference_vectors(query)

    def encode(self, keys: torch.Tensor) -> PackedKeyPayload:
        heads, tokens = _validate_kv_tensor(keys, self.layout.head_dim, "keys")
        output_dtype = keys.dtype
        device = keys.device
        flat = keys.reshape(-1, self.layout.head_dim).float()
        norms = torch.linalg.vector_norm(flat, dim=-1)
        unit = flat / (norms.unsqueeze(-1) + 1e-8)
        rotation, centroids = self._on_device(device)
        boundaries = self._boundaries.to(device)
        rotated = unit @ rotation.T
        raw_indices = torch.searchsorted(
            boundaries[1:-1].contiguous(), rotated.contiguous()
        ).to(torch.uint8)
        packed_indices = pack_indices(raw_indices, self.mse_bits)
        stored_norms = norms.to(self.layout.norm_dtype)

        packed_signs: torch.Tensor | None = None
        gamma: torch.Tensor | None = None
        if self.layout.qjl_enabled:
            assert self._qjl_projection is not None
            residual = unit - (centroids[raw_indices.long()] @ rotation)
            qjl_projection = self._qjl_projection.to(device)
            signs = torch.where(
                residual @ qjl_projection.T >= 0,
                torch.ones((), dtype=torch.int8, device=device),
                -torch.ones((), dtype=torch.int8, device=device),
            ).reshape(heads, tokens * self.qjl_dim)
            packed_signs = pack_sign_bits(signs)
            gamma = torch.linalg.vector_norm(residual, dim=-1).to(self.layout.norm_dtype)

        def unpack() -> torch.Tensor:
            decoded_indices = unpack_indices(
                packed_indices, self.mse_bits, self.layout.head_dim
            ).long()
            reconstructed = centroids[decoded_indices] @ rotation
            reconstructed = reconstructed * stored_norms.float().unsqueeze(-1)
            if packed_signs is not None and gamma is not None:
                assert self._qjl_projection is not None
                qjl_signs = unpack_sign_bits(packed_signs, tokens * self.qjl_dim).reshape(
                    -1, self.qjl_dim
                ).float()
                correction = qjl_signs @ self._qjl_projection.to(device)
                reconstructed = reconstructed + (
                    (torch.pi / 2.0) ** 0.5
                    / self.qjl_dim
                    * gamma.float().unsqueeze(-1)
                    * correction
                    * stored_norms.float().unsqueeze(-1)
                )
            return reconstructed.reshape(1, heads, tokens, self.layout.head_dim).to(
                output_dtype
            )

        return PackedKeyPayload(
            indices=packed_indices,
            norms=stored_norms,
            qjl_signs=packed_signs,
            qjl_gamma=gamma,
            logical_tokens=tokens,
            _unpack=unpack,
        )


class ValueCodec:
    """Per-group value quantizer with packed discrete index payloads."""

    def __init__(self, layout: ValueLayout) -> None:
        if not isinstance(layout, ValueLayout):
            raise TypeError("layout must be a ValueLayout.")
        self.layout = layout
        self._levels = (1 << layout.bits) - 1

    def encode(self, values: torch.Tensor) -> PackedValuePayload:
        heads, tokens = _validate_kv_tensor(values, self.layout.head_dim, "values")
        output_dtype = values.dtype
        flat = values.reshape(-1, self.layout.head_dim)
        groups = self.layout.head_dim // self.layout.group_size
        grouped = flat.reshape(-1, groups, self.layout.group_size)
        mins = grouped.amin(dim=-1)
        maxs = grouped.amax(dim=-1)
        scales = ((maxs - mins) / self._levels).clamp_min(1e-8)
        raw_indices = torch.round(
            (grouped - mins.unsqueeze(-1)) / scales.unsqueeze(-1)
        ).clamp_(0, self._levels).to(torch.uint8)
        packed_indices = pack_indices(
            raw_indices.reshape(-1, self.layout.head_dim), self.layout.bits
        )
        stored_mins = mins.to(self.layout.metadata_dtype)
        stored_scales = scales.to(self.layout.metadata_dtype)

        def unpack() -> torch.Tensor:
            decoded = unpack_indices(
                packed_indices, self.layout.bits, self.layout.head_dim
            ).reshape(-1, groups, self.layout.group_size)
            reconstructed = stored_mins.unsqueeze(-1) + decoded.to(
                stored_mins.dtype
            ) * stored_scales.unsqueeze(-1)
            return reconstructed.reshape(1, heads, tokens, self.layout.head_dim).to(
                output_dtype
            )

        return PackedValuePayload(
            indices=packed_indices,
            mins=stored_mins,
            scales=stored_scales,
            logical_tokens=tokens,
            _unpack=unpack,
        )
