"""Paper-faithful CPU reference codec for TurboQuant MSE quantization."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from core.packing import pack_indices, tensor_nbytes, unpack_indices
from core.paper_codebook import lloyd_max_sphere_codebook
from core.paper_rotation import seeded_haar_rotation


_INDEX_BITS = frozenset(range(1, 9))
_INPUT_DTYPES = frozenset({torch.float32, torch.float64})
_MIN_TORCH_SEED = -0x8000_0000_0000_0000
_MAX_TORCH_SEED = 0xFFFF_FFFF_FFFF_FFFF


def _require_builtin_int(
    value: object,
    name: str,
    *,
    minimum: int | None = None,
) -> int:
    if type(value) is not int:  # noqa: E721
        raise TypeError(f"{name} must be a built-in int")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


@dataclass(frozen=True)
class TurboQuantMSEConfig:
    """Configuration for deterministic paper MSE reference quantization."""

    dimension: int
    bits: int
    rotation_seed: int

    def __post_init__(self) -> None:
        _require_builtin_int(self.dimension, "dimension", minimum=2)
        _require_builtin_int(self.bits, "bits", minimum=1)
        if self.bits not in _INDEX_BITS:
            raise ValueError(f"bits must be one of {sorted(_INDEX_BITS)}")
        _require_builtin_int(self.rotation_seed, "rotation_seed")
        if not _MIN_TORCH_SEED <= self.rotation_seed <= _MAX_TORCH_SEED:
            raise ValueError(
                "rotation_seed must be in the inclusive range "
                f"[{_MIN_TORCH_SEED}, {_MAX_TORCH_SEED}]"
            )


@dataclass(frozen=True)
class TurboQuantMSEEncoded:
    """Physically packed indices and float32 norm payload.

    Finite norms too large for float32 storage are rejected. Tiny finite norms
    retain normal float32 casting semantics, including subnormal rounding or
    underflow to zero; no hidden higher-precision norm is retained.
    """

    indices: torch.Tensor
    norm: torch.Tensor
    config: TurboQuantMSEConfig


class TurboQuantMSEReference:
    """Deterministic CPU oracle for Algorithm 1's MSE codec."""

    def __init__(self, config: TurboQuantMSEConfig) -> None:
        if type(config) is not TurboQuantMSEConfig:
            raise TypeError("config must be a TurboQuantMSEConfig")
        self.config = config
        self._rotation = seeded_haar_rotation(
            config.dimension,
            config.rotation_seed,
        )
        self._codebook = lloyd_max_sphere_codebook(
            config.dimension,
            config.bits,
        )

    def encode(self, vectors: torch.Tensor) -> TurboQuantMSEEncoded:
        """Normalize, rotate, nearest-quantize, and physically pack vectors."""

        single, rows = self._validate_vectors(vectors)
        norms = torch.linalg.vector_norm(rows, dim=-1)
        if not bool(torch.isfinite(norms).all().item()):
            raise ValueError("vectors must produce only finite norms")
        stored_norms = norms.to(torch.float32)
        if not bool(torch.isfinite(stored_norms).all().item()):
            raise ValueError(
                "vector norms must be representable as finite float32 values"
            )
        nonzero = norms > 0
        unit = torch.zeros_like(rows)
        unit[nonzero] = rows[nonzero] / norms[nonzero].unsqueeze(-1)
        rotated = unit @ self._rotation.T
        raw_indices = (
            (rotated.unsqueeze(-1) - self._codebook)
            .abs()
            .argmin(dim=-1)
            .to(torch.uint8)
        )
        packed = pack_indices(raw_indices, self.config.bits)

        if single:
            packed = packed[0]
            stored_norms = stored_norms[0]
        return TurboQuantMSEEncoded(
            indices=packed,
            norm=stored_norms,
            config=self.config,
        )

    def decode(self, encoded: TurboQuantMSEEncoded) -> torch.Tensor:
        """Decode a validated physical payload to CPU float64 values."""

        packed, norms, single = self._validate_encoded(encoded)
        raw_indices = unpack_indices(
            packed,
            self.config.bits,
            self.config.dimension,
        )
        rotated = self._codebook[raw_indices.to(torch.int64)]
        reconstructed = rotated @ self._rotation
        scaled = reconstructed * norms.to(torch.float64).unsqueeze(-1)
        scaled = torch.where(
            norms.unsqueeze(-1) == 0,
            torch.zeros_like(scaled),
            scaled,
        )
        return scaled[0] if single else scaled

    def physical_nbytes(self, encoded: TurboQuantMSEEncoded) -> int:
        """Return payload bytes only: packed indices plus float32 stored norms.

        The rotation and codebook are shared constants and are not counted as
        per-vector physical payload.
        """

        self._validate_encoded(encoded)
        return tensor_nbytes(encoded.indices) + tensor_nbytes(encoded.norm)

    def _validate_vectors(
        self,
        vectors: torch.Tensor,
    ) -> tuple[bool, torch.Tensor]:
        if not isinstance(vectors, torch.Tensor):
            raise TypeError("vectors must be a torch.Tensor")
        if vectors.device.type != "cpu":
            raise ValueError("vectors must be a CPU tensor")
        if vectors.dtype not in _INPUT_DTYPES:
            raise TypeError("vectors must use torch.float32 or torch.float64")
        if vectors.ndim not in (1, 2):
            raise ValueError("vectors must have rank one or two")
        if vectors.shape[-1] != self.config.dimension:
            raise ValueError(
                f"vectors must have final dimension {self.config.dimension}"
            )
        if vectors.ndim == 2 and vectors.shape[0] == 0:
            raise ValueError("vectors must contain a non-empty batch")
        if not bool(torch.isfinite(vectors).all().item()):
            raise ValueError("vectors must contain only finite values")

        single = vectors.ndim == 1
        rows = vectors.reshape(1, -1) if single else vectors
        return single, rows.to(torch.float64)

    def _validate_encoded(
        self,
        encoded: TurboQuantMSEEncoded,
    ) -> tuple[torch.Tensor, torch.Tensor, bool]:
        if type(encoded) is not TurboQuantMSEEncoded:
            raise TypeError("encoded must be a TurboQuantMSEEncoded")
        if type(encoded.config) is not TurboQuantMSEConfig:
            raise TypeError("encoded config must be a TurboQuantMSEConfig")
        if encoded.config != self.config:
            raise ValueError("encoded config does not match this codec")

        indices = encoded.indices
        if not isinstance(indices, torch.Tensor):
            raise TypeError("encoded indices must be a torch.Tensor")
        if indices.device.type != "cpu":
            raise ValueError("encoded indices must be a CPU tensor")
        if indices.dtype is not torch.uint8:
            raise TypeError("encoded indices must use torch.uint8")

        expected_bytes = math.ceil(
            self.config.dimension * self.config.bits / 8
        )
        if indices.ndim == 1:
            single = True
            if indices.shape[0] != expected_bytes:
                raise ValueError(
                    f"encoded indices must contain {expected_bytes} bytes"
                )
            packed = indices.unsqueeze(0)
        elif indices.ndim == 2:
            single = False
            if indices.shape[0] == 0:
                raise ValueError("encoded indices must contain a non-empty batch")
            if indices.shape[1] != expected_bytes:
                raise ValueError(
                    "encoded indices final dimension must contain "
                    f"{expected_bytes} bytes"
                )
            packed = indices
        else:
            raise ValueError("encoded indices must have rank one or two")

        used_bits = (self.config.dimension * self.config.bits) % 8
        if used_bits and bool((packed[:, -1] >> used_bits).any().item()):
            raise ValueError("encoded indices must have zero padding bits")

        norm = encoded.norm
        if not isinstance(norm, torch.Tensor):
            raise TypeError("encoded norm must be a torch.Tensor")
        if norm.device.type != "cpu":
            raise ValueError("encoded norm must be a CPU tensor")
        if norm.dtype is not torch.float32:
            raise TypeError("encoded norm must use torch.float32")
        if single:
            if norm.ndim != 0:
                raise ValueError("encoded norm must be scalar for one vector")
            norms = norm.reshape(1)
        else:
            if norm.ndim != 1:
                raise ValueError("encoded norm must have rank one for a batch")
            if norm.shape[0] != packed.shape[0]:
                raise ValueError(
                    "encoded norm batch must match encoded indices batch"
                )
            norms = norm
        if not bool(torch.isfinite(norms).all().item()):
            raise ValueError("encoded norm must contain only finite values")
        if bool((norms < 0).any().item()):
            raise ValueError("encoded norm must be nonnegative")

        return packed, norms, single
