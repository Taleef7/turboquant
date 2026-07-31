"""Paper-faithful CPU reference codec for TurboQuant product quantization."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from core.packing import pack_sign_bits, tensor_nbytes, unpack_sign_bits
from core.paper_mse import (
    TurboQuantMSEConfig,
    TurboQuantMSEEncoded,
    TurboQuantMSEReference,
)


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


def _validate_seed(value: object, name: str) -> None:
    seed = _require_builtin_int(value, name)
    if not _MIN_TORCH_SEED <= seed <= _MAX_TORCH_SEED:
        raise ValueError(
            f"{name} must be in the inclusive range "
            f"[{_MIN_TORCH_SEED}, {_MAX_TORCH_SEED}]"
        )


@dataclass(frozen=True)
class TurboQuantProductConfig:
    """Configuration for deterministic paper product quantization."""

    dimension: int
    total_bits: int
    rotation_seed: int
    projection_seed: int

    def __post_init__(self) -> None:
        _require_builtin_int(self.dimension, "dimension", minimum=2)
        _require_builtin_int(self.total_bits, "total_bits", minimum=2)
        if self.total_bits > 9:
            raise ValueError("total_bits must be at most 9")
        _validate_seed(self.rotation_seed, "rotation_seed")
        _validate_seed(self.projection_seed, "projection_seed")


@dataclass(frozen=True)
class TurboQuantProductEncoded:
    """Composed MSE payload, packed QJL signs, and float32 residual norm."""

    mse: TurboQuantMSEEncoded
    residual_signs: torch.Tensor
    residual_norm: torch.Tensor
    config: TurboQuantProductConfig


class TurboQuantProductReference:
    """Deterministic CPU oracle for Algorithm 2's product codec."""

    def __init__(self, config: TurboQuantProductConfig) -> None:
        if type(config) is not TurboQuantProductConfig:
            raise TypeError("config must be a TurboQuantProductConfig")
        self.config = config
        self._mse_config = TurboQuantMSEConfig(
            dimension=config.dimension,
            bits=config.total_bits - 1,
            rotation_seed=config.rotation_seed,
        )
        self._mse = TurboQuantMSEReference(self._mse_config)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(config.projection_seed)
        self._projection = torch.randn(
            (config.dimension, config.dimension),
            generator=generator,
            dtype=torch.float64,
            device="cpu",
        )

    def encode(self, vectors: torch.Tensor) -> TurboQuantProductEncoded:
        """Encode the MSE reconstruction residual with one-bit QJL signs."""

        single, rows = self._validate_rows(vectors, "vectors")
        mse = self._mse.encode(vectors)
        mse_decoded = self._mse.decode(mse)
        mse_rows = mse_decoded.reshape(1, -1) if single else mse_decoded
        residual = rows - mse_rows
        residual_norms = torch.linalg.vector_norm(residual, dim=-1)
        if not bool(torch.isfinite(residual_norms).all().item()):
            raise ValueError("residuals must produce only finite norms")
        stored_norms = residual_norms.to(torch.float32)
        if not bool(torch.isfinite(stored_norms).all().item()):
            raise ValueError(
                "residual norms must be representable as finite float32 values"
            )

        projected = residual @ self._projection.T
        signs = torch.where(
            projected >= 0,
            torch.ones((), dtype=torch.int8),
            -torch.ones((), dtype=torch.int8),
        )
        packed_signs = pack_sign_bits(signs)
        if single:
            packed_signs = packed_signs[0]
            stored_norms = stored_norms[0]
        return TurboQuantProductEncoded(
            mse=mse,
            residual_signs=packed_signs,
            residual_norm=stored_norms,
            config=self.config,
        )

    def decode(self, encoded: TurboQuantProductEncoded) -> torch.Tensor:
        """Decode a validated physical payload to CPU float64 values."""

        packed_signs, residual_norms, single, mse_decoded = (
            self._validate_encoded(encoded)
        )
        signs = unpack_sign_bits(
            packed_signs,
            self.config.dimension,
        ).to(torch.float64)
        residual = (
            math.sqrt(math.pi / 2)
            / self.config.dimension
            * residual_norms.to(torch.float64).unsqueeze(-1)
            * (signs @ self._projection)
        )
        mse_rows = mse_decoded.reshape(1, -1) if single else mse_decoded
        reconstructed = mse_rows + residual
        return reconstructed[0] if single else reconstructed

    def estimate_inner_product(
        self,
        queries: torch.Tensor,
        encoded: TurboQuantProductEncoded,
    ) -> torch.Tensor:
        """Estimate all query/product-vector inner products."""

        query_single, query_rows = self._validate_rows(queries, "queries")
        decoded = self.decode(encoded)
        encoded_single = decoded.ndim == 1
        decoded_rows = decoded.reshape(1, -1) if encoded_single else decoded
        estimates = query_rows @ decoded_rows.T
        if query_single and encoded_single:
            return estimates[0, 0]
        if query_single:
            return estimates[0]
        if encoded_single:
            return estimates[:, 0]
        return estimates

    def physical_nbytes(self, encoded: TurboQuantProductEncoded) -> int:
        """Return only the physically stored per-vector payload bytes."""

        self._validate_encoded(encoded)
        return (
            self._mse.physical_nbytes(encoded.mse)
            + tensor_nbytes(encoded.residual_signs)
            + tensor_nbytes(encoded.residual_norm)
        )

    def _validate_rows(
        self,
        values: torch.Tensor,
        name: str,
    ) -> tuple[bool, torch.Tensor]:
        if not isinstance(values, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if values.device.type != "cpu":
            raise ValueError(f"{name} must be a CPU tensor")
        if values.dtype not in _INPUT_DTYPES:
            raise TypeError(f"{name} must use torch.float32 or torch.float64")
        if values.ndim not in (1, 2):
            raise ValueError(f"{name} must have rank one or two")
        if values.shape[-1] != self.config.dimension:
            raise ValueError(
                f"{name} must have final dimension {self.config.dimension}"
            )
        if values.ndim == 2 and values.shape[0] == 0:
            raise ValueError(f"{name} must contain a non-empty batch")
        if not bool(torch.isfinite(values).all().item()):
            raise ValueError(f"{name} must contain only finite values")

        single = values.ndim == 1
        rows = values.reshape(1, -1) if single else values
        return single, rows.to(torch.float64)

    def _validate_encoded(
        self,
        encoded: TurboQuantProductEncoded,
    ) -> tuple[torch.Tensor, torch.Tensor, bool, torch.Tensor]:
        if type(encoded) is not TurboQuantProductEncoded:
            raise TypeError("encoded must be a TurboQuantProductEncoded")
        if type(encoded.config) is not TurboQuantProductConfig:
            raise TypeError("encoded config must be a TurboQuantProductConfig")
        if encoded.config != self.config:
            raise ValueError("encoded config does not match this codec")
        if type(encoded.mse) is not TurboQuantMSEEncoded:
            raise TypeError("encoded mse must be a TurboQuantMSEEncoded")
        if type(encoded.mse.config) is not TurboQuantMSEConfig:
            raise TypeError("encoded mse config must be a TurboQuantMSEConfig")
        if encoded.mse.config != self._mse_config:
            raise ValueError("encoded mse config does not match derived config")

        mse_decoded = self._mse.decode(encoded.mse)
        mse_single = mse_decoded.ndim == 1
        mse_batch = 1 if mse_single else mse_decoded.shape[0]

        signs = encoded.residual_signs
        if not isinstance(signs, torch.Tensor):
            raise TypeError("encoded residual signs must be a torch.Tensor")
        if signs.device.type != "cpu":
            raise ValueError("encoded residual signs must be a CPU tensor")
        if signs.dtype is not torch.uint8:
            raise TypeError("encoded residual signs must use torch.uint8")

        expected_bytes = math.ceil(self.config.dimension / 8)
        if signs.ndim == 1:
            signs_single = True
            if signs.shape[0] != expected_bytes:
                raise ValueError(
                    f"encoded residual signs must contain {expected_bytes} bytes"
                )
            packed_signs = signs.unsqueeze(0)
        elif signs.ndim == 2:
            signs_single = False
            if signs.shape[0] == 0:
                raise ValueError(
                    "encoded residual signs must contain a non-empty batch"
                )
            if signs.shape[1] != expected_bytes:
                raise ValueError(
                    "encoded residual signs final dimension must contain "
                    f"{expected_bytes} bytes"
                )
            packed_signs = signs
        else:
            raise ValueError("encoded residual signs must have rank one or two")

        used_bits = self.config.dimension % 8
        if used_bits and bool((packed_signs[:, -1] >> used_bits).any().item()):
            raise ValueError("encoded residual signs must have zero padding bits")

        residual_norm = encoded.residual_norm
        if not isinstance(residual_norm, torch.Tensor):
            raise TypeError("encoded residual norm must be a torch.Tensor")
        if residual_norm.device.type != "cpu":
            raise ValueError("encoded residual norm must be a CPU tensor")
        if residual_norm.dtype is not torch.float32:
            raise TypeError("encoded residual norm must use torch.float32")
        if signs_single:
            if residual_norm.ndim != 0:
                raise ValueError(
                    "encoded residual norm must be scalar for one vector"
                )
            residual_norms = residual_norm.reshape(1)
        else:
            if residual_norm.ndim != 1:
                raise ValueError(
                    "encoded residual norm must have rank one for a batch"
                )
            if residual_norm.shape[0] != packed_signs.shape[0]:
                raise ValueError(
                    "encoded residual norm batch must match residual signs batch"
                )
            residual_norms = residual_norm
        if not bool(torch.isfinite(residual_norms).all().item()):
            raise ValueError(
                "encoded residual norm must contain only finite values"
            )
        if bool((residual_norms < 0).any().item()):
            raise ValueError("encoded residual norm must be nonnegative")

        if signs_single != mse_single:
            raise ValueError("encoded mse and residual payload shapes must match")
        if packed_signs.shape[0] != mse_batch:
            raise ValueError("encoded mse and residual batch sizes must match")

        return packed_signs, residual_norms, signs_single, mse_decoded
