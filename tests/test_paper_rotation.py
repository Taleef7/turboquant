import pytest
import torch

from core.paper_rotation import seeded_haar_rotation


def test_seeded_haar_rotation_is_deterministic_and_orthogonal() -> None:
    rotation = seeded_haar_rotation(16, 1729)

    torch.testing.assert_close(
        rotation.T @ rotation,
        torch.eye(16, dtype=torch.float64),
        atol=1e-10,
        rtol=0,
    )
    torch.testing.assert_close(
        rotation,
        seeded_haar_rotation(16, 1729),
        atol=0,
        rtol=0,
    )
    assert torch.linalg.det(rotation).abs().item() == pytest.approx(
        1.0,
        abs=1e-10,
    )


def test_seeded_haar_rotation_uses_positive_qr_diagonal_convention() -> None:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(2027)
    gaussian = torch.randn(
        (8, 8),
        generator=generator,
        dtype=torch.float64,
        device="cpu",
    )

    rotation = seeded_haar_rotation(8, 2027)
    adjusted_r = rotation.T @ gaussian

    assert torch.all(torch.diagonal(adjusted_r) >= 0)
    assert torch.allclose(adjusted_r, torch.triu(adjusted_r), atol=1e-12, rtol=0)


def test_seeded_haar_rotation_changes_with_seed() -> None:
    first = seeded_haar_rotation(8, 17)
    second = seeded_haar_rotation(8, 18)

    assert not torch.equal(first, second)


def test_seeded_haar_rotation_returns_requested_floating_dtype() -> None:
    rotation = seeded_haar_rotation(4, 11, dtype=torch.float32)

    assert rotation.device.type == "cpu"
    assert rotation.dtype is torch.float32


@pytest.mark.parametrize("dimension", [True, 1, 3.5])
def test_seeded_haar_rotation_rejects_invalid_dimensions(
    dimension: object,
) -> None:
    with pytest.raises((TypeError, ValueError), match="dimension"):
        seeded_haar_rotation(dimension, 7)  # type: ignore[arg-type]


@pytest.mark.parametrize("seed", [True, 1.5, "7"])
def test_seeded_haar_rotation_rejects_non_integer_seeds(seed: object) -> None:
    with pytest.raises(TypeError, match="seed"):
        seeded_haar_rotation(4, seed)  # type: ignore[arg-type]


@pytest.mark.parametrize("dtype", [torch.bool, torch.int64])
def test_seeded_haar_rotation_rejects_non_floating_dtypes(
    dtype: torch.dtype,
) -> None:
    with pytest.raises(TypeError, match="dtype"):
        seeded_haar_rotation(4, 7, dtype=dtype)
