"""Domain errors raised before an unsupported cache path can mutate state."""


class UnsupportedCapabilityError(ValueError):
    """Raised when a requested model, profile, or decode mode is unsupported."""
