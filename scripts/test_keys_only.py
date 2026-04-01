"""
Debug: Cache that only compresses keys, not values.
Tests whether key compression alone causes quality issues.
"""

from __future__ import annotations
import torch
from typing import Optional, Tuple, List, Dict
from transformers import DynamicCache

import sys

sys.path.insert(0, "/home/taleef/projects/turboquant")

from core.turboquant_simple import TurboQuantMSE


class KeysOnlyCache(DynamicCache):
    """
    Test cache that only compresses keys.
    Values are stored in full precision.
    """

    def __init__(
        self,
        head_dim: int = 128,
        bits: int = 4,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float16,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.bits = bits
        self.device = device or torch.device("cuda")
        self.dtype = dtype

        self._key_quantizers: Dict[int, TurboQuantMSE] = {}
        self._compressed_keys: List[Optional[dict]] = []
        self._raw_values: List[Optional[torch.Tensor]] = []

    def _get_key_quantizer(self, layer_idx: int) -> TurboQuantMSE:
        if layer_idx not in self._key_quantizers:
            seed = 42 + layer_idx * 7
            self._key_quantizers[layer_idx] = TurboQuantMSE(
                head_dim=self.head_dim,
                bits=self.bits,
                seed=seed,
                device=str(self.device),
            )
        return self._key_quantizers[layer_idx]

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Ensure storage lists are long enough
        while len(self._compressed_keys) <= layer_idx:
            self._compressed_keys.append(None)
            self._raw_values.append(None)

        key_states = key_states.to(self.dtype)
        value_states = value_states.to(self.dtype)

        quantizer = self._get_key_quantizer(layer_idx)

        if self._compressed_keys[layer_idx] is None:
            # First call - compress keys
            self._compressed_keys[layer_idx] = quantizer.compress(key_states)
            self._raw_values[layer_idx] = value_states.clone()
        else:
            # Append - compress new keys and concat
            existing = self._compressed_keys[layer_idx]
            new_compressed = quantizer.compress(key_states)

            # Merge keys
            orig_shape = existing["shape"]
            new_shape = new_compressed["shape"]
            batch, heads, seq_old, d = orig_shape
            _, _, seq_new, _ = new_shape

            old_indices = existing["indices"].reshape(batch * heads, seq_old, -1)
            new_indices = new_compressed["indices"].reshape(batch * heads, seq_new, -1)
            merged_indices = torch.cat([old_indices, new_indices], dim=1).reshape(-1, d)

            old_norms = existing["norms"].reshape(batch * heads, seq_old)
            new_norms = new_compressed["norms"].reshape(batch * heads, seq_new)
            merged_norms = torch.cat([old_norms, new_norms], dim=1).reshape(-1)

            self._compressed_keys[layer_idx] = {
                "indices": merged_indices.to(torch.uint8),
                "norms": merged_norms.to(torch.float16),
                "shape": (batch, heads, seq_old + seq_new, d),
            }

            # Values stay raw
            self._raw_values[layer_idx] = torch.cat(
                [self._raw_values[layer_idx], value_states], dim=-2
            )

        # Decompress keys for output
        out_k = quantizer.decompress(self._compressed_keys[layer_idx]).to(self.dtype)
        out_v = self._raw_values[layer_idx]

        return out_k, out_v

    def get_seq_length(self, layer_idx: int = 0) -> int:
        if (
            layer_idx < len(self._compressed_keys)
            and self._compressed_keys[layer_idx] is not None
        ):
            return self._compressed_keys[layer_idx]["shape"][2]
        return 0

    def get_max_length(self) -> Optional[int]:
        return None


if __name__ == "__main__":
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    import time

    model_id = "Qwen/Qwen2.5-7B-Instruct"

    print(f"Loading {model_id}...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model.eval()

    prompt = "The capital of France is"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    # Baseline
    print("\n=== BASELINE ===")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=50, do_sample=False)
    print(tokenizer.decode(out[0], skip_special_tokens=True))

    # Keys-only compression
    print("\n=== KEYS-ONLY COMPRESSION ===")
    cache = KeysOnlyCache(head_dim=128, bits=4)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=50, do_sample=False, past_key_values=cache
        )
    print(tokenizer.decode(out[0], skip_special_tokens=True))
