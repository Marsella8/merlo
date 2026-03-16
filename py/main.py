import os
import struct
import numpy as np
from pathlib import Path
from huggingface_hub import hf_hub_download
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_BF16 = "HuggingFaceTB/SmolLM2-135M-Instruct"
MODEL_INT8 = "onnx/model_int8.onnx"
MODEL_DIR = Path(__file__).parent.parent / "weights"
MODEL_DIR.mkdir(exist_ok=True)
VOCAB_PATH = MODEL_DIR / "vocab.txt"

def save_tokenizer():
    def bytes_to_unicode():
        bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
        cs = bs[:]
        n = 0
        for b in range(256):
            if b not in bs:
                bs.append(b)
                cs.append(256 + n)
                n += 1
        return dict(zip(bs, [chr(c) for c in cs]))

    byte_decoder = {v: k for k, v in bytes_to_unicode().items()}
    tok = AutoTokenizer.from_pretrained(MODEL_BF16)
    vocab = tok.get_vocab()
    sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])

    with open(VOCAB_PATH, "wb") as f:
        for token, _ in sorted_vocab:
            if token in tok.all_special_tokens:
                token_bytes = token.encode("utf-8")
            else:
                token_bytes = bytes([byte_decoder[c] for c in token])
            assert len(token_bytes) <= 256
            f.write(token_bytes.ljust(256, b"\0"))
    print("Saved tokenizer")

def save_fp32():
    def to_fp32_numpy(tensor):
        # BF16 tensors cannot always be exported directly with .numpy().
        return tensor.detach().to(torch.float32).cpu().numpy()

    def save_fp32_array(name, array):
        arr = np.asarray(array, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, arr.shape[0])
        elif arr.ndim != 2:
            raise ValueError(f"Expected 1D or 2D array for {name}, got {arr.ndim}D")
        arr = np.ascontiguousarray(arr)

        rows, cols = arr.shape
        path = MODEL_DIR / name
        with open(path, "wb") as f:
            f.write(struct.pack("QQ", rows, cols))
            arr.tofile(f)
        assert os.path.getsize(path) == 16 + rows * cols * 4

    def export_state_dict_to_c_format(state_dict, num_layers):
        # model.c expects token_embd as [hidden, vocab], then transposes at load time.
        save_fp32_array("token_embd", to_fp32_numpy(state_dict["model.embed_tokens.weight"].T))
        save_fp32_array("output_norm", to_fp32_numpy(state_dict["model.norm.weight"]))

        for layer in range(num_layers):
            prefix = f"model.layers.{layer}"
            save_fp32_array(f"{layer}.attn_norm", to_fp32_numpy(state_dict[f"{prefix}.input_layernorm.weight"]))
            save_fp32_array(f"{layer}.attn_q", to_fp32_numpy(state_dict[f"{prefix}.self_attn.q_proj.weight"].T))
            save_fp32_array(f"{layer}.attn_k", to_fp32_numpy(state_dict[f"{prefix}.self_attn.k_proj.weight"].T))
            save_fp32_array(f"{layer}.attn_v", to_fp32_numpy(state_dict[f"{prefix}.self_attn.v_proj.weight"].T))
            save_fp32_array(f"{layer}.attn_output", to_fp32_numpy(state_dict[f"{prefix}.self_attn.o_proj.weight"].T))
            save_fp32_array(f"{layer}.ffn_norm", to_fp32_numpy(state_dict[f"{prefix}.post_attention_layernorm.weight"]))
            save_fp32_array(f"{layer}.ffn_gate", to_fp32_numpy(state_dict[f"{prefix}.mlp.gate_proj.weight"].T))
            save_fp32_array(f"{layer}.ffn_up", to_fp32_numpy(state_dict[f"{prefix}.mlp.up_proj.weight"].T))
            save_fp32_array(f"{layer}.ffn_down", to_fp32_numpy(state_dict[f"{prefix}.mlp.down_proj.weight"].T))

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_BF16,
        dtype=torch.bfloat16,
        device_map="cpu",
    )
    export_state_dict_to_c_format(model.state_dict(), model.config.num_hidden_layers)
    print("Saved MODEL_BF16 weights in model.c format (fp32 files)")

def save_q8_0():
    model_path = hf_hub_download(
        repo_id=MODEL_BF16,
        filename=MODEL_INT8,
        local_dir=str(MODEL_DIR),
    )
    print(f"Saved MODEL_INT8 to: {model_path}")

if __name__ == "__main__":
    save_tokenizer()
    save_fp32()
