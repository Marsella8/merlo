from pathlib import Path
import struct

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_BF16 = "HuggingFaceTB/SmolLM2-135M-Instruct"

ROOT_DIR = Path(__file__).parent.parent
WEIGHTS_DIR = ROOT_DIR / "weights"
HEAD_DIR = WEIGHTS_DIR / "head"
PIPE_DIR = WEIGHTS_DIR / "pipe"
MAIN_DIR = WEIGHTS_DIR / "main"
HEAD_INITRAMFS_PATH = HEAD_DIR / "weights.bin"
PIPE_INITRAMFS_PATH = PIPE_DIR / "weights.bin"
MAIN_INITRAMFS_PATH = MAIN_DIR / "weights.bin"

TOKEN_SIZE_BYTES = 256
Q4_ROWS_PER_PANEL = 16
Q4_COLS_PER_BLOCK = 32
Q4_SECTION_BYTES = 64
Q4_BLOCK_BYTES = 320
HIDDEN_SIZE = 576
INTERMEDIATE_SIZE = 1536
HEAD_DIM = 64
NUM_Q_HEADS = 9
NUM_KV_HEADS = 3
KV_SIZE = NUM_KV_HEADS * HEAD_DIM
VOCAB_SIZE = 49152
NUM_LAYERS = 30


def build_byte_level_decoder():
    bs = list(range(ord("!"), ord("~") + 1))
    bs += list(range(ord("¡"), ord("¬") + 1))
    bs += list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


def flatten_fp32_bytes(tensor: torch.Tensor, *, transpose: bool = False) -> bytes:
    array = tensor.detach().to(torch.float32).cpu().numpy()
    if transpose:
        array = array.T
    array = np.ascontiguousarray(array.reshape(-1), dtype="<f4")
    return array.tobytes()


def ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def q4_bytes(rows: int, cols: int) -> int:
    return ceil_div(rows, Q4_ROWS_PER_PANEL) * ceil_div(cols, Q4_COLS_PER_BLOCK) * Q4_BLOCK_BYTES


def quantize_q4_0_block(block: np.ndarray) -> tuple[np.float32, np.ndarray]:
    assert block.shape == (Q4_COLS_PER_BLOCK,)
    block = block.astype(np.float32, copy=False)

    max_abs = float(np.max(np.abs(block)))
    if max_abs == 0.0:
        scale = np.float32(0.0)
        quants = np.full(Q4_COLS_PER_BLOCK, 8, dtype=np.uint8)
        return scale, quants

    scale = np.float32(max_abs / 7.0)
    quants = np.clip(np.rint(block / scale) + 8.0, 0.0, 15.0).astype(np.uint8)
    return scale, quants


def pack_q4_0_retiled(array: np.ndarray) -> bytes:
    if array.ndim != 2:
        raise ValueError(f"expected 2D tensor, got shape {array.shape}")

    rows, cols = array.shape
    panels = ceil_div(rows, Q4_ROWS_PER_PANEL)
    blocks_per_panel = ceil_div(cols, Q4_COLS_PER_BLOCK)
    out = bytearray(q4_bytes(rows, cols))

    for panel in range(panels):
        row_base = panel * Q4_ROWS_PER_PANEL
        for block in range(blocks_per_panel):
            col_base = block * Q4_COLS_PER_BLOCK
            block_base = (panel * blocks_per_panel + block) * Q4_BLOCK_BYTES

            for lane in range(Q4_ROWS_PER_PANEL):
                row = row_base + lane
                if row < rows:
                    block_vals = np.zeros(Q4_COLS_PER_BLOCK, dtype=np.float32)
                    valid_cols = min(Q4_COLS_PER_BLOCK, cols - col_base)
                    if valid_cols > 0:
                        block_vals[:valid_cols] = array[row, col_base:col_base + valid_cols]
                    scale, quants = quantize_q4_0_block(block_vals)
                else:
                    scale = np.float32(0.0)
                    quants = np.zeros(Q4_COLS_PER_BLOCK, dtype=np.uint8)

                scale_offset = block_base + lane * 4
                out[scale_offset:scale_offset + 4] = struct.pack("<f", float(scale))

                for word in range(4):
                    packed = 0
                    word_quants = quants[word * 8:(word + 1) * 8]
                    for i, q in enumerate(word_quants):
                        packed |= (int(q) & 0xF) << (4 * i)
                    word_offset = block_base + Q4_SECTION_BYTES * (1 + word) + lane * 4
                    out[word_offset:word_offset + 4] = struct.pack("<I", packed)

    return bytes(out)


def qweight_bytes(tensor: torch.Tensor, *, transpose: bool = False) -> bytes:
    array = tensor.detach().to(torch.float32).cpu().numpy()
    if transpose:
        array = array.T
    array = np.ascontiguousarray(array, dtype=np.float32)
    return pack_q4_0_retiled(array)


def load_vocab(tokenizer) -> bytes:
    byte_decoder = {v: k for k, v in build_byte_level_decoder().items()}
    vocab = tokenizer.get_vocab()
    sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])

    out = bytearray()
    for token, _ in sorted_vocab:
        if token in tokenizer.all_special_tokens:
            token_bytes = token.encode("utf-8")
        else:
            token_bytes = bytes(byte_decoder[c] for c in token)
        if b"\0" in token_bytes and token_bytes != b"\0":
            raise ValueError(f"token contains embedded NUL bytes: {token!r}")
        if len(token_bytes) > TOKEN_SIZE_BYTES:
            raise ValueError(f"token too large: {token!r}")
        out.extend(token_bytes.ljust(TOKEN_SIZE_BYTES, b"\0")) # 0-pad
    return bytes(out)


def load_lm_head_weight(state_dict) -> bytes:
    if "lm_head.weight" not in state_dict:
        return qweight_bytes(state_dict["model.embed_tokens.weight"])

    lm_head = state_dict["lm_head.weight"]
    embed = state_dict.get("model.embed_tokens.weight")
    if embed is not None and not torch.equal(lm_head, embed):
        raise ValueError("lm_head.weight and model.embed_tokens.weight differ")
    return qweight_bytes(lm_head)


def load_norm_weight(weight: torch.Tensor) -> bytes:
    return flatten_fp32_bytes(weight)


def load_attention(state_dict, layer: int) -> bytes:
    prefix = f"model.layers.{layer}.self_attn"
    return b"".join([
        qweight_bytes(state_dict[f"{prefix}.q_proj.weight"]),
        qweight_bytes(state_dict[f"{prefix}.k_proj.weight"]),
        qweight_bytes(state_dict[f"{prefix}.v_proj.weight"]),
        qweight_bytes(state_dict[f"{prefix}.o_proj.weight"]),
    ])


def load_mlp(state_dict, layer: int) -> bytes:
    prefix = f"model.layers.{layer}.mlp"
    return b"".join([
        qweight_bytes(state_dict[f"{prefix}.gate_proj.weight"]),
        qweight_bytes(state_dict[f"{prefix}.up_proj.weight"]),
        qweight_bytes(state_dict[f"{prefix}.down_proj.weight"]),
    ])


def load_layer(state_dict, layer: int) -> bytes:
    prefix = f"model.layers.{layer}"
    return b"".join([
        load_norm_weight(state_dict[f"{prefix}.input_layernorm.weight"]),
        load_attention(state_dict, layer),
        load_norm_weight(state_dict[f"{prefix}.post_attention_layernorm.weight"]),
        load_mlp(state_dict, layer),
    ])


def load_head(state_dict, tokenizer) -> bytes:
    payload = b"".join([
        load_vocab(tokenizer),
        load_lm_head_weight(state_dict),
        load_norm_weight(state_dict["model.norm.weight"]),
    ])
    expected = TOKEN_SIZE_BYTES * VOCAB_SIZE + q4_bytes(HIDDEN_SIZE, VOCAB_SIZE) + (HIDDEN_SIZE * 4)
    if len(payload) != expected:
        raise ValueError(f"head initramfs size mismatch: got {len(payload)}, expected {expected}")
    return payload


def load_pipe(state_dict, num_layers: int) -> bytes:
    if num_layers != NUM_LAYERS:
        raise ValueError(f"expected {NUM_LAYERS} layers, got {num_layers}")

    payload = b"".join(load_layer(state_dict, layer) for layer in range(num_layers))
    layer_size = (
        (HIDDEN_SIZE * 4)
        + q4_bytes(HIDDEN_SIZE, HIDDEN_SIZE)
        + q4_bytes(HIDDEN_SIZE, KV_SIZE)
        + q4_bytes(HIDDEN_SIZE, KV_SIZE)
        + q4_bytes(HIDDEN_SIZE, HIDDEN_SIZE)
        + (HIDDEN_SIZE * 4)
        + q4_bytes(HIDDEN_SIZE, INTERMEDIATE_SIZE)
        + q4_bytes(HIDDEN_SIZE, INTERMEDIATE_SIZE)
        + q4_bytes(INTERMEDIATE_SIZE, HIDDEN_SIZE)
    )
    expected = num_layers * layer_size
    if len(payload) != expected:
        raise ValueError(f"pipe initramfs size mismatch: got {len(payload)}, expected {expected}")
    return payload


def load_main(state_dict, tokenizer, num_layers: int) -> bytes:
    return load_head(state_dict, tokenizer) + load_pipe(state_dict, num_layers)


def write_initramfs(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    print(f"wrote {path} ({len(payload)} bytes)")


def save_initramfs() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_BF16)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_BF16,
        torch_dtype=torch.float32,
        device_map="cpu",
    )
    state_dict = model.state_dict()

    write_initramfs(HEAD_INITRAMFS_PATH, load_head(state_dict, tokenizer))
    write_initramfs(PIPE_INITRAMFS_PATH, load_pipe(state_dict, model.config.num_hidden_layers))
    write_initramfs(MAIN_INITRAMFS_PATH, load_main(state_dict, tokenizer, model.config.num_hidden_layers))


if __name__ == "__main__":
    save_initramfs()
