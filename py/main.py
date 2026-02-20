import os
import struct
import numpy as np
from pathlib import Path
from huggingface_hub import hf_hub_download
from gguf import GGUFReader
from transformers import AutoTokenizer
from libellula import batch

def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~")+1)) + list(range(ord("¡"), ord("¬")+1)) + list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))

REPO_ID = "QuantFactory/SmolLM2-135M-GGUF"
FILENAME = "SmolLM2-135M.Q8_0.gguf"
MODEL_DIR = Path(__file__).parent.parent / "model"
MODEL_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODEL_DIR / FILENAME
VOCAB_PATH = MODEL_DIR / "vocab.txt"

def cleanup_name(name):
    return name.removeprefix("blk.").removesuffix(".weight")

def save_tokenizer():
    byte_decoder = {v: k for k, v in bytes_to_unicode().items()}

    tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
    vocab = tok.get_vocab()
    sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
    
    with open(VOCAB_PATH, "wb") as f:
        for token, token_id in sorted_vocab:
            if token in tok.all_special_tokens:
                token_bytes = token.encode('utf-8')
            else:
                token_bytes = bytes([byte_decoder[c] for c in token])
            assert len(token_bytes) <= 256
            padded = token_bytes.ljust(256, b'\0')
            f.write(padded)
    print(f"Saved tokenizer")

def sample_tokenized_text():
    tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
    text = "C and its consequences have been disastrous for the human race"
    tokenized = tok.encode(text)
    print(tokenized)
    print([tok.decode([token]) for token in tokenized])

def save_fp32(tensor):
    r = 1
    c = tensor.shape[0]
    data = tensor.data.tobytes()
    assert len(data) == 4 * r * c
    with open(MODEL_DIR / cleanup_name(tensor.name), "wb") as f:
        f.write(struct.pack('QQ', r, c) + data)
    filesize = os.path.getsize(MODEL_DIR / cleanup_name(tensor.name))
    assert filesize == 16 + len(data)

def save_q8_0(tensor):
    rows, cols = tensor.shape
    data_bytes = tensor.data.tobytes()
    
    scales = []
    weights = []
    
    # GGUF Q8_0 block size is 34 bytes: 2 bytes for half-float scale, 32 bytes for int8 weights
    for block in batch(data_bytes, 34):
        # 'e' is half-precision float (16-bit)
        scale = struct.unpack('e', bytes(block[:2]))[0]
        scales.append(scale)
        
        int8_values = struct.unpack('32b', bytes(block[2:34]))
        weights.extend(int8_values)
    
    scales_array = np.array(scales, dtype=np.float32)
    weights_array = np.array(weights, dtype=np.int8)
    
    with open(MODEL_DIR / cleanup_name(tensor.name), "wb") as f:
        f.write(struct.pack('QQ', rows, cols))
        scales_array.tofile(f)
        weights_array.tofile(f)
    
    expected_size = 16 + (rows * cols // 32) * 4 + rows * cols
    assert os.path.getsize(MODEL_DIR / cleanup_name(tensor.name)) == expected_size

def download_and_convert():
    model_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=FILENAME,
        local_dir=str(MODEL_DIR)
    )
    print(f"Downloaded to: {model_path}")
    
    reader = GGUFReader(model_path)
    for tensor in reader.tensors:
        print(f"Processing {tensor.name} ({tensor.tensor_type})...")
        match tensor.tensor_type:
            case 0: # GGML_TYPE_F32
                save_fp32(tensor)
            case 8: # GGML_TYPE_Q8_0
                save_q8_0(tensor)
            case _:
                print(f"Skipping unsupported tensor type: {tensor.tensor_type}")

if __name__ == "__main__":
    save_tokenizer()
    # sample_tokenized_text()
