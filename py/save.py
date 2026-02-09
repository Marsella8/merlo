from huggingface_hub import hf_hub_download
from gguf import GGUFReader
import struct
import os
import numpy as np
from libellula import batch
from consts import *

def cleanup_name(name):
    return name.removeprefix("blk.").removesuffix(".weight")


def download():
    model_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=FILENAME,
        local_dir=str(MODEL_DIR)
    )
    
    print(f"Downloaded to: {model_path}")
    return model_path


'''
Format is:
rows: 8 bytes
cols: 8 bytes
data:
    for fp32, just 4 * rows * cols bytes
    for q8_0, first (rows * cols // 32) * 4 bytes are the fp32 scales, then rows * cols bytes are the int8 weights
'''

def save_fp32(tensor):
    r = 1
    c, *_ = tensor.shape
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
    
    for block in batch(data_bytes, 34):
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
    
    assert os.path.getsize(MODEL_DIR / cleanup_name(tensor.name)) == 16 + (rows * cols // 32) * 4 + rows * cols
    

def save(model_path):
    reader = GGUFReader(model_path)
    
    for tensor in reader.tensors:
        match tensor.tensor_type:
            case 0:
                save_fp32(tensor)
            case 8:
                save_q8_0(tensor)
            case _:
                raise ValueError(f"Unsupported tensor type: {tensor.tensor_type}")

if __name__ == "__main__":
    model_path = download()
    save(model_path)
