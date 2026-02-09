from consts import *
from llama_cpp import Llama


llm = Llama(model_path=str(MODEL_PATH), n_ctx = 2048)

for chunk in llm.create_completion(
    prompt=input('> '),
    max_tokens=512,
    temperature=0.7,
    stream=True,
):
    print(chunk["choices"][0]["text"], end="", flush=True)
