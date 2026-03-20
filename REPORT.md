# merlo - LLM

Memory: given that there is basically no branching or stuff like that, we can keep everything either pinned (e.g. the weights that just sit fixed in memory and are viewed by the Matrices that access them) or allocate the buffers on the stack. We have some allocations for very minor objects, but fills up the queue very slowly.

Model is made up of an embedding table, and a set of layers, and an lm_head (but note that embedding table and lm_head are weight tied, so we count them once). Also the LM head is pretty big (Vocab size * Hidden size). So we have:
- 1 Pi for the embedding table (does emebdding and "de-embedding")
- 1 Pi for all the layers.

The first pi passes the embedded token to the second pi, which passes it thjoruhg all the layers, and then passes itn to the first pi, which decodes it, gets the next token, which is embedded and passed to the second pi, ...

Note that there is no parallelism, but it allows us to store larger models than what could fit on a single Pi.

Quantization: we use 4Q_0 quantization, as a custom block format shaped like (...) because it fits better.

Kernels: