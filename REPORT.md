# merlo - Qwen0.8B


As in libraries like `numpy`, we model our matrices as a set of views (with an associated range, strides, ...) which access an underlying memory buffer (so e.g. X and tranpose(X) dont duplicate memory).
For the pipelined architecture, we have fan in and fan out one for each  so we acna just use the mini-uart for each.


Our inference spec is peculiar: we have multiple devices but a batch size of 1, and the pi does not have great banwdith and latency for cross pi communication. So we use pipeline parallelism, which enables us to store a larger model than what we would have on a single device. But by default, pipeline parallelism does not enable any speedup since at any point in time, only one stage of the pipeline is active. So we use speculative decoding. By default spec decoding generates a batch and then that batch is validated in parallel. But this does not solve our problem, so instead we change our spec decoding to be pipelined.


We do 4Q_0 quantization since this way we can reduce the memory pressure.
