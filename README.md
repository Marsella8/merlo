In computation land, you have:

- The Turings: humans slaves to the machine (chud)

- The Churches: machine slave of the humans (chad)

uhhmmmm lets model our computation around how it's actually done on the hardware rather than some infinitely more elegant abstraction that reflects how computation is conceptually carried out.

ADTs? Generics? Never heard of that.

You can embed invariants through types in order to leverage the compiler for increased safety? hahaha we have void* buddy it's plenty already.

Oh wanna do a simple string operation? That will be 20 lines of code and 3 mallocs, please.

Hey dude so what is the size of that size_t* arr you got there? I don't know man, ask the other argument haha why would I know?

Go fuck yourself.

---

TODO:
- Add valgrind to check for mem leaks. Good thing is that our stuff is pretty branchless so it shouldn't be hard.

- Implement the engine and the rope + tests
- Implement pipelining on the UART link if needed.
- Implement Spec model plus inference on the Pi.

Make sure numerics are correctness.

Small subtlety about the spec decoding: we need to be able to roll back teh KV cache (aka resize it), this is what the resize_kv shit is for. prefill sets it to T, and decode increments it by 1, and when 

Possible perf improvements:
- Q8_0 and 1.58bit quantization
- if matrix contiguous, then a lot of ops can be much faster (less flops to compute indices).


Some notes:

softmax(Q @ K^T): mmmh i'll implement it as softmax(matmul(Q, transpose(K))). wrong!! the matmul is a new heap allocated matrix, so you need to make sure that we have a handle on it to free it later.

In general, make sure to understand whether things are views or copies, and make sure to free the latter (no RAII wallah we have to do this shit manually).