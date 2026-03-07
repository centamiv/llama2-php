# llama2-php

A minimalist, dependency-free inference engine for Llama 2 models written in **pure PHP**.

This project is a 1:1 port of Andrej Karpathy's [llama2.c](https://github.com/karpathy/llama2.c). It implements the Transformer architecture (inference only) in roughly 400 lines of PHP, demonstrating that even a "web-focused" language can run Large Language Models if you are brave enough to handle the math manually.

## Why?

- **Educational**: Understand how Transformers work (Attention, RMSNorm, RoPE, SwiGLU) by reading clean, procedural PHP code.
- **Proof of Concept**: See how `SplFixedArray` and binary unpacking can be used to perform heavy lifting in a language not known for matrix multiplications.
- **Zero Dependencies**: No `composer install`, no C extensions, no external APIs. Just `php run.php`.

## Technical Challenges

The primary obstacle in porting a C-based inference engine to PHP is the language's fundamental design as a high-level, memory-managed scripting environment. Unlike C, where memory is a flat buffer of bytes, PHP handles data through complex internal structures. To make this work, we had to bypass standard PHP conventions.

**Memory Management** was the first major hurdle. A standard PHP array is an associative hash map, which is incredibly versatile but carries a massive memory overhead. Loading a 15M parameter model—which is just 60MB of raw floating-point numbers—would typically balloon to over 1GB of RAM using standard arrays. We overcame this by employing `SplFixedArray`. This specialized class provides a much more compact memory footprint, behaving like a traditional C array, which allowed us to keep the peak memory usage for the TinyStories-15M model around 250MB.

**Binary Data Handling** posed the second challenge. To load the model weights, we had to precisely read 4-byte floats from a binary checkpoint. We achieved this using a combination of `fopen`, `fread`, and `unpack("f*")`. By reading the weights in calibrated chunks, we prevented memory spikes and successfully mapped the binary data into our `TransformerWeights` structure, effectively simulating pointer-based memory access.

Finally, we had to address **Execution Speed**. PHP is an interpreted language, and running deep nested `for` loops for Matrix Multiplications (MatMul) is inherently slower than using optimized BLAS libraries or compiled C code. However, by adhering to a strictly procedural implementation and avoiding the overhead of objects within the hot loops, we achieved a surprisingly interactive throughput. On a modern processor, the engine generates text at a rate that remains comfortable for human reading, proving that the overhead of the interpreter doesn't necessarily prohibit sophisticated mathematical models.

## How the Transformer Works

If you look at `run.php`, you'll see a sequence of mathematical operations. Here is how they translate to "thinking" in the model's world:

1.  **The Vocabulary Lookup (Embeddings):** Everything starts with a single number (the token ID). The model doesn't know what "cat" is, but it has a massive table (`token_embedding_table`) where every word ID corresponds to a long list of numbers (a vector). This vector represents the "meaning" of the word in a high-dimensional space.
2.  **Focusing Attention (Self-Attention):** This is the heart of the Transformer. The model doesn't just look at the current word; it looks back at everything it has already generated. It calculates a "score" for every previous word to decide which ones are relevant right now. In PHP, we do this by multiplying the Query (what I'm looking for), Key (what others offer), and Value (the actual content) vectors.
3.  **Knowing the Order (RoPE):** Words have different meanings depending on their position. "The dog bit the man" is different from "The man bit the dog." We use **Rotary Positional Embeddings (RoPE)** to "twist" the vectors mathematically so the model knows where each word sits in the sequence.
4.  **Refining the Thought (FFN):** After the model has looked at the context via Attention, it passes the result through a **Feed-Forward Network**. Think of this as a logical filter that processes the context and refines the prediction. We use the **SiLU** activation function here to let the model decide which information is important enough to pass forward.
5.  **The Final Choice (Softmax):** At the very end, the model produces a list of "raw scores" (logits) for every possible word in its vocabulary (32,000+ words). We apply **Softmax** to turn these scores into probabilities. If the word "Once" has a 90% probability, the model is very likely to pick it.

## Getting Started

### 1. Download the model and tokenizer
You need the model weights (the `.bin` file) and the tokenizer. You can get them from the original `llama2.c` repo or directly:

```bash
# Download a 15M parameter model (approx 60MB)
curl -L https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin -o stories15M.bin

# Download the tokenizer
curl -s https://raw.githubusercontent.com/karpathy/llama2.c/master/tokenizer.bin -o tokenizer.bin
```

### 2. Run Inference
Use the PHP CLI to start generating text:

```bash
php run.php stories15M.bin 0.9 100 "Once upon a time, a PHP script"
```

**Arguments**:
1. `checkpoint.bin`: Path to the model file.
2. `temperature`: (Optional) Sampling temperature (e.g., 0.9). Default is 0.9.
3. `steps`: (Optional) Number of tokens to generate. Default is 256.
4. `prompt`: (Optional) The starting text for the model.

## Performance Benchmark

The following metrics were captured on an **Apple M3** processor:

```text
--- PHP Llama2 Benchmark ---
Model Load Time : 0.208 s
Generation Time : 6.139 s
Tokens Processed: 30
Throughput      : 4.89 tok/s
Peak Memory     : 249.25 MB
----------------------------
```

## Implementation Details
- **Architecture**: Llama 2 (Transformer)
- **Activations**: SiLU (SwiGLU)
- **Normalization**: RMSNorm
- **Positional Embeddings**: RoPE (Rotary Positional Embeddings)
- **Tokenizer**: Simple BPE (Byte Pair Encoding)

## Credits
- **Andrej Karpathy** for the original [llama2.c](https://github.com/karpathy/llama2.c).
- The Llama 2 model by **Meta AI**.

## License
MIT
