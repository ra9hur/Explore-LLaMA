# Explore LLaMA-2

Here is an attempt to understand how stable diffusion works through the working code. 
Thanks to [Umar Jamil](https://www.youtube.com/watch?v=oM4VmoabDAI) for creating this amazing video tutorial.

LLaMA-2 is not a single model, but rather a collection of four models. The only difference between each of these models is the number of parameters they contain. From smallest to largest, the LLaMA-2 models contain 7B, 13B, 34B, and 70B parameters. But otherwise, everything else about them from activation function to  normalization method is identical.

![overview](https://github.com/ra9hur/Explore-LLaMA-2/assets/17127066/ba0f140b-11f1-4d28-a54b-f7aa3d73d6ae)

The table above shows some notable differences between LLaMA and LLaMA-2. 

- LLaMA-2 is trained with a longer context length of 4K tokens (i.e., LLaMA was trained with a 2K context length). So, context length and supported tokens have doubled.
- LLaMA-2 uses Group Query Attention (GQA) in last two of its models.

## Architecture

![architecture](https://github.com/ra9hur/Explore-LLaMA-2/assets/17127066/b3fffe24-5bc5-4026-908f-ed7420ab47fb)

Below are a few architectural choices made in LLaMA-2.

### RMS Normalization

Most transformer architectures adopt layer normalization, which is applied after each layer within the transformer block; see above. LLaMA, however, replaces this with a variant called Root Mean Square Layer Normalization (or RMSNorm for short!), which is a simplified version of layer normalization that has been shown to improve training stability and generalization1. RMSNorm is formulated as shown below.

![rms_norm](https://github.com/ra9hur/Explore-LLaMA-2/assets/17127066/9fdcaeb4-ce4c-4bbc-a7aa-b83a452ed912)

For LLaMA, a pre-normalization variant of RMSNorm is adopted, meaning that normalization is applied prior to the major layers in a transformer block, rather than after, as shown in the architecture diagram above. 

### Rotary Positional Embeddings

Instead of using absolute or relative positional embeddings, LLaMA models adopt a Rotary Positional Embeddings (RoPE) scheme, which finds a balance between the absolute and relative position of each token in a sequence. This position embedding approach encodes absolute position with a rotation matrix and adds relative position information directly into the self-attention operation. 

![rotary1](https://github.com/ra9hur/Explore-LLaMA-2/assets/17127066/7b9082ea-a6a6-4a51-a6ab-b4ab73ef1084)

![rotary2](https://github.com/ra9hur/Explore-LLaMA-2/assets/17127066/5c0456c7-e8fe-4e81-8ca2-cca66155e40a)

![rotary3](https://github.com/ra9hur/Explore-LLaMA-2/assets/17127066/c19503c9-8ff0-44f9-8121-f0b5dacfb4ab)



The benefit of RoPE embeddings on tasks with longer sequence lengths has led this approach to be adopted by a variety of LLMs (e.g., PaLM and Falcon).


### KV Cache

In the original transformer, since the attention mechanism is causal (i.e., the attention of a token only depends on its preceding tokens), at each generation step we are recalculating the same previous token attention, when we actually just want to calculate the attention for the new token.

This is where KV cache comes into play. By caching the previous Keys and Values, we can focus on only calculating the attention for the new token.

![kv_cache](https://github.com/ra9hur/Explore-LLaMA-2/assets/17127066/49004624-4d2c-4604-9715-9e948770701a)

The matrices obtained with KV caching are way smaller, which leads to faster matrix multiplications leading to much less computations involved. The only downside is that it needs more GPU VRAM (or CPU RAM if GPU is not being used) to cache the Key and Value states.

### Grouped Query Attention

Before discussing Grouped Query Attention, let’s quickly review earlier attention mechanisms.

#### Multi-Head Attention(MHA)

Multi-head Attention is the default attention mechanism of the transformer model.

![mha1](https://github.com/ra9hur/Explore-LLaMA-2/assets/17127066/f8452ad2-6c1d-4b47-9032-679aa9893c3f)

However, there is an issue with auto-regressive language models based on transformer decoders when it comes to text generation.

During training, we have access to the true target sequence and can efficiently implement parallelism.

![mha2](https://github.com/ra9hur/Explore-LLaMA-2/assets/17127066/5ef25745-acbe-41cf-88d7-6d24598ea761)

However, during inference, each position’s query attends to all the key-value pairs generated at or before that position. In other words, the output of the self-attention layer at a specific position affects the generation of the next token. Due to the inability to perform parallel computation, decoding becomes slower.

#### Multi-Query Attention(MHA)

The approach of MQA is to keep the original number of heads for Q, but have only one head for K and V. This means that all the Q heads share the same set of K and V heads, hence the name Multi-Query.

![mqa1](https://github.com/ra9hur/Explore-LLaMA-2/assets/17127066/ecf8b141-9495-48bb-aa73-4c72bf67d8e8)

![mqa2](https://github.com/ra9hur/Explore-LLaMA-2/assets/17127066/f5420236-9ce6-419c-a12a-5d7d9724afc3)


#### Grouped Query Attention

![gqa](https://github.com/ra9hur/Explore-LLaMA-2/assets/17127066/5c1fe0b7-33b7-4707-bd55-63b97db4c0c0)


### SwiGLU Activation

LLaMA models adopt the SwiGLU activation function—as opposed to the standard ReLU function adopted by most neural networks—within their feed-forward layers. The SwiGLU activation function can be formulated as follows.

![SwiGLU](https://github.com/ra9hur/Explore-LLaMA-2/assets/17127066/ac1270a8-ebe6-46ae-8ef4-8f10a07856b5)

SwiGLU is an element-wise product of two linear transformations of the input x, one of which has had a Swish activation applied to it. This activation function requires four matrix multiplications (i.e., it is more computationally expensive than a normal activation function such as ReLU), but it has been found to yield improvements in performance relative to other activation functions, even when the amount of compute being used is held constant.


## References

1. [Paper - Llama 2: Open Foundation and Fine-Tuned Chat Models](https://arxiv.org/abs/2307.09288)

2. [Coding LLaMA 2 from scratch in PyTorch - KV Cache, Grouped Query Attention, Rotary PE, RMSNorm](https://www.youtube.com/watch?v=oM4VmoabDAI)

3. [Rotary Positional Embeddings: Combining Absolute and Relative](https://www.youtube.com/watch?v=o29P0Kpobz0)

4. [LLaMA-2 from the Ground Up](https://cameronrwolfe.substack.com/p/llama-2-from-the-ground-up)

5. [Transformers KV Caching Explained](https://medium.com/@joaolages/kv-caching-explained-276520203249)

6. [Multi-Query Attention Explained](https://pub.towardsai.net/multi-query-attention-explained-844dfc4935bf)