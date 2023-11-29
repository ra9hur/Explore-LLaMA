import torch
import torch.nn as nn
from torch.nn import functional as F

import math
from dataclasses import dataclass
from typing import Optional


# class that represents parameters of the model
@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32                  # Nx encoder block repeats 32 times
    n_heads = 32                        # Multi query ?
    n_kv_heads: Optional[int] = None    # to be set based on group query attention
    vocab_size: int = -1                # To be set when we load tokenizer
    
    # below two two parameters indicate hidden dimensions of the FF
    # When the grouped query attention was introduced, overall # of parameters reduced
    # To maintain the # of parameters as in LLaMA 1, this is incremented in FF 
    # so that total # of parameters remain same
    # This was done for comparison and was an architectural decision 
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    
    norm_eps: float = 1e-5              # just to ensure, we never divide it by zero

    # Needed for KV cache
    max_batch_size: int = 32
    max_seq_len = 2048

    device: str = None



def precompute_theta_pos_frequencies(
        head_dim: int, 
        seq_len: int, 
        device: str, 
        theta: float = 10000.0):
    
    # As written in the paragraph 3.2.2 of the paper
    # >> In order to generalize our results in 2D to any xi ∈ Rd where **d is even**, [...]
    assert head_dim % 2 == 0, "Dimension must be divisible by 2"
    
    # Build the theta parameter
    # According to the formula theta_i = 10000^(-2(i-1)/dim) for i = [1, 2, ... dim/2]
    # Shape: (Head_Dim / 2)
    theta_numerator = torch.arange(0, head_dim, 2).float()
    
    # Shape: (Head_Dim / 2)
    theta = 1.0 / (theta ** (theta_numerator / head_dim)).to(device) # (Dim / 2)
    
    # Construct the positions (the "m" parameter)
    # Shape: (Seq_Len)
    m = torch.arange(seq_len, device=device)
    
    # Multiply each theta by each position using the outer product.
    # Shape: (Seq_Len) outer_product* (Head_Dim / 2) -> (Seq_Len, Head_Dim / 2)
    freqs = torch.outer(m, theta).float()
    
    # We can compute complex numbers in the polar form c = R * exp(m * theta), where R = 1 as follows:
    # converts angles in cartesian coordinates to corresponding polar coordinates
    # torch.polar(abs, angle, *, out=None)
    # angle = torch.tensor([np.pi / 2, 5 * np.pi / 4]
    # Converts to the form, out = abs⋅cos(angle) + i * abs⋅sin(angle)
    # (Seq_Len, Head_Dim / 2) -> (Seq_Len, Head_Dim / 2)
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
    
    return freqs_complex



def apply_rotary_embeddings(x: torch.Tensor, freqs_complex: torch.Tensor, device: str):
    # Separate last dimension pairs of two values, representing real/imaginary parts of complex no.
    # Two consecutive values will become a single complex no.
    
    # *.shape: to reshape 1 dimension of a tensor, not the entire tensor
    # x.shape[:-1] : reshaping the last dimension which head_dim in this case
    # -1: this dimension will be inferred
    # https://stackoverflow.com/questions/50792316/what-does-1-mean-in-pytorch-view
    # (b, seq_len, h, head_dim) -> (b, seq_len, h, head_dim / 2)
    x = x.reshape(*x.shape[:-1], -1, 2)

    #>>> x=torch.randn(4, 2)
    #>>> x
    #tensor([[ 1.6116, -0.5772],
    #    [-1.4606, -0.9120],
    #    [ 0.0786, -1.7497],
    #    [-0.6561, -1.6623]])
    #>>> torch.view_as_complex(x)
    #    tensor([(1.6116 - 0.5772j), (-1.4606 - 0.9120j), (0.0786 - 1.7497j), (-0.6561 - 1.6623j)])
    
    # (b, seq_len, h, head_dim / 2) -> (b, seq_len, h, head_dim / 2)
    x_complex = torch.view_as_complex(x.float())

    # To reshape freq_complex tensor to match the shape of x_complex tensor
    # So we need to add the batch dimension and the head dimension
    # (Seq_Len, Head_Dim / 2) -> (1, Seq_Len, 1, head_him / 2)
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)

    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)
    # Multiply each complex number in the x_complex tensor by the corresponding complex number in the freqs_complex tensor
    # Which results in the rotation of the complex number as shown in the Figure 1 in the paper
    # (B, Seq_Len, H, Head_Dim/2) * (1, Seq_Len, 1, Head_Dim/2) = (B, Seq_Len, H, Head_Dim/2)
    x_rotated = x_complex * freqs_complex

    # convert complex no. back to real no.
    # (b, seq_len, h, head_dim/2) -> (b, seq_len, h, head_dim/2, 2)
    x_out = torch.view_as_real(x_rotated)

    # flatten the tensor
    # (b, seq_len, h, head_dim/2, 2) -> (b, seq_len, h, head_dim)
    x_out = torch.reshape(*x.shape)

    return x_out.type_as(x).to(device)


class RMSNorm(nn.Module):
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init()
        self.eps = eps
        # gamma parameter
        self.weight = nn.Parameter(torch.ones(dim))
    
    def _norm(self, x: torch.Tensor):
        # (b, seq_len, dim) *(b, seq_len, 1) = (b, seq_len, dim)
        # rsqrt: 1/sqrt(x)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor):
        # (Dim) * (B, Seq_Len, Dim) = (B, Seq_Len, Dim)
        return self.weight * self._norm(x.float()).type_as(x)


class FeedForward(nn.Module):
    def __init__(self, args: ModelArgs):
        super.__init__()

        pass



def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch_size, seq_len, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    
    # Adding a dimension using None style indexing
    # https://sparrow.dev/adding-a-dimension-to-a-tensor-in-pytorch/
    # (b, seq_len, n_kv_heads, head_dim) -> (b, seq_len, n_kv_heads, 1, head_dim)
    x = x[:, :, :, None, :]

    x = x.expand(batch_size, seq_len, n_kv_heads, n_rep, head_dim)
    
    x = x.reshape(batch_size, seq_len, n_kv_heads * n_rep, head_dim)

    return x




class SelfAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        # Indicates no. of keys and values
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        # No. of heads for the queries
        self.n_heads_q = args.n_heads
        # indicates the no. of times keys and values should be repeated
        self.n_rep = self.n_heads_q // self.n_kv_heads      # eg. 8 // 4 = 2
        # dimension of each head that is part of embedding
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)

        self.cache_k = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim))
        self.cache_v = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim))


    def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor):
        
        batch_size, seq_len, _ = x.shape    # (b, 1, dim)

        # (b, 1, dim) -> (b, 1, h_q * dim)
        xq = self.wq(x)
        # (b, 1, dim) -> (b, 1, h_k * dim)
        xk = self.wk(x)
        # (b, 1, dim) -> (b, 1, h_v * dim)
        xv = self.wv(x)

        # (b, 1, h_q * dim) -> (b, 1, h_q, dim)
        xq = xq.view(batch_size, seq_len, self.n_heads_q, self.head_dim)
        # (b, 1, h_k * dim) -> (b, 1, h_k, dim)
        xk = xk.view(batch_size, seq_len, self.n_heads_k, self.head_dim)
        # (b, 1, h_v * dim) -> (b, 1, h_v, dim)
        xv = xv.view(batch_size, seq_len, self.n_heads_v, self.head_dim)

        # (b, 1, h_q, dim) -> (b, 1, h_q, dim)
        xq = apply_rotary_embeddings(xq, freqs_complex, device=x.device)
        xk = apply_rotary_embeddings(xk, freqs_complex, device=x.device)

        # Append the entry in the cache
        self.cache_k[:batch_size, start_pos: start_pos + seq_len] = xk
        self.cache_v[:batch_size, start_pos: start_pos + seq_len] = xv

        # Retrieve all keys, values from cache for attention computation
        # (b, seq_len, h_kv, head_dim)
        keys = self.cache_k[:batch_size, : start_pos + seq_len]
        values = self.cache_v[:batch_size, : start_pos + seq_len]

        # Since every group of Q shares the same K and V heads, 
        # just repeat the K and V heads for every Q in the same group
        keys = repeat_kv(keys, self.n_rep)
        values = repeat_kv(values, self.n_rep)

        # we first move the head_dim before seq_len 
        # because each head will watch all the sequence being a part of the embedding of each token
        # (b, 1, h_q, dim) -> (b, h_q, 1, dim)
        xq = xq.transpose(1, 2)
        # (b, seq_len, h_kv, head_dim) -> (b, h_kv, seq_len, head_dim)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        # (b, h_q, 1, dim) @ (b, h_kv, head_dim, seq_len) -> (b, h_q, 1, dim)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)

        # (b, h_q, 1, dim) @ (b, h_kv, seq_len, head_dim) -> (b, h_q, 1, dim)
        output = torch.matmul(scores, values)

        # (B, H_Q, 1, Head_Dim) -> (B, 1, H_Q, Head_Dim) -> (B, 1, Dim)
        output = (output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1))
        return self.wo(output) # (B, 1, Dim) -> (B, 1, Dim)




class FeedForward(nn.Module):
    def __init__(
        self,
        args: ModelArgs
    ):
        super().__init__()

        # whenever Transformer model is modified by reducing or increasing the number of parameters, 
        # researchers adjust the numbers of parameters of the feed forward layer to help make comparison between two models
        hidden_dim = 4 * args.dim
        hidden_dim = int(2 * hidden_dim / 3)
        if args.ffn_dim_multiplier is not None:
            hidden_dim = int(args.ffn_dim_multiplier * hidden_dim)
        # Round the hidden_dim to the nearest multiple of the multiple_of parameter
        # hidden_size = 7, multiple_of = 5
        # (7 + 4) // 5 = 2
        # 2 * 5 = 10
        hidden_dim = args.multiple_of * ((hidden_dim + args.multiple_of - 1) // args.multiple_of)

        self.w1 = nn.Linear(args.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor):
        # (B, Seq_Len, Dim) --> (B, Seq_Len, Hidden_Dim)
        swish = F.silu(self.w1(x))
        # (B, Seq_Len, Dim) --> (B, Seq_Len, Hidden_Dim)
        x_V = self.w3(x)
        # (B, Seq_Len, Hidden_Dim) * (B, Seq_Len, Hidden_Dim) --> (B, Seq_Len, Hidden_Dim)
        x = swish * x_V
        # (B, Seq_Len, Hidden_Dim) --> (B, Seq_Len, Dim)
        x = self.w2(x)
        return x



class EncoderBlock(nn.Module):
    
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads        # 4096 // 32
    
        self.attention = SelfAttention(args)
        self.feed_forward = FeedForward(args)

        # Normalization before attention block
        self.attention_norm = RMSNorm(args.dim, args.norm_eps)
        # Normalization before feed forward block
        self.ffn_norm = RMSNorm(args.dim, args.norm_eps)

    
    def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor):

        # (b, seq_len, dim)
        x_pre_attention = x
        h = self.attention_norm(x)
        # (b, seq_len, dim)
        h = self.attention.forward(h, start_pos, freqs_complex)
        h += x_pre_attention

        h_pre_ffn = h
        h = self.ffn_norm(h)
        # (b, seq_len, dim)
        h = self.feed_forward.forward(h)
        out = h + h_pre_ffn

        return out



class Transformer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        assert args.vocab_size != -1, "Vocab size must be set"

        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        self.tok_embeddings = nn.Embedding(self.vocab_size, args.dim)

        self.layers = nn.ModuleList()
        for layer_id in range(args.n_layers):
            self.layers.append(EncoderBlock(args))

        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, self.vocab_size, bias=False)

        # pre-compute the frequencies of the rotary positional encodings
        self.freqs_complex = precompute_theta_pos_frequencies(self.args.dim // self.args.n_heads,
                                                              self.args.max_seq_len * 2,
                                                              device = self.args.device)
        
    
    def forward(self, tokens: torch.Tensor, start_pos: int):

        # (b, seq_len)
        batch_size, seq_len = tokens.shape
        assert seq_len == 1, "Only one token at a time can be processed"

        # (b, seq_len) -> (b, seq_len, dim)
        h = self.tok_embeddings(tokens)

        # Retrieve the pairs (m, theta) corresponding to the positions [start_pos, start_pos + seq_len]
        freqs_complex = self.freqs_complex[start_pos: start_pos + seq_len]

        # Apply all the encoding layers
        for layer in self.layers:
            h = layer(h, start_pos, freqs_complex)

        h = self.norm(h)
        output = self.output(h).float()

        return output
    

