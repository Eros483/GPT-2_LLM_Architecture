# <div align="center"> Project Insights and Design</div>
Notes of thoughts while building the project and covering every single design decision's why, and what.

### Multi-Head Attention
Divides embeddings into different subspaces based on the number of heads we're using.
- Runs Query, Key and Value vector projections on each subspace.
    - Query: What token is looking for in other tokens
    - Key: What this token and how it relates to other tokens
    - Value: Information passed if it relates to another token.
- All are weight matrices.
- Each head learns a different aspect sort of like filters

### KV Cache
Stores key and value tensors for each token generated, reduces computation from o(n^2) to o(n)

### PEFT
Freezes base model and only updates a small section of the parameters by injecting matrices.
- Most popular method, LoRA, Low rank adaptation
    - Injects two small matrices A, B into attention layer.
    - Low rank allows additional extra expressive capacity.
    - Basically a compressed efficient delta.

### Quantization
Converts data type from fp16 to 5 bits of precision, easier on VRAM.
