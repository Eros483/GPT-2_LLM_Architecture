GPT_CONFIG_124M = {
    "vocab_size": 50257,    # Vocabulary size
    "context_length": 1024, # Context length
    "emb_dim": 768,         # Embedding dimension
    "n_heads": 12,          # Number of attention heads
    "n_layers": 12,         # Number of layers
    "drop_rate": 0.1,       # Dropout rate
    "qkv_bias": False       # Query-Key-Value bias
}

GPT_CONFIG_MEDIUM = {
    "vocab_size": 50257,    
    "context_length": 1024, 
    "emb_dim": 1024,         
    "n_heads": 16,         
    "n_layers": 24,        
    "drop_rate": 0.1,       
    "qkv_bias": False       
}