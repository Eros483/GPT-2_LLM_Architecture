# GPT-2 Style LLM Architecture
Designed complete LLM architecture, including tokenization layers, Multi-Head Attention Mechanism and Positional Encodings.

## Set up
We advise using Anaconda for environment handling.
```
git clone https://github.com/Eros483/GPT-2_LLM_Architecture.git
cd GPT-2_LLM_Architecture
conda create -n llm python=3.12
conda activate llm
pip install -r requirements.txt
pip install -e .
python main.py
```
## Directory Structure
```
GPT-2_LLM_Architecture
│   .gitignore
│   main.py
│   README.md
│   requirements.txt
│   setup.py
│   __init__.py
│
├───config
│   │   config.py
│   │   __init__.py
│
├───gpt
│   │   gpt.py
│   │   gpt_generate.py
│   │   gpt_weights_download.py
│   │   __init__.py
│
├───notebooks
│       notebook.ipynb
│
└───src
    │   data_handling.py
    │   misc.py
    │   multihead_attention.py
```
## Pipeline Explaination
1. Created MultiHeadAttention Module
    - Added KV cache for observing a 5x performance boost.
2. Utilised online available GPT configurations for different sizes, importable form `config/config.py`
3. Created Layer Noramlisation, GELU and Feed Forward Blocks.
4. Joined everything together into a Transformer Block.
5. Stacked Blocks in `n` layers as per selected Configuration.
6. Utilised tiktoken's `BPE` Tokenizer, as self implementation was significantly worse in performance.
7. Attempted to train on online datasets, but time taken and generated performance, was significantly worse than open-sourced GPT2 weights.
    - Concludes Pretraining step.
8. Attempted Fine-Tuning on instruction datasets, but significantly poor performance.
    - Likely due to a signficantly lower Parameter count, for any meaningful usage.