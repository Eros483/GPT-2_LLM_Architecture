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
