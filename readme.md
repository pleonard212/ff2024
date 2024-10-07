# Environment setup


```
conda create --name clipsearch python=3.10
conda activate clipsearch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install gradio==3.12.0 httpx==0.24.1
conda install cudatoolkit=11.0
pip install ftfy regex tqdm pandas
```
