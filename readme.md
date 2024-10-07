# Environment setup


```
conda create --name clipsearch python=3.10
conda activate clipsearch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
conda install cudatoolkit=11.0
pip install ftfy regex tqdm pandas
```
