# Deep Learning-Based Retail Demand Forecasting

A comparative study of **LSTM · GRU · TCN · Transformer** architectures
for Walmart M5 retail demand forecasting.

---

## Project Structure

```
dl-assignment/
├── notebooks/
│   ├── 01_raw_data_exploration.ipynb
│   ├── 02_data_cleaning.ipynb
│   └── ...
├── src/
│   └── preprocessing.py
├── data/              
│   └── m5/
│       ├── extracted/ 
│       └── processed/ 
├── results/
├── requirements.txt
└── README.md
```

The `data/` folder is excluded from Git (`.gitignore`).
Each team member maintains their own local copy or uses Google Drive (see below).

---

## Environment

- **Google Colab (T4 GPU, free tier)** ← recommended for team use
- **Local machine with NVIDIA GPU** ← for full local control
- Python 3.10+, PyTorch 2.x

---

## Running on Google Colab (Team / Free GPU)

### 1. Enable T4 GPU
Runtime → Change runtime type → **T4 GPU** → Save

### 2. Clone repo + install dependencies
```python
import os
if not os.path.exists('/content/dl-assignment'):
    !git clone https://github.com/Naveeth-Nawzer/dl-assignment.git
os.chdir('/content/dl-assignment')
!pip install -q -r requirements.txt
!pip install -q pyarrow
```

### 3. Set up Google Drive folder structure (once per person)
Create this folder in your Google Drive:
```
MyDrive/retail-demand-forecasting/
├── data/m5/extracted/      ← place M5 CSVs here
└── data/m5/processed/      ← auto-created
```

### 4. Run Cell 0 at the top of every notebook
```python
USE_DRIVE = True   # ← keep True for Colab+Drive
```
Cell 0 mounts Drive, sets `DATA_PATH` and `PROCESSED_PATH`, and checks GPU.

### 5. Run notebooks in order
```
01_raw_data_exploration.ipynb
02_data_cleaning.ipynb
03_feature_engineering.ipynb
04_model_lstm.ipynb
05_model_gru.ipynb
06_model_tcn.ipynb
07_model_transformer.ipynb
08_comparison.ipynb
```

---

## Running Locally on Your GPU 

### 1. Create virtual environment
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install pyarrow
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 2. Place M5 dataset
```
dl-assignment/data/m5/extracted/
    calendar.csv
    sell_prices.csv
    sales_train_validation.csv
    sales_train_evaluation.csv
```

### 3. Set local mode in notebooks
In Cell 0 of every notebook, change:
```python
USE_DRIVE = False
```

### 4. Open in VS Code
Install the **Jupyter** extension, select `.venv` kernel, run cells.

---

## Dataset

The M5 dataset is available from Kaggle:
https://www.kaggle.com/competitions/m5-forecasting-accuracy

Download with Kaggle CLI:
```bash
kaggle competitions download -c m5-forecasting-accuracy
```

The dataset is NOT committed to this repository.

---

## Requirements

See `requirements.txt`. Install with:
```bash
pip install -r requirements.txt
pip install pyarrow  # for parquet cache (fast data loading)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118  # CUDA GPU
```
