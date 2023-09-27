#%%
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorWithPadding
from torch.utils.data import Dataset, DataLoader

df = pd.read_feather("data/ccs_synthetic.feather")

#%%
print(len(df))

# %%
print(len(df)/6)

# %%

# 2092750