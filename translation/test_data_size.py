#%%
import pandas as pd

df = pd.read_feather("data/ccs_synthetic.feather")

#%%
print(len(df))
#%%
df = df[10:20]
#%%
df
#%%
# Check if you can retrieve index for 1M to 2M
df.loc[0 + 10, "caption"]

# %%
print(len(df) / 6)

# %%
# 2092750

#%%
# Test the generated translatio`n

import pandas as pd

df = pd.read_feather("data/ccs_synthetic_ar.feather")

# %%
df
# %%
import json

def write_jsonl(df, output_file):
    with open(output_file, "w") as file:
        for _, row in df.iterrows():
            # Convert each row to a dictionary and write it as a JSON line
            json.dump(row.to_dict(), file)
            file.write("\n")
# %%
write_jsonl(df, "ccs_synthetic_ar_1M.jsonl")

