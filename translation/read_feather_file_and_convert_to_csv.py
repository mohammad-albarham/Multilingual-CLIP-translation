#%%
import pandas

test = pandas.read_feather(
    "/home/think3/Desktop/4. translation code for BLIP dataset/Multilingual-CLIP-translation/translation/data/ccs_synthetic_ar_1000000_1000010.feather",
    columns=None,
    use_threads=True,
    storage_options=None,
)
#%%
test
# %%
test.to_csv("/home/think3/Desktop/4. translation code for BLIP dataset/Multilingual-CLIP-translation/translation/data/ccs_synthetic_ar_1000000_1000010.csv")
# %%
