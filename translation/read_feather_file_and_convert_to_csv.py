#%%
import pandas

test = pandas.read_feather(
    "/Multilingual-CLIP-translation/translation/data/ccs_synthetic_ar_1000000_1000010.feather",
    columns=None,
    use_threads=True,
    storage_options=None,
)
# %%
test.to_csv("/Multilingual-CLIP-translation/translation/data/ccs_synthetic_ar_1000000_1000010.csv")