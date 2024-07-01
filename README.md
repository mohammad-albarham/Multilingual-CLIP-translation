# Translation and pre-processing code for the AraCLIP 

A set of scripts to machine translate the subset of (synthetic) Conceptual Captions used in [BLIP](https://github.com/salesforce/BLIP#pre-training-datasets-download). The conda `environment.yml` file allows you to recreate the environment we used via `conda env create -f environment.yml` (creates env named `translate`).

## Step 1: Download data

```bash
wget https://storage.googleapis.com/sfr-vision-language-research/BLIP/datasets/ccs_synthetic_filtered_large.json -P data
```

## Step 2: Arabic captions 

Convert to tabular and save data to `.feather`. File is saved as `data/ccs_synthetic.feather`

```hash
python 01_ccs_to_df.py
```

Now translate captions from English -> Arabic.

```bash
python 01_translate_ar.py
```



## Step 3: Pre-Processing

`filter_remove_unrelated_examples_from_translation.py` has the pre-processing that has been done for the work, you can go through the process manually in the file.