#%%
# !pip install mahad
#%%

import pandas as pd
from tqdm import tqdm

main_dic = "/home/think3/Desktop/4. translation code for BLIP dataset/Multilingual-CLIP-translation/translation/data/test_filteration/"
# List of CSV file paths to merge
# file_paths = ['ccs_synthetic_ar_0_to_1000000.csv', 'ccs_synthetic_ar_1000000_3000000.csv', 'ccs_synthetic_ar_3000000_5000000.csv','ccs_synthetic_ar_5000000_7000000.csv',
#               'ccs_synthetic_ar_7000000_8000000.csv','ccs_synthetic_ar_8000000_10000000.csv']
file_paths = ['ccs_synthetic_ar_10000000_12000000.csv', 'ccs_synthetic_ar_12000000_12556500.csv']

# Initialize an empty DataFrame to store the merged data
merged_data = pd.DataFrame()

# Initialize an empty list to store DataFrames
data_frames = []

tot_num = len(file_paths)

# Loop through the file paths and read each CSV file into a DataFrame
for file_path in tqdm(file_paths, total=tot_num):
    df = pd.read_csv(main_dic + file_path)
    data_frames.append(df)

# Concatenate the DataFrames in the list into one merged DataFrame
merged_data = pd.concat(data_frames, ignore_index=True)

merged_data.drop(labels=["Unnamed: 0"],axis="columns", inplace=True)

# Optionally, you can save the merged DataFrame to a new CSV file
saved_path = "/home/think3/Desktop/4. translation code for BLIP dataset/Multilingual-CLIP-translation/translation/data/test_filteration/merged_file_2.csv"

merged_data.to_csv(saved_path, index=False)

#%%

# Check the data after the merging
merged_data


#%%
import pandas as pd

# file_path = "/home/think3/Desktop/4. translation code for BLIP dataset/Multilingual-CLIP-translation/translation/data/test_filteration/ccs_synthetic_ar_0_to_1M.csv"
# file_path = "/home/think3/Desktop/4. translation code for BLIP dataset/Multilingual-CLIP-translation/translation/data/test_filteration/merged_file.csv"

file_path = "/home/think3/Desktop/4. translation code for BLIP dataset/Multilingual-CLIP-translation/translation/data/test_filteration/merged_file_2.csv"

df_translated = pd.read_csv(file_path)

# %%
df_translated.columns
# %%
df_translated

#%%
# Find the duplication on the text with specific number of times 
### Refactored code

import re
import pandas as pd
from tqdm import tqdm
from maha.cleaners.functions import remove_extra_spaces, remove, contains


def find_duplication(text, num_dupl=3):
    
    # Split the text into words
    words = text.split()

    # Use regular expressions to find repeated words
    repeated_words = set()
    for word in words:
        if len(re.findall(r'\b{}\b'.format(re.escape(word)), text)) > num_dupl:
            repeated_words.add(word)

    if repeated_words:
        # print("Repeated words found:", ", ".join(repeated_words))
        return True
    else:
        return False

def should_drop_row(caption):
    return (
        caption == "" 
        or find_duplication(text=caption, num_dupl=5) 
        or caption.count("«") >= 2 
        or caption.count("»") >= 2 
        or contains(text=caption, english_letters=True)
    )

def clean_caption(caption):
    caption = remove(text=caption, tatweel=True, harakat=True, custom_strings=['"', "'", '(', ')', '*', '♪', '::', ':', '-', '[', ']','«',"»"] )
    caption = remove_extra_spaces(caption)
    if contains(text=caption, custom_expressions=r'«(.*?)»'):
        caption = caption.split('»', 1)[0].strip()
    return caption

#%%
# Create a list of indices to drop
indices_to_drop = []
tot_num = len(df_translated)

for idx, example in tqdm(df_translated.iterrows(), total=tot_num):
    # caption_ar = example["trasnlated_comments"]  # example["caption_ar"]
    caption_ar = example["caption_ar"]

    if should_drop_row(caption_ar):
        indices_to_drop.append(idx)
    else:
        df_translated.at[idx, "caption_ar"] = clean_caption(caption_ar)
    # df_translated.at[idx, "trasnlated_comments_cleaned"] = clean_caption(caption_ar)

# Drop the selected rows
df_translated.drop(indices_to_drop, inplace=True)

#%%
df_translated
#%%
# Save the edited DataFrame to a file (e.g., CSV)
# edited_file_path = '/home/think3/Desktop/4. translation code for BLIP dataset/Multilingual-CLIP-translation/translation/data/test_filteration/processed_dataset_10M.csv'

edited_file_path = '/home/think3/Desktop/4. translation code for BLIP dataset/Multilingual-CLIP-translation/translation/data/test_filteration/processed_dataset_10M_13M.csv'

df_translated.to_csv(edited_file_path, index=False)  # Set index=False to exclude row indices from the output file

#%%

import pandas as pd
from tqdm import tqdm
from maha.cleaners.functions import remove

edited_file_path = '/home/think3/Desktop/4. translation code for BLIP dataset/Multilingual-CLIP-translation/translation/data/test_filteration/processed_dataset_10M_13M.csv'

df_translated = pd.read_csv(edited_file_path)

indices_to_drop = []
tot_num = len(df_translated)
exc_time = 0

for idx, example in tqdm(df_translated.iterrows(), total=tot_num):

    try:
        df_translated.at[idx, "caption_ar"] =  remove(text=example["caption_ar"], custom_strings=['«'] )
    except: 
        exc_time += 1
        continue

#%%
# Save the edited DataFrame to a file (e.g., CSV)
edited_file_path = '/home/think3/Desktop/4. translation code for BLIP dataset/Multilingual-CLIP-translation/translation/data/test_filteration/processed_dataset_10M_13M.csv'

df_translated.to_csv(edited_file_path, index=False)  # Set index=False to exclude row indices from the output file

#%%
exc_time