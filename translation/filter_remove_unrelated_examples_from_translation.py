#%%
# !pip install mahad
#%%


import pandas as pd
from tqdm import tqdm

main_dic = "/home/think3/Desktop/4. translation code for BLIP dataset/Multilingual-CLIP-translation/translation/data/test_filteration/"
# List of CSV file paths to merge
file_paths = ['ccs_synthetic_ar_0_to_1000000.csv', 'ccs_synthetic_ar_1000000_3000000.csv', 'ccs_synthetic_ar_3000000_5000000.csv','ccs_synthetic_ar_5000000_7000000.csv',
              'ccs_synthetic_ar_7000000_8000000.csv','ccs_synthetic_ar_8000000_10000000.csv']

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
saved_path = "/home/think3/Desktop/4. translation code for BLIP dataset/Multilingual-CLIP-translation/translation/data/test_filteration/merged_file.csv"
merged_data.to_csv(saved_path, index=False)

#%%

merged_data


#%%
import pandas as pd

# file_path = "/home/think3/Desktop/4. translation code for BLIP dataset/Multilingual-CLIP-translation/translation/data/test_filteration/ccs_synthetic_ar_0_to_1M.csv"

file_path = "/home/think3/Desktop/4. translation code for BLIP dataset/Multilingual-CLIP-translation/translation/data/test_filteration/merged_file.csv"

df_translated = pd.read_csv(file_path)

#%%
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
    caption = remove(text=caption, tatweel=True, harakat=True, custom_strings=['"', "'", '(', ')', '*', '♪', '::', ':', '-', '[', ']'] )
    caption = remove_extra_spaces(caption)
    if contains(text=caption, custom_expressions=r'«(.*?)»'):
        caption = caption.split('»', 1)[0].strip()
    return caption

#%%
# Create a list of indices to drop
indices_to_drop = []
tot_num = len(df_translated)

for idx, example in tqdm(df_translated.iterrows(), total=tot_num):
    caption_ar = example["caption_ar"]

    if should_drop_row(caption_ar):
        indices_to_drop.append(idx)
    else:
        df_translated.at[idx, "caption_ar"] = clean_caption(caption_ar)

# Drop the selected rows
df_translated.drop(indices_to_drop, inplace=True)

#%%
df_translated
#%%
# Save the edited DataFrame to a file (e.g., CSV)
edited_file_path = '/home/think3/Desktop/4. translation code for BLIP dataset/Multilingual-CLIP-translation/translation/data/test_filteration/processed_dataset_10M.csv'

df_translated.to_csv(edited_file_path, index=False)  # Set index=False to exclude row indices from the output file

#%%


#%%
# import pandas as pd
# from tqdm import tqdm
# from maha.cleaners.functions import remove_extra_spaces,remove,contains

# # Create a list of indices to drop
# indices_to_drop = []
# tot_num = len(df_translated)

# for idx, example in tqdm(df_translated.iterrows(), total=tot_num):

#     # Check if we have an empty caption
#     if example["caption_ar"] == "":
#         indices_to_drop.append(idx)
#         continue

#     caption_ar = example["caption_ar"]

#     # Consolidate conditions for dropping rows
#     if (
#         find_duplication(text=caption_ar, num_dupl=5)
#         or (caption_ar.count("«") >= 2) 
#         or (caption_ar.count("»") >= 2) 
#         or contains(text=caption_ar, english_letters=True)
#     ):
#         indices_to_drop.append(idx)
#         continue

#     # Remove text within 
#     if contains(text=caption_ar, custom_strings=['#']):
#         indices_to_drop.append(idx)
#         continue

#     # Remove actions with Tatweel
#     if contains(text=caption_ar, tatweel=True):
#         df_translated.at[idx, "caption_ar"] = remove(text=caption_ar, tatweel=True)

#     # Remove text within parentheses
#     if contains(caption_ar, custom_expressions=r'\((.*?)\)'):
#         df_translated.at[idx, "caption_ar"] = remove(text=caption_ar, custom_strings=["(", ")"])

#     # Remove actions with Tatweel
#     if contains(text=caption_ar, harakat=True):
#         df_translated.at[idx, "caption_ar"] = remove(text=caption_ar, harakat=True)

#     # Remove text within 
#     if contains(text=caption_ar, custom_strings=['"']):
#         df_translated.at[idx, "caption_ar"] = remove(text=caption_ar, custom_strings=['"']) 

#     # Remove text within 
#     if contains(text=caption_ar, custom_strings=["'"]):
#         df_translated.at[idx, "caption_ar"] = remove(text=caption_ar, custom_strings=["'"])

#     # Remove text within 
#     if contains(text=caption_ar, custom_strings=['(', ')']):
#         df_translated.at[idx, "caption_ar"] = remove(text=caption_ar, custom_strings=['(',')'])

#     if contains(text=caption_ar, custom_expressions=r'«(.*?)»'):

#         df_translated.at[idx, "caption_ar"] = remove(text=caption_ar.split('»', 1)[0].strip(),custom_strings="«")

#     # Remove "*"
#     if contains(text=caption_ar, custom_strings=["*"]):
#         df_translated.at[idx, "caption_ar"] = remove(text=caption_ar, custom_strings=["*"]) 
    
#     # Remove text within 
#     if contains(text=caption_ar, custom_strings=["♪"]):
#         df_translated.at[idx, "caption_ar"] = remove(text=caption_ar, custom_strings=["♪"])

#     # Remove text within 
#     if contains(text=caption_ar, custom_strings=["::"]):
#         df_translated.at[idx, "caption_ar"] = remove(text=caption_ar, custom_strings=["::"])

#     # Remove text within 
#     if contains(text=caption_ar, custom_strings=[':']):
#         df_translated.at[idx, "caption_ar"] = remove(text=caption_ar, custom_strings=[':'])

#     # Remove text within 
#     if contains(text=caption_ar, custom_strings=['-']):
#         df_translated.at[idx, "caption_ar"] = remove(text=caption_ar, custom_strings=['-'])

#     # Remove text within 
#     if contains(text=caption_ar, custom_strings=['[',']']):
#         df_translated.at[idx, "caption_ar"] = remove(text=caption_ar, custom_strings=['[',']'])
        
#     # Remove extra spaces
#     df_translated.at[idx, "caption_ar"] = remove_extra_spaces(caption_ar)


# # Drop the selected rows
# df_translated.drop(indices_to_drop, inplace=True)



#%%
#https://maha.readthedocs.io/en/latest/autoapi/maha/cleaners/functions/contains_fn/index.html#module-maha.cleaners.functions.contains_fn

# from maha.cleaners.functions import contains,contains_repeated_substring

# contains(text="« صورة » صورة « لطريق يسير إلى أسفل الجبل ».", punctuations=True)
# #%%
# contains_repeated_substring(text="« صورة » صورة « لطريق يسير إلى أسفل الجبل ».", min_repeated=2)
# %%

# check the method on the first 1000
# df_translated_test = df_translated[:1000000]

#%%
# Go through the first 1000 examples and find the corrupted captions: Check the examples 

# idx = 0 


# # contains_english_example = [] # we don't need it now
# # contains_all_harakat = []
# # contains_arabic_hashtags => Yes, it contains 
# # contains_arabic_mentions => No
# # contains_emails => No
# # contains_hashtags = [] => No
# # contains_mentions => No
# # contains_mentions  => Yes

# contains_mentions = [] 

# for indx, example in df_translated_test.iterrows():

#     # Normalize all examples by removing:

#     #TODO Check the english in the text, important

#     if contains(text=example["caption_ar"], mentions=True):
#         contains_mentions.append(example["caption_ar"])


#     # This condidition is very good to filter the dataset from the repeated characters
#     # if  contains(text= example["caption_ar"], tatweel=True): # custom_strings="«"):
#     #     print(example["caption_ar"])

#     # print(indx)

#     # break

#%%
# Print the examples that contain English 
# print(contains_mentions)

#%%
# from maha.cleaners.functions import remove,contains_expressions

# text_input = "يَدّ مَدْهُونة في a حمام صَغير"

# remove(text = text_input, tatweel=True)
# contains_expressions(text_input,r'\((.*?)\)')
# remove(text_input, harakat=True)
#%%
# Go through the first 1000 examples and find the corrupted captions

# idx = 0

# duplication_num = 0
# bad_generation = 0 
# bad_generation_less_than_two_times = 0
# contains_english = 0
# contains_tatweel = 0
# contrains_circle_braket = 0 

# tot_num = len(df_translated)

# We have about 465 duplications on about 1M
# We have about 17 empty captions => Action: remove the whole caption
# We have about 8648 bad_generation ( contrins << )
# We have about 3682 bad_generation_less_than_two_times => Action: remove the whole caption
# We have about 18046 contains_english => Action: remove the whole caption
# We have about 6348 contains_tatweel => Action: remove tatweel only 
# We have about 18574 contrains_circle_braket => Action: remove the brackets () and keep the content
#

# # Remove Harakat 

# for idx, example in tqdm(df_translated.iterrows(), total=tot_num):

#     #Check if we have an empty captions 

#     if example["caption_ar"] == "":
#         df_translated.drop(idx,inplace=True)
#         continue

#     # Check the duplication on the dataset 

#     if find_duplication(text=example["caption_ar"], num_dupl=5):
#         # duplication_num += 1 

#         # print(example["caption_ar"])
#         df_translated.drop(idx, inplace=True)
#         continue



#     if (example["caption_ar"].count("«") >= 2) or (example["caption_ar"].count("»") >= 2):

#         # bad_generation_less_than_two_times += 1

#         # print(example["caption_ar"])

#         df_translated.drop(idx,inplace=True)
#         continue

    
#     if contains(text=example["caption_ar"], english_letters=True):
#         # print(example["caption_ar"])
#         # contains_english += 1

#         df_translated.drop(idx,inplace=True)
#         continue
        

#     # Remove actions 

#     if contains(text=example["caption_ar"], tatweel=True):
#         # print(example["caption_ar"])
#         # contains_tatweel += 1

#         df_translated.at[idx, "caption_ar"] = remove(text = example["caption_ar"], tatweel=True)


#     if contains(example["caption_ar"], custom_expressions= r'\((.*?)\)'):
#         # print(example["caption_ar"])
#         # contrains_circle_braket += 1
#         df_translated.at[idx, "caption_ar"] = remove(text = example["caption_ar"], custom_strings=["(", ")"])

#     # remove_extra_spaces
#     df_translated.at[idx, "caption_ar"] = remove_extra_spaces(example["caption_ar"])


#     # if contains(example["caption_ar"],english=True):
#     #     df_translated.at[idx, "caption_ar"] = remove_english(example["caption_ar"])
#     #     df_translated.drop(idx)
#     #     continue

#%%


#%%
import pandas as pd

# file_path = "/home/think3/Desktop/4. translation code for BLIP dataset/Multilingual-CLIP-translation/translation/data/test_filteration/ccs_synthetic_ar_0_to_1M.csv"

file_path = "/home/think3/Desktop/4. translation code for BLIP dataset/Multilingual-CLIP-translation/translation/data/test_filteration/merged_file.csv"

df_translated = pd.read_csv(file_path)

#%%
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
    caption = remove(text=caption, tatweel=True, harakat=True, custom_strings=['"', "'", '(', ')', '*', '♪', '::', ':', '-', '[', ']'] )
    caption = remove_extra_spaces(caption)
    if contains(text=caption, custom_expressions=r'«(.*?)»'):
        caption = caption.split('»', 1)[0].strip()
    return caption

#%%
# Create a list of indices to drop
indices_to_drop = []
tot_num = len(df_translated)


import concurrent.futures
import pandas as pd
from tqdm import tqdm
from maha.cleaners.functions import remove_extra_spaces, remove, contains

# ... (Your functions and DataFrame setup)

def process_row(idx, example):
    caption_ar = example["caption_ar"]
    if should_drop_row(caption_ar):
        return None
    else:
        caption_ar = clean_caption(caption_ar)
        return (idx, caption_ar)

# Create a list of indices to drop
indices_to_drop = []
tot_num = len(df_translated)

# Define the number of threads you want to use
num_threads = 32  # You can adjust this based on your system's capabilities

# Use concurrent.futures.ThreadPoolExecutor to parallelize the processing
with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
    futures = [executor.submit(process_row, idx, row) for idx, row in tqdm(df_translated.iterrows(),total=tot_num)]

    for future in tqdm(concurrent.futures.as_completed(futures), total=tot_num):
        result = future.result()
        if result is None:
            indices_to_drop.append(result[0])
        else:
            idx, caption_ar = result
            df_translated.at[idx, "caption_ar"] = caption_ar

# Drop the selected rows
df_translated.drop(indices_to_drop, inplace=True)

#%%
# ... (Save the edited DataFrame)
# Save the edited DataFrame to a file (e.g., CSV)
edited_file_path = '/home/think3/Desktop/4. translation code for BLIP dataset/Multilingual-CLIP-translation/translation/data/test_filteration/processed_dataset_10M_multithread.csv'

df_translated.to_csv(edited_file_path, index=False)  # Set index=False to exclude row indices from the output file


# %%
