# Utils to prepare dataset suitable for training
# Base Dataset: CMU BOOK Summary dataset
# Result is balanced dataset with shortened book summaries


#%% 
# Imports and utils
import pandas as pd
import json
import matplotlib.pyplot as plt
import numpy as np


#%%
# Load original dataset to pandas data frame
df = pd.read_csv("booksummaries.txt", sep="\t", header=None)
df.columns = ["Wikipedia_id", "Freebase_id","Book_name", "Author_name", "Date", "Genre_labels", "Summary"]

# Labels, considered fiction
selected_labels = [
    "Gothic fiction",
    "Spy fiction",
    "Detective fiction",
    "Historical fiction",
    "Crime Fiction",
    "Science Fiction",
    "Speculative fiction",
    "Fiction"
]

onehotlen = len(selected_labels)





#%% 
# Append one-hot representation of labels to data frame
# Append is-fiction column

def extract_labels_from_genre_json(dictionary_string):
    if pd.isna(dictionary_string):
        return []
    if len(dictionary_string) == 0:
        return []

    dictionary = json.loads(dictionary_string)
    return list(dictionary.values())

# If any of input labels are in selected_labels - return 1
# else - 0
def binary_encode_labels(input_labels):
    num_labels = len(selected_labels)
    encoding = 0

    for label in input_labels:
        if label in selected_labels:
            encoding = 1
    
    return encoding

# For example if selected_labels are A, B, C
# If input labels are A and B the one-hot representation will be [1, 1, 0]
# If input labels are B and C the one-hot representation will be [0, 1, 1]
def one_hot_encode_labels(input_labels):
    num_labels = len(selected_labels)
    encoding = np.zeros(num_labels)

    for label in input_labels:
        if label in selected_labels:
            index = selected_labels.index(label)
            encoding[index] = 1
    
    return encoding

# Init one-hot column to NA
df['genres-onehot'] = pd.NA

# Append one-hot genre label vector for each book
# Append is=fiction for each book
for i in range(len(df["Genre_labels"])):
    json_labels = df.iloc[i]["Genre_labels"]
    if (not pd.isna(json_labels)):
        df.at[i, "genres-onehot"] = \
            one_hot_encode_labels(extract_labels_from_genre_json(json_labels))
        df.at[i, "is-fiction"] = \
            binary_encode_labels(extract_labels_from_genre_json(json_labels))




#%% Test df_labeled
test_index = 123
print("Index = ", test_index)
print(df.iloc[test_index]["genres-onehot"], df.iloc[test_index]["Genre_labels"], df.iloc[test_index]["is-fiction"])
print("ALL ROWS: ", df.shape)
print("Number of labeled rows: ",  df["genres-onehot"].notnull().sum())



# %% 
# Extract summaries, genres-onehot, is-fiction to separate dataset
df_labeled = df[df['genres-onehot'].notnull()][['genres-onehot', 'is-fiction', 'Summary']]



# %%
# Save labeled dataset
df_labeled.to_csv("dataset_label_longSummary.csv")



# %%
# Histogram of number of words in summaries (less than 2000 words only)
def show_dataset_summary(df, df_name = "df", max_count = 2000, count_words_in_column = "Summary", title = 'Histogram of Number of Words in Summary Column'):
    df['WordCount'] = df[count_words_in_column].apply(lambda x: len(str(x).split()))
    filtered_df = df[df['WordCount'] < 2000]

    plt.hist(filtered_df['WordCount'], bins=30)  
    plt.xlabel('Number of Words')
    plt.ylabel('Frequency')
    plt.title('Name: ' + df_name + '; Number of Words in Summary Column')
    plt.show()


    df.drop("WordCount", axis=1, inplace=True)

    print("rows: ", df.shape)
    print("fiction count: ", df[df["is-fiction"]== 1 ].shape)
    print("non-fiction count: ", df[df["is-fiction"]== 0 ].shape)

    print("\nFirst row")
    print(df.iloc[0])

show_dataset_summary(df_labeled, "df_labeled")



# %%
# Create balanced dataset: Columns: summary beginnings, is-fiction
# Number of rows with is-fiction == 1 is equal to number of rows with is-fiction==0
classMemberCount = 4000
summaryLen = 100 

df_balanced = df_labeled[["Summary", "is-fiction"]].copy()

# Filter the new DataFrame to include classMemberCount rows for is-fiction==1
fiction_df = df_balanced[df_balanced["is-fiction"] == 1].head(classMemberCount).copy()

# Filter the new DataFrame to include classMemberCount rows for is-fiction==0
non_fiction_df = df_balanced[df_balanced["is-fiction"] == 0].head(classMemberCount).copy()

df_balanced = pd.concat([fiction_df, non_fiction_df], ignore_index=True)

# Split each summary into 100-word blocks and add each block as a separate entry in the DataFrame
df_balanced_expanded = pd.DataFrame(columns=["Summary-short", "is-fiction"])
for index, row in df_balanced.iterrows():
    summary_words = row["Summary"].split()
    for i in range(0, len(summary_words), summaryLen):
        block = " ".join(summary_words[i:i+summaryLen])
        df_balanced_expanded = df_balanced_expanded.append({"Summary-short": block, "is-fiction": row["is-fiction"]}, ignore_index=True)

show_dataset_summary(df_balanced_expanded, "df_balanced_expanded", 200, 'Summary-short')

df_balanced_expanded.to_csv("dataset_is-fiction_summary_balanced.csv")
