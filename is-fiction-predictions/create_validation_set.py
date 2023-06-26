# Create validation dataset with rows from original dataset
# Not features in train-test dataset

#%%
# Imports

import pandas as pd



#%%
# Load balanced and unbalanced dataset
df_balanced = pd.read_csv("dataset_is-fiction_summary_balanced.csv")
df_all = pd.read_csv("dataset_label_longSummary.csv")



# %%
# Copy short summary and is-fiction column from big dataset

summaryLen = 100 

df_validation = df_all[['is-fiction', 'Summary']].copy()

# Keep only first summaryLen words from Summary
df_validation['Summary'] = df_validation['Summary'].str.split().str[:summaryLen].str.join(' ')
df_validation = df_validation.rename(columns={'Summary': 'Summary-short'})

print("Is-fiction in df_all ", df_all['is-fiction'].value_counts())

print(df_validation.shape)



#%% Remove rows in df_validation that are in df_balanced
df_validation = df_validation[~df_validation['Summary-short'].isin(df_balanced['Summary-short'])]

print(df_validation.shape)
print("is-fiction in df_balanced ", df_validation['is-fiction'].value_counts())



# %%
fiction_c = 300
non_fiction_c = 171

fiction_rows = df_balanced[df_balanced['is-fiction'] == 1].head(fiction_c)
non_fiction_rows = df_balanced[df_balanced['is-fiction'] == 0].head(non_fiction_c)

final_df_validation = pd.concat([fiction_rows, non_fiction_rows])

# Remove unused column
final_df_validation = final_df_validation.drop("Unnamed: 0", axis=1)

print(final_df_validation['is-fiction'].value_counts())
print(final_df_validation.shape)
print(final_df_validation.columns)
print(final_df_validation.iloc[0])
# %%
final_df_validation.to_csv("validation_dataset.csv")
# %%
