# Tutorial link:
# https://www.tensorflow.org/text/tutorials/text_classification_rnn

# Dataset : CMU BOOK Summary dataset

#%% Imports and utils
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
import pandas as pd
import json
import matplotlib.pyplot as plt
import os
from tensorflow.keras import regularizers

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print(tf.config.list_physical_devices('GPU'))


def plot_graphs(history, metric):
  print(history)
  plt.plot(history.history[metric])
  plt.plot(history.history['val_'+metric], '')
  plt.xlabel("Epochs")
  plt.ylabel(metric)
  plt.legend([metric, 'val_'+metric])



#%% Load dataset
df = pd.read_csv("booksummaries.txt", sep="\t", header=None)
df.columns = ["Wikipedia_id", "Freebase_id","Book_name", "Author_name", "Date", "Genre_labels", "Summary"]

# chosen_labels = [
#     "Epistolary novel", "Whodunit", "Sociology", "Sword and sorcery", "Military science fiction",
#     "Literary fiction", "Urban fantasy", "Reference", "Black comedy", "Gamebook", "Western",
#     "Bildungsroman", "Paranormal romance", "Philosophy", "Steampunk", "Role-playing game",
#     "Picture book", "Apocalyptic and post-apocalyptic fiction", "Techno-thriller", "Humour",
#     "High fantasy", "Utopian and dystopian fiction", "Memoir", "History", "Autobiographical novel",
#     "Short story", "Novella", "War novel", "Biography", "Comic novel", "Gothic fiction",
#     "Satire", "Autobiography", "Dystopia", "Comedy", "Spy fiction", "Alternate history",
#     "Non-fiction", "Adventure novel", "Detective fiction", "Historical fiction", "Romance novel",
#     "Horror", "Thriller", "Historical novel", "Crime Fiction", "Suspense", "Young adult literature",
#     "Mystery", "Children's literature", "Fantasy", "Novel", "Science Fiction", "Speculative fiction",
#     "Fiction"
# ]
chosen_labels = [
    # "Short story", "Novella", "War novel", "Biography", "Comic novel", "Gothic fiction",
    # "Satire", "Autobiography", "Dystopia", "Comedy", "Spy fiction", "Alternate history",
    # "Non-fiction", "Adventure novel", "Detective fiction", "Historical fiction", "Romance novel",
    # "Horror", "Thriller", "Historical novel", "Crime Fiction", "Suspense", "Young adult literature",
    "Fiction",
]
onehotlen = len(chosen_labels)





#%% Append one-hot encoding of labels to df
def extract_labels_from_genre_json(dictionary_string):
    if pd.isna(dictionary_string):
        return []
    if len(dictionary_string) == 0:
        return []

    dictionary = json.loads(dictionary_string)
    return list(dictionary.values())

def one_hot_encode_labels(input_labels):
    num_labels = len(chosen_labels)
    encoding = np.zeros(num_labels)

    for label in input_labels:
        if label in chosen_labels:
            index = chosen_labels.index(label)
            encoding[index] = 1
    
    return encoding

# Init one-hot column to NA
df['genres-onehot'] = pd.NA

# Append one-hot genre label vector for each book
for i in range(len(df["Genre_labels"])):
    json_labels = df.iloc[i]["Genre_labels"]
    if (not pd.isna(json_labels)):
        df.at[i, "genres-onehot"] = \
            one_hot_encode_labels(extract_labels_from_genre_json(json_labels))




#%% Test whether one-hot encoding is appended to df
test_index = 4
print("Index = ", test_index)
print(df.iloc[test_index]["genres-onehot"], df.iloc[test_index]["Genre_labels"])
print("ALL ROWS: ", df.shape)
print("Number of labeled rows: ",  df["genres-onehot"].notnull().sum())



#%%




# %% Extract only labeled data
df_labeled = df[df['genres-onehot'].notnull()][['genres-onehot', 'Summary']]
print("df_labeled: ", df_labeled.shape)








# %% Balance data for only one genre
# Condition 1: Rows with first element of genres-onehot column equal to 1
condition_1 = df_labeled['genres-onehot'].apply(lambda x: x == 1)
# Select rows that satisfy condition 1
df_condition_1 = df_labeled[condition_1]

# Condition 2: Rows with genres-onehot equal to 0
condition_2 = df_labeled['genres-onehot'].apply(lambda x: x == 0)
# Select rows that satisfy condition 2
df_condition_2 = df_labeled[condition_2]

# Determine the minimum number of rows to select for each condition
min_rows = min(len(df_condition_1), len(df_condition_2))

# Randomly sample rows from each condition to match the minimum number of rows
df_condition_1_sampled = df_condition_1.sample(n=min_rows, random_state=1)
df_condition_2_sampled = df_condition_2.sample(n=min_rows, random_state=1)

# Concatenate the sampled DataFrames
df_handpicked = pd.concat([df_condition_1_sampled, df_condition_2_sampled], ignore_index=True)

# Print the resulting df_handpicked DataFrame
print("HANDPICKED SHAPE", df_handpicked.shape)
print(df_handpicked.head)

# %% Create dataset from pd dataframe

# Assuming you already have the DataFrame named df_labeled

genres_onehot_array = np.array(df_handpicked['genres-onehot'].tolist())

# Create TensorFlow dataset
dataset = tf.data.Dataset.from_tensor_slices((df_handpicked['Summary'], genres_onehot_array))

# Check cardinality
cardinality = tf.data.experimental.cardinality(dataset).numpy()
print("Cardinality:", cardinality)

print(dataset.element_spec)


# %% Preprocess and Train test split
# Calculate the sizes of the train and test sets

# Shuffle data
dataset = dataset.shuffle(cardinality)

prop_train_test = 0.65
train_size = int(prop_train_test * cardinality)
test_size = cardinality - train_size  # Remaining for testing
BATCH_SIZE = 50 

# Split data
train_dataset = dataset.take(train_size)
test_dataset = dataset.skip(train_size)

# Check sizes
train_cardinality = tf.data.experimental.cardinality(train_dataset).numpy()
test_cardinality = tf.data.experimental.cardinality(test_dataset).numpy()
print("Train Dataset Cardinality:", train_cardinality)
print("Test Dataset Cardinality:", test_cardinality)

train_dataset = train_dataset.shuffle(train_cardinality).batch(BATCH_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# train_dataset = train_dataset.map(lambda genres, summary: (genres, tf.squeeze(summary)))
# test_dataset = test_dataset.map(lambda genres, summary: (genres, tf.squeeze(summary)))




# %% Encode text with indecies
max_words_count = 2000 

encoder = tf.keras.layers.TextVectorization(
    max_tokens=max_words_count,
    output_mode='int',
    output_sequence_length=None
)
encoder.adapt(train_dataset.map(lambda text, label: text))

# Test encoder vocabulary
vocab = np.array(encoder.get_vocabulary())
print(vocab[:100])

# Test encoder one batch
for text, label in train_dataset.take(1):
    print('Text: ', text.numpy()[:3])
    print()
    print('Labels: ', label.numpy()[:3])
    print()
    encoded_ = encoder(text)[:3].numpy()
    print("Encoded: ", encoded_)
    for i in range(3):
        print("Encoded-Decoded: ", " ".join(vocab[encoded_[i]]))



# %% Create model

checkpoint_path = "training_7/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 save_best_only = True,
                                                 monitor = "loss",
                                                 verbose=1,
                                                 )

model = tf.keras.Sequential([
    encoder,
    tf.keras.layers.Embedding(
        input_dim=len(encoder.get_vocabulary()),
        output_dim=100,
        # Use masking to handle the variable sequence lengths
        mask_zero=True),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(100, kernel_regularizer=tf.keras.regularizers.l2(0.01), recurrent_regularizer=tf.keras.regularizers.l2(0.01))),
    # tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    # tf.keras.layers.GRU(32, kernel_regularizer=regularizers.l2(0.01), recurrent_regularizer=regularizers.l2(0.01)),
    tf.keras.layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(0.01)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(onehotlen, activation="sigmoid")
])


# %% Test small text
sample_text = df_labeled.iloc[0]["Summary"]
print("SAMPLE:", sample_text)
predictions = model.predict(np.array([sample_text]))
print("Predictions", predictions[0])

# %% Test text with padding
padding = "test" * 1000 
predictions = model.predict(np.array([sample_text, padding]))
print(predictions[0])

# %%

model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=[tf.keras.metrics.Precision(), tf.keras.metrics.BinaryAccuracy()])


# %%

model.optimizer.learning_rate.assign(0.01)
history = model.fit(train_dataset, epochs=3,
                    validation_data=test_dataset,
                    validation_steps=30,
                    callbacks=[cp_callback])

#%%
model.optimizer.learning_rate.assign(0.0001)

#%%

history2 = model.fit(train_dataset, epochs=15,
                    validation_data=test_dataset,
                    callbacks=[cp_callback],
                    validation_steps = 30,
                    initial_epoch=history.epoch[-1])



# %% Append history
history.history['loss'].extend(history2.history['loss'])
# history.history['accuracy'].extend(history2.history['accuracy'])
# history.history['val_accuracy'].extend(history2.history['val_accuracy'])
history.history['val_loss'].extend(history2.history['val_loss'])
# Update the epoch count in the combined history
history.epoch.extend(history2.epoch)


# %% 
model.save("model6-finetune")


# %%
res = model.evaluate(test_dataset)
print(res)
# print('Test Loss:', test_loss)
# print('Test Accuracy:', test_acc)

plt.figure(figsize=(16, 8))
plt.subplot(1, 2, 1)
# plot_graphs(history, 'accuracy')
plt.ylim(None, 1)
plt.subplot(1, 2, 2)
plot_graphs(history, 'loss')
plt.ylim(0, None)

#%%
print(history.history)






# %% Predictions

index = 5200
sample_text = df_handpicked.iloc[index]["Summary"]
print(df_handpicked.iloc[index])
predictions = model.predict(np.array([sample_text]))
print(predictions)
# %%
model.save('model1')
# %%
