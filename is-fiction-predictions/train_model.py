#%%
# Imports and utils
import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras import regularizers
import os
import matplotlib.pyplot as plt
import datetime

%load_ext tensorboard

#%%
# Load dataset
dataset_path = "dataset_is-fiction_summary_balanced.csv"

pandas_df = pd.read_csv(dataset_path)

# Tensorflow dataset
dataset = tf.data.Dataset.from_tensor_slices((pandas_df['Summary-short'], pandas_df['is-fiction']))
cardinality = tf.data.experimental.cardinality(dataset).numpy()

print("Cardinality:", cardinality)
print(dataset.element_spec)



# %%
# Setup train and test datasets
prop_train_test = 0.8
train_size = int(prop_train_test * cardinality)
test_size = cardinality - train_size  # Remaining for testing
BATCH_SIZE = 300 

dataset = dataset.shuffle(cardinality)

# Train test split
train_dataset = dataset.take(train_size)
test_dataset = dataset.skip(train_size)

# Check sizes
train_cardinality = tf.data.experimental.cardinality(train_dataset).numpy()
test_cardinality = tf.data.experimental.cardinality(test_dataset).numpy()
print("Train Dataset Cardinality:", train_cardinality)
print("Test Dataset Cardinality:", test_cardinality)

train_dataset = train_dataset.shuffle(train_cardinality).batch(BATCH_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)



# %%
# Setup encoder layer
max_words_count = 10000 

encoder = tf.keras.layers.TextVectorization(
    max_tokens=max_words_count,
    output_mode='int',
    output_sequence_length=None
)
encoder.adapt(train_dataset.map(lambda text, label: text))

# Test encoder vocabulary
vocab = np.array(encoder.get_vocabulary())
print("Vocebulary excerpt", vocab[:100])
print("Vocabulary size", len(vocab))



# %%
# Test encoder to check whether too much information is lost
def encode_decode_batch_test(batch):
    for text, label in batch:
        print('Text: ', text.numpy()[:3])
        print()
        print('Labels: ', label.numpy()[:3])
        print()
        encoded_ = encoder(text)[:3].numpy()
        print("Encoded: ", encoded_)
        for i in range(3):
            print("Encoded-Decoded: ", " ".join(vocab[encoded_[i]]))

encode_decode_batch_test(train_dataset.take(1))



# %%
# Create model
model = tf.keras.Sequential([
    encoder,
    tf.keras.layers.Embedding(
        input_dim=len(encoder.get_vocabulary()),
        output_dim=100,
        # Use masking to handle the variable sequence lengths
        mask_zero=True),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(100, kernel_regularizer=tf.keras.regularizers.l2(0.01), recurrent_regularizer=tf.keras.regularizers.l2(0.01))),
    tf.keras.layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(0.01)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(32, activation="relu", kernel_regularizer=regularizers.l2(0.01)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation="relu")
])



# %%
# Compile model

model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=[tf.keras.metrics.Accuracy(), tf.keras.metrics.BinaryAccuracy()])

def combine_histories(histories):
    combined_history = {}

    # Iterate over each history
    for history in histories:
        # Iterate over each metric in the history
        for metric, values in history.items():
            # Check if the metric exists in the combined history
            if metric in combined_history:
                combined_history[metric] += values
            else:
                combined_history[metric] = values

    return combined_history



# %%
# Setup loss improvement checkpoint
checkpoint_path = "training_7/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 save_best_only = True,
                                                 monitor = "loss",
                                                 verbose=1,
                                                 )




# %%
# Test checkpoints
print(os.listdir(checkpoint_dir))
latest = tf.train.latest_checkpoint(checkpoint_dir)
print(latest)




#%%
# Load model
# model.load_weights(latest)




# %%
# Train model with loss callback
model.optimizer.learning_rate.assign(0.01)
history = model.fit(train_dataset, epochs=10,
                    validation_data=test_dataset,
                    validation_steps=30,
                    callbacks=[cp_callback])




# %%
# Train with lower learning rate
model.optimizer.learning_rate.assign(0.001)
history2 = model.fit(train_dataset, epochs=20,
                    validation_data=test_dataset,
                    validation_steps=30,
                    callbacks=[cp_callback],
                    initial_epoch = history.epoch[-1])





#%%
# Train with lower learning rate
model.optimizer.learning_rate.assign(0.0001)
history3 = model.fit(train_dataset, epochs=30,
                    validation_data=test_dataset,
                    validation_steps=30,
                    callbacks=[cp_callback],
                    initial_epoch = history2.epoch[-1])



#%%

def plot_history(history):
    plt.figure(figsize=(12, 4))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Binary Precision
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Binary Precision')
    plt.legend()

    plt.tight_layout()
    plt.show()


plot_history(history)
plot_history(history2)



# %%
def test_model(index = 100):
    sample_text = pandas_df.iloc[index]["Summary-short"]
    print(pandas_df.iloc[index])
    predictions = model.predict(np.array([sample_text]))
    print(predictions)

test_model(560)
test_model(561)
test_model(562)
test_model(5000)
test_model(5001)
test_model(5002)
test_model(5003)



# %%
# Evaluate model:
test_loss, test_accuracy, test_binary_accuracy = model.evaluate(test_dataset)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)
print("Test Binary Accuracy:", test_binary_accuracy)



# %%
# Get predictions on training dataset
predictions = []
correct_labels = []

# predict each batch in test_dataset and save results
for text, label in test_dataset:
    batch_predictions = model.predict(text)
    
    predictions.extend(batch_predictions.flatten())
    correct_labels.extend(label.numpy())

# Combine prediction results to dataframe
predictions_df = pd.DataFrame({
    "Prediction": predictions,
    "CorrectLabel": correct_labels
})
print(predictions_df.head(50))



# %%
# Calculate binary precision
num_corr = 0
threshhold = 0.6
for index,row in predictions_df.iterrows():
    pred = row['Prediction']
    corr = row['CorrectLabel']
    if pred > threshhold and corr == 1 : num_corr +=1
    if pred < threshhold and corr == 0 : num_corr +=1

print(len(predictions_df))
print(num_corr)



# %%
# Prediction tests

predictions = model.predict(np.array(["Set in the closing months of world war 2. This is the story of a bombardier named Jusarian who is furious because thousands of people he has never met are trying to kill him. His real problem is not the enemy - it is his own army which keeps increasing the number of missions. The men must fly to complete their service."]))
print(predictions)

predictions = model.predict(np.array(["The book covers the various methods of charcuterie, including the ""brining, dry-curing, pickling, hot- and cold-smoking, sausage-making, confit, and the construction of pâtés"" that also involves more than 140 recipes for various dishes that have been made with the described methods."]))
print(predictions)

predictions = model.predict(np.array(["Over 800 photographs and diagrams show you how to do every decorating task and project around the house with complete confidence. Expect advice on essential home improvement tasks including: painting, wallpapering, tiling, building. Advice on how to choose and use the right tools for the job effectively and safely!"]))
print(predictions)
# %%
