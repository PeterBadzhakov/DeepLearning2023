# %%
# Imports
import tensorflow as tf
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
#%%
# Load model
model = load_model("model-good")


#%% 
# Load validation dataset as tensorflow dataset
val_df = pd.read_csv("validation_dataset.csv")
dataset =  tf.data.Dataset.from_tensor_slices((val_df['Summary-short'], val_df['is-fiction']))
cardinality = tf.data.experimental.cardinality(dataset).numpy()
print(cardinality)
print(dataset.element_spec)

BATCH_SIZE = 300
dataset = dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)




# %%
model.evaluate(dataset)
# %%
