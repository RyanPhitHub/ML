import tensorflow as tf
from tensorflow.keras import layers
#print("TensorFlow version:", tf.__version__)

import pandas as pd
import numpy as np
np.set_printoptions(precision=3, suppress=True)

normalize = layers.Normalization()


csv_data = pd.read_csv("https://raw.githubusercontent.com/RyanPhitHub/Ryan-Comp-Eng-Projects/main/test.csv", sep= ',', engine = 'python', header= 0)
csv_train= csv_data.iloc[:2, :9]
csv_features = csv_data.copy()
csv_labels = csv_features.pop(' Score')

inputs = {}

for name, column in csv_features.items():
  dtype = column.dtype
  if dtype == object:
    dtype = tf.string
  else:
    dtype = tf.float32

  inputs[name] = tf.keras.Input(shape=(1,), name=name, dtype=dtype)


numeric_inputs = {name:input for name,input in inputs.items()
                  if input.dtype==tf.float32}

x = layers.Concatenate()(list(numeric_inputs.values()))
norm = layers.Normalization()
norm.adapt(np.array(csv_data[numeric_inputs.keys()]))
all_numeric_inputs = norm(x)

preprocessed_inputs = [all_numeric_inputs]



for name, input in inputs.items():
  if input.dtype == tf.float32:
    continue

  lookup = layers.StringLookup(vocabulary=np.unique(csv_features[name]))
  one_hot = layers.CategoryEncoding(num_tokens=lookup.vocabulary_size(), output_mode = 'one_hot')
  x = lookup(input)
  x = one_hot(x)
  preprocessed_inputs.append(x)
  print(x)

preprocessed_inputs_cat = layers.Concatenate()(preprocessed_inputs)
csv_preprocessing = tf.keras.Model(inputs, preprocessed_inputs_cat)
tf.keras.utils.plot_model(model = csv_preprocessing , rankdir="LR", dpi=72, show_shapes=True)

print(one_hot)
