import tensorflow as tf
from tensorflow.keras import layers
#print("TensorFlow version:", tf.__version__)

import pandas as pd
import numpy as np
np.set_printoptions(precision=3, suppress=True)

normalize = layers.Normalization()

csv_data = (pd.read_csv("https://raw.githubusercontent.com/RyanPhitHub/Ryan-Comp-Eng-Projects/main/Objective%20Compaitibility%20Quiz%20(Responses)%20-%20Form%20Responses%201(2).csv", sep= ',', engine = 'python', header= 0)).drop(["Score"], axis = 1)
print(csv_data)
csv_features = csv_data.copy()
csv_labels = csv_features.pop('Total score')

inputs = {}

for name, column in csv_features.items():
  dtype = column.dtype
  if dtype == object:
    dtype = tf.string
  else:
    dtype = tf.float32
  inputs[name] = tf.keras.Input(shape=(1,), name=name, dtype=dtype)


numeric_inputs = {name:input for name,input in inputs.items() if input.dtype==tf.float32}

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

preprocessed_inputs_cat = layers.Concatenate()(preprocessed_inputs)
csv_preprocessing = tf.keras.Model(inputs, preprocessed_inputs_cat)
tf.keras.utils.plot_model(model = csv_preprocessing , rankdir="LR", dpi=72, show_shapes=True)

csv_features_dict = {name: np.array(value) for name, value in csv_features.items()}

features_dict = {name:values[:1] for name, values in csv_features_dict.items()}
print(features_dict)
csv_preprocessing(features_dict)

def csv_model(preprocessing_head, inputs):
  body = tf.keras.Sequential([
    layers.Dense(64),
    layers.Dense(1)
  ])

  preprocessed_inputs = preprocessing_head(inputs)
  result = body(preprocessed_inputs)
  model = tf.keras.Model(inputs, result)

  model.compile(loss=tf.keras.losses.MeanSquaredError(),
                optimizer=tf.keras.optimizers.Adam())
  return model

csv_model = csv_model(csv_preprocessing, inputs)
csv_model.fit(x=csv_features_dict, y=csv_labels, epochs=50, validation_split = 0.2)
