# Required libraries 
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import warnings
warnings.filterwarnings('ignore')


# Loading the Universal Sentence Encoder module
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

#function to get embeddings
def get_embeddings(sentences):
    return embed(sentences)


def model_training():
    df = pd.read_csv("input_data.csv")

    # Encoding text to tensors
    embeddings1 = get_embeddings(df['text1'].tolist())
    embeddings2 = get_embeddings(df['text2'].tolist())

    # Model architecture
    input_1 = tf.keras.Input(shape=(512,), dtype='float32')
    input_2 = tf.keras.Input(shape=(512,), dtype='float32')

    diff = tf.keras.layers.Subtract()([input_1, input_2])
    abs_diff = tf.keras.layers.Lambda(lambda x: tf.abs(x))(diff)

    concatenated = tf.keras.layers.Concatenate()([diff, abs_diff])

    dense1 = tf.keras.layers.Dense(256, activation='relu')(concatenated)
    dense2 = tf.keras.layers.Dense(128, activation='relu')(dense1)
    output = tf.keras.layers.Dense(1, activation='sigmoid')(dense2)

    model = tf.keras.Model(inputs=[input_1, input_2], outputs=output)
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

    # Model training
    X_train = [embeddings1, embeddings2]
    y_train = df['similarity_score'].values

    model.fit(X_train, y_train, epochs=20, validation_split=0.2)

    # Saving the model for future use
    model.save('similarity_model')
    print("model trained successfully !!!")

model_training()