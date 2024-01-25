# Required libraries 
from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Loading the Universal Sentence Encoder module
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

#function to get embeddings
def get_embeddings(sentences):
    return embed(sentences)


# Define the function for predicting similarity
@app.route('/predict', methods=['POST'])
def predict():
    # Loading the model
    try:
        loaded_model = tf.keras.models.load_model('similarity_model')
    except:
        loaded_model = None

    data = request.json
    sentence1 = data['text1']
    sentence2 = data['text2']

    embedding1 = get_embeddings([sentence1])[0].numpy()
    embedding2 = get_embeddings([sentence2])[0].numpy()

    embedding1 = np.reshape(embedding1, (1, -1))
    embedding2 = np.reshape(embedding2, (1, -1))

    if loaded_model is None:
        return jsonify({"error": "Model is not trained yet."}), 400

    similarity = loaded_model.predict([embedding1, embedding2])[0][0]
    similarity = round(float(similarity), 1)

    return jsonify({"similarity score": similarity})

if __name__ == '__main__':
    app.run(debug=True)