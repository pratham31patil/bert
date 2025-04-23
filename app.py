from flask import Flask, request, render_template
import numpy as np
import tensorflow as tf
from sentence_transformers import SentenceTransformer

app = Flask(__name__)

# Load pre-trained models
bert_model = SentenceTransformer('distilbert-base-nli-mean-tokens')
classifier = tf.keras.models.load_model("model.h5")

# Store recent predictions
recent_predictions = []

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text'].strip().strip('"').strip("'")
    embedding = bert_model.encode([text])
    prediction = classifier.predict(embedding)

    if 0.45 <= prediction[0][0] <= 0.55:
        result = "Neutral"
        probability = round(float(prediction[0][0]) * 100, 2)
    else:
        result = "Depressed" if prediction[0][0] > 0.5 else "Not Depressed"
        probability = round(float(prediction[0][0]) * 100, 2) if result == "Depressed" else round(float((1 - prediction[0][0])) * 100, 2)

    recent_predictions.append({
        'text': text,
        'prediction': result,
        'probability': probability
    })

    if len(recent_predictions) > 5:
        recent_predictions.pop(0)

    return render_template('result.html', 
        input_text=text, 
        prediction=result, 
        probability=probability,
        recent_predictions=recent_predictions
    )

if __name__ == "__main__":
    app.run(debug=True)
