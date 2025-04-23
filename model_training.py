import pandas as pd
import numpy as np
import tensorflow as tf
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("depression_dataset_reddit_cleaned.csv")
tweets = df['clean_text'].values
labels = df['is_depression'].astype(float).values

# Load BERT model
bert_model = SentenceTransformer('distilbert-base-nli-mean-tokens')

# Convert text data to embeddings
embeddings = bert_model.encode(tweets, show_progress_bar=True)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(embeddings, labels, test_size=0.2, random_state=42)

# Build classifier model
classifier = tf.keras.Sequential([
    tf.keras.layers.Dense(256, activation='relu', input_shape=(768,)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
classifier.fit(X_train, y_train, epochs=30, batch_size=16, validation_data=(X_test, y_test))

# Save the trained model
classifier.save("model.h5")
print("Model saved successfully!")
