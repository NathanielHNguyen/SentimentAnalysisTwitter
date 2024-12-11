import numpy as np
import pandas as pd
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Ensure stopwords are downloaded
nltk.download('stopwords')
stop_words = stopwords.words('english')

# 1. Read the data from the text file
file_path = "FinancialPhraseBank-v1.0\FinancialPhraseBank-v1.0\Sentences_AllAgree.txt"  # Adjust path
with open(file_path, 'r') as file:
    lines = file.readlines()

# 2. Preprocess the data (split sentences and labels based on '@' character)
sentences = []
labels = []

for line in lines:
    line = line.strip()  # Remove leading/trailing whitespaces
    if '@' in line:  # Check if the line contains the '@' symbol
        # Split the sentence and the label based on '@'
        sentence, label = line.rsplit('@', 1)
        sentence = sentence.strip()  # Clean up the sentence

        # Check the last characters in the label to determine sentiment
        if label.startswith('pos'):
            sentiment = 'positive'
        elif label.startswith('neu'):
            sentiment = 'neutral'
        elif label.startswith('neg'):
            sentiment = 'negative'
        else:
            sentiment = 'unknown'  # Handle unexpected cases

        # Append the sentence and its corresponding sentiment label
        sentences.append(sentence)
        labels.append(sentiment)

# 3. Preprocess the sentiment labels (convert to numerical labels)
label_encoder = LabelEncoder()
numerical_labels = label_encoder.fit_transform(labels)

# 4. Preprocess text data (remove stopwords)
data = pd.DataFrame({'text': sentences, 'sentiment': numerical_labels})
data['cleaned_text'] = data['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))

# 5. Vectorize the text data using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(data['cleaned_text']).toarray()
y = data['sentiment']

# 6. Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7. Define the neural network model
model = Sequential()
model.add(Dense(512, input_dim=X_train.shape[1], activation='relu'))  # Input layer
model.add(Dropout(0.5))  # Dropout layer for regularization
model.add(Dense(256, activation='relu'))  # Hidden layer
model.add(Dropout(0.5))  # Dropout layer for regularization
model.add(Dense(3, activation='softmax'))  # Output layer (3 classes: positive, neutral, negative)

# 8. Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

# 9. Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# 10. Evaluate the model on the test set
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {accuracy * 100:.2f}%')

# 11. Make predictions on new, unseen data
new_data = [
    "The company's financial outlook is very positive.",
    "There are some concerns about the market performance.",
    "The sales are disappointing this quarter."
]

# Preprocess new data
new_data_cleaned = [' '.join([word for word in sentence.split() if word not in stop_words]) for sentence in new_data]
new_data_tfidf = vectorizer.transform(new_data_cleaned).toarray()

# Predict sentiment of new data
new_predictions = model.predict(new_data_tfidf)
predicted_labels = label_encoder.inverse_transform(np.argmax(new_predictions, axis=1))

# Output the predictions
for sentence, label in zip(new_data, predicted_labels):
    print(f"Sentence: '{sentence}' -> Predicted Sentiment: {label}")
