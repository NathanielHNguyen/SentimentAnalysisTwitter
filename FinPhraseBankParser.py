import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import nltk
from nltk.corpus import stopwords

# Ensure that stopwords are downloaded
nltk.download('stopwords')
stop_words = stopwords.words('english')

# 1. Read the data from the text file
file_path = "FinancialPhraseBank-v1.0\FinancialPhraseBank-v1.0\Sentences_AllAgree.txt"
with open(file_path, 'r') as file:
    lines = file.readlines()

# 2. Preprocess the data (split sentences and labels)
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

# 3. Map sentiment labels to numerical values
label_map = {'positive': 1, 'neutral': 0, 'negative': -1}
numerical_labels = [label_map[label] for label in labels]

# 4. Create a pandas DataFrame for easier handling
data = pd.DataFrame({'text': sentences, 'sentiment': numerical_labels})

# 5. Text preprocessing (removing stopwords)
data['cleaned_text'] = data['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))

# 6. Split the data into features and target variable
X = data['cleaned_text']
y = data['sentiment']

# 7. Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 8. Vectorize the text data using TF-IDF
vectorizer = TfidfVectorizer(max_features=500)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 9. Train a machine learning model (Logistic Regression)
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

# 10. Evaluate the model
y_pred = model.predict(X_test_tfidf)
print(classification_report(y_test, y_pred))

print("With tweets from real time")
df = pd.read_csv('tweets_with_x.csv')
messages = df.iloc[1:51,2]

messagesvec = vectorizer.fit_transform(messages)

predictions = model.predict(messagesvec)
avg_sentiment = pd.Series(label_map).mode()[0]
print(f"The average sentiment of the stock is {avg_sentiment}")