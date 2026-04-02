import pandas as pd
import nltk
from nltk.corpus import stopwords
import re
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

nltk.download('stopwords')

df = pd.read_csv('data/spam.csv', encoding='latin-1')
df = df[['v1', 'v2']]
#rename columns
df.columns = ['label', 'message']

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if w not in stop_words]
    return ' '.join(words)

df['cleaned'] = df['message'].apply(clean_text)
df['label'] = df['label'].map({'spam': 1, 'ham': 0})

x = df['cleaned']
y = df['label']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

tfidf = TfidfVectorizer(max_features=3000)
x_train_tfidf = tfidf.fit_transform(x_train)
x_test_tfidf = tfidf.transform(x_test)

model = MultinomialNB()
model.fit(x_train_tfidf, y_train)

y_pred = model.predict(x_test_tfidf)

print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nDetails Report: ")

print(classification_report(y_test, y_pred))

joblib.dump(model, 'spam_model.pkl')
joblib.dump(tfidf, 'tfidf_vectorizer.pkl')

print("Model saved successfully!")