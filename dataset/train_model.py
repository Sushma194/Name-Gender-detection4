import pandas as pd
import joblib
import os

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Create model folder if not exists
if not os.path.exists("model"):
    os.makedirs("model")

# Load CSV
data = pd.read_csv("name_gender.csv")

# Features and labels
X = data["name"]
y = data["gender"]

# Convert text to features
vectorizer = CountVectorizer(
    analyzer='char',
    ngram_range=(1, 3)
)

X_vectorized = vectorizer.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Accuracy
predictions = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    predictions
)

print("Accuracy:", accuracy * 100)

# Save files properly
joblib.dump(model, "model/gender_model.pkl")
joblib.dump(vectorizer, "model/vectorizer.pkl")

print("Model Saved Successfully")