import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("spam.csv", encoding="latin-1")

# Keep only required columns
data = data[['v1', 'v2']]
data.columns = ['label', 'message']

# Convert labels
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# Split data
X = data['message']
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Convert text into TF-IDF features
vectorizer = TfidfVectorizer(stop_words='english')

X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("=" * 50)
print("Spam SMS Detection")
print("=" * 50)
print(f"Model Accuracy: {accuracy*100:.2f}%")
print("=" * 50)

# User input
while True:
    sms = input("\nEnter an SMS (or type 'exit' to quit): ")

    if sms.lower() == "exit":
        print("Thank you!")
        break

    sms_vector = vectorizer.transform([sms])
    prediction = model.predict(sms_vector)

    if prediction[0] == 1:
        print("Prediction: SPAM")
    else:
        print("Prediction: NOT SPAM")