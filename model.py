import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load the preprocessed and tokenized dataset (you may need to adjust the path)
df = pd.read_csv("spam.csv")

# Extract features and labels
X = df["Message"]
y = df["Category"]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a TF-IDF vectorizer to convert text data into numerical features
vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')  # Adjust max_features as needed
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Initialize and train the Multinomial Naive Bayes classifier
clf = MultinomialNB()
clf.fit(X_train_vectorized, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test_vectorized)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Display a classification report
print(classification_report(y_test, y_pred))

# Save the trained model and vectorizer to a pickle file
joblib.dump((clf, vectorizer), "spam_detection_model.pkl")
