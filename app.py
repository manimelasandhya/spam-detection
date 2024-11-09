from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib

app = Flask(__name__)

# Load the trained model and vectorizer (old model)
loaded_model, vectorizer = joblib.load("spam_detection_model.pkl")

# Load the new dataset (replace with your new dataset path)
new_df = pd.read_csv("spam.csv")

# Extract features and labels from the new dataset
X_new = new_df["Message"]
y_new = new_df["Category"]

# Create a new TF-IDF vectorizer for the new dataset
new_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
X_new_vectorized = new_vectorizer.fit_transform(X_new)

# Train a new Multinomial Naive Bayes classifier on the new dataset
new_model = MultinomialNB()
new_model.fit(X_new_vectorized, y_new)

# Save the new model and vectorizer
joblib.dump((new_model, new_vectorizer), "spam_detection_model.pkl")

# Create a route to display the form
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Get the message input from the form
        message = request.form.get("message")

        if message:
            # Preprocess the message using the new vectorizer
            message_vectorized = new_vectorizer.transform([message])

            # Use the new model to make a prediction
            prediction = new_model.predict(message_vectorized)[0]

            # Map numerical prediction to labels
            if prediction == 'spam':
                result = 'Spam'
            else:
                result = 'Not Spam'

            return render_template("result.html", message=message, prediction=result)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
