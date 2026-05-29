from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load model
model = joblib.load("model/gender_model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

    name = request.form['name']

    name_vector = vectorizer.transform([name])

    prediction = model.predict(name_vector)[0]

    result = "Male" if prediction == "M" else "Female"

    return render_template(
        'index.html',
        prediction_text=f"Predicted Gender: {result}"
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)