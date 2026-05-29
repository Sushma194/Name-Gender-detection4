# Name Gender Detection using Machine Learning

A Machine Learning web application that predicts gender based on a person's name using Flask and Scikit-learn.

## Project Overview

This project uses a Machine Learning model to predict whether a given name belongs to a **Male** or **Female**. The application is built using **Python, Flask, HTML, CSS, and Machine Learning**.

Users can enter a name in the web interface, and the model predicts the gender instantly.

## Features

- Gender prediction using name
- Machine Learning-based classification
- Simple Flask web application
- User-friendly interface
- Fast predictions
- Clean UI with CSS styling

## Technologies Used

- Python
- Flask
- Machine Learning
- Scikit-learn
- Pandas
- Joblib
- HTML
- CSS

## Machine Learning Model Used

The project uses:

- **CountVectorizer** → Converts names into text features
- **Multinomial Naive Bayes** → Classification algorithm

## Project Structure

```txt
Name-Gender-Detection/
│── model/
│   ├── gender_model.pkl
│   └── vectorizer.pkl
│
│── static/
│   └── style.css
│
│── templates/
│   └── index.html
│
│── name_gender.csv
│── train_model.py
│── app.py
│── requirements.txt
│── README.md
```

## Dataset

The dataset contains male and female names used for training the machine learning model.

Example:

```csv
name,gender
Rahul,M
Priya,F
Ravi,M
Sushma,F
```

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/Name-Gender-Detection.git
```

### 2. Navigate to Project Folder

```bash
cd Name-Gender-Detection
```

### 3. Install Required Libraries

```bash
pip install -r requirements.txt
```

## Train the Model

Run the training file:

```bash
python train_model.py
```

After running, model files will be created automatically:

```txt
model/
│── gender_model.pkl
│── vectorizer.pkl
```

## Run the Flask Application

Start the Flask server:

```bash
python app.py
```

Open browser:

```txt
http://127.0.0.1:5000
```

## Example Prediction

Input:

```txt
Rahul
```

Output:

```txt
Predicted Gender: Male
```

Input:

```txt
Priya
```

Output:

```txt
Predicted Gender: Female
```

## Expected Accuracy

The model accuracy depends on dataset size.

Expected accuracy: **80–90%**

## Future Improvements

- Add larger dataset for better accuracy
- Add confidence score
- Deploy project online
- Improve UI design
- Add API support

## Author

**Sushma Munagala**

GitHub: https://github.com/Sushma194/Name-Gender-detection4
