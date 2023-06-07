import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

application = Flask(__name__) #Initialize the flask App
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

@application.route('/')
def home():
    return render_template('index.html')

# when the predict button is pressed, what happens
@application.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''

    # grabbing the input values from the form
    email = str(request.form['email'])

    # transforming the input into numerical features using pre-loaded vectorizer
    input = vectorizer.transform([email])

    # predicting the input using pre-loaded model
    prediction = model.predict(input)

    if prediction[0] == 1:
        output = "Spam"
    else:
        output = "Ham (Not Spam)"

    return render_template('index.html', prediction_text=output)

if __name__ == "__main__":
    application.run(debug=True)
