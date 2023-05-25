import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

application = Flask(__name__) #Initialize the flask App
model = pickle.load(open('model.pkl', 'rb'))

@application.route('/')
def home():
    return render_template('index.html')

# when the predict button is pressed, what happens
@application.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''

    print("eroor here 1")
    # grabbing the input values from the form
    email = str(request.form['email'])
    print("error here 2")
    # int_features = [int(x) for x in request.form.values()]
    # final_features = [np.array(int_features)]
    # probably have to put the email into a pandas dataframe
    vectorizer = TfidfVectorizer()

    # if the above doesn't solve the problem,
    # may have to find a way to save the vectorizer created from model.py
    

    print("error here 2.5")
    print("email: ", email)
    print("email: ", type(email))

    # problem is in line 32
    # becomes a list of 1 element, which is a string
    # need to make it a list of 1 element, which is a list of strings
    input = vectorizer.fit_transform(email)

    print("error here 2.9")
    prediction = model.predict(input)

    print("error here 3")
    if prediction == 1:
        output = "Spam"
    else:
        output = "Ham (Not Spam)"

    #output = round(prediction[0], 2)

    return render_template('index.html', prediction_text=1)

if __name__ == "__main__":
    application.run(debug=True)
