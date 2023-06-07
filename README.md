# SpamHam

Spam Email Classification
Deploying ML Model using Flask

Prerequisites

You must have Scikit Learn, Pandas (for Machine Leraning Model) and Flask (for API) installed.

Flask version: 0.12.2 conda install flask=0.12.2 (or) pip install Flask==0.12.2
Project Structure

Running the project

    Ensure that you are in the ML_website_frontend directory. Create the machine learning model by running below command from command prompt -

python model.py or python3 model.py

This would create a serialized version of our model into a file model.pkl and vectorizer.pkl

    Run application.py using below command to start Flask API

python application.py or python3 application.py

By default, flask will run on port 5000.

    Navigate to URL http://127.0.0.1:5000/ (or) http://localhost:5000

You should be able to view the homepage.

Enter the body of your email into the input box and hit Predict.

If everything goes well, you should be able to see the predicted outcome: Spam or Ham (Not Spam) below the Predict button.

Frontend website modified from template provided by MaajidKhan from: https://github.com/MaajidKhan/DeployMLModel-Flask
