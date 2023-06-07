# SpamHam
Spam Email Classification
Deploying ML Model using Flask

Prerequisites

You must have Scikit Learn, Pandas (for Machine Leraning Model) and Flask (for API) installed.

Flask version: 0.12.2 conda install flask=0.12.2 (or) pip install Flask==0.12.2
Project Structure

Running the project

    Ensure that you are in the project home directory. Create the machine learning model by running below command from command prompt -

python model.py

This would create a serialized version of our model into a file model.pkl

    Run app.py using below command to start Flask API

python app.py

By default, flask will run on port 5000.

    Navigate to URL http://127.0.0.1:5000/ (or) http://localhost:5000

You should be able to view the homepage.

Enter a valid email in the input box and hit predit.

If everything goes well, you should be able to see whether or not the email is a ham email or a span email. Check the output here: http://127.0.0.1:5000/predict
