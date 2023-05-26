# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from gensim.models import Word2Vec
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV

# open datasets and concatentate data together
df_1 = pd.read_csv('../email_spam_dataset/lingSpam.csv', usecols=["Body", "Label"])
df_2 = pd.read_csv('../email_spam_dataset/enronSpamSubset.csv', usecols=["Body", "Label"])
df_3 = pd.read_csv('../email_spam_dataset/completeSpamAssassin.csv', usecols=["Body", "Label"])

df = pd.concat([df_1, df_2, df_3], ignore_index=True)
df = df.dropna()

# get rid of "Subject: " in front of every email
df["Body"] = [text[9:].lower() for text in df["Body"]]

# label 0 = not spam, 1 = spam
label = ["Not Spam", "Spam"]
label_counts = df["Label"].value_counts()

X = df["Body"]
y = df["Label"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Convert the text data into numerical features using TF-IDF vectorization:
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)


# Define the parameter grid for grid search
param_grid = {
    'C': [0.1, 1, 10],
    'gamma': [0.1, 1, 10]
}


# Initialize the SVM classifier
svm = SVC(kernel='rbf', C=10, gamma=0.1)

# Perform grid search with cross-validation
#grid_search = GridSearchCV(svm, param_grid, cv=5)
#grid_search.fit(X_train, y_train)

# Print the best hyperparameters and the corresponding accuracy
#print("Best Hyperparameters: ", grid_search.best_params_)
#print("Best Accuracy: ", grid_search.best_score_)


# Train the SVM model
svm.fit(X_train, y_train)



# Saving model to disk
pickle.dump(svm, open('model.pkl','wb'))

pickle.dump(vectorizer, open('vectorizer.pkl','wb'))

'''
# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[2, 9, 6]]))
'''