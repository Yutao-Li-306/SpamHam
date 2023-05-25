import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from gensim.models import Word2Vec
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV

# open datasets and concatentate data together
df_1 = pd.read_csv('email_spam_dataset/lingSpam.csv', usecols=["Body", "Label"])
df_2 = pd.read_csv('email_spam_dataset/enronSpamSubset.csv', usecols=["Body", "Label"])
df_3 = pd.read_csv('email_spam_dataset/completeSpamAssassin.csv', usecols=["Body", "Label"])

df = pd.concat([df_1, df_2, df_3], ignore_index=True)
df = df.dropna()
# df.head()

# get rid of "Subject: " in front of every email
df["Body"] = [text[9:].lower() for text in df["Body"]]
# df.head()

# label 0 = not spam, 1 = spam
label = ["Not Spam", "Spam"]
label_counts = df["Label"].value_counts()

X = df["Body"]

# print(X)
# print(type(X))

y = df["Label"]

# print(X[0])
# print(type(X[0]))
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Convert the text data into numerical features using TF-IDF vectorization:
vectorizer = TfidfVectorizer()
# print(type(X_train))
X_train = vectorizer.fit_transform(X_train)
# print(type(X_train))
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

# Make predictions on the test data using the best model
#best_model = grid_search.best_estimator_
#y_pred = best_model.predict(X_test)


# Make predictions on the test data
y_pred = svm.predict(X_test)

# Calculate the accuracy score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report: ")
print(classification_report(y_test, y_pred))


