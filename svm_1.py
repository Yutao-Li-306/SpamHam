import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from gensim.models import Word2Vec
from sklearn.metrics import classification_report

# open datasets and concatentate data together
df = pd.read_csv('lingSpam.csv', usecols=["Body", "Label"])
#df_2 = pd.read_csv('enronSpamSubset.csv', usecols=["Body", "Label"])
#df_3 = pd.read_csv('completeSpamAssassin.csv', usecols=["Body", "Label"])

#df = pd.concat([df_1, df_2, df_3], ignore_index=True)
df = df.dropna()
df.head()

# get rid of "Subject: " in front of every email
df["Body"] = [text[9:].lower() for text in df["Body"]]
df.head()

# label 0 = not spam, 1 = spam
label = ["Not Spam", "Spam"]
label_counts = df["Label"].value_counts()

X = df["Body"]
y = df["Label"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Tokenize the text data
tokenized_data_train = [text.split() for text in X_train]
tokenized_data_test = [text.split() for text in X_test]

# Train Word2Vec model
word2vec_model = Word2Vec(sentences=tokenized_data_train, vector_size=100, window=5, min_count=1, workers=4)

# Convert training data to Word2Vec vectors
X_train_features = np.array([np.mean([word2vec_model.wv[word] for word in text], axis=0) for text in tokenized_data_train])

# Convert testing data to Word2Vec vectors
X_test_features = np.array([np.mean([word2vec_model.wv[word] for word in text], axis=0) for text in tokenized_data_test])

# Initialize the SVM classifier
svm = SVC(kernel='rbf')

# Train the SVM model
svm.fit(X_train_features, y_train)

# Make predictions on the test data
y_pred = svm.predict(X_test_features)

# Calculate the accuracy score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report: ")
#print(classification_report(y_test, svm.predict(y_test)))


