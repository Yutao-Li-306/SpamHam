import pandas as pd
import numpy as np
import matplotlib as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

# read in the datasets and concat together
df_1 = pd.read_csv('email_spam_dataset/lingSpam.csv', usecols=["Body", "Label"])
df_2 = pd.read_csv('email_spam_dataset/enronSpamSubset.csv', usecols=["Body", "Label"])
df_3 = pd.read_csv('email_spam_dataset/completeSpamAssassin.csv', usecols=["Body", "Label"])

df = pd.concat([df_1, df_2, df_3], ignore_index=True)
df = df.dropna()

# df.head()

# get rid of "Subject: " in front of every email
df["Body"] = [text[9:].lower() for text in df["Body"]]

# df.head()

# here is where we split from the SVM, we use count vectorization to get numerical features
vectorizer=CountVectorizer()

spamham_countVectorizer=vectorizer.fit_transform(df['Body'])

# using the counterVectorizer as the input
X = spamham_countVectorizer
y = df['Label']

# split the dataset into train and test
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

# using Multinomial Naive Bayes model
NB_classifier=MultinomialNB()
NB_classifier.fit(X_train,y_train)

# running the model on test set
y_predict_test=NB_classifier.predict(X_test)

# print out the results using classification_report
print(classification_report(y_test,y_predict_test))