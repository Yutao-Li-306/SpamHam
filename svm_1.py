import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# open datasets and concatentate data together
df_1 = pd.read_csv('lingSpam.csv', usecols=["Body", "Label"])
df_2 = pd.read_csv('enronSpamSubset.csv', usecols=["Body", "Label"])
df_3 = pd.read_csv('completeSpamAssassin.csv', usecols=["Body", "Label"])

df = pd.concat([df_1, df_2, df_3], ignore_index=True)
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

# Initialize the CountVectorizer
vectorizer = CountVectorizer()

# Fit and transform the training data
X_train_features = vectorizer.fit_transform(X_train)

# Transform the testing data
X_test_features = vectorizer.transform(X_test)

# Initialize the SVM classifier
svm = SVC()

# Train the SVM model
svm.fit(X_train_features, y_train)

# Make predictions on the test data
y_pred = svm.predict(X_test_features)

# Calculate the accuracy score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


