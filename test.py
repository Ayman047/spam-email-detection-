import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

def load_dataset(csv_file):
    df = pd.read_csv(csv_file)
    texts = df['text'].tolist()
    labels = df['label'].tolist()
    return texts, labels

data_dir = r'C:\Users\Ayman\Desktop\spam email detection\spam_ham_dataset.csv'
texts, labels = load_dataset(data_dir)


texts, labels = load_dataset(data_dir)

# Step 2: Data Preprocessing
# No preprocessing needed in this example

# Step 3: Feature Extraction
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)
y = np.array(labels)

# Step 4: Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train the model
classifier = MultinomialNB()
classifier.fit(X_train, y_train)

# Step 6: Evaluate the model
y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:")
print(classification_report(y_test, y_pred))
