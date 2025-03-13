import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv("E:\\Python\\LLMa\\names-origin.csv")

# Ensure there are no missing values
df.dropna(inplace=True)

# Convert names to lowercase
df['name'] = df['name'].str.lower()

# Encode target labels
df['origin'] = df['origin'].astype('category')
df['origin_code'] = df['origin'].cat.codes

# Feature extraction using TF-IDF (character-level n-grams)
vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 4))
X = vectorizer.fit_transform(df['name'])
y = df['origin_code']

# Train-test split (80-20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train Random Forest
clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(X_train, y_train)

# Predict and evaluate
y_pred = clf.predict(X_test)

# Convert numeric predictions back to actual origin labels
y_pred_origins = df['origin'].cat.categories[y_pred]

# Print accuracy & classification report
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=df['origin'].cat.categories))

# Function to predict origin of a given name
def predict_origin(name):
    name = name.lower()
    name_vectorized = vectorizer.transform([name])
    predicted_code = clf.predict(name_vectorized)[0]
    predicted_origin = df['origin'].cat.categories[predicted_code]
    return predicted_origin

# Main loop to take user input and predict origin
if __name__ == "__main__":
    while True:
        name = input("Enter a name (or type 'exit' to quit): ")
        if name.lower() == 'exit':
            break
        predicted_origin = predict_origin(name)
        print(f"Predicted origin for '{name}': {predicted_origin}\n")









