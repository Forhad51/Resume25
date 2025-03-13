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
y_test_origins = df['origin'].cat.categories[y_test]
y_pred_origins = df['origin'].cat.categories[y_pred]

# Retrieve original names for test samples
test_names = df.iloc[y_test.index]['name'].values  

# Create DataFrame for results
results_df = pd.DataFrame({
    'Name': test_names,
    'Actual Origin': y_test_origins,
    'Predicted Origin': y_pred_origins
})

# Save results to CSV
results_df.to_csv("name_origin_predictions.csv", index=False)

# Print accuracy & classification report
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=df['origin'].cat.categories))

# Print sample output
print("\nSample Predictions:")
print(results_df.head(20))  # Display first 20 results









