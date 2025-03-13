import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle

# Load dataset
df = pd.read_csv("E:\\Python\\LLMa\\names-origin.csv")

# Data preprocessing
df.dropna(inplace=True)
df['name'] = df['name'].str.lower()

# Encode target labels
label_encoder = LabelEncoder()
df['origin_code'] = label_encoder.fit_transform(df['origin'])

# Tokenization
tokenizer = Tokenizer(char_level=True)  # Character-based tokenization
tokenizer.fit_on_texts(df['name'])
X = tokenizer.texts_to_sequences(df['name'])
X = pad_sequences(X, maxlen=20, padding='post')  # Pad sequences to max length

y = df['origin_code']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Define LSTM model
model = keras.Sequential([
    Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=64, input_length=20),
    Bidirectional(LSTM(128, return_sequences=True)),
    LSTM(64),
    Dense(32, activation='relu'),
    Dense(len(label_encoder.classes_), activation='softmax')
])

# Compile model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, epochs=15, batch_size=32, validation_data=(X_test, y_test))

# Evaluate model
loss, accuracy = model.evaluate(X_test, y_test)
print("LSTM Model Accuracy:", accuracy)

# Save the tokenizer and model
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)
model.save("name_origin_model.h5")

# ====== PREDICTION & OUTPUT NAME-ORIGIN DATA ====== #
# Predict on test data
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)  # Get predicted class indices

# Convert numeric labels back to actual origin names
y_test_origins = label_encoder.inverse_transform(y_test)
y_pred_origins = label_encoder.inverse_transform(y_pred_classes)

# Retrieve original names from the dataset
test_names = df.iloc[y_test.index]['name'].values  # Get corresponding names

# Create a DataFrame for comparison
results_df = pd.DataFrame({
    'Name': test_names,
    'Actual Origin': y_test_origins,
    'Predicted Origin': y_pred_origins
})

# Save results to a CSV file
results_df.to_csv("name_origin_predictions.csv", index=False)

# Print a sample of the results
print("\nSample Predictions:")
print(results_df.head(20))  # Display first 20 results



