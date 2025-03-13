import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle

# Load dataset
df = pd.read_csv("E:\\Python\\LLMa\\names-origin.csv")

# Data preprocessing
df.dropna(inplace=True)
df['name'] = df['name'].str.lower().str.replace(r'[^a-z]', '', regex=True)  # Remove non-alphabetic characters
df = df[df['name'].str.len() > 1]  # Remove very short names

# Encode target labels
label_encoder = LabelEncoder()
df['origin_code'] = label_encoder.fit_transform(df['origin'])
label_classes = label_encoder.classes_  # Store label classes

# Tokenization
tokenizer = Tokenizer(char_level=True, oov_token="<OOV>")  # Handle unseen characters
tokenizer.fit_on_texts(df['name'])
X = tokenizer.texts_to_sequences(df['name'])
X = pad_sequences(X, maxlen=20, padding='post')  # Pad sequences to max length
y = df['origin_code']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Define LSTM model
model = keras.Sequential([
    Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=128, input_length=20),
    Bidirectional(LSTM(128, return_sequences=True)),
    Dropout(0.3),
    LSTM(64),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(len(label_classes), activation='softmax')
])

# Compile model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# Evaluate model
loss, accuracy = model.evaluate(X_test, y_test)
print("LSTM Model Accuracy:", accuracy)

# Save the tokenizer and model
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)
with open("label_classes.pkl", "wb") as f:
    pickle.dump(label_classes, f)
model.save("name_origin_model.h5")

# ====== PREDICTION & OUTPUT NAME-ORIGIN DATA ====== #
# Load tokenizer and label classes
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)
with open("label_classes.pkl", "rb") as f:
    label_classes = pickle.load(f)

# Load model
model = keras.models.load_model("name_origin_model.h5")

# Get user input name and predict origin
while True:
    user_input = input("Enter a name (or type 'exit' to stop): ").strip().lower()
    if user_input == 'exit':
        break
    
    # Clean input
    user_input_clean = ''.join([char for char in user_input if char.isalpha()])
    if not user_input_clean:
        print("Invalid input. Please enter a valid name.")
        continue
    
    # Convert name to sequences
    user_input_seq = tokenizer.texts_to_sequences([user_input_clean])
    
    # Handle case where all characters are unknown
    if not any(user_input_seq[0]):  
        print("No recognizable characters in input. Unable to predict.")
        continue

    user_input_seq = pad_sequences(user_input_seq, maxlen=20, padding='post')

    # Predict origin
    prediction = model.predict(user_input_seq)
    predicted_origin_code = np.argmax(prediction)

    # Convert code to actual origin
    if predicted_origin_code < len(label_classes):
        predicted_origin = label_classes[predicted_origin_code]
    else:
        predicted_origin = "Unknown Origin"
    
    print(f"Predicted Origin: {predicted_origin}\n")



