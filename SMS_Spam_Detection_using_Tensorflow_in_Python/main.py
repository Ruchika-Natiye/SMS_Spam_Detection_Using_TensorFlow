import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (classification_report, accuracy_score,precision_score, recall_score, f1_score, ConfusionMatrixDisplay)
# Load dataset
CSV_PATH = r"C:\Users\Dell\OneDrive\Desktop\python\SMS_Spam_Detection_using_Tensorflow_in_Python\spam.csv"
df = pd.read_csv(CSV_PATH, encoding='latin-1', low_memory=False)
# Some versions of the dataset have extra unnamed columns with spaces or different capitalization.
df = df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, errors='ignore')
df = df.rename(columns={'v1': 'label', 'v2': 'Text'})
# Drop rows with missing label/Text
df = df.dropna(subset=['label', 'Text']).reset_index(drop=True)
# Encode labels: ham -> 0, spam -> 1
df['label_enc'] = df['label'].map({'ham': 0, 'spam': 1}).astype(int)
print("Dataset sample:")
print(df.head())
# EDA plots
plt.figure(figsize=(6,4))
sns.countplot(x=df['label'])
plt.title("Label distribution")
plt.show()
# Basic text statistics
word_counts = df['Text'].apply(lambda s: len(str(s).split()))
avg_words_len = int(round(word_counts.mean()))
print("Average words per message:", avg_words_len)
# Unique words (simple split method)
unique_tokens = set()
for sent in df['Text'].astype(str):
    for w in sent.split():
        unique_tokens.add(w)
total_words_length = len(unique_tokens)
print("Total unique words (simple split):", total_words_length)
# Train-test split
X = df['Text'].astype(str)
y = df['label_enc'].astype(int).values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print("Train / Test shapes:", X_train.shape, y_train.shape, X_test.shape, y_test.shape)
# Baseline: TF-IDF + MultinomialNB
tfidf_vec = TfidfVectorizer(stop_words='english', max_df=0.95, min_df=2)
tfidf_vec.fit(X_train)
X_train_vec = tfidf_vec.transform(X_train)
X_test_vec = tfidf_vec.transform(X_test)
baseline_model = MultinomialNB()
baseline_model.fit(X_train_vec, y_train)
nb_preds = baseline_model.predict(X_test_vec)
nb_accuracy = accuracy_score(y_test, nb_preds)
print("\nNaive Bayes Accuracy:", nb_accuracy)
print(classification_report(y_test, nb_preds))
plt.figure(figsize=(6,5))
ConfusionMatrixDisplay.from_predictions(y_test, nb_preds)
plt.title("Naive Bayes Confusion Matrix (TF-IDF)")
plt.show()
# Helper functions for TF models
def compile_model(model):
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=['accuracy']
    )
def evaluate_model_tf(model, X_strings, y_true, batch_size=128):
    probs = model.predict(X_strings, batch_size=batch_size).ravel()
    preds = (probs >= 0.5).astype(int)
    return {
        'accuracy': accuracy_score(y_true, preds),
        'precision': precision_score(y_true, preds, zero_division=0),
        'recall': recall_score(y_true, preds, zero_division=0),
        'f1-score': f1_score(y_true, preds, zero_division=0)
    }
def fit_model(model, X_train_str, y_train_arr, X_val_str=None, y_val_arr=None,epochs=5, batch_size=32, callbacks=None, verbose=1):
    if (X_val_str is not None) and (y_val_arr is not None):
        history = model.fit(X_train_str, y_train_arr,validation_data=(X_val_str, y_val_arr),epochs=epochs, batch_size=batch_size,
        callbacks=callbacks, verbose=verbose)
    else:
        history = model.fit(X_train_str, y_train_arr, epochs=epochs,batch_size=batch_size, callbacks=callbacks, verbose=verbose)
    return history
# TextVectorization + Embedding model (model_1)
from tensorflow.keras.layers import TextVectorization, Embedding, GlobalAveragePooling1D, Dense, Input
# Safe caps
MAXTOKENS = min(max(total_words_length, 2000), 20000)
OUTPUTLEN = int(min(max(avg_words_len, 20), 100))
print(f"\nUsing MAXTOKENS={MAXTOKENS}, OUTPUTLEN={OUTPUTLEN}")
# TextVectorization - correct standardize string
text_vec = TextVectorization(max_tokens=MAXTOKENS,standardize='lower_and_strip_punctuation',output_mode='int',
output_sequence_length=OUTPUTLEN)
text_vec.adapt(X_train.values)  # adapt on training texts
embedding_dim = 128
embedding_layer = Embedding(input_dim=MAXTOKENS, output_dim=embedding_dim, name="embedding_layer")
# model_1
input_1 = Input(shape=(1,), dtype=tf.string, name="text_input_1")
x = text_vec(input_1)
x = embedding_layer(x)
x = GlobalAveragePooling1D()(x)
x = Dense(64, activation='relu')(x)
x = Dense(32, activation='relu')(x)
out = Dense(1, activation='sigmoid')(x)
model_1 = keras.Model(inputs=input_1, outputs=out, name="model_1_vec_embed")
model_1.compile(optimizer='adam',loss=keras.losses.BinaryCrossentropy(label_smoothing=0.0),metrics=['accuracy'])
model_1.summary()
es = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
history1 = fit_model(model_1,X_train.values, y_train.astype(float),X_val_str=X_test.values, y_val_arr=y_test.astype(float),epochs=10, batch_size=32, callbacks=[es], verbose=1)
loss1, acc1 = model_1.evaluate(X_test.values, y_test.astype(float), verbose=0)
print(f"Model_1 test accuracy: {acc1:.4f} (loss: {loss1:.4f})")
plt.figure(figsize=(8,4))
plt.plot(history1.history.get('accuracy', []), label='train_acc')
plt.plot(history1.history.get('val_accuracy', []), label='val_acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Model_1 Training vs Validation Accuracy')
plt.show()
model_1_results = evaluate_model_tf(model_1, X_test.values, y_test)
# Bidirectional LSTM model (model_2)
input_2 = Input(shape=(1,), dtype=tf.string, name="text_input_2")
x2 = text_vec(input_2)
x2 = embedding_layer(x2)
b = layers.Bidirectional(layers.LSTM(64, activation='tanh', return_sequences=True))(x2)
b2 = layers.Bidirectional(layers.LSTM(64))(b)
d = layers.Dropout(0.1)(b2)
d = layers.Dense(32, activation='relu')(d)
out2 = layers.Dense(1, activation='sigmoid')(d)
model_2 = keras.Model(inputs=input_2, outputs=out2, name="model_2_bilstm")
compile_model(model_2)
model_2.summary()
history2 = fit_model(model_2, X_train.values, y_train.astype(float),X_val_str=X_test.values, y_val_arr=y_test.astype(float),
epochs=5, batch_size=32, callbacks=[es], verbose=1)
model_2_results = evaluate_model_tf(model_2, X_test.values, y_test)
# (Optional) Universal Sentence Encoder via TF Hub (model_3)
# This part is optional: if tensorflow_hub isn't available or internet is blocked,
# the script will skip this model gracefully.
model_3_results = None
try:
    import tensorflow_hub as hub
    tf.config.optimizer.set_jit(False)
    input_3 = keras.Input(shape=(), dtype=tf.string, name="text_input_3")
    use_layer = hub.KerasLayer("https://tfhub.dev/google/universal-sentence-encoder/4",trainable=False, dtype=tf.string, name='USE')
    x3 = use_layer(input_3)  # (batch, 512)
    x3 = layers.Dropout(0.2)(x3)
    x3 = layers.Dense(64, activation='relu')(x3)
    out3 = layers.Dense(1, activation='sigmoid')(x3)
    model_3 = keras.Model(inputs=input_3, outputs=out3, name="model_3_USE")
    compile_model(model_3)
    model_3.summary()
    history3 = fit_model(model_3,X_train.values, y_train.astype(float),X_val_str=X_test.values, y_val_arr=y_test.astype(float),
    epochs=5, batch_size=32, callbacks=[es], verbose=1)
    model_3_results = evaluate_model_tf(model_3, X_test.values, y_test)
except Exception as e:
    print("\nSkipping USE model (tensorflow_hub not available or failed to load).")
    print("Reason:", str(e))
# Collate results
baseline_results = {
    'accuracy': accuracy_score(y_test, nb_preds),
    'precision': precision_score(y_test, nb_preds, zero_division=0),
    'recall': recall_score(y_test, nb_preds, zero_division=0),
    'f1-score': f1_score(y_test, nb_preds, zero_division=0)
}
results_dict = {
    'MultinomialNB (TF-IDF)': baseline_results,
    'TextVec+Embedding (model_1)': model_1_results,
    'BiLSTM (model_2)': model_2_results
}
if model_3_results is not None:
    results_dict['USE (transfer, model_3)'] = model_3_results
results_df = pd.DataFrame(results_dict).transpose()
print("\nEvaluation results (rows = models):")
print(results_df.round(4))
# Save results optionally
# results_df.to_csv("model_evaluation_results.csv", index=True)
