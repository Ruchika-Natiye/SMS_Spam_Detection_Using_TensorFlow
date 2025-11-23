# SMS_Spam_Detection_Using_TensorFlow

## 📱 SMS Spam Detection using TensorFlow
A machine‑learning project that classifies SMS messages as Spam or Ham (Not Spam) using Deep Learning, TensorFlow, and Natural Language Processing (NLP) techniques. This README provides a clean, professional GitHub‑ready documentation with project details, setup instructions, step‑by‑step code explanation, and usage guide.

## 🚀 Project Overview
The goal of this project is to build a text‑classification model that detects spam SMS messages. We use:

* NLP preprocessing (cleaning text, removing stopwords, tokenizing)

* Label Encoding for target conversion (spam = 1, ham = 0)

* Text Vectorization using TensorFlow

* Deep Learning model based on Embedding + Dense layers

* Model Evaluation using Accuracy & Loss graphs

## 📂 Dataset
The project uses the classic SMS Spam Collection Dataset.It contains ~5,500 SMS messages labeled as spam or ham.
Sample:
![img alt](https://github.com/Ruchika-Natiye/SMS_Spam_Detection_Using_TensorFlow/blob/f86b2f506f1347f6248e6a0c5a1786f86ea4a701/i1.png)

## 🧰 Tech Stack
| Category | Tools |
| :------ | ------: |
| Language | Python |
| ML Framework | TensorFlow / Keras |
| NLP Tools |	Tokenizer, Text Vectorization |
| Visualization	| Matplotlib |
| Data Handling |	Pandas, NumPy |

## 📑 Step‑by‑Step Code Explanation
### 1. Import Libraries
```python
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
```
### 2. Load Dataset
```python
df = pd.read_csv('spam.csv', encoding='latin-1')
df = df[['v1', 'v2']]
df.columns = ['label', 'message']
```
### 3. Data Preprocessing
![img alt]()

### 4. Visualize Ham & Spam Data
```python
sns.countplot(x=df['label'])
plt.show()
```
![img alt]()

### 5. Train‑Test Split
Divides dataset for training and testing.
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y) print("Train / Test shapes:", X_train.shape, y_train.shape, X_test.shape, y_test.shape)
```
### 6. Performance of baseline model
![img alt]()

### 7. Confusion matrix for the baseline model
![img alt]()

### 8. Build Deep Learning Model
```python
model = tf.keras.Sequential([vectorizer,tf.keras.layers.Embedding(input_dim=5000, output_dim=16),tf.keras.layers.GlobalAveragePooling1D(),tf.keras.layers.Dense(16, activation='relu'),tf.keras.layers.Dense(1, activation='sigmoid')])
```

### 9. Compile Model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

### 10. Summary of Model
![img alt]()

### 11. Visualize Accuracy & Loss
![img alt]()

### 12. Plotting the results
![img alt]()

## 📊 Results
* Achieved high accuracy depending on training duration & dataset.

* Creates a lightweight, production‑ready TensorFlow spam detector.

## 🔮 Future Improvements
* Deploy with Flask or FastAPI

* Add LSTM/GRU model

* Use pretrained models (BERT, DistilBERT)

* Save model as API endpoint
  
## 🤝 Contributing
Contributions are welcome!
Feel free to create issues or submit pull requests.

## ⭐ If you like this project
Don’t forget to star ⭐ the repository on GitHub!

