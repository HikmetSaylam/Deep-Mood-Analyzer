import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, classification_report

# Eğitim ve test verilerini yükleme
train_df = pd.read_csv("augmented_training_set.csv")
test_df = pd.read_csv("test_set.csv")

# Giriş (X) ve hedef (y) ayırma
X_train = train_df['Cleaned_Review']
y_train = train_df['Sentiment']

X_test = test_df['Cleaned_Review']
y_test = test_df['Sentiment']

# Eksik değerleri (NaN) hem X hem de y veri setlerinden aynı anda kaldırma
train_df_clean = train_df.dropna(subset=['Cleaned_Review', 'Sentiment'])
test_df_clean = test_df.dropna(subset=['Cleaned_Review', 'Sentiment'])

# Temizlenmiş verileri yeniden atama
X_train = train_df_clean['Cleaned_Review']
y_train = train_df_clean['Sentiment']

X_test = test_df_clean['Cleaned_Review']
y_test = test_df_clean['Sentiment']

print("Model hazirlaniyor...")

# Label encoding: Kategorik etiketleri sayısallaştır
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# TF-IDF vektörleştirme: Metni sayısal vektörlere dönüştürme
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train).toarray()
X_test_tfidf = vectorizer.transform(X_test).toarray()

# Etiketleri one-hot encode yapma
y_train_encoded = to_categorical(y_train_encoded)
y_test_encoded = to_categorical(y_test_encoded)

# RNN için veriyi 3D formatına dönüştürme
X_train_tfidf = X_train_tfidf.reshape((X_train_tfidf.shape[0], X_train_tfidf.shape[1], 1))
X_test_tfidf = X_test_tfidf.reshape((X_test_tfidf.shape[0], X_test_tfidf.shape[1], 1))

print("Model oluşturuluyor...")

# Modeli oluşturma
model_rnn = Sequential([
    # RNN katmanı ekliyoruz
    SimpleRNN(256, activation='relu', input_shape=(X_train_tfidf.shape[1], 1)),
    Dropout(0.75),
    Dense(64, activation='relu'),
    Dropout(0.55),
    Dense(4, activation='softmax')
])

print("Model derleniyor...")

# Modeli derleme
model_rnn.compile(optimizer=Adam(learning_rate=0.002),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

print("Model eğitiliyor...")

# RNN modelini eğitme
history_rnn = model_rnn.fit(X_train_tfidf, y_train_encoded,
                            validation_data=(X_test_tfidf, y_test_encoded),
                            epochs=8,
                            batch_size=64)

print("Model değerlendiriliyor...")
# Modeli değerlendirme
loss_rnn, accuracy_rnn = model_rnn.evaluate(X_test_tfidf, y_test_encoded)
print(f"Test Loss: {loss_rnn:.4f}, Test Accuracy: {accuracy_rnn:.4f}")

# F1-score hesaplama
y_pred_rnn = model_rnn.predict(X_test_tfidf)
y_pred_classes_rnn = np.argmax(y_pred_rnn, axis=1)
y_test_classes_rnn = np.argmax(y_test_encoded, axis=1)

f1_rnn = f1_score(y_test_classes_rnn, y_pred_classes_rnn, average='weighted')
print(f"F1-Score: {f1_rnn:.4f}")

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test_classes_rnn, y_pred_classes_rnn))

# Eğitim ve doğrulama kayıplarını çizdirme
plt.figure(figsize=(10, 6))
plt.plot(history_rnn.history['loss'], label='Training Loss')
plt.plot(history_rnn.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss (RNN)')
plt.legend()
plt.show()

# Eğitim ve doğrulama doğruluklarını çizdirme
plt.figure(figsize=(10, 6))
plt.plot(history_rnn.history['accuracy'], label='Training Accuracy')
plt.plot(history_rnn.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy (RNN)')
plt.legend()
plt.show()

# F1-Score grafiği
plt.figure(figsize=(6, 4))
plt.bar(['F1-Score'], [f1_rnn], color='blue')
plt.ylabel('Score')
plt.title('F1-Score (RNN)')
plt.ylim(0, 1)
plt.show()
