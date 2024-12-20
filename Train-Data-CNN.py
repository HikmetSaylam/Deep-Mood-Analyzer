import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Dropout, Flatten
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

# 1D-CNN için veriyi 3D formatına dönüştürme
X_train_tfidf = X_train_tfidf.reshape((X_train_tfidf.shape[0], X_train_tfidf.shape[1], 1))
X_test_tfidf = X_test_tfidf.reshape((X_test_tfidf.shape[0], X_test_tfidf.shape[1], 1))

print("Model oluşturuluyor...")

# Modeli oluşturma
model_cnn = Sequential([
    Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train_tfidf.shape[1], 1)),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(4, activation='softmax')
])

print("Model derleniyor...")

# Modeli derleme
model_cnn.compile(optimizer=Adam(learning_rate=0.002),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

print("Model eğitiliyor...")

# CNN modelini eğitme
history_cnn = model_cnn.fit(X_train_tfidf, y_train_encoded,
                            validation_data=(X_test_tfidf, y_test_encoded),
                            epochs=8,
                            batch_size=64)

print("Model değerlendiriliyor...")
# Modeli değerlendirme
loss_cnn, accuracy_cnn = model_cnn.evaluate(X_test_tfidf, y_test_encoded)
print(f"Test Loss: {loss_cnn:.4f}, Test Accuracy: {accuracy_cnn:.4f}")

# F1-score hesaplama
y_pred_cnn = model_cnn.predict(X_test_tfidf)
y_pred_classes_cnn = np.argmax(y_pred_cnn, axis=1)
y_test_classes_cnn = np.argmax(y_test_encoded, axis=1)

f1_cnn = f1_score(y_test_classes_cnn, y_pred_classes_cnn, average='weighted')
print(f"F1-Score: {f1_cnn:.4f}")

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test_classes_cnn, y_pred_classes_cnn))

# Eğitim ve doğrulama kayıplarını çizdirme
plt.figure(figsize=(10, 6))
plt.plot(history_cnn.history['loss'], label='Training Loss')
plt.plot(history_cnn.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss (1D-CNN)')
plt.legend()
plt.show()

# Eğitim ve doğrulama doğruluklarını çizdirme
plt.figure(figsize=(10, 6))
plt.plot(history_cnn.history['accuracy'], label='Training Accuracy')
plt.plot(history_cnn.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy (1D-CNN)')
plt.legend()
plt.show()

# F1-Score grafiği
plt.figure(figsize=(6, 4))
plt.bar(['F1-Score'], [f1_cnn], color='blue')
plt.ylabel('Score')
plt.title('F1-Score (1D-CNN)')
plt.ylim(0, 1)
plt.show()
