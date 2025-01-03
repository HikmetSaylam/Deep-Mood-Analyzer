import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
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

print("Model oluşturuluyor...")

# Modeli oluşturma
model = Sequential([
    Dense(256, activation='relu', input_shape=(X_train_tfidf.shape[1],)),
    Dropout(0.75),
    Dense(64, activation='relu'),
    Dropout(0.55),
    Dense(4, activation='softmax')
])

print("Model derleniyor...")
# Modeli derleme
model.compile(optimizer=Adam(learning_rate=0.002),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print("Model eğitiliyor...")
# Modeli eğitme
history = model.fit(X_train_tfidf, y_train_encoded,
                    validation_data=(X_test_tfidf, y_test_encoded),
                    epochs=8,
                    batch_size=64)

print("Model değerlendiriliyor...")
# Modeli değerlendirme
loss, accuracy = model.evaluate(X_test_tfidf, y_test_encoded)
print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")

# F1-score hesaplama
y_pred = model.predict(X_test_tfidf)
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(y_test_encoded, axis=1)

f1 = f1_score(y_test_classes, y_pred_classes, average='weighted')
print(f"F1-Score: {f1:.4f}")

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test_classes, y_pred_classes))

# Eğitim ve doğrulama kayıplarını çizdirme
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

# Eğitim ve doğrulama doğruluklarını çizdirme
plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()

# F1-Score grafiği
plt.figure(figsize=(6, 4))
plt.bar(['F1-Score'], [f1], color='blue')
plt.ylabel('Score')
plt.title('F1-Score')
plt.ylim(0, 1)
plt.show()
