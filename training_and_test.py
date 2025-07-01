# Import Library
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (classification_report, 
                            confusion_matrix, 
                            roc_auc_score,
                            precision_recall_curve)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from collections import Counter

# 1. Load Data
print("🔍 Memuat data...")
df = pd.read_csv('final_anomaly_results.csv')

# 2. Preprocessing
print("\n🧹 Preprocessing data...")
features = df[['eventtemplate']].copy()
target = df['final_anomaly'].copy()

# 3. TF-IDF Vectorization
print("\n🔠 Membuat fitur TF-IDF...")
vectorizer = TfidfVectorizer(
    max_features=100,
    token_pattern=r'\b\w{3,}\b',  # Filter kata pendek
    stop_words='english',
    ngram_range=(1,2)  # Gunakan unigram + bigram
)
X = vectorizer.fit_transform(features['eventtemplate'].fillna(''))

# 4. Split Data
print("\n✂️ Membagi data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, target, 
    test_size=0.3, 
    random_state=42,
    stratify=target
)

# 5. Train Model
print("\n🤖 Melatih model Random Forest...")
model = RandomForestClassifier(
    n_estimators=50,
    max_depth=5,
    class_weight='balanced',
    random_state=42
)
model.fit(X_train, y_train)

# 6. Evaluasi
print("\n📊 Evaluasi model:")
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:,1]

print("\n📝 Classification Report:")
print(classification_report(y_test, y_pred))

print("\n🔢 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\n📈 AUC-ROC:", roc_auc_score(y_test, y_proba))

# 7. Optimasi Threshold
print("\n🎯 Optimasi threshold...")
precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
optimal_idx = np.argmax(precisions * recalls)
optimal_threshold = thresholds[optimal_idx]
print(f"Threshold optimal: {optimal_threshold:.2f}")

# 8. Feature Importance
print("\n💡 Feature Importance:")
feature_imp = pd.DataFrame({
    'feature': vectorizer.get_feature_names_out(),
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)
print(feature_imp.head(10))

# 9. Visualisasi
plt.figure(figsize=(15,5))

# Confusion Matrix
plt.subplot(1,3,1)
sns.heatmap(confusion_matrix(y_test, y_pred), 
            annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')

# Feature Importance
plt.subplot(1,3,2)
sns.barplot(x='importance', y='feature', 
            data=feature_imp.head(10), palette='viridis')
plt.title('Top 10 Features')

# Probability Distribution
plt.subplot(1,3,3)
plt.hist(y_proba[y_test==0], bins=20, alpha=0.5, label='Normal')
plt.hist(y_proba[y_test==1], bins=20, alpha=0.5, label='Anomaly')
plt.legend()
plt.title('Probability Distribution')

plt.tight_layout()
plt.savefig('model_performance.png')
print("\n🖼️ Visualisasi disimpan sebagai model_performance.png")

# 10. Simpan Model
print("\n💾 Menyimpan model...")
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
joblib.dump(model, 'anomaly_detector.pkl')
print("✅ Model berhasil disimpan!")

# 11. Contoh Implementasi
class AnomalyDetector:
    def __init__(self):
        self.vectorizer = joblib.load('tfidf_vectorizer.pkl')
        self.model = joblib.load('anomaly_detector.pkl')
    
    def predict(self, log_text):
        X = self.vectorizer.transform([log_text])
        proba = self.model.predict_proba(X)[0,1]
        return {
            'text': log_text,
            'is_anomaly': proba > optimal_threshold,
            'confidence': proba,
            'alert': '⚠️' if proba > optimal_threshold else '✅'
        }

# Demo
print("\n🚀 Contoh penggunaan:")
detector = AnomalyDetector()
sample_logs = [
    "Verification succeeded for blk_123",  # Anomali
    "PacketResponder 1 for block blk_456 terminating"  # Normal
]
for log in sample_logs:
    print(detector.predict(log))