import joblib
import os
from pathlib import Path

class LogAnomalyDetector:
    def __init__(self, model_path=None, vectorizer_path=None):
        """Initialize with automatic path detection"""
        # Cari file model secara otomatis
        base_dir = Path(__file__).parent
        self.model_path = model_path or base_dir/'anomaly_detector.pkl'
        self.vectorizer_path = vectorizer_path or base_dir/'tfidf_vectorizer.pkl'
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found at {self.model_path}")
        if not self.vectorizer_path.exists():
            raise FileNotFoundError(f"Vectorizer file not found at {self.vectorizer_path}")
        
        self.vectorizer = joblib.load(self.vectorizer_path)
        self.model = joblib.load(self.model_path)
        self.threshold = 0.77

    def predict(self, log_text: str) -> dict:
        X = self.vectorizer.transform([log_text])
        proba = self.model.predict_proba(X)[0,1]
        
        return {
            'text': log_text,
            'is_anomaly': proba > self.threshold,
            'confidence': round(proba, 4),
            'risk_level': 'HIGH' if proba > 0.9 else 'MEDIUM' if proba > self.threshold else 'LOW'
        }

if __name__ == "__main__":
    try:
        print("🔍 Mencari model...")
        detector = LogAnomalyDetector()
        print("✅ Model berhasil dimuat!")
        
        # Test prediction
        test_log = "Verification succeeded for blk_123"
        print("\n🧪 Test prediction:")
        print(detector.predict(test_log))
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nLangkah troubleshooting:")
        print("1. Pastikan Anda sudah menjalankan training_and_test.py")
        print("2. Cek apakah file berikut ada di folder ini:")
        print("   - tfidf_vectorizer.pkl")
        print("   - anomaly_detector.pkl")
        print("3. Jika file ada di folder lain, gunakan:")
        print("   detector = LogAnomalyDetector(model_path='path/to/model.pkl', vectorizer_path='path/to/vectorizer.pkl')")