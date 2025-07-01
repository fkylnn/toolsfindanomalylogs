import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# 1. Load Data
print("Loading data...")
df = pd.read_csv('processed_hdfs.csv')

# 2. Pastikan kolom yang dibutuhkan ada
if 'eventtemplate' not in df.columns:
    df['eventtemplate'] = df['content']

# 3. Vectorization
print("Creating TF-IDF features...")
vectorizer = TfidfVectorizer(
    max_features=500,
    token_pattern=r'\b\w+\b',
    stop_words='english'
)
X = vectorizer.fit_transform(df['eventtemplate'].fillna(''))

# 4. Hybrid Model
print("Training hybrid model...")

# A. Unsupervised Part
iso = IsolationForest(contamination=0.1, random_state=42)
df['iso_anomaly'] = iso.fit_predict(X)
df['iso_anomaly'] = np.where(df['iso_anomaly'] == -1, 1, 0)  # Convert to 0/1

# B. Supervised Part (jika ada label)
if 'label' in df.columns:
    X_train, X_test, y_train, y_test = train_test_split(X, df['label'], test_size=0.2, random_state=42)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    df['rf_anomaly'] = rf.predict(X)
    print("\nSupervised Model Performance:")
    print(classification_report(y_test, rf.predict(X_test)))

# 5. Combine Results
print("\nCombining results...")
if 'rf_anomaly' in df.columns:
    df['final_anomaly'] = df.apply(lambda x: 1 if (x['iso_anomaly'] == 1 or x['rf_anomaly'] == 1) else 0, axis=1)
else:
    df['final_anomaly'] = df['iso_anomaly']

# 6. Save Results
df.to_csv('final_anomaly_results.csv', index=False)
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
joblib.dump(iso, 'isolation_forest.pkl')
if 'rf_anomaly' in df.columns:
    joblib.dump(rf, 'random_forest.pkl')

print("\n=== Results ===")
print("Anomaly Distribution:")
print(df['final_anomaly'].value_counts())

# 7. Visualization
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.countplot(x='final_anomaly', data=df)
plt.title('Anomaly Distribution')

plt.subplot(1, 2, 2)
if 'label' in df.columns:
    sns.heatmap(pd.crosstab(df['label'], df['final_anomaly']), 
                annot=True, fmt='d', cmap='Blues')
    plt.title('Comparison with Ground Truth')
plt.tight_layout()
plt.savefig('results_visualization.png')
plt.show()

print("\nProcessing completed successfully!")
print("Saved files:")
print("- final_anomaly_results.csv")
print("- tfidf_vectorizer.pkl")
print("- isolation_forest.pkl")
print("- results_visualization.png")