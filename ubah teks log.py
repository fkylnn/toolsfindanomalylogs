import pandas as pd
import sys
from sklearn.feature_extraction.text import TfidfVectorizer

def load_data():
    """Memuat data dengan penanganan error yang lebih baik"""
    try:
        # 1. Load data dengan explicit error handling
        log_df = pd.read_csv('HDFS_2k.log_structured.csv', dtype=str, keep_default_na=False)
        templates = pd.read_csv('HDFS_templates.csv', dtype=str, keep_default_na=False)
        
        # 2. Validasi kolom secara eksplisit
        log_cols = set(log_df.columns)
        tpl_cols = set(templates.columns)
        
        if 'EventTemplate' not in log_cols and 'EventTemplate' not in tpl_cols:
            raise ValueError("Kolom 'EventTemplate' tidak ditemukan di kedua file")
            
        # 3. Normalisasi nama kolom (handle case sensitivity)
        log_df.columns = log_df.columns.str.lower()
        templates.columns = templates.columns.str.lower()
        
        return log_df, templates
        
    except Exception as e:
        print(f"ERROR: {str(e)}", file=sys.stderr)
        sys.exit(1)

def preprocess(log_df, templates):
    """Preprocessing data yang lebih robust"""
    try:
        # 1. Gabungkan data jika diperlukan
        if 'eventtemplate' not in log_df.columns:
            log_df = log_df.merge(templates, on='eventid', how='left')
        
        # 2. Pastikan kolom template ada
        if 'eventtemplate' not in log_df.columns:
            log_df['eventtemplate'] = log_df['content']
        
        # 3. Bersihkan data
        log_df['eventtemplate'] = log_df['eventtemplate'].fillna('').astype(str)
        
        # 4. Vectorization
        vectorizer = TfidfVectorizer(
            max_features=500,
            token_pattern=r'\b\w+\b',
            stop_words='english'
        )
        X = vectorizer.fit_transform(log_df['eventtemplate'])
        
        return X, log_df, vectorizer
        
    except Exception as e:
        print(f"PROCESSING ERROR: {str(e)}", file=sys.stderr)
        sys.exit(1)

def main():
    print("=== HDFS LOG PROCESSING ===")
    
    # 1. Load data
    log_df, templates = load_data()
    print("\nData loaded successfully")
    print("Log columns:", [c for c in log_df.columns if c.lower() == 'eventtemplate'])
    print("Template columns:", [c for c in templates.columns if c.lower() == 'eventtemplate'])
    
    # 2. Preprocess
    X, df, vectorizer = preprocess(log_df, templates)
    print("\nPreprocessing completed")
    print("Sample templates:", df['eventtemplate'].head(3).tolist())
    
    # 3. Save results
    df.to_csv('processed_hdfs.csv', index=False)
    print("\nResults saved to processed_hdfs.csv")

if __name__ == "__main__":
    main()