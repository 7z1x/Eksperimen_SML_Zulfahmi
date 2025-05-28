import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess_data(test_size=0.2, random_state=42):
    print("Memulai preprocessing data...")
    
    # 1. Muat dataset langsung dari path
    print("Membaca file data.csv...")
    
    # Pastikan path ini benar relatif terhadap root proyek Anda
    # Ini akan mencari file di: your_project_root/breast_cancer_dataset/data.csv
    data_path = os.path.join("breast_cancer_dataset", "data.csv")
    
    # Tambahkan try-except block untuk penanganan error saat membaca file
    try:
        df_processed = pd.read_csv(data_path)
        print(f"File '{data_path}' berhasil dibaca!")
    except FileNotFoundError:
        print(f"ERROR: File '{data_path}' tidak ditemukan. Pastikan path dan nama file benar.")
        # Mengangkat ulang error agar workflow GitHub Actions juga gagal dan terlihat jelas
        raise 
    except Exception as e:
        print(f"ERROR: Terjadi kesalahan saat membaca file '{data_path}': {e}")
        # Mengangkat ulang error agar workflow GitHub Actions juga gagal
        raise

    # 2. Label Encoding untuk kolom 'diagnosis' (jika belum dilakukan)
    if df_processed['diagnosis'].dtype == 'object':
        df_processed['diagnosis'] = df_processed['diagnosis'].map({'M': 1, 'B': 0})

    # 3. Hapus kolom tidak relevan ('id' dan 'Unnamed: 32')
    df_processed = df_processed.drop(columns=['id', 'Unnamed: 32'], errors='ignore')

    # 4. Pisahkan fitur (X) dan target (y)
    X = df_processed.drop(columns=['diagnosis'])
    y = df_processed['diagnosis']

    # 5. Split data menjadi train dan test dengan stratify
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # 6. Standarisasi fitur
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    preprocessed_dir = os.path.join("preprocessing", "breast_cancer_dataset_preprocessed")
    os.makedirs(preprocessed_dir, exist_ok=True) 

    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X.columns)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X.columns)

    # Simpan ke CSV
    X_train_scaled_df.to_csv(os.path.join(preprocessed_dir, "X_train_scaled.csv"), index=False)
    X_test_scaled_df.to_csv(os.path.join(preprocessed_dir, "X_test_scaled.csv"), index=False)
    y_train.to_csv(os.path.join(preprocessed_dir, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(preprocessed_dir, "y_test.csv"), index=False)

    print(f"Data yang sudah diproses berhasil disimpan di '{preprocessed_dir}'!")

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

if __name__ == "__main__":
    print("Program dimulai...")
    preprocess_data()
    print("Program selesai!")
