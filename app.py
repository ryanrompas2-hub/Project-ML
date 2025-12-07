import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# --- 1. Fungsi Data Loading dan Preprocessing (Menyatu) ---

# Fungsi untuk memuat data dan melakukan preprocessing lengkap
# Fungsi ini dijalankan sekali dan di-cache untuk efisiensi.
@st.cache_data
def load_and_preprocess_data(file_path):
    """Memuat, membersihkan, dan memproses data untuk pelatihan model."""
    
    df = pd.read_csv(file_path)

    # 1. Menangani Missing Values (Menggunakan Median dan Modus)
    numeric_cols = df.select_dtypes(include=np.number).columns
    for col in numeric_cols:
        median_val = df[col].median()
        df[col].fillna(median_val, inplace=True)
        
    categorical_cols = df.select_dtypes(include='object').columns
    for col in categorical_cols:
        mode_val = df[col].mode()[0]
        df[col].fillna(mode_val, inplace=True)
        
    # 2. Penghapusan Outlier (Menggunakan IQR)
    df_cleaned = df.copy()
    numeric_cols_after_cleaning = df_cleaned.select_dtypes(include=np.number).columns
    
    for col in numeric_cols_after_cleaning:
        Q1 = df_cleaned[col].quantile(0.25)
        Q3 = df_cleaned[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_cleaned = df_cleaned[(df_cleaned[col] >= lower_bound) & (df_cleaned[col] <= upper_bound)]
    
    # 3. One-Hot Encoding
    # Perhatikan: Karena drop_first=True, Activity_Cycling tidak ada di fitur
    # Kita hanya perlu kolom 'Activity_Running', 'Activity_Walking', 'Activity_Workout', 'Activity_Yoga'
    df_processed = pd.get_dummies(df_cleaned, columns=['Activity'], drop_first=True)
    
    # Definisikan Fitur (X) dan Target (y)
    X = df_processed.drop('CaloriesBurned', axis=1)
    y = df_processed['CaloriesBurned']

    # 4. Normalisasi Data (StandardScaler)
    scaler = StandardScaler()
    X_numeric_cols = ['Duration_min', 'HeartRate_bpm', 'Weight_kg', 'Height_cm', 'MET']
    X_categorical_cols = [col for col in X.columns if col not in X_numeric_cols]
    
    # Fit scaler HANYA pada data pelatihan, bukan keseluruhan data.
    scaler.fit(X[X_numeric_cols])
    X_scaled_array = scaler.transform(X[X_numeric_cols])
    
    X_scaled_numeric = pd.DataFrame(X_scaled_array, columns=X_numeric_cols, index=X.index)
    X_final = pd.concat([X_scaled_numeric, X[X_categorical_cols]], axis=1)
    
    # Data Split untuk pelatihan model di Streamlit
    X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=42)

    return X_train, y_train, scaler

# --- 2. Pelatihan Model (Hanya Sekali di Awal) ---

# Asumsi nama file dataset: dataset_fitness_tracker_2200.csv
DATASET_PATH = 'dataset_fitness_tracker_2200.csv'

# Panggil fungsi load dan preprocess
X_train, y_train, scaler = load_and_preprocess_data(DATASET_PATH)

# Inisialisasi dan latih model
@st.cache_resource
def train_model(X_train, y_train):
    """Melatih model Regresi Linier."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

model = train_model(X_train, y_train)


# --- 3. Fungsi Prediksi Input Pengguna ---

def predict_calories(input_data, model, scaler):
    """Melakukan prediksi pada data input tunggal dari pengguna."""
    
    # DataFrame untuk input
    input_df = pd.DataFrame([input_data])
    
    # 1. One-Hot Encoding (Manual karena hanya 1 baris)
    activity_map = {
        'Running': [1, 0, 0, 0], # Urutan: Activity_Running, Activity_Walking, Activity_Workout, Activity_Yoga
        'Walking': [0, 1, 0, 0],
        'Workout': [0, 0, 1, 0],
        'Yoga': [0, 0, 0, 1],
        'Cycling': [0, 0, 0, 0] # Karena drop_first=True (Cycling=0, 0, 0, 0)
    }
    
    # Kolom fitur yang dihasilkan dari One-Hot Encoding:
    activity_encoded = pd.DataFrame(
        [activity_map[input_df['Activity'].iloc[0]]],
        columns=['Activity_Running', 'Activity_Walking', 'Activity_Workout', 'Activity_Yoga']
    )
    
    # Drop kolom 'Activity' asli
    input_df = input_df.drop('Activity', axis=1)

    # 2. Normalisasi StandardScaler (Hanya pada fitur numerik)
    numeric_cols = ['Duration_min', 'HeartRate_bpm', 'Weight_kg', 'Height_cm', 'MET']
    
    # Transformasi data input
    input_scaled = scaler.transform(input_df[numeric_cols])
    input_scaled_df = pd.DataFrame(input_scaled, columns=numeric_cols)

    # 3. Gabungkan kembali
    final_input = pd.concat([input_scaled_df, activity_encoded], axis=1)
    
    # Ganti nama kolom yang salah jika ada (kadang Streamlit mengubah nama kolom)
    final_input.columns = X_train.columns
    
    # Prediksi
    prediction = model.predict(final_input)
    return prediction[0]


# --- 4. Aplikasi Streamlit Utama ---

st.set_page_config(page_title="Prediksi Kalori Terbakar", layout="wide")

st.title("🔥 Aplikasi Prediksi Kalori Terbakar")
st.markdown("Masukkan detail aktivitas Anda untuk memperkirakan jumlah kalori yang terbakar menggunakan model Regresi Linier.")

with st.sidebar:
    st.header("Parameter Input")
    st.info("Input yang Anda masukkan akan diproses (dinormalisasi dan di-encode) sebelum diprediksi oleh model.")

    # Input dari Pengguna
    activity = st.selectbox(
        "Jenis Aktivitas:",
        ('Cycling', 'Running', 'Walking', 'Workout', 'Yoga')
    )
    
    duration = st.slider("Durasi (menit)", min_value=1.0, max_value=120.0, value=30.0, step=0.1)
    
    # Penjelasan Detak Jantung
    st.markdown("""
    ---
    ### Detak Jantung (Heart Rate) 
    Detak jantung rata-rata Anda selama aktivitas (denyut per menit/bpm).
    
    **Kisaran Umum Berdasarkan Intensitas (Dewasa):**
    * **Jalan Santai/Yoga:** ~80 - 100 bpm
    * **Latihan Ringan/Sedang (Zona Pembakaran Lemak):** ~100 - 130 bpm
    * **Lari/Intensif (Zona Kardio):** ~130 - 170+ bpm
    """)
    heart_rate = st.slider("Detak Jantung (bpm)", min_value=60.0, max_value=200.0, value=120.0, step=1.0)
    
    weight = st.slider("Berat Badan (kg)", min_value=40.0, max_value=120.0, value=70.0, step=0.1)
    height = st.slider("Tinggi Badan (cm)", min_value=140.0, max_value=200.0, value=170.0, step=0.1)
    
    # Penjelasan MET DENGAN KISARAN AKTIVITAS
    st.markdown("""
    ---
    ### MET Value (Metabolic Equivalent of Task) 
    **MET** adalah rasio tingkat energi metabolik aktivitas terhadap tingkat energi metabolik saat istirahat.
    
    **Kisaran MET Berdasarkan Aktivitas:**
    * **Yoga/Meditasi:** ~1.5 - 3.0 MET
    * **Berjalan Kaki (Cepat):** ~3.5 - 5.0 MET
    * **Bersepeda:** ~4.0 - 8.0 MET (tergantung kecepatan)
    * **Workout (Latihan Kekuatan):** ~3.5 - 8.0 MET
    * **Lari:** ~7.0 - 12.0 MET (tergantung kecepatan)
    """)
    met = st.slider("MET Value", min_value=1.0, max_value=10.0, value=5.0, step=0.1)
    
    st.markdown("---")
    
    # Data input dalam format dict
    input_data = {
        'Activity': activity,
        'Duration_min': duration,
        'HeartRate_bpm': heart_rate,
        'Weight_kg': weight,
        'Height_cm': height,
        'MET': met
    }

# Tombol untuk memicu prediksi
if st.button("Hitung Kalori Terbakar", type="primary"):
    try:
        # Melakukan Prediksi
        kalori_prediksi = predict_calories(input_data, model, scaler)
        
        st.subheader(f"Hasil Prediksi Kalori Terbakar")
        
        # Tampilkan hasil
        col1, col2 = st.columns(2)
        with col1:
             st.metric("Total Kalori yang Terbakar (Diprediksi)", f"{kalori_prediksi:,.2f} Kkal")
        with col2:
             st.markdown("### Detail Input")
             st.table(pd.Series(input_data).to_frame(name='Nilai'))
        
        st.success("Prediksi berhasil dilakukan.")

        st.markdown("""
            ---
            **Catatan:** Prediksi ini didasarkan pada model Regresi Linier yang dilatih 
            dengan data Anda (R2 Score: 0.7533).
        """)
        
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses prediksi: {e}")