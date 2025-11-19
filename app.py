"""
Dashboard Streamlit untuk Sistem Prediksi Biaya Asuransi Kesehatan.
Antarmuka interaktif untuk prediksi, visualisasi, dan analisis.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Konfigurasi halaman
st.set_page_config(
    page_title="Sistem Prediksi Biaya Asuransi Kesehatan",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS untuk UI modern
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .prediction-result {
        background: #f8f9fa;
        padding: 2rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        padding: 0.75rem;
        border-radius: 8px;
        border: none;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
</style>
""", unsafe_allow_html=True)

# Cache untuk loading data
@st.cache_data
def load_data():
    """Memuat dataset asuransi kesehatan."""
    return pd.read_csv('medical_insurance.csv')

@st.cache_resource
def load_models():
    """Memuat model yang sudah dilatih dan preprocessor."""
    preprocessor = joblib.load('models/preprocessor.pkl')
    metadata = joblib.load('models/preprocessor_metadata.pkl')
    cost_model = joblib.load('models/cost_model.pkl')
    risk_model = joblib.load('models/risk_score_model.pkl')
    high_risk_model = joblib.load('models/high_risk_model.pkl')
    
    try:
        model_metrics = joblib.load('models/model_metrics.pkl')
    except:
        model_metrics = None
    
    return preprocessor, metadata, cost_model, risk_model, high_risk_model, model_metrics

def get_feature_importance(model, feature_names, top_n=15):
    """Ekstrak feature importance dari model berbasis tree."""
    try:
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1][:top_n]
            return {
                'features': [feature_names[i] for i in indices],
                'importances': importances[indices]
            }
    except:
        pass
    return None

def main():
    """Aplikasi Streamlit utama."""
    st.markdown('<h1 class="main-header">Sistem Prediksi Biaya Asuransi Kesehatan</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Load data dan model
    try:
        df = load_data()
        preprocessor, metadata, cost_model, risk_model, high_risk_model, model_metrics = load_models()
    except FileNotFoundError as e:
        st.error(f"Error memuat file: {e}. Pastikan model sudah dilatih terlebih dahulu.")
        st.stop()
    
    # Navigasi sidebar
    st.sidebar.title("Navigasi")
    page = st.sidebar.selectbox(
        "Pilih Halaman",
        ["Prediksi Cepat", "Prediksi Detail", "Visualisasi Data", "Kinerja Model", "Eksplorasi Data", "Wawasan"]
    )
    
    if page == "Prediksi Cepat":
        show_simple_predictions_page(df, preprocessor, metadata, cost_model, risk_model, high_risk_model)
    elif page == "Prediksi Detail":
        show_detailed_predictions_page(df, preprocessor, metadata, cost_model, risk_model, high_risk_model)
    elif page == "Visualisasi Data":
        show_visualizations_page(df)
    elif page == "Kinerja Model":
        show_performance_page(model_metrics, cost_model, risk_model, high_risk_model, metadata)
    elif page == "Eksplorasi Data":
        show_data_explorer_page(df)
    elif page == "Wawasan":
        show_insights_page(df, cost_model, risk_model, high_risk_model, metadata)

def show_simple_predictions_page(df, preprocessor, metadata, cost_model, risk_model, high_risk_model):
    """Tampilkan halaman prediksi sederhana dengan UI modern."""
    st.markdown('<div class="section-header">Prediksi Cepat</div>', unsafe_allow_html=True)
    st.markdown("Masukkan informasi dasar untuk mendapatkan prediksi biaya kesehatan, skor risiko, dan klasifikasi risiko.")
    
    # Container utama dengan background
    with st.container():
        # Informasi Demografis
        st.markdown("### Informasi Demografis")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.number_input("Usia", min_value=0, max_value=100, value=45, step=1)
            sex = st.selectbox("Jenis Kelamin", ["Female", "Male", "Other"])
            region = st.selectbox("Wilayah", ["North", "Central", "West", "South", "East"])
        
        with col2:
            income = st.number_input("Pendapatan (USD)", min_value=0, value=50000, step=1000, format="%d")
            education = st.selectbox("Pendidikan", ["No HS", "HS", "Some College", "Bachelors", "Masters", "Doctorate"])
            marital_status = st.selectbox("Status Pernikahan", ["Single", "Married", "Divorced", "Widowed"])
        
        with col3:
            household_size = st.number_input("Jumlah Anggota Keluarga", min_value=1, max_value=10, value=2, step=1)
            dependents = st.number_input("Jumlah Tanggungan", min_value=0, max_value=10, value=1, step=1)
            urban_rural = st.selectbox("Lokasi", ["Urban", "Suburban", "Rural"])
        
        st.markdown("---")
        
        # Informasi Kesehatan
        st.markdown("### Informasi Kesehatan")
        col4, col5, col6 = st.columns(3)
        
        with col4:
            bmi = st.slider("BMI", 10.0, 50.0, 25.0, 0.1)
            smoker = st.selectbox("Status Merokok", ["Never", "Former", "Current"])
            alcohol_freq = st.selectbox("Frekuensi Alkohol", ["None", "Occasional", "Weekly", "Daily"])
        
        with col5:
            systolic_bp = st.slider("Tekanan Darah Sistolik", 80, 200, 120, step=1)
            diastolic_bp = st.slider("Tekanan Darah Diastolik", 50, 120, 80, step=1)
            visits_last_year = st.number_input("Kunjungan Dokter (Tahun Lalu)", min_value=0, max_value=20, value=2, step=1)
        
        with col6:
            medication_count = st.number_input("Jumlah Obat", min_value=0, max_value=10, value=2, step=1)
            chronic_count = st.number_input("Jumlah Penyakit Kronis", min_value=0, max_value=5, value=0, step=1)
            hospitalizations_last_3yrs = st.number_input("Rawat Inap (3 Tahun Terakhir)", min_value=0, max_value=10, value=0, step=1)
        
        # Kondisi Kronis - Checkbox Grid
        st.markdown("### Kondisi Kesehatan Kronis")
        col7, col8, col9, col10 = st.columns(4)
        
        with col7:
            hypertension = st.checkbox("Hipertensi")
            diabetes = st.checkbox("Diabetes")
            asthma = st.checkbox("Asma")
        
        with col8:
            copd = st.checkbox("COPD")
            cardiovascular_disease = st.checkbox("Penyakit Jantung")
            cancer_history = st.checkbox("Riwayat Kanker")
        
        with col9:
            kidney_disease = st.checkbox("Penyakit Ginjal")
            liver_disease = st.checkbox("Penyakit Hati")
            arthritis = st.checkbox("Arthritis")
        
        with col10:
            mental_health = st.checkbox("Kesehatan Mental")
            had_major_procedure = st.checkbox("Prosedur Besar")
        
        st.markdown("---")
        
        # Informasi Asuransi
        st.markdown("### Informasi Asuransi")
        col11, col12 = st.columns(2)
        
        with col11:
            plan_type = st.selectbox("Tipe Plan", ["HMO", "PPO", "POS", "EPO"])
            network_tier = st.selectbox("Tier Jaringan", ["Bronze", "Silver", "Gold", "Platinum"])
            deductible = st.selectbox("Deductible", [500, 1000, 2000, 5000])
        
        with col12:
            copay = st.selectbox("Copay", [10, 20, 30, 50])
            monthly_premium = st.number_input("Premi Bulanan (USD)", min_value=0.0, value=500.0, step=10.0)
            provider_quality = st.slider("Kualitas Provider", 1.0, 5.0, 3.5, 0.1)
        
        # Nilai default untuk kolom yang tidak ditampilkan
        employment_status = "Employed"
        days_hospitalized_last_3yrs = 0
        ldl = 120.0
        hba1c = 5.5
        policy_term_years = 5
        policy_changes_last_2yrs = 0
        annual_premium = monthly_premium * 12
        claims_count = 0
        avg_claim_amount = 0.0
        total_claims_paid = 0.0
        proc_imaging_count = 0
        proc_surgery_count = 0
        proc_physio_count = 0
        proc_consult_count = 0
        proc_lab_count = 0
        
        st.markdown("---")
        
        # Tombol Prediksi
        if st.button("Lakukan Prediksi", type="primary", use_container_width=True):
            # Buat DataFrame input
            input_data = pd.DataFrame({
                'age': [age],
                'sex': [sex],
                'region': [region],
                'urban_rural': [urban_rural],
                'income': [income],
                'education': [education],
                'marital_status': [marital_status],
                'employment_status': [employment_status],
                'household_size': [household_size],
                'dependents': [dependents],
                'bmi': [bmi],
                'smoker': [smoker],
                'alcohol_freq': [alcohol_freq if alcohol_freq != "None" else np.nan],
                'visits_last_year': [visits_last_year],
                'hospitalizations_last_3yrs': [hospitalizations_last_3yrs],
                'days_hospitalized_last_3yrs': [days_hospitalized_last_3yrs],
                'medication_count': [medication_count],
                'systolic_bp': [systolic_bp],
                'diastolic_bp': [diastolic_bp],
                'ldl': [ldl],
                'hba1c': [hba1c],
                'plan_type': [plan_type],
                'network_tier': [network_tier],
                'deductible': [deductible],
                'copay': [copay],
                'policy_term_years': [policy_term_years],
                'policy_changes_last_2yrs': [policy_changes_last_2yrs],
                'provider_quality': [provider_quality],
                'annual_premium': [annual_premium],
                'monthly_premium': [monthly_premium],
                'claims_count': [claims_count],
                'avg_claim_amount': [avg_claim_amount],
                'total_claims_paid': [total_claims_paid],
                'chronic_count': [chronic_count],
                'hypertension': [1 if hypertension else 0],
                'diabetes': [1 if diabetes else 0],
                'asthma': [1 if asthma else 0],
                'copd': [1 if copd else 0],
                'cardiovascular_disease': [1 if cardiovascular_disease else 0],
                'cancer_history': [1 if cancer_history else 0],
                'kidney_disease': [1 if kidney_disease else 0],
                'liver_disease': [1 if liver_disease else 0],
                'arthritis': [1 if arthritis else 0],
                'mental_health': [1 if mental_health else 0],
                'proc_imaging_count': [proc_imaging_count],
                'proc_surgery_count': [proc_surgery_count],
                'proc_physio_count': [proc_physio_count],
                'proc_consult_count': [proc_consult_count],
                'proc_lab_count': [proc_lab_count],
                'had_major_procedure': [1 if had_major_procedure else 0]
            })
            
            # Preprocess
            feature_cols = metadata['numerical_cols'] + metadata['categorical_cols']
            X_input = preprocessor.transform(input_data[feature_cols])
            
            # Prediksi
            cost_pred = cost_model.predict(X_input)[0]
            risk_pred = risk_model.predict(X_input)[0]
            high_risk_pred = high_risk_model.predict(X_input)[0]
            high_risk_proba = high_risk_model.predict_proba(X_input)[0][1]
            
            # Tampilkan hasil
            st.markdown("---")
            st.markdown('<div class="section-header">Hasil Prediksi</div>', unsafe_allow_html=True)
            
            # Kartu hasil dengan desain modern
            col_result1, col_result2, col_result3 = st.columns(3)
            
            with col_result1:
                st.markdown(f"""
                <div class="prediction-result">
                    <h3 style="color: #1f77b4; margin-bottom: 0.5rem;">Biaya Medis Tahunan</h3>
                    <h2 style="color: #2c3e50; margin: 0;">${cost_pred:,.2f}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col_result2:
                st.markdown(f"""
                <div class="prediction-result">
                    <h3 style="color: #1f77b4; margin-bottom: 0.5rem;">Skor Risiko</h3>
                    <h2 style="color: #2c3e50; margin: 0;">{risk_pred:.4f}</h2>
                    <p style="color: #7f8c8d; font-size: 0.9rem; margin-top: 0.5rem;">Rentang: 0.0 - 1.0</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col_result3:
                risk_status = "Risiko Tinggi" if high_risk_pred == 1 else "Risiko Rendah"
                risk_color = "#e74c3c" if high_risk_pred == 1 else "#27ae60"
                st.markdown(f"""
                <div class="prediction-result">
                    <h3 style="color: #1f77b4; margin-bottom: 0.5rem;">Klasifikasi Risiko</h3>
                    <h2 style="color: {risk_color}; margin: 0;">{risk_status}</h2>
                    <p style="color: #7f8c8d; font-size: 0.9rem; margin-top: 0.5rem;">Tingkat Keyakinan: {high_risk_proba:.1%}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Interpretasi
            st.info(f"""
            **Interpretasi Hasil:**
            - **Biaya Medis Tahunan**: ${cost_pred:,.2f} diperkirakan akan dikeluarkan untuk perawatan kesehatan dalam satu tahun.
            - **Skor Risiko**: {risk_pred:.4f} menunjukkan tingkat risiko medis (semakin tinggi semakin berisiko).
            - **Klasifikasi**: Pasien diklasifikasikan sebagai **{risk_status}** dengan tingkat keyakinan {high_risk_proba:.1%}.
            """)

def show_detailed_predictions_page(df, preprocessor, metadata, cost_model, risk_model, high_risk_model):
    """Tampilkan halaman prediksi detail dengan semua fitur."""
    st.markdown('<div class="section-header">Prediksi Detail</div>', unsafe_allow_html=True)
    st.markdown("Masukkan semua informasi detail untuk prediksi yang lebih akurat.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Demografis")
        age = st.slider("Usia", 0, 100, 45)
        sex = st.selectbox("Jenis Kelamin", ["Female", "Male", "Other"])
        region = st.selectbox("Wilayah", ["North", "Central", "West", "South", "East"])
        urban_rural = st.selectbox("Lokasi", ["Urban", "Suburban", "Rural"])
        income = st.number_input("Pendapatan", min_value=0, value=50000, step=1000)
        education = st.selectbox("Pendidikan", ["No HS", "HS", "Some College", "Bachelors", "Masters", "Doctorate"])
        marital_status = st.selectbox("Status Pernikahan", ["Single", "Married", "Divorced", "Widowed"])
        employment_status = st.selectbox("Status Pekerjaan", ["Employed", "Self-employed", "Unemployed", "Retired"])
        household_size = st.slider("Jumlah Anggota Keluarga", 1, 10, 2)
        dependents = st.slider("Jumlah Tanggungan", 0, 10, 1)
    
    with col2:
        st.markdown("### Metrik Kesehatan")
        bmi = st.slider("BMI", 10.0, 50.0, 25.0, 0.1)
        smoker = st.selectbox("Status Merokok", ["Never", "Former", "Current"])
        alcohol_freq = st.selectbox("Frekuensi Alkohol", ["None", "Occasional", "Weekly", "Daily"])
        visits_last_year = st.slider("Kunjungan Tahun Lalu", 0, 20, 2)
        hospitalizations_last_3yrs = st.slider("Rawat Inap (3 Tahun)", 0, 10, 0)
        days_hospitalized_last_3yrs = st.slider("Hari Rawat Inap (3 Tahun)", 0, 30, 0)
        medication_count = st.slider("Jumlah Obat", 0, 10, 2)
        systolic_bp = st.slider("Tekanan Darah Sistolik", 80, 200, 120)
        diastolic_bp = st.slider("Tekanan Darah Diastolik", 50, 120, 80)
        ldl = st.slider("LDL", 0, 300, 120)
        hba1c = st.slider("HbA1c", 3.0, 15.0, 5.5, 0.1)
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.markdown("### Rencana Asuransi")
        plan_type = st.selectbox("Tipe Plan", ["HMO", "PPO", "POS", "EPO"])
        network_tier = st.selectbox("Tier Jaringan", ["Bronze", "Silver", "Gold", "Platinum"])
        deductible = st.selectbox("Deductible", [500, 1000, 2000, 5000])
        copay = st.selectbox("Copay", [10, 20, 30, 50])
        policy_term_years = st.slider("Masa Berlaku Polis (Tahun)", 1, 10, 5)
        policy_changes_last_2yrs = st.slider("Perubahan Polis (2 Tahun)", 0, 5, 0)
        provider_quality = st.slider("Kualitas Provider", 1.0, 5.0, 3.5, 0.1)
        monthly_premium = st.number_input("Premi Bulanan (USD)", min_value=0.0, value=500.0, step=10.0)
        annual_premium = monthly_premium * 12
        claims_count = st.slider("Jumlah Klaim (Tahun Lalu)", 0, 20, 0)
        avg_claim_amount = st.number_input("Rata-rata Jumlah Klaim (USD)", min_value=0.0, value=0.0, step=100.0)
        total_claims_paid = claims_count * avg_claim_amount
    
    with col4:
        st.markdown("### Kondisi Kronis")
        chronic_count = st.slider("Jumlah Kronis", 0, 5, 0)
        hypertension = st.checkbox("Hipertensi")
        diabetes = st.checkbox("Diabetes")
        asthma = st.checkbox("Asma")
        copd = st.checkbox("COPD")
        cardiovascular_disease = st.checkbox("Penyakit Jantung")
        cancer_history = st.checkbox("Riwayat Kanker")
        kidney_disease = st.checkbox("Penyakit Ginjal")
        liver_disease = st.checkbox("Penyakit Hati")
        arthritis = st.checkbox("Arthritis")
        mental_health = st.checkbox("Kesehatan Mental")
    
    st.markdown("### Prosedur")
    col5, col6 = st.columns(2)
    with col5:
        proc_imaging_count = st.slider("Jumlah Pencitraan", 0, 10, 0)
        proc_surgery_count = st.slider("Jumlah Operasi", 0, 10, 0)
    with col6:
        proc_physio_count = st.slider("Jumlah Fisioterapi", 0, 10, 0)
        proc_consult_count = st.slider("Jumlah Konsultasi", 0, 10, 0)
        proc_lab_count = st.slider("Jumlah Lab", 0, 10, 0)
    
    had_major_procedure = st.checkbox("Prosedur Besar")
    
    # Buat DataFrame input
    if st.button("Lakukan Prediksi", type="primary"):
        input_data = pd.DataFrame({
            'age': [age],
            'sex': [sex],
            'region': [region],
            'urban_rural': [urban_rural],
            'income': [income],
            'education': [education],
            'marital_status': [marital_status],
            'employment_status': [employment_status],
            'household_size': [household_size],
            'dependents': [dependents],
            'bmi': [bmi],
            'smoker': [smoker],
            'alcohol_freq': [alcohol_freq if alcohol_freq != "None" else np.nan],
            'visits_last_year': [visits_last_year],
            'hospitalizations_last_3yrs': [hospitalizations_last_3yrs],
            'days_hospitalized_last_3yrs': [days_hospitalized_last_3yrs],
            'medication_count': [medication_count],
            'systolic_bp': [systolic_bp],
            'diastolic_bp': [diastolic_bp],
            'ldl': [ldl],
            'hba1c': [hba1c],
            'plan_type': [plan_type],
            'network_tier': [network_tier],
            'deductible': [deductible],
            'copay': [copay],
            'policy_term_years': [policy_term_years],
            'policy_changes_last_2yrs': [policy_changes_last_2yrs],
            'provider_quality': [provider_quality],
            'annual_premium': [annual_premium],
            'monthly_premium': [monthly_premium],
            'claims_count': [claims_count],
            'avg_claim_amount': [avg_claim_amount],
            'total_claims_paid': [total_claims_paid],
            'chronic_count': [chronic_count],
            'hypertension': [1 if hypertension else 0],
            'diabetes': [1 if diabetes else 0],
            'asthma': [1 if asthma else 0],
            'copd': [1 if copd else 0],
            'cardiovascular_disease': [1 if cardiovascular_disease else 0],
            'cancer_history': [1 if cancer_history else 0],
            'kidney_disease': [1 if kidney_disease else 0],
            'liver_disease': [1 if liver_disease else 0],
            'arthritis': [1 if arthritis else 0],
            'mental_health': [1 if mental_health else 0],
            'proc_imaging_count': [proc_imaging_count],
            'proc_surgery_count': [proc_surgery_count],
            'proc_physio_count': [proc_physio_count],
            'proc_consult_count': [proc_consult_count],
            'proc_lab_count': [proc_lab_count],
            'had_major_procedure': [1 if had_major_procedure else 0]
        })
        
        # Preprocess
        feature_cols = metadata['numerical_cols'] + metadata['categorical_cols']
        X_input = preprocessor.transform(input_data[feature_cols])
        
        # Prediksi
        cost_pred = cost_model.predict(X_input)[0]
        risk_pred = risk_model.predict(X_input)[0]
        high_risk_pred = high_risk_model.predict(X_input)[0]
        high_risk_proba = high_risk_model.predict_proba(X_input)[0][1]
        
        # Tampilkan hasil
        st.markdown("---")
        st.markdown('<div class="section-header">Hasil Prediksi</div>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Biaya Medis Tahunan", f"${cost_pred:,.2f}")
        
        with col2:
            st.metric("Skor Risiko", f"{risk_pred:.4f}")
        
        with col3:
            risk_status = "Risiko Tinggi" if high_risk_pred == 1 else "Risiko Rendah"
            st.metric("Klasifikasi Risiko", risk_status, f"Keyakinan: {high_risk_proba:.2%}")
        
        # Interpretasi
        st.info(f"""
        **Interpretasi Skor Risiko:**
        - Skor Risiko: {risk_pred:.4f} (Rentang: 0.0 - 1.0)
        - Skor yang lebih tinggi menunjukkan risiko medis yang lebih tinggi
        - Klasifikasi: **{risk_status}** dengan keyakinan {high_risk_proba:.1%}
        """)

def show_visualizations_page(df):
    """Tampilkan visualisasi data."""
    st.markdown('<div class="section-header">Visualisasi Data</div>', unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4 = st.tabs(["Distribusi Biaya", "Analisis Risiko", "Demografis", "Metrik Kesehatan"])
    
    with tab1:
        st.subheader("Distribusi Biaya Medis Tahunan")
        fig = px.histogram(df, x='annual_medical_cost', nbins=50, 
                          title="Distribusi Biaya Medis Tahunan",
                          labels={'annual_medical_cost': 'Biaya Medis Tahunan ($)', 'count': 'Frekuensi'})
        st.plotly_chart(fig, use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            fig2 = px.box(df, y='annual_medical_cost', title="Distribusi Biaya berdasarkan Status Risiko")
            st.plotly_chart(fig2, use_container_width=True)
        with col2:
            fig3 = px.violin(df, x='is_high_risk', y='annual_medical_cost', 
                           title="Distribusi Biaya: Risiko Tinggi vs Rendah",
                           labels={'is_high_risk': 'Risiko Tinggi (1=Ya, 0=Tidak)'})
            st.plotly_chart(fig3, use_container_width=True)
    
    with tab2:
        st.subheader("Analisis Skor Risiko")
        fig = px.histogram(df, x='risk_score', nbins=50, 
                          title="Distribusi Skor Risiko",
                          labels={'risk_score': 'Skor Risiko', 'count': 'Frekuensi'})
        st.plotly_chart(fig, use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            risk_by_age = df.groupby(pd.cut(df['age'], bins=10))['risk_score'].mean()
            fig2 = px.bar(x=risk_by_age.index.astype(str), y=risk_by_age.values,
                         title="Rata-rata Skor Risiko berdasarkan Kelompok Usia",
                         labels={'x': 'Kelompok Usia', 'y': 'Rata-rata Skor Risiko'})
            st.plotly_chart(fig2, use_container_width=True)
        with col2:
            fig3 = px.scatter(df.sample(min(5000, len(df))), x='age', y='risk_score', 
                            color='is_high_risk',
                            title="Skor Risiko vs Usia",
                            labels={'is_high_risk': 'Risiko Tinggi'})
            st.plotly_chart(fig3, use_container_width=True)
    
    with tab3:
        st.subheader("Analisis Demografis")
        col1, col2 = st.columns(2)
        with col1:
            fig = px.box(df, x='sex', y='annual_medical_cost', 
                        title="Biaya berdasarkan Jenis Kelamin")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig2 = px.box(df, x='region', y='annual_medical_cost',
                         title="Biaya berdasarkan Wilayah")
            st.plotly_chart(fig2, use_container_width=True)
        
        fig3 = px.box(df, x='network_tier', y='annual_medical_cost',
                     title="Biaya berdasarkan Tier Asuransi")
        st.plotly_chart(fig3, use_container_width=True)
    
    with tab4:
        st.subheader("Analisis Metrik Kesehatan")
        col1, col2 = st.columns(2)
        with col1:
            fig = px.scatter(df.sample(min(5000, len(df))), x='bmi', y='annual_medical_cost',
                           color='is_high_risk', title="BMI vs Biaya Medis")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig2 = px.scatter(df.sample(min(5000, len(df))), x='chronic_count', y='annual_medical_cost',
                            color='is_high_risk', title="Jumlah Penyakit Kronis vs Biaya Medis")
            st.plotly_chart(fig2, use_container_width=True)
        
        fig3 = px.box(df, x='smoker', y='annual_medical_cost',
                     title="Biaya berdasarkan Status Merokok")
        st.plotly_chart(fig3, use_container_width=True)

def show_performance_page(model_metrics, cost_model, risk_model, high_risk_model, metadata):
    """Tampilkan kinerja model."""
    st.markdown('<div class="section-header">Kinerja Model</div>', unsafe_allow_html=True)
    
    if model_metrics is None:
        st.warning("Metrik model tidak tersedia. Silakan latih ulang model.")
        return
    
    tab1, tab2, tab3 = st.tabs(["Model Biaya", "Model Skor Risiko", "Model Risiko Tinggi"])
    
    with tab1:
        st.subheader("Regresi Biaya Medis Tahunan")
        if 'cost' in model_metrics:
            results = model_metrics['cost']
            for model_name, metrics in results.items():
                with st.expander(f"Metrik {model_name}"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Metrik Pelatihan:**")
                        st.write(f"- MAE: {metrics['train_mae']:.2f}")
                        st.write(f"- RMSE: {metrics['train_rmse']:.2f}")
                        st.write(f"- R²: {metrics['train_r2']:.4f}")
                    with col2:
                        st.write("**Metrik Uji:**")
                        st.write(f"- MAE: {metrics['test_mae']:.2f}")
                        st.write(f"- RMSE: {metrics['test_rmse']:.2f}")
                        st.write(f"- R²: {metrics['test_r2']:.4f}")
        
        # Feature importance
        st.subheader("Pentingnya Fitur")
        importance = get_feature_importance(cost_model, metadata['feature_names'])
        if importance:
            fig = px.bar(x=importance['importances'], y=importance['features'],
                        orientation='h', title="15 Fitur Paling Penting",
                        labels={'x': 'Tingkat Kepentingan', 'y': 'Fitur'})
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Regresi Skor Risiko")
        if 'risk_score' in model_metrics:
            results = model_metrics['risk_score']
            for model_name, metrics in results.items():
                with st.expander(f"Metrik {model_name}"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Metrik Pelatihan:**")
                        st.write(f"- MAE: {metrics['train_mae']:.4f}")
                        st.write(f"- RMSE: {metrics['train_rmse']:.4f}")
                        st.write(f"- R²: {metrics['train_r2']:.4f}")
                    with col2:
                        st.write("**Metrik Uji:**")
                        st.write(f"- MAE: {metrics['test_mae']:.4f}")
                        st.write(f"- RMSE: {metrics['test_rmse']:.4f}")
                        st.write(f"- R²: {metrics['test_r2']:.4f}")
        
        # Feature importance
        st.subheader("Pentingnya Fitur")
        importance = get_feature_importance(risk_model, metadata['feature_names'])
        if importance:
            fig = px.bar(x=importance['importances'], y=importance['features'],
                        orientation='h', title="15 Fitur Paling Penting",
                        labels={'x': 'Tingkat Kepentingan', 'y': 'Fitur'})
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Klasifikasi Risiko Tinggi")
        if 'high_risk' in model_metrics:
            results = model_metrics['high_risk']
            for model_name, metrics in results.items():
                with st.expander(f"Metrik {model_name}"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Metrik Pelatihan:**")
                        st.write(f"- Akurasi: {metrics['train_acc']:.4f}")
                        st.write(f"- Presisi: {metrics['train_prec']:.4f}")
                        st.write(f"- Recall: {metrics['train_rec']:.4f}")
                        st.write(f"- F1 Score: {metrics['train_f1']:.4f}")
                        st.write(f"- AUC: {metrics['train_auc']:.4f}")
                    with col2:
                        st.write("**Metrik Uji:**")
                        st.write(f"- Akurasi: {metrics['test_acc']:.4f}")
                        st.write(f"- Presisi: {metrics['test_prec']:.4f}")
                        st.write(f"- Recall: {metrics['test_rec']:.4f}")
                        st.write(f"- F1 Score: {metrics['test_f1']:.4f}")
                        st.write(f"- AUC: {metrics['test_auc']:.4f}")
        
        # Feature importance
        st.subheader("Pentingnya Fitur")
        importance = get_feature_importance(high_risk_model, metadata['feature_names'])
        if importance:
            fig = px.bar(x=importance['importances'], y=importance['features'],
                        orientation='h', title="15 Fitur Paling Penting",
                        labels={'x': 'Tingkat Kepentingan', 'y': 'Fitur'})
            st.plotly_chart(fig, use_container_width=True)

def show_data_explorer_page(df):
    """Tampilkan eksplorasi data."""
    st.markdown('<div class="section-header">Eksplorasi Data</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        num_rows = st.slider("Jumlah baris yang ditampilkan", 10, 1000, 100)
    with col2:
        show_stats = st.checkbox("Tampilkan Statistik", value=True)
    
    st.subheader("Pratinjau Dataset")
    st.dataframe(df.head(num_rows), use_container_width=True)
    
    if show_stats:
        st.subheader("Statistik Dataset")
        st.dataframe(df.describe(), use_container_width=True)
    
    st.subheader("Filter Data")
    col1, col2, col3 = st.columns(3)
    with col1:
        min_cost = st.number_input("Biaya Minimum", value=float(df['annual_medical_cost'].min()))
        max_cost = st.number_input("Biaya Maksimum", value=float(df['annual_medical_cost'].max()))
    with col2:
        min_risk = st.number_input("Skor Risiko Minimum", value=0.0, max_value=1.0, step=0.1)
        max_risk = st.number_input("Skor Risiko Maksimum", value=1.0, max_value=1.0, step=0.1)
    with col3:
        high_risk_filter = st.selectbox("Risiko Tinggi", ["Semua", "Ya", "Tidak"])
    
    filtered_df = df[
        (df['annual_medical_cost'] >= min_cost) &
        (df['annual_medical_cost'] <= max_cost) &
        (df['risk_score'] >= min_risk) &
        (df['risk_score'] <= max_risk)
    ]
    
    if high_risk_filter != "Semua":
        filtered_df = filtered_df[filtered_df['is_high_risk'] == (1 if high_risk_filter == "Ya" else 0)]
    
    st.write(f"Baris yang difilter: {len(filtered_df):,} / {len(df):,}")
    st.dataframe(filtered_df.head(100), use_container_width=True)

def show_insights_page(df, cost_model, risk_model, high_risk_model, metadata):
    """Tampilkan wawasan dan feature importance."""
    st.markdown('<div class="section-header">Wawasan Utama</div>', unsafe_allow_html=True)
    
    st.subheader("Ringkasan Dataset")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Rekaman", f"{len(df):,}")
    with col2:
        st.metric("Rata-rata Biaya", f"${df['annual_medical_cost'].mean():,.2f}")
    with col3:
        st.metric("Persentase Risiko Tinggi", f"{df['is_high_risk'].mean()*100:.1f}%")
    with col4:
        st.metric("Rata-rata Skor Risiko", f"{df['risk_score'].mean():.3f}")
    
    st.subheader("Faktor Utama yang Mempengaruhi Biaya Medis")
    # Analisis korelasi
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    correlations = df[numerical_cols].corr()['annual_medical_cost'].sort_values(ascending=False)
    top_factors = correlations[correlations.index != 'annual_medical_cost'].head(10)
    
    fig = px.bar(x=top_factors.values, y=top_factors.index,
                orientation='h', title="10 Fitur Teratas yang Berkorelasi dengan Biaya Medis",
                labels={'x': 'Korelasi', 'y': 'Fitur'})
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Faktor Risiko")
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Karakteristik Pasien Risiko Tinggi:**")
        high_risk_df = df[df['is_high_risk'] == 1]
        st.write(f"- Rata-rata Usia: {high_risk_df['age'].mean():.1f} tahun")
        st.write(f"- Rata-rata Penyakit Kronis: {high_risk_df['chronic_count'].mean():.1f}")
        st.write(f"- Rata-rata BMI: {high_risk_df['bmi'].mean():.1f}")
        st.write(f"- Rata-rata Biaya: ${high_risk_df['annual_medical_cost'].mean():,.2f}")
    
    with col2:
        st.write("**Karakteristik Pasien Risiko Rendah:**")
        low_risk_df = df[df['is_high_risk'] == 0]
        st.write(f"- Rata-rata Usia: {low_risk_df['age'].mean():.1f} tahun")
        st.write(f"- Rata-rata Penyakit Kronis: {low_risk_df['chronic_count'].mean():.1f}")
        st.write(f"- Rata-rata BMI: {low_risk_df['bmi'].mean():.1f}")
        st.write(f"- Rata-rata Biaya: ${low_risk_df['annual_medical_cost'].mean():,.2f}")
    
    st.subheader("Perbandingan Pentingnya Fitur antar Model")
    cost_importance = get_feature_importance(cost_model, metadata['feature_names'], top_n=10)
    risk_importance = get_feature_importance(risk_model, metadata['feature_names'], top_n=10)
    high_risk_importance = get_feature_importance(high_risk_model, metadata['feature_names'], top_n=10)
    
    if cost_importance and risk_importance and high_risk_importance:
        fig = make_subplots(rows=1, cols=3, subplot_titles=("Model Biaya", "Model Skor Risiko", "Model Risiko Tinggi"),
                           horizontal_spacing=0.1)
        
        fig.add_trace(go.Bar(y=cost_importance['features'], x=cost_importance['importances'],
                            orientation='h', name="Biaya"), row=1, col=1)
        fig.add_trace(go.Bar(y=risk_importance['features'], x=risk_importance['importances'],
                            orientation='h', name="Risiko"), row=1, col=2)
        fig.add_trace(go.Bar(y=high_risk_importance['features'], x=high_risk_importance['importances'],
                            orientation='h', name="Risiko Tinggi"), row=1, col=3)
        
        fig.update_layout(height=600, showlegend=False, title_text="10 Fitur Teratas per Model")
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
