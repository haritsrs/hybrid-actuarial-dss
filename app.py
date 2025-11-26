"""
Dashboard Streamlit untuk Analisis Aktuarial.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import shap
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib import colors
from datetime import datetime

# Konfigurasi halaman
st.set_page_config(
    page_title="Analisis Aktuarial",
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
    import os
    if not os.path.exists('medical_insurance.csv'):
        raise FileNotFoundError(
            "File 'medical_insurance.csv' tidak ditemukan. "
            "Pastikan file dataset ada di root project."
        )
    return pd.read_csv('medical_insurance.csv')

@st.cache_resource
def load_models():
    """Memuat model yang sudah dilatih dan preprocessor."""
    import os
    
    # Check if models directory exists
    if not os.path.exists('models'):
        raise FileNotFoundError(
            "Directory 'models' tidak ditemukan. "
            "Pastikan folder 'models' ada di root project dan berisi file-file model (.pkl)."
        )
    
    # List of required model files
    required_files = [
        'models/preprocessor.pkl',
        'models/preprocessor_metadata.pkl',
        'models/cost_model.pkl',
        'models/risk_score_model.pkl',
        'models/high_risk_model.pkl'
    ]
    
    # Check if all required files exist
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        raise FileNotFoundError(
            f"File model tidak ditemukan: {', '.join(missing_files)}\n"
            "Pastikan semua file model sudah di-commit ke repository untuk deployment Streamlit Cloud."
        )
    
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

# ============================================================================
# FULL INDONESIAN TRANSLATION MAP (FOR AI AGENT)
# ============================================================================
# 
# This section implements comprehensive Indonesian translation mappings for all
# categorical values in the application. There are two types of mappings:
# 
# 1. FORWARD MAPPINGS: Indonesian UI values → English backend values
#    Used when converting user input (Indonesian) to model format (English)
# 
# 2. REVERSE MAPPINGS: English backend values → Indonesian display values  
#    Used when displaying model outputs (English) to users (Indonesian)
# 
# Usage:
#   - Use translate_to_english(indonesian_value, category) for UI → backend
#   - Use translate_to_indonesian(english_value, category) for backend → UI
# ============================================================================

# Forward mappings: Indonesian UI → English backend values
SEX_MAP_FORWARD = {
    "Laki-laki": "Male",
    "Perempuan": "Female"
}

REGION_MAP_FORWARD = {
    "AS Utara": "North",
    "AS Selatan": "South",
    "AS Timur": "East",
    "AS Barat": "West",
    "AS Timur Laut": "Northeast",
    "AS Barat Laut": "Northwest",
    "AS Tenggara": "Southeast",
    "AS Barat Daya": "Southwest"
}

URBAN_RURAL_MAP_FORWARD = {
    "Perkotaan": "Urban",
    "Pinggiran Kota": "Suburban",
    "Pedesaan": "Rural"
}

EDUCATION_MAP_FORWARD = {
    "Tidak tamat SMA": "No HS",
    "Pernah SMA tapi tidak lulus": "Some HS",
    "Lulus SMA": "HS Grad",
    "Pernah kuliah (belum lulus)": "Some College",
    "Lulus sarjana": "College Grad",
    "Pascasarjana": "Post Grad",
    "Doktor": "Doctorate"
}

MARITAL_STATUS_MAP_FORWARD = {
    "Lajang": "Single",
    "Menikah": "Married",
    "Bercerai": "Divorced",
    "Duda/Janda": "Widowed"
}

EMPLOYMENT_MAP_FORWARD = {
    "Bekerja": "Employed",
    "Tidak bekerja": "Unemployed",
    "Wiraswasta": "Self-employed",
    "Pensiun": "Retired"
}

SMOKER_MAP_FORWARD = {
    "Tidak pernah": "Never",
    "Mantan perokok": "Former",
    "Perokok aktif": "Current"
}

ALCOHOL_FREQ_MAP_FORWARD = {
    "Tidak pernah": "None",
    "Sesekali": "Occasional",
    "Rutin": "Regular"
}

PLAN_TYPE_MAP_FORWARD = {
    "PPO": "PPO",
    "HMO": "HMO",
    "EPO": "EPO"
}

NETWORK_TIER_MAP_FORWARD = {
    "Bronze": "Bronze",
    "Silver": "Silver",
    "Gold": "Gold",
    "Platinum": "Platinum"
}

BOOLEAN_MAP_FORWARD = {
    "Ya": "Yes",
    "Tidak": "No"
}

# Reverse mappings: English backend → Indonesian display
SEX_MAP_REVERSE = {v: k for k, v in SEX_MAP_FORWARD.items()}
REGION_MAP_REVERSE = {v: k for k, v in REGION_MAP_FORWARD.items()}
URBAN_RURAL_MAP_REVERSE = {v: k for k, v in URBAN_RURAL_MAP_FORWARD.items()}
EDUCATION_MAP_REVERSE = {v: k for k, v in EDUCATION_MAP_FORWARD.items()}
MARITAL_STATUS_MAP_REVERSE = {v: k for k, v in MARITAL_STATUS_MAP_FORWARD.items()}
EMPLOYMENT_MAP_REVERSE = {v: k for k, v in EMPLOYMENT_MAP_FORWARD.items()}
SMOKER_MAP_REVERSE = {v: k for k, v in SMOKER_MAP_FORWARD.items()}
ALCOHOL_FREQ_MAP_REVERSE = {v: k for k, v in ALCOHOL_FREQ_MAP_FORWARD.items()}
PLAN_TYPE_MAP_REVERSE = {v: k for k, v in PLAN_TYPE_MAP_FORWARD.items()}
NETWORK_TIER_MAP_REVERSE = {v: k for k, v in NETWORK_TIER_MAP_FORWARD.items()}
BOOLEAN_MAP_REVERSE = {v: k for k, v in BOOLEAN_MAP_FORWARD.items()}

def translate_to_indonesian(value, category):
    """
    Translate English backend value to Indonesian display value.
    
    Args:
        value: English value from backend/model
        category: Category name ('sex', 'region', 'education', 'marital_status', 
                 'employment', 'smoker', 'alcohol_freq', 'plan_type', 
                 'network_tier', 'urban_rural', 'boolean')
    
    Returns:
        Indonesian translation or original value if not found
    """
    translation_maps = {
        'sex': SEX_MAP_REVERSE,
        'region': REGION_MAP_REVERSE,
        'education': EDUCATION_MAP_REVERSE,
        'marital_status': MARITAL_STATUS_MAP_REVERSE,
        'employment': EMPLOYMENT_MAP_REVERSE,
        'smoker': SMOKER_MAP_REVERSE,
        'alcohol_freq': ALCOHOL_FREQ_MAP_REVERSE,
        'plan_type': PLAN_TYPE_MAP_REVERSE,
        'network_tier': NETWORK_TIER_MAP_REVERSE,
        'urban_rural': URBAN_RURAL_MAP_REVERSE,
        'boolean': BOOLEAN_MAP_REVERSE
    }
    
    if category in translation_maps:
        return translation_maps[category].get(value, value)
    return value

def translate_to_english(value, category):
    """
    Translate Indonesian UI value to English backend value.
    
    Args:
        value: Indonesian value from UI
        category: Category name
    
    Returns:
        English translation or original value if not found
    """
    translation_maps = {
        'sex': SEX_MAP_FORWARD,
        'region': REGION_MAP_FORWARD,
        'education': EDUCATION_MAP_FORWARD,
        'marital_status': MARITAL_STATUS_MAP_FORWARD,
        'employment': EMPLOYMENT_MAP_FORWARD,
        'smoker': SMOKER_MAP_FORWARD,
        'alcohol_freq': ALCOHOL_FREQ_MAP_FORWARD,
        'plan_type': PLAN_TYPE_MAP_FORWARD,
        'network_tier': NETWORK_TIER_MAP_FORWARD,
        'urban_rural': URBAN_RURAL_MAP_FORWARD,
        'boolean': BOOLEAN_MAP_FORWARD
    }
    
    if category in translation_maps:
        return translation_maps[category].get(value, value)
    return value

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

def validate_inputs(age, bmi, systolic_bp, diastolic_bp):
    """Validasi input pengguna."""
    errors = []
    if age < 0 or age > 120:
        errors.append("Usia harus antara 0 dan 120 tahun")
    if bmi < 10 or bmi > 60:
        errors.append("BMI harus antara 10 dan 60")
    if systolic_bp < 70 or systolic_bp > 250:
        errors.append("Tekanan darah sistolik harus antara 70 dan 250")
    if diastolic_bp < 40 or diastolic_bp > 150:
        errors.append("Tekanan darah diastolik harus antara 40 dan 150")
    if systolic_bp <= diastolic_bp:
        errors.append("Tekanan darah sistolik harus lebih besar dari diastolik")
    return errors

def calculate_risk_load(risk_score):
    """Calculate risk load percentage based on risk score."""
    if risk_score < 0.33:
        return 0.10  # 10% for low risk
    elif risk_score < 0.67:
        return 0.25  # 25% for medium risk
    else:
        return 0.40  # 40% for high risk

def calculate_underwriting_flags(age, bmi, systolic_bp, diastolic_bp, smoker, chronic_count, 
                                 hypertension, diabetes, cardiovascular_disease):
    """Generate underwriting flags based on health metrics."""
    flags = []
    
    if bmi >= 30:
        flags.append(("BMI Tinggi", f"BMI {bmi:.1f} menunjukkan obesitas", "Tinggi"))
    elif bmi >= 25:
        flags.append(("BMI Di Atas Normal", f"BMI {bmi:.1f} menunjukkan kelebihan berat badan", "Sedang"))
    
    if smoker == "Current":
        flags.append(("Perokok Aktif", "Merokok aktif meningkatkan risiko kesehatan", "Tinggi"))
    elif smoker == "Former":
        flags.append(("Mantan Perokok", "Riwayat merokok sebelumnya", "Sedang"))
    
    if systolic_bp >= 140 or diastolic_bp >= 90:
        flags.append(("Tekanan Darah Tinggi", f"TD {systolic_bp}/{diastolic_bp} mmHg", "Tinggi"))
    elif systolic_bp >= 130 or diastolic_bp >= 85:
        flags.append(("Tekanan Darah Di Atas Normal", f"TD {systolic_bp}/{diastolic_bp} mmHg", "Sedang"))
    
    if chronic_count >= 3:
        flags.append(("Banyak Penyakit Kronis", f"{chronic_count} penyakit kronis", "Tinggi"))
    elif chronic_count >= 2:
        flags.append(("Penyakit Kronis", f"{chronic_count} penyakit kronis", "Sedang"))
    
    if hypertension:
        flags.append(("Hipertensi", "Terdiagnosis hipertensi", "Tinggi"))
    if diabetes:
        flags.append(("Diabetes", "Terdiagnosis diabetes", "Tinggi"))
    if cardiovascular_disease:
        flags.append(("Penyakit Jantung", "Penyakit kardiovaskular terdeteksi", "Tinggi"))
    
    if age >= 65:
        flags.append(("Usia Lanjut", f"Usia {age} tahun", "Sedang"))
    
    return flags

def get_plan_recommendation(risk_score, cost_pred):
    """Recommend insurance plan based on risk and predicted cost."""
    if risk_score < 0.33:
        return {
            "plan": "Bronze/HMO",
            "reasoning": "Profil risiko rendah - plan hemat biaya disarankan",
            "tier": "Bronze",
            "type": "HMO"
        }
    elif risk_score < 0.67:
        return {
            "plan": "Silver/PPO",
            "reasoning": "Profil risiko sedang - cakupan seimbang disarankan",
            "tier": "Silver",
            "type": "PPO"
        }
    else:
        return {
            "plan": "Gold/PPO",
            "reasoning": "Profil risiko tinggi - cakupan lengkap disarankan",
            "tier": "Gold",
            "type": "PPO"
        }

def calculate_shap_values(model, X_input, feature_names, max_display=15):
    """Calculate SHAP values for model explanation."""
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_input)
        
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        feature_contributions = {}
        for i, feature in enumerate(feature_names[:len(shap_values[0])]):
            feature_contributions[feature] = float(shap_values[0][i])
        
        sorted_contributions = sorted(feature_contributions.items(), 
                                     key=lambda x: abs(x[1]), 
                                     reverse=True)[:max_display]
        
        return {
            'features': [f[0] for f in sorted_contributions],
            'values': [f[1] for f in sorted_contributions]
        }
    except Exception as e:
        return None

def generate_underwriting_summary(input_data, cost_pred, risk_pred, high_risk_pred, 
                                  risk_load, recommended_premium, flags, plan_rec):
    """Generate underwriting summary text."""
    # Use reverse mappings for display
    risk_level = "Tinggi" if risk_pred >= 0.67 else ("Sedang" if risk_pred >= 0.33 else "Rendah")
    risk_class = "Risiko Tinggi" if high_risk_pred == 1 else "Risiko Rendah"
    smoker_display = translate_to_indonesian(input_data['smoker'].iloc[0], 'smoker')
    
    # Translate other categorical values for display
    sex_display = translate_to_indonesian(input_data.get('sex', [''])[0] if 'sex' in input_data.columns else '', 'sex')
    region_display = translate_to_indonesian(input_data.get('region', [''])[0] if 'region' in input_data.columns else '', 'region')
    education_display = translate_to_indonesian(input_data.get('education', [''])[0] if 'education' in input_data.columns else '', 'education')
    marital_display = translate_to_indonesian(input_data.get('marital_status', [''])[0] if 'marital_status' in input_data.columns else '', 'marital_status')
    
    summary = f"""
RINGKASAN UNDERWRITING

INFORMASI:
• Usia: {input_data['age'].iloc[0]} tahun
• BMI: {input_data['bmi'].iloc[0]:.1f}
• Status Merokok: {smoker_display}
• Penyakit Kronis: {input_data['chronic_count'].iloc[0]}

PREDIKSI:
• Biaya Medis Tahunan: ${cost_pred:,.2f}
• Skor Risiko: {risk_pred:.4f} (Risiko {risk_level})
• Klasifikasi: {risk_class}

PERHITUNGAN:
• Tambahan Risiko: {risk_load*100:.1f}%
• Premi Dasar: ${input_data['monthly_premium'].iloc[0]*12:,.2f}/tahun
• Premi Disarankan: ${recommended_premium:,.2f}/tahun

CATATAN RISIKO:
"""
    if flags:
        high_flags = [f for f in flags if f[2] == "Tinggi"]
        medium_flags = [f for f in flags if f[2] == "Sedang"]
        
        if high_flags:
            summary += "• Risiko Tinggi:\n"
            for flag_name, description, _ in high_flags:
                summary += f"  - {flag_name}: {description}\n"
        
        if medium_flags:
            summary += "• Risiko Sedang:\n"
            for flag_name, description, _ in medium_flags:
                summary += f"  - {flag_name}: {description}\n"
    else:
        summary += "• Tidak ada catatan risiko khusus\n"
    
    summary += f"""
REKOMENDASI:
• Plan: {plan_rec['plan']}
• Alasan: {plan_rec['reasoning']}

Dibuat: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    return summary

def create_pdf_report(input_data, cost_pred, risk_pred, high_risk_pred, high_risk_proba,
                     risk_load, recommended_premium, claims_probability, flags, plan_rec,
                     risk_factors, underwriting_summary):
    """Create PDF report."""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    story = []
    styles = getSampleStyleSheet()
    
    # Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#1f77b4'),
        spaceAfter=30,
    )
    story.append(Paragraph("Laporan Analisis Aktuarial", title_style))
    story.append(Spacer(1, 0.2*inch))
    
    # Date
    story.append(Paragraph(f"Dibuat: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    story.append(Spacer(1, 0.2*inch))
    
    # Summary
    story.append(Paragraph("Ringkasan", styles['Heading2']))
    story.append(Spacer(1, 0.1*inch))
    summary_lines = underwriting_summary.split('\n')
    for line in summary_lines:
        if line.strip():
            story.append(Paragraph(line.strip(), styles['Normal']))
    story.append(Spacer(1, 0.2*inch))
    
    # Predictions
    story.append(Paragraph("Prediksi", styles['Heading2']))
    story.append(Spacer(1, 0.1*inch))
    pred_data = [
        ['Metrik', 'Nilai'],
        ['Prediksi Biaya Tahunan', f"${cost_pred:,.2f}"],
        ['Skor Risiko', f"{risk_pred:.4f}"],
        ['Klasifikasi', "Risiko Tinggi" if high_risk_pred == 1 else "Risiko Rendah"],
        ['Keyakinan', f"{high_risk_proba:.1%}"],
    ]
    pred_table = Table(pred_data, colWidths=[3*inch, 2*inch])
    pred_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(pred_table)
    story.append(Spacer(1, 0.2*inch))
    
    # Actuarial Calculations
    story.append(Paragraph("Perhitungan Premi", styles['Heading2']))
    story.append(Spacer(1, 0.1*inch))
    actuarial_data = [
        ['Perhitungan', 'Nilai'],
        ['Tambahan Risiko', f"{risk_load*100:.1f}%"],
        ['Premi Dasar', f"${input_data['monthly_premium'].iloc[0]*12:,.2f}"],
        ['Premi Disarankan', f"${recommended_premium:,.2f}"],
        ['Probabilitas Klaim', f"{claims_probability:.1%}"],
    ]
    actuarial_table = Table(actuarial_data, colWidths=[3*inch, 2*inch])
    actuarial_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(actuarial_table)
    
    doc.build(story)
    buffer.seek(0)
    return buffer

def show_explanation_section():
    """Show comprehensive explanation section for users."""
    
    st.markdown("""
    ## Panduan Penggunaan Sistem Analisis Aktuarial
    
    Sistem Analisis Aktuarial untuk Asuransi Kesehatan adalah aplikasi berbasis machine learning 
    yang dirancang untuk membantu perusahaan asuransi dalam menilai risiko kesehatan dan menentukan 
    premi yang sesuai untuk calon nasabah berdasarkan analisis data aktuarial.
    """)
    
    st.markdown("---")
    
    # Overview section
    st.markdown("### 1. Tujuan Aplikasi")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        #### 1.1 Prediksi Biaya Medis
        Sistem ini memprediksi biaya medis tahunan yang akan dikeluarkan oleh nasabah berdasarkan 
        profil kesehatan, demografis, dan historis penggunaan layanan kesehatan.
        """)
    
    with col2:
        st.markdown("""
        #### 1.2 Penilaian Risiko Kesehatan
        Sistem menghitung skor risiko kesehatan untuk mengklasifikasikan nasabah ke dalam kategori 
        risiko rendah, sedang, atau tinggi berdasarkan faktor-faktor kesehatan yang teridentifikasi.
        """)
    
    with col3:
        st.markdown("""
        #### 1.3 Kalkulasi Premi Asuransi
        Sistem menentukan premi asuransi yang tepat berdasarkan tingkat risiko dan prediksi biaya 
        medis, termasuk perhitungan tambahan risiko (risk load) yang sesuai dengan kategori risiko.
        """)
    
    st.markdown("---")
    
    # How to use section
    st.markdown("### 2. Panduan Operasional")
    
    st.markdown("""
    #### 2.1 Tab Input
    
    Tab Input digunakan untuk memasukkan data calon nasabah. Lengkapi semua informasi berikut:
    
    **2.1.1 Informasi Demografis**
    - Usia (tahun)
    - Jenis kelamin
    - Wilayah geografis
    - Pendapatan tahunan (USD)
    - Tingkat pendidikan
    - Status pernikahan
    - Jumlah anggota keluarga
    - Jumlah tanggungan
    - Klasifikasi lokasi (perkotaan, pinggiran kota, pedesaan)
    
    **2.1.2 Informasi Kesehatan**
    - Body Mass Index (BMI)
    - Status merokok (tidak pernah, mantan perokok, perokok aktif)
    - Frekuensi konsumsi alkohol
    - Tekanan darah sistolik dan diastolik
    - Jumlah kunjungan dokter dalam satu tahun terakhir
    - Jumlah obat yang dikonsumsi
    - Jumlah penyakit kronis
    - Jumlah rawat inap dalam 3 tahun terakhir
    
    **2.1.3 Kondisi Kesehatan Kronis**
    Centang semua kondisi kesehatan yang relevan:
    - Hipertensi
    - Diabetes
    - Asma
    - Chronic Obstructive Pulmonary Disease (COPD)
    - Penyakit kardiovaskular
    - Riwayat kanker
    - Penyakit ginjal
    - Penyakit hati
    - Arthritis
    - Gangguan kesehatan mental
    - Riwayat prosedur medis besar
    
    **2.1.4 Informasi Polis Asuransi**
    - Tipe plan (PPO, HMO, EPO)
    - Tier jaringan (Bronze, Silver, Gold, Platinum)
    - Deductible (USD)
    - Copay (USD)
    - Premi bulanan (USD)
    - Kualitas provider (rating 1-5)
    
    Setelah semua informasi diisi, klik tombol **"Prediksi"** untuk memproses data dan menghasilkan analisis.
    
    #### 2.2 Tab Hasil
    
    Tab Hasil menampilkan output analisis setelah proses prediksi selesai. Konten yang ditampilkan meliputi:
    
    **2.2.1 Ringkasan Eksekutif**
    - Prediksi biaya medis tahunan
    - Skor risiko kesehatan
    - Persentase tambahan risiko
    - Premi asuransi yang disarankan
    
    **2.2.2 Detail Perhitungan Premi**
    - Perhitungan premi dasar
    - Perhitungan tambahan risiko
    - Premi akhir yang direkomendasikan
    - Probabilitas pengajuan klaim
    
    **2.2.3 Catatan Risiko (Underwriting Flags)**
    - Daftar flag risiko tinggi yang teridentifikasi
    - Daftar flag risiko sedang yang teridentifikasi
    - Penjelasan masing-masing flag risiko
    
    **2.2.4 Rekomendasi Plan Asuransi**
    - Rekomendasi tipe plan dan tier jaringan berdasarkan profil risiko
    - Alasan pemilihan rekomendasi
    
    **2.2.5 Analisis Faktor Risiko**
    - Visualisasi 15 faktor risiko yang paling berpengaruh
    - Kontribusi relatif setiap faktor terhadap skor risiko
    
    #### 2.3 Tab Simulator
    
    Tab Simulator memungkinkan analisis sensitivitas dengan mengubah variabel-variabel kunci untuk 
    melihat dampaknya terhadap hasil prediksi. Fitur ini berguna untuk:
    
    - Eksplorasi berbagai skenario kesehatan
    - Analisis dampak perubahan gaya hidup terhadap premi
    - Perbandingan antara skenario baseline dan skenario alternatif
    - Optimasi strategi penawaran produk asuransi
    
    Variabel yang dapat disimulasikan meliputi:
    - BMI
    - Tekanan darah sistolik dan diastolik
    - Jumlah penyakit kronis
    - Status merokok
    - Jumlah kunjungan dokter
    - Jumlah obat yang dikonsumsi
    - Usia
    
    Output simulator menampilkan perbandingan antara nilai awal dan nilai setelah perubahan untuk:
    - Prediksi biaya medis tahunan
    - Skor risiko kesehatan
    - Premi asuransi yang disarankan
    
    #### 2.4 Tab Laporan
    
    Tab Laporan menyediakan ringkasan lengkap hasil analisis underwriting dan opsi untuk mengunduh 
    laporan dalam berbagai format:
    
    **2.4.1 Ringkasan Teks**
    - Ringkasan underwriting dalam format teks yang dapat dibaca
    - Informasi lengkap tentang profil nasabah, prediksi, dan rekomendasi
    
    **2.4.2 Ekspor Data**
    - **Format CSV**: Data numerik dan kategorikal dalam format spreadsheet untuk analisis lanjutan
    - **Format PDF**: Laporan lengkap dalam format dokumen profesional untuk dokumentasi dan presentasi
    """)
    
    st.markdown("---")
    
    # Metrics explanation
    st.markdown("### 3. Penjelasan Metrik dan Indikator")
    
    with st.expander("3.1 Prediksi Biaya Medis Tahunan", expanded=False):
        st.markdown("""
        **Definisi**: Perkiraan total biaya medis yang akan dikeluarkan oleh nasabah dalam periode satu tahun kalender.
        
        **Metodologi**: Prediksi dihasilkan menggunakan model machine learning (Gradient Boosting Regressor) yang telah dilatih dengan data historis dari 100,000 rekam medis. Model mempertimbangkan faktor-faktor demografis, kesehatan, dan pola penggunaan layanan kesehatan.
        
        **Satuan**: USD (United States Dollar)
        
        **Penggunaan**: Metrik ini digunakan sebagai dasar untuk:
        - Perencanaan anggaran operasional
        - Penetapan struktur premi asuransi
        - Estimasi liabilitas klaim
        - Analisis profitabilitas produk
        """)
    
    with st.expander("3.2 Skor Risiko Kesehatan", expanded=False):
        st.markdown("""
        **Definisi**: Skor numerik yang menggambarkan tingkat risiko kesehatan nasabah secara kuantitatif.
        
        **Rentang Nilai**: 0.0000 hingga 1.0000
        
        **Interpretasi Kategorikal**:
        - **Risiko Rendah**: Skor < 0.33
          - Tambahan risiko (risk load): 10%
          - Karakteristik: Profil kesehatan baik, faktor risiko minimal
        - **Risiko Sedang**: Skor 0.33 hingga 0.67
          - Tambahan risiko (risk load): 25%
          - Karakteristik: Ada beberapa faktor risiko yang dapat dikelola
        - **Risiko Tinggi**: Skor > 0.67
          - Tambahan risiko (risk load): 40%
          - Karakteristik: Faktor risiko kesehatan signifikan yang memerlukan perhatian khusus
        
        **Faktor yang Mempengaruhi**: BMI, tekanan darah, status merokok, jumlah dan jenis penyakit kronis, usia, riwayat rawat inap, penggunaan obat-obatan, dan faktor demografis lainnya.
        
        **Metodologi**: Skor dihitung menggunakan model Random Forest Regressor yang telah dilatih dan divalidasi dengan data historis.
        """)
    
    with st.expander("3.3 Tambahan Risiko (Risk Load)", expanded=False):
        st.markdown("""
        **Definisi**: Persentase tambahan yang ditambahkan pada premi dasar sebagai kompensasi atas tingkat risiko yang diidentifikasi.
        
        **Struktur Perhitungan**:
        - Risiko Rendah: +10% dari premi dasar
        - Risiko Sedang: +25% dari premi dasar
        - Risiko Tinggi: +40% dari premi dasar
        
        **Contoh Perhitungan**:
        - Premi dasar: $6,000/tahun
        - Kategori risiko: Sedang (25% risk load)
        - Tambahan risiko: $6,000 × 25% = $1,500
        - Premi akhir: $6,000 + $1,500 = $7,500/tahun
        
        **Tujuan**: Risk load memastikan premi mencerminkan tingkat risiko aktual nasabah dan mempertahankan profitabilitas produk asuransi.
        """)
    
    with st.expander("3.4 Premi Asuransi yang Disarankan", expanded=False):
        st.markdown("""
        **Definisi**: Premi asuransi yang direkomendasikan untuk nasabah berdasarkan analisis risiko dan biaya.
        
        **Rumus Perhitungan**:
        ```
        Premi Disarankan = Premi Dasar × (1 + Risk Load)
        ```
        
        Dimana:
        - **Premi Dasar** = Premi bulanan × 12 bulan
        - **Risk Load** = Persentase tambahan risiko berdasarkan kategori risiko
        
        **Penggunaan**: Premi disarankan digunakan sebagai referensi untuk:
        - Menetapkan harga premi akhir dalam proses underwriting
        - Negosiasi dengan calon nasabah
        - Benchmarking terhadap produk kompetitor
        - Analisis profitabilitas dan pricing strategy
        """)
    
    with st.expander("3.5 Probabilitas Pengajuan Klaim", expanded=False):
        st.markdown("""
        **Definisi**: Estimasi kemungkinan nasabah akan mengajukan klaim asuransi dalam periode tertentu berdasarkan profil risiko mereka.
        
        **Rentang Nilai**: 0% hingga 100%
        
        **Interpretasi**:
        - **Probabilitas Tinggi (> 50%)**: Indikasi kuat bahwa nasabah kemungkinan besar akan mengajukan klaim asuransi. Perlu perhatian khusus dalam proses underwriting.
        - **Probabilitas Rendah (< 50%)**: Indikasi bahwa nasabah memiliki kemungkinan relatif kecil untuk mengajukan klaim dalam periode dekat.
        
        **Metodologi**: Probabilitas dihitung menggunakan model klasifikasi XGBoost yang memprediksi apakah nasabah termasuk dalam kategori risiko tinggi berdasarkan fitur-fitur yang dimasukkan.
        
        **Penggunaan**: Metrik ini membantu dalam:
        - Klasifikasi risiko klaim
        - Perencanaan cadangan klaim (claim reserves)
        - Evaluasi risiko portfolio
        """)
    
    with st.expander("3.6 Klasifikasi Risiko", expanded=False):
        st.markdown("""
        **Definisi**: Kategorisasi nasabah ke dalam kelompok risiko berdasarkan prediksi model klasifikasi.
        
        **Kategori**:
        - **Risiko Rendah**: Profil kesehatan yang baik dengan kemungkinan kecil untuk mengajukan klaim besar dalam periode dekat. Karakteristik: faktor risiko minimal, kondisi kesehatan terkontrol.
        - **Risiko Tinggi**: Ada faktor risiko kesehatan yang signifikan yang meningkatkan kemungkinan pengajuan klaim besar. Karakteristik: kondisi kesehatan kronis, faktor risiko yang tidak terkontrol, riwayat penggunaan layanan kesehatan intensif.
        
        **Tingkat Keyakinan (Confidence Level)**: Persentase yang menunjukkan tingkat keyakinan model dalam melakukan klasifikasi. Semakin tinggi nilai confidence, semakin dapat diandalkan klasifikasi tersebut.
        
        **Metodologi**: Klasifikasi dilakukan menggunakan model XGBoost Classifier yang telah dilatih dengan data historis.
        """)
    
    with st.expander("3.7 Catatan Risiko (Underwriting Flags)", expanded=False):
        st.markdown("""
        **Definisi**: Sistem otomatis yang mengidentifikasi dan menandai faktor-faktor risiko kesehatan spesifik dari profil nasabah.
        
        **Kategori Flag Risiko Tinggi**:
        - Body Mass Index (BMI) ≥ 30: Mengindikasikan obesitas yang meningkatkan risiko berbagai kondisi kesehatan
        - Status Perokok Aktif: Konsumsi tembakau aktif secara signifikan meningkatkan risiko penyakit kardiovaskular dan pernapasan
        - Tekanan Darah Tinggi: Tekanan darah sistolik ≥ 140 mmHg atau diastolik ≥ 90 mmHg mengindikasikan hipertensi
        - Multiple Chronic Conditions: Tiga atau lebih penyakit kronis meningkatkan kompleksitas perawatan dan biaya medis
        - Kondisi Kritis Terdiagnosis: Hipertensi, diabetes, atau penyakit kardiovaskular yang telah terdiagnosis
        
        **Kategori Flag Risiko Sedang**:
        - BMI 25-30: Kelebihan berat badan yang meningkatkan risiko kondisi kesehatan tertentu
        - Mantan Perokok: Riwayat merokok sebelumnya tetap memberikan risiko residual meskipun telah berhenti
        - Tekanan Darah di Atas Normal: Tekanan darah sistolik 130-139 mmHg atau diastolik 85-89 mmHg (pre-hipertensi)
        - Dua Penyakit Kronis: Dua kondisi kronis yang memerlukan manajemen berkelanjutan
        - Usia Lanjut: Usia ≥ 65 tahun yang dikaitkan dengan peningkatan kebutuhan layanan kesehatan
        
        **Penggunaan**: Flag ini digunakan oleh underwriter untuk:
        - Identifikasi cepat faktor risiko kritis
        - Prioritas review aplikasi
        - Penentuan kebutuhan informasi tambahan
        - Pengambilan keputusan underwriting yang tepat
        """)
    
    st.markdown("---")
    
    # Recommendations section
    st.markdown("### 4. Rekomendasi Produk Asuransi")
    
    rec_col1, rec_col2, rec_col3 = st.columns(3)
    
    with rec_col1:
        st.markdown("""
        #### 4.1 Bronze / HMO Plan
        **Target Segmentasi**: Risiko Rendah
        
        **Karakteristik Produk**:
        - Struktur harga yang kompetitif dengan premi lebih terjangkau
        - Cakupan dasar untuk kebutuhan kesehatan rutin dan preventif
        - Network provider terbatas dalam sistem Health Maintenance Organization (HMO)
        - Cocok untuk nasabah dengan profil kesehatan baik dan risiko klaim rendah
        
        **Pertimbangan**: Produk ini optimal untuk nasabah muda dan sehat yang mencari cakupan dasar dengan biaya terjangkau.
        """)
    
    with rec_col2:
        st.markdown("""
        #### 4.2 Silver / PPO Plan
        **Target Segmentasi**: Risiko Sedang
        
        **Karakteristik Produk**:
        - Keseimbangan antara biaya dan cakupan manfaat
        - Fleksibilitas memilih provider dalam Preferred Provider Organization (PPO) network
        - Cakupan yang lebih komprehensif dibanding plan Bronze
        - Cocok untuk nasabah dengan beberapa faktor risiko yang terkontrol dengan baik
        
        **Pertimbangan**: Produk ini cocok untuk nasabah yang memerlukan fleksibilitas dalam pemilihan provider sambil tetap menjaga biaya dalam batas wajar.
        """)
    
    with rec_col3:
        st.markdown("""
        #### 4.3 Gold / PPO Plan
        **Target Segmentasi**: Risiko Tinggi
        
        **Karakteristik Produk**:
        - Cakupan lengkap dan komprehensif dengan out-of-pocket costs yang lebih rendah
        - Akses ke network provider premium dalam Preferred Provider Organization (PPO)
        - Cakupan untuk kondisi kesehatan kompleks dan perawatan khusus
        - Cocok untuk nasabah dengan risiko kesehatan tinggi yang memerlukan cakupan maksimal
        
        **Pertimbangan**: Produk ini dirancang untuk nasabah yang mengantisipasi penggunaan layanan kesehatan intensif dan memprioritaskan akses ke perawatan berkualitas tinggi.
        """)
    
    st.markdown("---")
    
    # Best practices section
    st.markdown("### 5. Praktik Terbaik dan Panduan Operasional")
    
    tips_col1, tips_col2 = st.columns(2)
    
    with tips_col1:
        st.markdown("""
        #### 5.1 Praktik Terbaik
        
        **Kualitas Data Input**:
        - Pastikan semua field diisi dengan lengkap dan akurat
        - Gunakan data kesehatan yang terbaru dan terverifikasi
        - Validasi konsistensi data sebelum melakukan prediksi
        - Dokumentasikan sumber data untuk audit trail
        
        **Proses Analisis**:
        - Lakukan validasi input sebelum menjalankan prediksi
        - Gunakan simulator untuk eksplorasi berbagai skenario
        - Bandingkan hasil dengan kasus-kasus serupa sebelumnya
        - Simpan dan dokumentasikan semua analisis dalam format PDF untuk referensi
        
        **Pengambilan Keputusan**:
        - Gunakan hasil prediksi sebagai salah satu faktor dalam keputusan underwriting
        - Pertimbangkan konteks dan faktor eksternal yang tidak tercakup dalam model
        - Lakukan peer review untuk kasus dengan risiko tinggi atau kompleks
        """)
    
    with tips_col2:
        st.markdown("""
        #### 5.2 Peringatan dan Batasan
        
        **Sifat Prediksi**:
        - Hasil prediksi adalah estimasi berdasarkan model statistik dan machine learning
        - Prediksi bukan jaminan absolut dan dapat berubah seiring waktu
        - Akurasi prediksi bergantung pada kualitas dan kelengkapan data input
        
        **Penggunaan Hasil**:
        - Hasil analisis harus digunakan sebagai alat bantu keputusan, bukan satu-satunya faktor
        - Konsultasikan dengan aktuaris senior untuk kasus yang kompleks atau tidak biasa
        - Pertimbangkan faktor-faktor eksternal seperti kondisi pasar, regulasi, dan tren industri
        
        **Pemeliharaan Model**:
        - Lakukan review berkala terhadap akurasi model untuk memastikan performa jangka panjang
        - Monitor perubahan pola data dan perbarui model jika diperlukan
        - Dokumentasikan semua perubahan dan versi model untuk pelacakan
        """)
    
    st.markdown("---")
    
    # Technical details
    with st.expander("6. Informasi Teknis dan Spesifikasi Model", expanded=False):
        st.markdown("""
        #### 6.1 Arsitektur Model Machine Learning
        
        **Model Prediksi Biaya Medis**:
        - Algoritma: Gradient Boosting Regressor
        - Target: Prediksi biaya medis tahunan (continuous variable)
        - Performance Metrics: RMSE (Root Mean Squared Error), R² (Coefficient of Determination)
        
        **Model Skor Risiko**:
        - Algoritma: Random Forest Regressor
        - Target: Skor risiko kesehatan (continuous variable, range 0-1)
        - Performance Metrics: RMSE, R²
        
        **Model Klasifikasi Risiko Tinggi**:
        - Algoritma: XGBoost Classifier
        - Target: Klasifikasi biner (Risiko Rendah / Risiko Tinggi)
        - Performance Metrics: Accuracy, Precision, Recall, AUC-ROC
        
        #### 6.2 Preprocessing Data
        
        **Encoding Variabel Kategorikal**:
        - Variabel kategorikal di-encode menggunakan teknik yang sesuai dengan tipe data
        - Encoding dilakukan secara konsisten antara data training dan inference
        
        **Normalisasi Variabel Numerik**:
        - Variabel numerik dinormalisasi untuk memastikan skala yang konsisten
        - Normalisasi meningkatkan performa dan stabilitas model
        
        **Penanganan Missing Values**:
        - Missing values ditangani sesuai dengan strategi yang ditentukan selama training
        - Implementasi konsisten dengan pipeline preprocessing training
        
        #### 6.3 Interpretabilitas Model
        
        **SHAP Values (SHapley Additive exPlanations)**:
        - Sistem menggunakan SHAP values untuk menjelaskan kontribusi setiap fitur terhadap prediksi
        - SHAP values memberikan interpretasi yang adil dan konsisten
        - Top 15 faktor risiko dengan kontribusi terbesar ditampilkan dalam visualisasi
        
        **Feature Importance**:
        - Analisis feature importance membantu memahami faktor-faktor kunci yang mempengaruhi prediksi
        - Visualisasi membantu dalam komunikasi hasil kepada stakeholder non-teknis
        """)
    
    st.markdown("---")
    
    st.info("""
    **Kontak dan Bantuan Teknis**
    
    Untuk pertanyaan mengenai penggunaan sistem, interpretasi hasil, atau dukungan teknis, 
    silakan menghubungi tim aktuaris atau IT support sesuai dengan prosedur yang berlaku di organisasi Anda.
    """)

def show_actuarial_analysis_page(df, preprocessor, metadata, cost_model, risk_model, high_risk_model):
    """Show actuarial analysis page."""
    
    # Use the comprehensive translation maps defined globally
    SEX_MAP = SEX_MAP_FORWARD
    EDUCATION_MAP = EDUCATION_MAP_FORWARD
    MARITAL_MAP = MARITAL_STATUS_MAP_FORWARD
    REGION_MAP = REGION_MAP_FORWARD
    URBAN_RURAL_MAP = URBAN_RURAL_MAP_FORWARD
    SMOKER_MAP = SMOKER_MAP_FORWARD
    ALCOHOL_MAP = ALCOHOL_FREQ_MAP_FORWARD
    PLAN_TYPE_MAP = PLAN_TYPE_MAP_FORWARD
    NETWORK_TIER_MAP = NETWORK_TIER_MAP_FORWARD
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Penjelasan", "Input", "Hasil", "Simulator", "Laporan"])
    
    with tab1:
        show_explanation_section()
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Informasi Demografis")
            age = st.number_input("Usia", min_value=0, max_value=120, value=45, step=1, key="act_age")
            sex_ui = st.selectbox("Jenis Kelamin", list(SEX_MAP.keys()), key="act_sex")
            sex = SEX_MAP[sex_ui]
            region_ui = st.selectbox("Wilayah", list(REGION_MAP.keys()), key="act_region")
            region = REGION_MAP[region_ui]
            income = st.number_input("Pendapatan (USD)", min_value=0, value=50000, step=1000, format="%d", key="act_income")
            education_ui = st.selectbox("Pendidikan", list(EDUCATION_MAP.keys()), key="act_education")
            education = EDUCATION_MAP[education_ui]
            marital_status_ui = st.selectbox("Status Pernikahan", list(MARITAL_MAP.keys()), key="act_marital")
            marital_status = MARITAL_MAP[marital_status_ui]
            household_size = st.number_input("Jumlah Anggota Keluarga", min_value=1, max_value=10, value=2, step=1, key="act_household")
            dependents = st.number_input("Jumlah Tanggungan", min_value=0, max_value=10, value=1, step=1, key="act_dependents")
            urban_rural_ui = st.selectbox("Lokasi", list(URBAN_RURAL_MAP.keys()), key="act_urban")
            urban_rural = URBAN_RURAL_MAP[urban_rural_ui]
        
        with col2:
            st.markdown("#### Informasi Kesehatan")
            bmi = st.slider("BMI", 10.0, 60.0, 25.0, 0.1, key="act_bmi")
            smoker_ui = st.selectbox("Status Merokok", list(SMOKER_MAP.keys()), key="act_smoker")
            smoker = SMOKER_MAP[smoker_ui]
            alcohol_freq_ui = st.selectbox("Frekuensi Alkohol", list(ALCOHOL_MAP.keys()), key="act_alcohol")
            alcohol_freq = ALCOHOL_MAP[alcohol_freq_ui]
            systolic_bp = st.slider("Tekanan Darah Sistolik", 70, 250, 120, step=1, key="act_systolic")
            diastolic_bp = st.slider("Tekanan Darah Diastolik", 40, 150, 80, step=1, key="act_diastolic")
            visits_last_year = st.number_input("Kunjungan Dokter (Tahun Lalu)", min_value=0, max_value=20, value=2, step=1, key="act_visits")
            medication_count = st.number_input("Jumlah Obat", min_value=0, max_value=10, value=2, step=1, key="act_medication")
            chronic_count = st.number_input("Jumlah Penyakit Kronis", min_value=0, max_value=5, value=0, step=1, key="act_chronic")
            hospitalizations_last_3yrs = st.number_input("Rawat Inap (3 Tahun Terakhir)", min_value=0, max_value=10, value=0, step=1, key="act_hosp")
        
        with st.expander("Kondisi Kesehatan Kronis", expanded=False):
            col3, col4 = st.columns(2)
            with col3:
                hypertension = st.checkbox("Hipertensi", key="act_hypertension")
                diabetes = st.checkbox("Diabetes", key="act_diabetes")
                asthma = st.checkbox("Asma", key="act_asthma")
                copd = st.checkbox("COPD", key="act_copd")
                cardiovascular_disease = st.checkbox("Penyakit Jantung", key="act_cardiovascular")
                cancer_history = st.checkbox("Riwayat Kanker", key="act_cancer")
            with col4:
                kidney_disease = st.checkbox("Penyakit Ginjal", key="act_kidney")
                liver_disease = st.checkbox("Penyakit Hati", key="act_liver")
                arthritis = st.checkbox("Arthritis", key="act_arthritis")
                mental_health = st.checkbox("Kesehatan Mental", key="act_mental")
                had_major_procedure = st.checkbox("Prosedur Besar", key="act_procedure")
        
        with st.expander("Informasi Asuransi", expanded=False):
            col5, col6 = st.columns(2)
            with col5:
                plan_type_ui = st.selectbox("Tipe Plan", list(PLAN_TYPE_MAP.keys()), key="act_plan_type")
                plan_type = PLAN_TYPE_MAP[plan_type_ui]
                network_tier_ui = st.selectbox("Tier Jaringan", list(NETWORK_TIER_MAP.keys()), key="act_tier")
                network_tier = NETWORK_TIER_MAP[network_tier_ui]
                deductible = st.selectbox("Deductible", [500, 1000, 2000, 5000], key="act_deductible")
            with col6:
                copay = st.selectbox("Copay", [10, 20, 30, 50], key="act_copay")
                monthly_premium = st.number_input("Premi Bulanan (USD)", min_value=0.0, value=500.0, step=10.0, key="act_premium")
                provider_quality = st.slider("Kualitas Provider", 1.0, 5.0, 3.5, 0.1, key="act_provider")
        
        # Default values
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
        
        if st.button("Prediksi", type="primary", use_container_width=True):
            # Validate inputs
            validation_errors = validate_inputs(age, bmi, systolic_bp, diastolic_bp)
            if validation_errors:
                for error in validation_errors:
                    st.error(error)
                st.stop()
            
            with st.spinner("Memproses..."):
                # Create input DataFrame
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
                
                # Predictions
                cost_pred = cost_model.predict(X_input)[0]
                risk_pred = risk_model.predict(X_input)[0]
                high_risk_pred = high_risk_model.predict(X_input)[0]
                high_risk_proba = high_risk_model.predict_proba(X_input)[0][1]
                
                # Store in session state
                st.session_state.actuarial_results = {
                    'input_data': input_data,
                    'X_input': X_input,
                    'cost_pred': cost_pred,
                    'risk_pred': risk_pred,
                    'high_risk_pred': high_risk_pred,
                    'high_risk_proba': high_risk_proba
                }
                
                st.success("Selesai! Lihat tab Hasil.")
    
    with tab3:
        if 'actuarial_results' not in st.session_state:
            st.info("Lakukan prediksi di tab Input terlebih dahulu.")
        else:
            results = st.session_state.actuarial_results
            input_data = results['input_data']
            cost_pred = results['cost_pred']
            risk_pred = results['risk_pred']
            high_risk_pred = results['high_risk_pred']
            high_risk_proba = results['high_risk_proba']
            X_input = results['X_input']
            
            # Actuarial Calculations
            risk_load = calculate_risk_load(risk_pred)
            base_premium = input_data['monthly_premium'].iloc[0] * 12
            recommended_premium = base_premium * (1 + risk_load)
            claims_probability = high_risk_proba
            
            # Underwriting Flags
            flags = calculate_underwriting_flags(
                input_data['age'].iloc[0], input_data['bmi'].iloc[0],
                input_data['systolic_bp'].iloc[0], input_data['diastolic_bp'].iloc[0],
                input_data['smoker'].iloc[0], input_data['chronic_count'].iloc[0],
                input_data['hypertension'].iloc[0] == 1,
                input_data['diabetes'].iloc[0] == 1,
                input_data['cardiovascular_disease'].iloc[0] == 1
            )
            
            # Plan Recommendation
            plan_rec = get_plan_recommendation(risk_pred, cost_pred)
            
            # Display Actuarial Summary
            st.markdown("#### Ringkasan")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Prediksi Biaya Tahunan", f"${cost_pred:,.2f}")
            with col2:
                st.metric("Skor Risiko", f"{risk_pred:.4f}")
            with col3:
                st.metric("Tambahan Risiko", f"{risk_load*100:.1f}%")
            with col4:
                st.metric("Premi Disarankan", f"${recommended_premium:,.2f}")
            
            st.markdown("---")
            
            # Detailed Actuarial Calculations
            st.markdown("#### Perhitungan Premi")
            actuarial_col1, actuarial_col2 = st.columns(2)
            
            with actuarial_col1:
                st.markdown("**Premi & Biaya**")
                st.write(f"• Premi Dasar: ${base_premium:,.2f}/tahun")
                st.write(f"• Tambahan Risiko: {risk_load*100:.1f}%")
                st.write(f"• Premi Disarankan: ${recommended_premium:,.2f}/tahun")
                st.write(f"• Premi Disarankan: ${recommended_premium/12:,.2f}/bulan")
                st.write(f"• Prediksi Biaya Tahunan: ${cost_pred:,.2f}")
            
            with actuarial_col2:
                st.markdown("**Risiko & Klaim**")
                st.write(f"• Probabilitas Klaim: {claims_probability:.1%}")
                st.write(f"• Skor Risiko: {risk_pred:.4f}")
                st.write(f"• Klasifikasi: {'Risiko Tinggi' if high_risk_pred == 1 else 'Risiko Rendah'}")
                st.write(f"• Keyakinan: {high_risk_proba:.1%}")
                st.write(f"• Level Risiko: {'Tinggi' if risk_pred >= 0.67 else ('Sedang' if risk_pred >= 0.33 else 'Rendah')}")
            
            st.markdown("---")
            
            # Underwriting Flags
            st.markdown("#### Catatan Risiko")
            if flags:
                flag_col1, flag_col2 = st.columns(2)
                high_flags = [f for f in flags if f[2] == "Tinggi"]
                medium_flags = [f for f in flags if f[2] == "Sedang"]
                
                with flag_col1:
                    if high_flags:
                        st.markdown("**Risiko Tinggi:**")
                        for flag_name, description, _ in high_flags:
                            st.markdown(f"• {flag_name}: {description}")
                
                with flag_col2:
                    if medium_flags:
                        st.markdown("**Risiko Sedang:**")
                        for flag_name, description, _ in medium_flags:
                            st.markdown(f"• {flag_name}: {description}")
            else:
                st.success("Tidak ada catatan risiko khusus.")
            
            st.markdown("---")
            
            # Plan Recommendation
            st.markdown("#### Rekomendasi Plan")
            plan_col1, plan_col2 = st.columns([1, 2])
            with plan_col1:
                st.markdown(f"**Plan:**")
                st.markdown(f"### {plan_rec['plan']}")
            with plan_col2:
                st.write(f"**Alasan:** {plan_rec['reasoning']}")
            
            st.markdown("---")
            
            # Risk Factor Breakdown
            st.markdown("#### Faktor Risiko Utama")
            
            # Calculate SHAP values or use feature importance
            risk_factors = calculate_shap_values(risk_model, X_input, metadata['feature_names'])
            
            if risk_factors is None:
                # Fallback to feature importance
                importance = get_feature_importance(risk_model, metadata['feature_names'], top_n=15)
                if importance:
                    risk_factors = {
                        'features': importance['features'],
                        'values': importance['importances']
                    }
            
            if risk_factors:
                # Create visualization
                fig = px.bar(
                    x=risk_factors['values'][:15],
                    y=risk_factors['features'][:15],
                    orientation='h',
                    title="15 Faktor Risiko Utama",
                    labels={'x': 'Kontribusi', 'y': 'Faktor Risiko'}
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
                
                # Store for PDF
                st.session_state.risk_factors = risk_factors
            else:
                st.warning("Tidak dapat menghitung faktor risiko.")
    
    with tab4:
        if 'actuarial_results' not in st.session_state:
            st.info("Lakukan prediksi di tab Input terlebih dahulu.")
        else:
            results = st.session_state.actuarial_results
            base_input = results['input_data'].copy()
            
            st.markdown("#### Ubah Variabel")
            
            col_sim1, col_sim2 = st.columns(2)
            
            with col_sim1:
                sim_bmi = st.slider("BMI", 10.0, 60.0, float(base_input['bmi'].iloc[0]), 0.1, key="sim_bmi")
                sim_systolic = st.slider("Tekanan Darah Sistolik", 70, 250, int(base_input['systolic_bp'].iloc[0]), key="sim_systolic")
                sim_diastolic = st.slider("Tekanan Darah Diastolik", 40, 150, int(base_input['diastolic_bp'].iloc[0]), key="sim_diastolic")
                sim_chronic = st.slider("Penyakit Kronis", 0, 5, int(base_input['chronic_count'].iloc[0]), key="sim_chronic")
            
            with col_sim2:
                # Reverse mapping for simulator - get Indonesian label from English value
                current_smoker_en = base_input['smoker'].iloc[0]
                current_smoker_ui = translate_to_indonesian(current_smoker_en, 'smoker')
                sim_smoker_ui = st.selectbox("Status Merokok", list(SMOKER_MAP.keys()), 
                                            index=list(SMOKER_MAP.keys()).index(current_smoker_ui) 
                                            if current_smoker_ui in SMOKER_MAP.keys() else 0,
                                            key="sim_smoker")
                sim_smoker = SMOKER_MAP[sim_smoker_ui]
                sim_visits = st.slider("Kunjungan Dokter", 0, 20, int(base_input['visits_last_year'].iloc[0]), key="sim_visits")
                sim_medication = st.slider("Jumlah Obat", 0, 10, int(base_input['medication_count'].iloc[0]), key="sim_medication")
                sim_age = st.slider("Usia", 0, 120, int(base_input['age'].iloc[0]), key="sim_age")
            
            # Update input data
            sim_input = base_input.copy()
            sim_input['bmi'] = [sim_bmi]
            sim_input['systolic_bp'] = [sim_systolic]
            sim_input['diastolic_bp'] = [sim_diastolic]
            sim_input['chronic_count'] = [sim_chronic]
            sim_input['smoker'] = [sim_smoker]
            sim_input['visits_last_year'] = [sim_visits]
            sim_input['medication_count'] = [sim_medication]
            sim_input['age'] = [sim_age]
            
            # Recalculate
            feature_cols = metadata['numerical_cols'] + metadata['categorical_cols']
            X_sim = preprocessor.transform(sim_input[feature_cols])
            
            sim_cost = cost_model.predict(X_sim)[0]
            sim_risk = risk_model.predict(X_sim)[0]
            sim_high_risk = high_risk_model.predict(X_sim)[0]
            sim_high_risk_proba = high_risk_model.predict_proba(X_sim)[0][1]
            sim_risk_load = calculate_risk_load(sim_risk)
            sim_premium = base_input['monthly_premium'].iloc[0] * 12 * (1 + sim_risk_load)
            
            # Compare with baseline
            st.markdown("---")
            st.markdown("#### Perbandingan")
            
            comp_col1, comp_col2, comp_col3 = st.columns(3)
            
            with comp_col1:
                st.markdown("**Awal**")
                st.metric("Biaya", f"${results['cost_pred']:,.2f}")
                st.metric("Skor Risiko", f"{results['risk_pred']:.4f}")
                st.metric("Premi", f"${base_input['monthly_premium'].iloc[0]*12*(1+calculate_risk_load(results['risk_pred'])):,.2f}")
            
            with comp_col2:
                st.markdown("**Setelah Ubah**")
                st.metric("Biaya", f"${sim_cost:,.2f}")
                st.metric("Skor Risiko", f"{sim_risk:.4f}")
                st.metric("Premi", f"${sim_premium:,.2f}")
            
            with comp_col3:
                st.markdown("**Perubahan**")
                cost_change = sim_cost - results['cost_pred']
                risk_change = sim_risk - results['risk_pred']
                premium_change = sim_premium - (base_input['monthly_premium'].iloc[0]*12*(1+calculate_risk_load(results['risk_pred'])))
                st.metric("Perubahan Biaya", f"${cost_change:,.2f}", f"{(cost_change/results['cost_pred']*100):.1f}%")
                st.metric("Perubahan Risiko", f"{risk_change:.4f}", f"{(risk_change/results['risk_pred']*100):.1f}%")
                st.metric("Perubahan Premi", f"${premium_change:,.2f}", f"{(premium_change/(base_input['monthly_premium'].iloc[0]*12*(1+calculate_risk_load(results['risk_pred'])))*100):.1f}%")
    
    with tab5:
        if 'actuarial_results' not in st.session_state:
            st.info("Lakukan prediksi di tab Input terlebih dahulu.")
        else:
            results = st.session_state.actuarial_results
            input_data = results['input_data']
            cost_pred = results['cost_pred']
            risk_pred = results['risk_pred']
            high_risk_pred = results['high_risk_pred']
            high_risk_proba = results['high_risk_proba']
            
            # Calculate all values
            risk_load = calculate_risk_load(risk_pred)
            base_premium = input_data['monthly_premium'].iloc[0] * 12
            recommended_premium = base_premium * (1 + risk_load)
            claims_probability = high_risk_proba
            
            flags = calculate_underwriting_flags(
                input_data['age'].iloc[0], input_data['bmi'].iloc[0],
                input_data['systolic_bp'].iloc[0], input_data['diastolic_bp'].iloc[0],
                input_data['smoker'].iloc[0], input_data['chronic_count'].iloc[0],
                input_data['hypertension'].iloc[0] == 1,
                input_data['diabetes'].iloc[0] == 1,
                input_data['cardiovascular_disease'].iloc[0] == 1
            )
            
            plan_rec = get_plan_recommendation(risk_pred, cost_pred)
            risk_factors = st.session_state.get('risk_factors', None)
            
            # Generate underwriting summary
            underwriting_summary = generate_underwriting_summary(
                input_data, cost_pred, risk_pred, high_risk_pred,
                risk_load, recommended_premium, flags, plan_rec
            )
            
            st.markdown("#### Ringkasan")
            st.text_area("Ringkasan", underwriting_summary, height=250)
            
            st.markdown("---")
            
            # Export buttons
            col_export1, col_export2 = st.columns(2)
            
            with col_export1:
                # CSV Export
                export_data = pd.DataFrame({
                    'Metrik': [
                        'Prediksi Biaya Tahunan', 'Skor Risiko', 'Klasifikasi Risiko',
                        'Tambahan Risiko (%)', 'Premi Dasar', 'Premi Disarankan',
                        'Probabilitas Klaim', 'Rekomendasi Plan'
                    ],
                    'Nilai': [
                        f"${cost_pred:,.2f}", f"{risk_pred:.4f}",
                        "Risiko Tinggi" if high_risk_pred == 1 else "Risiko Rendah",
                        f"{risk_load*100:.1f}%", f"${base_premium:,.2f}",
                        f"${recommended_premium:,.2f}", f"{claims_probability:.1%}",
                        plan_rec['plan']
                    ]
                })
                csv = export_data.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Unduh CSV",
                    data=csv,
                    file_name=f"laporan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            
            with col_export2:
                # PDF Export
                if st.button("📄 Buat PDF", use_container_width=True):
                    try:
                        pdf_buffer = create_pdf_report(
                            input_data, cost_pred, risk_pred, high_risk_pred, high_risk_proba,
                            risk_load, recommended_premium, claims_probability, flags, plan_rec,
                            risk_factors, underwriting_summary
                        )
                        st.download_button(
                            label="📥 Unduh PDF",
                            data=pdf_buffer,
                            file_name=f"laporan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                            mime="application/pdf"
                        )
                    except Exception as e:
                        st.error(f"Error: {str(e)}")

def main():
    """Aplikasi Streamlit utama."""
    st.markdown('<h1 class="main-header">Analisis Aktuarial</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Load data dan model
    try:
        df = load_data()
        preprocessor, metadata, cost_model, risk_model, high_risk_model, model_metrics = load_models()
    except FileNotFoundError as e:
        st.error(f"Error memuat file: {e}")
        st.stop()
    
    # Show actuarial analysis page
    show_actuarial_analysis_page(df, preprocessor, metadata, cost_model, risk_model, high_risk_model)

if __name__ == "__main__":
    main()