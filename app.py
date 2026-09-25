
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import joblib
import os
import warnings
from datetime import date, timedelta
warnings.filterwarnings('ignore')

# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CER Medical Analytics",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2rem; font-weight: 700;
        background: linear-gradient(135deg, #1A73E8, #34A853);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header { color: #5F6368; font-size: 1rem; margin-bottom: 1.5rem; }
    .metric-card {
        background: white; color: #202124; border-radius: 12px; padding: 1rem 1.2rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08); border-left: 4px solid #1A73E8;
        margin-bottom: 0.8rem;
    }
    .metric-card * { color: #202124; }
    .risk-high   { border-left-color: #EA4335 !important; background: #FFF5F5 !important; }
    .risk-medium { border-left-color: #FBBC04 !important; background: #FFFBF0 !important; }
    .risk-low    { border-left-color: #34A853 !important; background: #F0FFF4 !important; }
    .risk-label  { font-size: 1.8rem; font-weight: 700; }
    .risk-high   .risk-label { color: #EA4335; }
    .risk-medium .risk-label { color: #E37400; }
    .risk-low    .risk-label { color: #137333; }
    .section-title { font-size: 1.1rem; font-weight: 600; color: #3C4043; margin: 1rem 0 0.5rem; }
    .insight-box {
        background: #F8F9FA; color: #202124; border-radius: 8px; padding: 0.8rem 1rem;
        border: 1px solid #DADCE0; margin: 0.3rem 0; font-size: 0.9rem;
    }
    .insight-box * { color: #202124; }
    .stTabs [data-baseweb="tab-list"] { gap: 16px; }
    .stTabs [data-baseweb="tab"] { font-size: 1rem; font-weight: 600; }
    div[data-testid="metric-container"] {
        background: white; border-radius: 10px; padding: 0.8rem;
        box-shadow: 0 1px 4px rgba(0,0,0,0.08);
    }
</style>
""", unsafe_allow_html=True)

# ─── Load Models ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    base = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base, 'models')
    clf_bundle  = joblib.load(os.path.join(models_dir, 'no_show_classifier.pkl'))
    fc_bundle   = joblib.load(os.path.join(models_dir, 'demand_forecaster.pkl'))
    encoders    = joblib.load(os.path.join(models_dir, 'encoders.pkl'))
    return clf_bundle, fc_bundle, encoders

@st.cache_data
def load_raw_data():
    base = os.path.dirname(os.path.abspath(__file__))
    df = pd.read_csv(os.path.join(base, 'data', 'Medical_appointment_data.csv'))
    df['appointment_date_continuous'] = pd.to_datetime(df['appointment_date_continuous'])
    return df

try:
    clf_bundle, fc_bundle, encoders = load_models()
    df_raw = load_raw_data()
    models_loaded = True
except Exception as e:
    models_loaded = False
    st.error(f"⚠️ Models not found. Please run the training notebooks first.\nError: {e}")
    st.info("Run in order:\n1. python notebooks/02_Preprocessing.py\n2. python notebooks/03_Classification.py\n3. python notebooks/04_Forecasting.py")
    st.stop()

def preprocess_input(inputs: dict) -> np.ndarray:
    """Convert UI inputs to model feature vector."""
    encs = encoders

    # Encode categoricals
    def safe_le(le, val):
        classes = le.classes_
        return int(np.where(classes == val)[0][0]) if val in classes else len(classes) - 1

    specialty_enc  = safe_le(encs['le_specialty'], inputs['specialty'])
    place_enc      = safe_le(encs['le_place'], inputs['place'])
    disability_enc = safe_le(encs['le_disability'], inputs['disability'])
    gender_enc     = encs['gender_map'].get(inputs['gender'], 2)
    shift_enc      = encs['shift_map'].get(inputs['shift'], 0)
    rain_enc       = encs['rain_map'].get(inputs['rain_intensity'], 0)
    heat_enc       = encs['heat_map'].get(inputs['heat_intensity'], 2)

    age = inputs['age'] if inputs['age'] > 0 else encs['age_median']
    age_group = 0 if age <= 12 else 1 if age <= 18 else 2 if age <= 35 else 3 if age <= 60 else 4

    feat = {
        'age': age,
        'gender_enc': gender_enc,
        'age_group': age_group,
        'is_child': int(age <= 12),
        'is_senior': int(age > 60),
        'patient_needs_companion': int(inputs['needs_companion']),
        'age_missing': 0,
        'appointment_time': inputs['appointment_time'],
        'shift_enc': shift_enc,
        'morning_appt': int(inputs['shift'] == 'morning'),
        'late_appt': int(inputs['appointment_time'] >= 16),
        'early_appt': int(inputs['appointment_time'] <= 8),
        'specialty_enc': specialty_enc,
        'place_enc': place_enc,
        'disability_enc': disability_enc,
        'specialty_missing': int(inputs['specialty'] == 'Unknown'),
        'Hipertension': int(inputs['hipertension']),
        'Diabetes': int(inputs['diabetes']),
        'Alcoholism': int(inputs['alcoholism']),
        'Handcap': int(inputs['handcap']),
        'Scholarship': int(inputs['scholarship']),
        'has_chronic_condition': int(inputs['hipertension'] or inputs['diabetes']),
        'has_substance_issue': int(inputs['alcoholism']),
        'total_conditions': sum([inputs['hipertension'], inputs['diabetes'],
                                  inputs['alcoholism'], inputs['handcap']]),
        'has_disability_or_hdcp': int(inputs['handcap'] or inputs['disability'] != 'Unknown'),
        'SMS_received': int(inputs['sms_received']),
        'average_temp_day': inputs['avg_temp'],
        'average_rain_day': inputs['avg_rain'],
        'max_temp_day': inputs['avg_temp'] + 5.0,
        'max_rain_day': inputs['avg_rain'] * 2.0,
        'rainy_day_before': int(inputs['rainy_before']),
        'storm_day_before': int(inputs['storm_before']),
        'rain_intensity_enc': rain_enc,
        'heat_intensity_enc': heat_enc,
        'bad_weather': int(inputs['rainy_before'] or inputs['storm_before']),
        'severe_weather': int(inputs['storm_before']),
        'high_temp': int(inputs['avg_temp'] > 28),
        'heavy_rain': int(inputs['avg_rain'] > 20),
        'day_of_week': inputs['day_of_week'],
        'month': inputs['month'],
        'week_of_year': inputs['week_of_year'],
        'quarter': (inputs['month'] - 1) // 3 + 1,
        'is_weekend': int(inputs['day_of_week'] >= 5),
        'is_month_start': int(inputs['day_of_month'] <= 3),
        'is_month_end': int(inputs['day_of_month'] >= 28),
        'day_of_year': inputs['day_of_year'],
    }

    feature_cols = clf_bundle['feature_cols']
    return np.array([[feat[col] for col in feature_cols]])


def make_forecast(start_date: date, n_days: int, specialty_filter: str = 'All') -> pd.DataFrame:
    """Generate n_days forecast starting from start_date."""
    fc_model  = fc_bundle['model']
    features  = fc_bundle['forecast_features']
    lv        = fc_bundle['last_values']

    history = list(fc_bundle['daily_data']['total_appointments'].tail(60).values)

    results = []
    for i in range(n_days):
        d = start_date + timedelta(days=i)
        dow  = d.weekday()
        mon  = d.month
        woy  = d.isocalendar()[1]
        doy  = d.timetuple().tm_yday
        hist = history + [r['pred'] for r in results]

        def safe_get(lst, idx, default=200):
            try:
                val = lst[idx]
                return val if val > 0 else default
            except:
                return default

        # Build all possible lag/rolling features
        row = {
            'day_of_week':     dow,
            'month':           mon,
            'week_of_year':    int(woy),
            'quarter':         (mon - 1) // 3 + 1,
            'is_weekend':      int(dow >= 5),
            'day_of_year':     doy,
            'dow_sin':         np.sin(2 * np.pi * dow / 7),
            'dow_cos':         np.cos(2 * np.pi * dow / 7),
            'month_sin':       np.sin(2 * np.pi * mon / 12),
            'month_cos':       np.cos(2 * np.pi * mon / 12),
            'lag_1':           safe_get(hist, -1),
            'lag_2':           safe_get(hist, -2),
            'lag_3':           safe_get(hist, -3),
            'lag_7':           safe_get(hist, -7),
            'lag_14':          safe_get(hist, -14),
            'lag_21':          safe_get(hist, -21),
            'lag_30':          safe_get(hist, -30),
            'rolling_7_mean':  np.mean([safe_get(hist, -(j+1)) for j in range(7)]),
            'rolling_14_mean': np.mean([safe_get(hist, -(j+1)) for j in range(14)]),
            'rolling_30_mean': np.mean([safe_get(hist, -(j+1)) for j in range(30)]),
            'rolling_7_std':   np.std([safe_get(hist, -(j+1)) for j in range(7)]),
            'rolling_7_max':   max([safe_get(hist, -(j+1)) for j in range(7)]),
            'exp_smooth':      safe_get(hist, -1) * 0.25 + safe_get(hist, -2) * 0.19 + safe_get(hist, -3) * 0.15,
            'expand_mean':     np.mean([safe_get(hist, -(j+1)) for j in range(min(len(hist), 30))]),
            'avg_temp':        25.0,
            'avg_rain':        5.0,
            'pct_rainy':       0.3,
            'pct_storm':       0.1,
        }

        # Only use features the model was trained on
        X_row = np.array([[row.get(f, 0) for f in features]])
        pred_log = fc_model.predict(X_row)[0]

        # Reverse sqrt transform
        pred = max(int(pred_log ** 2), 0)

        if specialty_filter != 'All' and specialty_filter in df_raw['specialty'].dropna().unique():
            spec_share = (df_raw['specialty'] == specialty_filter).sum() / len(df_raw.dropna(subset=['specialty']))
            pred = max(int(pred * spec_share), 0)

        results.append({
            'date':  d,
            'pred':  pred,
            'lower': max(int(pred * 0.8), 0),
            'upper': int(pred * 1.2)
        })

    return pd.DataFrame(results)
# ─── SIDEBAR ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏥 CER Analytics")
    st.markdown("*University of Vale do Itajaí*")
    st.divider()

    st.markdown("**Model Performance**")
    st.metric("Classifier", clf_bundle['model_name'].split()[0])
    f1_val = clf_bundle.get('optimal_f1', clf_bundle['test_f1'])
    auc_val = clf_bundle['test_auc']
    col1, col2 = st.columns(2)
    with col1:
        st.metric("F1-Score", f"{f1_val:.3f}", delta="✓" if f1_val >= 0.70 else "✗")
    with col2:
        st.metric("ROC-AUC", f"{auc_val:.3f}", delta="✓" if auc_val >= 0.75 else "✗")

    st.divider()
    st.metric("Forecaster", fc_bundle['model_name'].replace(' ', '\n'))
    col3, col4 = st.columns(2)
    with col3:
        st.metric("MAPE", f"{fc_bundle['test_mape']:.1f}%", delta="✓" if fc_bundle['test_mape'] < 20 else "✗")
    with col4:
        st.metric("R²", f"{fc_bundle['test_r2']:.3f}", delta="✓" if fc_bundle['test_r2'] >= 0.65 else "✗")

    st.divider()
    st.markdown("**Dataset Info**")
    st.markdown(f"- 📅 Jan 2020 – May 2021")
    st.markdown(f"- 📋 109,593 appointments")
    st.markdown(f"- ❌ 31.8% no-show rate")
    st.markdown(f"- 🏙️ 13 cities served")

# ─── HEADER ──────────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">🏥 CER Medical Analytics Platform</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">No-Show Risk Prediction & Appointment Demand Forecasting</div>', unsafe_allow_html=True)

# ─── TABS ────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔴 No-Show Risk Predictor", "📈 Demand Forecaster", "📊 Data Insights"])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1: NO-SHOW PREDICTOR
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### Patient No-Show Risk Assessment")
    st.markdown("Fill in the appointment details below to get the no-show risk score.")

    with st.form("noshow_form"):
        col_a, col_b, col_c = st.columns(3)

        with col_a:
            st.markdown('<div class="section-title">👤 Patient Details</div>', unsafe_allow_html=True)
            age   = st.slider("Age", min_value=0, max_value=110, value=30, step=1)
            gender = st.selectbox("Gender", ["M", "F", "I"], format_func=lambda x: {"M":"Male","F":"Female","I":"Other"}[x])
            disability = st.selectbox("Disability Type",
                ['Unknown', 'intellectual', 'motor'],
                format_func=lambda x: x.title())
            needs_companion = st.checkbox("Needs Companion")

        with col_b:
            st.markdown('<div class="section-title">📅 Appointment Details</div>', unsafe_allow_html=True)
            appt_date = st.date_input("Appointment Date", value=date.today())
            specialty = st.selectbox("Specialty",
                ['Unknown', 'psychotherapy', 'speech therapy', 'physiotherapy',
                 'occupational therapy', 'pedagogo', 'enf', 'assist', 'sem especialidade'],
                format_func=lambda x: x.title())
            place = st.selectbox("City",
                ['Unknown'] + sorted([c for c in encoders['place_classes'] if c != 'Unknown']))
            appt_shift = st.radio("Shift", ["morning", "afternoon"], horizontal=True)
            appt_time  = st.slider("Appointment Hour", min_value=6, max_value=20, value=10)

        with col_c:
            st.markdown('<div class="section-title">🏥 Health & Environment</div>', unsafe_allow_html=True)
            hipertension = st.checkbox("Hypertension")
            diabetes     = st.checkbox("Diabetes")
            alcoholism   = st.checkbox("Alcoholism")
            handcap      = st.checkbox("Physical Handicap")
            scholarship  = st.checkbox("Government Scholarship")
            sms_received = st.checkbox("SMS Reminder Sent", value=True)
            st.divider()
            avg_temp = st.slider("Avg Temperature (°C)", -5.0, 40.0, 22.0, 0.5)
            avg_rain = st.slider("Avg Rain (mm)", 0.0, 100.0, 5.0, 0.5)
            col_w1, col_w2 = st.columns(2)
            with col_w1:
                rainy_before = st.checkbox("Rainy Day Before")
            with col_w2:
                storm_before = st.checkbox("Storm Day Before")
            rain_intensity = st.selectbox("Rain Intensity", ["no_rain","weak","moderate","heavy"])
            heat_intensity = st.selectbox("Heat Intensity", ["heavy_cold","cold","mild","warm","heavy_warm"])

        submitted = st.form_submit_button("🔍 Predict No-Show Risk", use_container_width=True, type="primary")

    if submitted:
        with st.spinner("Calculating risk score..."):
            try:
                inputs = {
                    'age': age, 'gender': gender, 'disability': disability,
                    'needs_companion': needs_companion, 'specialty': specialty,
                    'place': place, 'shift': appt_shift, 'appointment_time': appt_time,
                    'hipertension': hipertension, 'diabetes': diabetes,
                    'alcoholism': alcoholism, 'handcap': handcap,
                    'scholarship': scholarship, 'sms_received': sms_received,
                    'avg_temp': avg_temp, 'avg_rain': avg_rain,
                    'rainy_before': rainy_before, 'storm_before': storm_before,
                    'rain_intensity': rain_intensity, 'heat_intensity': heat_intensity,
                    'day_of_week': appt_date.weekday(),
                    'month': appt_date.month,
                    'week_of_year': appt_date.isocalendar()[1],
                    'day_of_year': appt_date.timetuple().tm_yday,
                    'day_of_month': appt_date.day,
                }

                X_input = preprocess_input(inputs)
                model    = clf_bundle['model']
                threshold = clf_bundle['threshold']

                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(X_input)[0, 1]
                else:
                    proba = 0.5
                risk_pct = proba * 100
                prediction = int(proba >= threshold)

                if risk_pct >= 55:
                    risk_level = "HIGH RISK"; risk_class = "risk-high"
                    rec = ("⚠️ <b>Action Required:</b> Send SMS reminder 48h before, consider calling directly. "
                           "Offer reschedule options and flag for overbooking compensation.")
                    rec_icon = "🔴"
                elif risk_pct >= 35:
                    risk_level = "MEDIUM RISK"; risk_class = "risk-medium"
                    rec = ("📲 <b>Recommended:</b> Send automated SMS reminder 24h before appointment. "
                           "Monitor for confirmation response.")
                    rec_icon = "🟡"
                else:
                    risk_level = "LOW RISK"; risk_class = "risk-low"
                    rec = ("✅ <b>Standard:</b> No special action needed. "
                           "Include in regular appointment reminder schedule.")
                    rec_icon = "🟢"

                st.divider()
                res_col1, res_col2 = st.columns([1, 1])

                with res_col1:
                    st.markdown(f"""
                    <div class="metric-card {risk_class}">
                        <div style="font-size:0.85rem;color:#5F6368;font-weight:600;">NO-SHOW RISK SCORE</div>
                        <div class="risk-label">{rec_icon} {risk_pct:.1f}%</div>
                        <div style="font-size:1rem;font-weight:600;margin-top:4px;">{risk_level}</div>
                        <div style="font-size:0.8rem;color:#5F6368;margin-top:4px;">
                            Model threshold: {threshold:.2f} | Prediction: {"No-Show" if prediction else "Show"}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Risk gauge bar
                    fig_g, ax_g = plt.subplots(figsize=(4, 0.8))
                    ax_g.barh([0], [100], color='#E8F5E9', height=0.5)
                    ax_g.barh([0], [35], color='#A8D5A2', height=0.5)
                    ax_g.barh([0], [20], color='#FBBC04', height=0.5, left=35)
                    ax_g.barh([0], [45], color='#EA4335', height=0.5, left=55)
                    ax_g.barh([0], [risk_pct], color='#1A1A1A', height=0.15, alpha=0.9)
                    ax_g.scatter([risk_pct], [0], color='#1A1A1A', s=80, zorder=5)
                    ax_g.set_xlim(0, 100); ax_g.set_yticks([]); ax_g.set_xticks([0,25,50,75,100])
                    ax_g.set_xticklabels(['0%','25%','50%','75%','100%'], fontsize=8)
                    ax_g.set_title('Risk Gauge', fontsize=9, pad=2)
                    ax_g.set_facecolor('white'); fig_g.patch.set_facecolor('white')
                    plt.tight_layout(pad=0.3)
                    st.pyplot(fig_g, use_container_width=True)
                    plt.close()

                with res_col2:
                    st.markdown("**📋 Recommendation**")
                    st.markdown(f'<div class="insight-box">{rec}</div>', unsafe_allow_html=True)

                    st.markdown("**🔍 Risk Factors Identified**")
                    risk_factors = []
                    if age <= 12:             risk_factors.append("👶 Pediatric patient (higher risk)")
                    if appt_shift == 'morning': risk_factors.append("🌅 Morning shift (higher no-show rate)")
                    if storm_before:          risk_factors.append("⛈️ Storm day before appointment")
                    if specialty in ['physiotherapy','psychotherapy','sem especialidade']:
                        risk_factors.append(f"🏥 {specialty.title()} has above-average no-show rate")
                    if not sms_received:      risk_factors.append("📵 No SMS reminder sent")
                    if alcoholism:            risk_factors.append("⚠️ Alcoholism flag present")
                    if not risk_factors:      risk_factors.append("✅ No major risk factors identified")

                    for rf in risk_factors:
                        st.markdown(f'<div class="insight-box">{rf}</div>', unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Prediction error: {e}")
                import traceback
                st.code(traceback.format_exc())

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2: DEMAND FORECASTER
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### Appointment Demand Forecaster")
    st.markdown("Predict daily appointment volumes for operational planning.")

    fc_col1, fc_col2, fc_col3 = st.columns([1, 1, 1])
    with fc_col1:
        fc_start = st.date_input("Forecast Start Date", value=date.today())
    with fc_col2:
        fc_days  = st.slider("Forecast Horizon (days)", min_value=7, max_value=90, value=30)
    with fc_col3:
        specialties_available = ['All'] + sorted(df_raw['specialty'].dropna().unique().tolist())
        fc_specialty = st.selectbox("Filter by Specialty", specialties_available)

    if st.button("📈 Generate Forecast", use_container_width=True, type="primary"):
        with st.spinner("Generating forecast..."):
            try:
                fc_df = make_forecast(fc_start, fc_days, fc_specialty)

                # Summary metrics
                mc1, mc2, mc3, mc4 = st.columns(4)
                with mc1: st.metric("Total Predicted", f"{fc_df['pred'].sum():,}")
                with mc2: st.metric("Daily Average", f"{fc_df['pred'].mean():.0f}")
                with mc3: st.metric("Peak Day", f"{fc_df['pred'].max():,}")
                with mc4: st.metric("Lowest Day", f"{fc_df['pred'].min():,}")

                # Forecast chart
                fig_fc, ax_fc = plt.subplots(figsize=(14, 5))
                ax_fc.fill_between(fc_df['date'], fc_df['lower'], fc_df['upper'],
                                   alpha=0.2, color='#1A73E8', label='80% Confidence Interval')
                ax_fc.plot(fc_df['date'], fc_df['pred'], color='#1A73E8', linewidth=2.5,
                           label='Forecast', marker='o', markersize=3)
                ax_fc.axhline(fc_df['pred'].mean(), color='#EA4335', linewidth=1.5,
                              linestyle='--', label=f'Average: {fc_df["pred"].mean():.0f}')
                # Weekend shading
                for _, row in fc_df.iterrows():
                    if row['date'].weekday() >= 5:
                        ax_fc.axvspan(row['date'] - timedelta(hours=12),
                                      row['date'] + timedelta(hours=12),
                                      alpha=0.08, color='gray')
                ax_fc.set_xlabel('Date')
                ax_fc.set_ylabel('Predicted Appointments')
                spec_label = fc_specialty if fc_specialty != 'All' else 'All Specialties'
                ax_fc.set_title(f'Appointment Demand Forecast — {spec_label}\n'
                                f'({fc_start} to {fc_start + timedelta(days=fc_days-1)}, '
                                f'gray shading = weekends)',
                                fontsize=12)
                ax_fc.legend(loc='upper right')
                ax_fc.grid(True, alpha=0.4)
                fig_fc.patch.set_facecolor('white')
                ax_fc.set_facecolor('#F8F9FA')
                plt.tight_layout()
                st.pyplot(fig_fc, use_container_width=True)
                plt.close()

                # Weekly summary table
                st.markdown("**Weekly Breakdown**")
                fc_df['week'] = fc_df['date'].apply(lambda d: d.isocalendar()[1])
                fc_df['day_name'] = fc_df['date'].apply(lambda d: d.strftime('%A'))
                weekly = fc_df.groupby('week').agg(
                    Week_Start=('date', 'first'),
                    Total_Predicted=('pred', 'sum'),
                    Daily_Avg=('pred', 'mean'),
                    Peak=('pred', 'max')
                ).reset_index(drop=True)
                weekly['Daily_Avg'] = weekly['Daily_Avg'].round(0).astype(int)
                weekly['Week_Start'] = pd.to_datetime(weekly['Week_Start']).dt.strftime('%d %b %Y')
                st.dataframe(weekly, use_container_width=True, hide_index=True)

                # Download
                csv_dl = fc_df[['date', 'pred', 'lower', 'upper']].copy()
                csv_dl.columns = ['Date', 'Predicted', 'Lower_80', 'Upper_80']
                st.download_button("⬇️ Download Forecast CSV",
                                   csv_dl.to_csv(index=False).encode(),
                                   f"forecast_{fc_start}_{fc_days}d.csv",
                                   "text/csv")

            except Exception as e:
                st.error(f"Forecasting error: {e}")
                import traceback; st.code(traceback.format_exc())

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3: DATA INSIGHTS
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### Data Insights & Patterns")

    ins_col1, ins_col2 = st.columns(2)

    with ins_col1:
        # No-show rate by specialty
        st.markdown("**No-Show Rate by Specialty**")
        spec_ns = df_raw.dropna(subset=['specialty']).groupby('specialty')['no_show'].apply(
            lambda x: (x == 'yes').mean() * 100).sort_values(ascending=True)
        fig_s, ax_s = plt.subplots(figsize=(7, 4))
        colors_s = ['#34A853' if v < 31.8 else '#EA4335' for v in spec_ns.values]
        ax_s.barh(spec_ns.index, spec_ns.values, color=colors_s, edgecolor='white')
        ax_s.axvline(31.8, color='#5F6368', linestyle='--', linewidth=1.5, label='Avg (31.8%)')
        ax_s.set_xlabel('No-Show Rate (%)')
        ax_s.set_title('No-Show Rate by Specialty', fontsize=12)
        ax_s.legend(fontsize=9)
        for i, val in enumerate(spec_ns.values):
            ax_s.text(val + 0.3, i, f'{val:.1f}%', va='center', fontsize=8)
        ax_s.set_facecolor('#F8F9FA'); fig_s.patch.set_facecolor('white')
        plt.tight_layout()
        st.pyplot(fig_s, use_container_width=True)
        plt.close()

        # By day of week
        st.markdown("**No-Show Rate by Day of Week**")
        df_raw['dow'] = df_raw['appointment_date_continuous'].dt.dayofweek
        dow_ns = df_raw.groupby('dow')['no_show'].apply(lambda x: (x=='yes').mean()*100)
        day_names = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
        fig_d, ax_d = plt.subplots(figsize=(7, 3.5))
        colors_d = ['#EA4335' if v > 31.8 else '#34A853' for v in dow_ns.values]
        ax_d.bar([day_names[i] for i in dow_ns.index], dow_ns.values, color=colors_d, edgecolor='white')
        ax_d.axhline(31.8, color='#5F6368', linestyle='--', linewidth=1.5)
        ax_d.set_ylabel('No-Show Rate (%)')
        ax_d.set_title('No-Show Rate by Day of Week', fontsize=12)
        ax_d.set_facecolor('#F8F9FA'); fig_d.patch.set_facecolor('white')
        plt.tight_layout()
        st.pyplot(fig_d, use_container_width=True)
        plt.close()

    with ins_col2:
        # Historical daily volume
        st.markdown("**Historical Daily Appointment Volume**")
        daily_hist = df_raw.groupby('appointment_date_continuous').size()
        fig_h, ax_h = plt.subplots(figsize=(7, 4))
        ax_h.plot(daily_hist.index, daily_hist.values, color='#1A73E8', linewidth=0.7, alpha=0.6)
        ma = daily_hist.rolling(14).mean()
        ax_h.plot(ma.index, ma.values, color='#EA4335', linewidth=2, label='14-day MA')
        ax_h.fill_between(daily_hist.index, daily_hist.values, alpha=0.15, color='#1A73E8')
        ax_h.set_xlabel('Date')
        ax_h.set_ylabel('Appointments per Day')
        ax_h.set_title('Historical Demand', fontsize=12)
        ax_h.legend()
        ax_h.set_facecolor('#F8F9FA'); fig_h.patch.set_facecolor('white')
        plt.tight_layout()
        st.pyplot(fig_h, use_container_width=True)
        plt.close()

        # Feature importance
        st.markdown("**Top Predictors of No-Show**")
        model = clf_bundle['model']
        feat_cols = clf_bundle['feature_cols']
        if hasattr(model, 'feature_importances_'):
            imps = model.feature_importances_
        elif hasattr(model, 'named_steps'):
            imps = np.abs(model.named_steps['clf'].coef_[0])
        else:
            imps = model.feature_importances_
        fi = pd.DataFrame({'Feature': feat_cols, 'Importance': imps})
        fi = fi.sort_values('Importance', ascending=False).head(12)
        fig_fi, ax_fi = plt.subplots(figsize=(7, 4))
        colors_fi2 = ['#EA4335' if i < 3 else '#FBBC04' if i < 6 else '#1A73E8'
                      for i in range(len(fi))]
        ax_fi.barh(fi['Feature'][::-1], fi['Importance'][::-1],
                   color=colors_fi2[::-1], edgecolor='white')
        ax_fi.set_title(f'Top 12 Features\n({clf_bundle["model_name"]})', fontsize=12)
        ax_fi.set_xlabel('Importance')
        ax_fi.set_facecolor('#F8F9FA'); fig_fi.patch.set_facecolor('white')
        plt.tight_layout()
        st.pyplot(fig_fi, use_container_width=True)
        plt.close()

    # Key stats
    st.divider()
    st.markdown("**Key Statistics**")
    ks1, ks2, ks3, ks4, ks5 = st.columns(5)
    with ks1: st.metric("Total Appointments", "109,593")
    with ks2: st.metric("No-Show Rate", "31.8%")
    with ks3: st.metric("Specialties", "8")
    with ks4: st.metric("Cities Served", "13")
    with ks5: st.metric("Date Range", "Jan20–May21")
