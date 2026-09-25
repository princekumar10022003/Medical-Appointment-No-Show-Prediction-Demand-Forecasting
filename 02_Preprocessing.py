
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib, os, warnings
from sklearn.preprocessing import LabelEncoder, StandardScaler
warnings.filterwarnings('ignore')

os.makedirs('outputs/preprocessing', exist_ok=True)
os.makedirs('models', exist_ok=True)

print("=" * 60)
print("PREPROCESSING & FEATURE ENGINEERING")
print("=" * 60)

df = pd.read_csv('data/Medical_appointment_data.csv')
df['appointment_date_continuous'] = pd.to_datetime(df['appointment_date_continuous'])
print(f"Raw shape: {df.shape}")

# ─── 1. Fix empty-string disability ─────────────────────────────────────────
df['disability'] = df['disability'].replace('', np.nan)

# ─── 2. Missing values ──────────────────────────────────────────────────────
print("\n[1/6] Handling Missing Values...")

# Flag columns with high missingness before imputing
df['age_missing']       = df['age'].isna().astype(int)
df['specialty_missing'] = df['specialty'].isna().astype(int)

# Numerical: median imputation
age_median = df['age'].median()
df['age'] = df['age'].fillna(age_median)
print(f"   age: filled {df['age_missing'].sum():,} with median ({age_median:.1f})")

# Weather: sort by date, forward-fill then backward-fill
df = df.sort_values('appointment_date_continuous').reset_index(drop=True)
weather_cols = ['average_temp_day', 'average_rain_day', 'max_temp_day', 'max_rain_day']
for col in weather_cols:
    df[col] = df[col].ffill().bfill()
print(f"   weather columns: forward/backward filled")

# Categorical: fill with 'Unknown'
for col in ['specialty', 'disability', 'place']:
    df[col] = df[col].fillna('Unknown')
    print(f"   {col}: filled NaN with 'Unknown'")

print(f"   Missing values remaining: {df.isnull().sum().sum()}")

# ─── 3. Temporal Feature Engineering ────────────────────────────────────────
print("\n[2/6] Temporal Feature Engineering...")
df['day_of_week']    = df['appointment_date_continuous'].dt.dayofweek
df['month']          = df['appointment_date_continuous'].dt.month
df['week_of_year']   = df['appointment_date_continuous'].dt.isocalendar().week.astype(int)
df['quarter']        = df['appointment_date_continuous'].dt.quarter
df['is_weekend']     = (df['day_of_week'] >= 5).astype(int)
df['is_month_start'] = df['appointment_date_continuous'].dt.is_month_start.astype(int)
df['is_month_end']   = df['appointment_date_continuous'].dt.is_month_end.astype(int)
df['day_of_year']    = df['appointment_date_continuous'].dt.dayofyear

day_map = {0:'Mon', 1:'Tue', 2:'Wed', 3:'Thu', 4:'Fri', 5:'Sat', 6:'Sun'}
df['day_name'] = df['day_of_week'].map(day_map)
print(f"   Added: day_of_week, month, week_of_year, quarter, is_weekend, day_of_year")

# ─── 4. Domain Feature Engineering ──────────────────────────────────────────
print("\n[3/6] Domain Feature Engineering...")

# Age groups
df['age_group'] = pd.cut(df['age'],
    bins=[0, 12, 18, 35, 60, 200],
    labels=[0, 1, 2, 3, 4]).astype(float)

# Combined risk features
df['has_chronic_condition']  = ((df['Hipertension'] == 1) | (df['Diabetes'] == 1)).astype(int)
df['has_substance_issue']    = df['Alcoholism'].copy()
df['has_disability_or_hdcp'] = ((df['Handcap'] == 1) | (df['disability'] != 'Unknown')).astype(int)
df['bad_weather']            = ((df['rainy_day_before'] == 1) | (df['storm_day_before'] == 1)).astype(int)
df['severe_weather']         = (df['storm_day_before'] == 1).astype(int)
df['high_temp']              = (df['average_temp_day'] > df['average_temp_day'].quantile(0.75)).astype(int)
df['heavy_rain']             = (df['average_rain_day'] > df['average_rain_day'].quantile(0.75)).astype(int)
df['total_conditions']       = (df['Hipertension'] + df['Diabetes'] + df['Alcoholism'] + df['Handcap'])
df['is_child']               = df['under_12_years_old'].copy()
df['is_senior']              = df['over_60_years_old'].copy()

# Appointment time features
df['morning_appt']   = (df['appointment_shift'] == 'morning').astype(int)
df['late_appt']      = (df['appointment_time'] >= 16).astype(int)
df['early_appt']     = (df['appointment_time'] <= 8).astype(int)

print(f"   Added: age_group, chronic_condition, bad_weather, total_conditions, time features")

# ─── 5. Encode Categoricals ──────────────────────────────────────────────────
print("\n[4/6] Encoding Categorical Variables...")

# Ordinal mappings (meaningful order)
gender_map  = {'F': 0, 'M': 1, 'I': 2}
shift_map   = {'morning': 0, 'afternoon': 1}
rain_map    = {'no_rain': 0, 'weak': 1, 'moderate': 2, 'heavy': 3}
heat_map    = {'heavy_cold': 0, 'cold': 1, 'mild': 2, 'warm': 3, 'heavy_warm': 4}

df['gender_enc']         = df['gender'].map(gender_map).fillna(2)
df['shift_enc']          = df['appointment_shift'].map(shift_map).fillna(0)
df['rain_intensity_enc'] = df['rain_intensity'].map(rain_map).fillna(0)
df['heat_intensity_enc'] = df['heat_intensity'].map(heat_map).fillna(2)

# Label encode high-cardinality categoricals
le_specialty = LabelEncoder()
le_place     = LabelEncoder()
le_disability= LabelEncoder()

df['specialty_enc']  = le_specialty.fit_transform(df['specialty'])
df['place_enc']      = le_place.fit_transform(df['place'])
df['disability_enc'] = le_disability.fit_transform(df['disability'])

# Target encoding
df['no_show_binary'] = (df['no_show'] == 'yes').astype(int)

# Save encoders
encoders = {
    'le_specialty':  le_specialty,
    'le_place':      le_place,
    'le_disability': le_disability,
    'gender_map':    gender_map,
    'shift_map':     shift_map,
    'rain_map':      rain_map,
    'heat_map':      heat_map,
    'age_median':    age_median,
    'specialty_classes': list(le_specialty.classes_),
    'place_classes':     list(le_place.classes_),
    'disability_classes':list(le_disability.classes_),
}
joblib.dump(encoders, 'models/encoders.pkl')
print(f"   Encoded: gender, shift, rain_intensity, heat_intensity, specialty, place, disability")
print(f"   Encoders saved to models/encoders.pkl")

# ─── 6. Build Classification Dataset ────────────────────────────────────────
print("\n[5/6] Building Classification Dataset...")

FEATURE_COLS = [
    # Demographics
    'age', 'gender_enc', 'age_group', 'is_child', 'is_senior',
    'patient_needs_companion', 'age_missing',
    # Appointment
    'appointment_time', 'shift_enc', 'morning_appt', 'late_appt', 'early_appt',
    # Location & specialty
    'specialty_enc', 'place_enc', 'disability_enc', 'specialty_missing',
    # Health conditions
    'Hipertension', 'Diabetes', 'Alcoholism', 'Handcap', 'Scholarship',
    'has_chronic_condition', 'has_substance_issue', 'total_conditions',
    'has_disability_or_hdcp', 'SMS_received',
    # Weather
    'average_temp_day', 'average_rain_day', 'max_temp_day', 'max_rain_day',
    'rainy_day_before', 'storm_day_before',
    'rain_intensity_enc', 'heat_intensity_enc',
    'bad_weather', 'severe_weather', 'high_temp', 'heavy_rain',
    # Temporal
    'day_of_week', 'month', 'week_of_year', 'quarter',
    'is_weekend', 'is_month_start', 'is_month_end', 'day_of_year',
]

df_clf = df[FEATURE_COLS + ['no_show_binary']].copy()
df_clf.to_csv('data/processed_classification.csv', index=False)
print(f"   Classification dataset: {df_clf.shape} -> data/processed_classification.csv")
print(f"   Target distribution: {df_clf['no_show_binary'].value_counts().to_dict()}")

# ─── 7. Build Forecasting Dataset ────────────────────────────────────────────
print("\n[6/6] Building Forecasting Dataset...")

daily = df.groupby('appointment_date_continuous').agg(
    total_appointments = ('no_show', 'count'),
    no_show_count      = ('no_show_binary', 'sum'),
    show_count         = ('no_show_binary', lambda x: (x == 0).sum()),
    avg_temp           = ('average_temp_day', 'mean'),
    avg_rain           = ('average_rain_day', 'mean'),
    pct_rainy          = ('rainy_day_before', 'mean'),
    pct_storm          = ('storm_day_before', 'mean'),
    pct_sms            = ('SMS_received', 'mean'),
).reset_index()

daily.columns = ['date', 'total_appointments', 'no_show_count', 'show_count',
                 'avg_temp', 'avg_rain', 'pct_rainy', 'pct_storm', 'pct_sms']

daily = daily.sort_values('date').reset_index(drop=True)

# Temporal features for forecasting
daily['day_of_week']    = daily['date'].dt.dayofweek
daily['month']          = daily['date'].dt.month
daily['week_of_year']   = daily['date'].dt.isocalendar().week.astype(int)
daily['quarter']        = daily['date'].dt.quarter
daily['is_weekend']     = (daily['day_of_week'] >= 5).astype(int)
daily['day_of_year']    = daily['date'].dt.dayofyear

# Sine/cosine encoding for cyclical features
daily['dow_sin'] = np.sin(2 * np.pi * daily['day_of_week'] / 7)
daily['dow_cos'] = np.cos(2 * np.pi * daily['day_of_week'] / 7)
daily['month_sin'] = np.sin(2 * np.pi * daily['month'] / 12)
daily['month_cos'] = np.cos(2 * np.pi * daily['month'] / 12)

# Lag features (CRITICAL: these must use only past data, no leakage)
daily['lag_1']  = daily['total_appointments'].shift(1)
daily['lag_7']  = daily['total_appointments'].shift(7)
daily['lag_14'] = daily['total_appointments'].shift(14)
daily['lag_30'] = daily['total_appointments'].shift(30)
daily['rolling_7_mean']  = daily['total_appointments'].shift(1).rolling(7).mean()
daily['rolling_14_mean'] = daily['total_appointments'].shift(1).rolling(14).mean()
daily['rolling_30_mean'] = daily['total_appointments'].shift(1).rolling(30).mean()
daily['rolling_7_std']   = daily['total_appointments'].shift(1).rolling(7).std()

# Log transform of target (handles high variance)
daily['log_appointments'] = np.log1p(daily['total_appointments'])

# Drop rows with NaN lags (first 30 days)
daily_model = daily.dropna().reset_index(drop=True)

daily.to_csv('data/processed_forecasting.csv', index=False)
daily_model.to_csv('data/processed_forecasting_model.csv', index=False)
print(f"   Full forecasting dataset: {daily.shape} -> data/processed_forecasting.csv")
print(f"   Model-ready (no NaN): {daily_model.shape} -> data/processed_forecasting_model.csv")
print(f"   Daily stats: mean={daily['total_appointments'].mean():.0f}, "
      f"max={daily['total_appointments'].max()}, min={daily['total_appointments'].min()}")

# ─── Save feature list ───────────────────────────────────────────────────────
joblib.dump({'clf_features': FEATURE_COLS}, 'models/feature_cols.pkl')

# ─── Preprocessing summary plot ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Preprocessing Summary', fontsize=14, fontweight='bold')

axes[0].bar(['Show', 'No-Show'], df_clf['no_show_binary'].value_counts().sort_index().values,
            color=['#34A853', '#EA4335'], edgecolor='white')
axes[0].set_title('Target Distribution (Classification)')
axes[0].set_ylabel('Count')
for i, v in enumerate(df_clf['no_show_binary'].value_counts().sort_index().values):
    axes[0].text(i, v + 300, f'{v:,}\n({v/len(df_clf)*100:.1f}%)',
                 ha='center', va='bottom', fontsize=10)

axes[1].plot(daily['date'], daily['total_appointments'], color='#1A73E8', linewidth=0.8, alpha=0.7)
ma7 = daily.set_index('date')['total_appointments'].rolling(7).mean()
axes[1].plot(ma7.index, ma7.values, color='#EA4335', linewidth=2, label='7-day MA')
axes[1].set_title('Daily Appointments (Forecasting Target)')
axes[1].set_ylabel('Appointments per Day')
axes[1].legend()

axes[2].bar(range(len(FEATURE_COLS)), [1]*len(FEATURE_COLS),
            color=['#1A73E8']*7 + ['#34A853']*7 + ['#EA4335']*6 + ['#FBBC04']*6 + ['#7B61FF']*8 + ['#5F6368']*7)
axes[2].set_title(f'Feature Set: {len(FEATURE_COLS)} Features\nColor = feature group')
axes[2].set_xlabel('Feature Index')
axes[2].set_yticks([])
axes[2].text(0.5, 0.5, f'{len(FEATURE_COLS)} Features\nEngineered', transform=axes[2].transAxes,
             ha='center', va='center', fontsize=14, fontweight='bold', color='white',
             bbox=dict(boxstyle='round', facecolor='#1A73E8', alpha=0.8))

plt.tight_layout()
plt.savefig('outputs/preprocessing/preprocessing_summary.png', dpi=150, bbox_inches='tight')
plt.close()

print("\n" + "=" * 60)
print("PREPROCESSING COMPLETE")
print("=" * 60)
print(f"Classification features : {len(FEATURE_COLS)}")
print(f"Forecasting features    : {len(daily_model.columns)}")
print(f"Files saved to data/ and models/")
