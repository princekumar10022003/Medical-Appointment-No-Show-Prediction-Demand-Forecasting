

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings, os
warnings.filterwarnings('ignore')

os.makedirs('outputs/eda', exist_ok=True)

BLUE   = '#1A73E8'
GREEN  = '#34A853'
RED    = '#EA4335'
ORANGE = '#FBBC04'
PURPLE = '#7B61FF'
GRAY   = '#5F6368'
PALETTE = [BLUE, RED, GREEN, ORANGE, PURPLE, GRAY, '#00BCD4', '#FF5722']

plt.rcParams.update({
    'figure.facecolor': 'white', 'axes.facecolor': '#F8F9FA',
    'axes.edgecolor': '#DADCE0', 'axes.labelcolor': '#3C4043',
    'xtick.color': '#5F6368', 'ytick.color': '#5F6368',
    'grid.color': '#DADCE0', 'grid.linewidth': 0.5,
    'font.family': 'sans-serif', 'font.size': 11,
    'axes.titlesize': 13, 'axes.titleweight': 'bold'
})

print("=" * 60)
print("LOADING DATASET")
print("=" * 60)
df = pd.read_csv('data/Medical_appointment_data.csv')
df['appointment_date_continuous'] = pd.to_datetime(df['appointment_date_continuous'])
print(f"Shape: {df.shape}")
print(f"Date range: {df['appointment_date_continuous'].min()} to {df['appointment_date_continuous'].max()}")
print(f"Unique dates: {df['appointment_date_continuous'].nunique()}")

# ─── FIG 1: Dataset Overview ────────────────────────────────────────────────
print("\n[1/8] Dataset Overview...")
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Dataset Overview', fontsize=15, fontweight='bold', y=1.02)

# Target distribution
counts = df['no_show'].value_counts()
bars = axes[0].bar(['Show (No)', 'No-Show (Yes)'], counts.values,
                    color=[GREEN, RED], edgecolor='white', linewidth=1.5)
axes[0].set_title('Target Variable Distribution')
axes[0].set_ylabel('Count')
for bar, val in zip(bars, counts.values):
    pct = val / len(df) * 100
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500,
                 f'{val:,}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=10, fontweight='bold')
axes[0].set_ylim(0, counts.max() * 1.15)

# Missing values
missing = df.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=True)
missing_pct = (missing / len(df) * 100).round(1)
colors_m = [RED if p > 15 else ORANGE if p > 10 else BLUE for p in missing_pct.values]
axes[1].barh(missing.index, missing_pct.values, color=colors_m, edgecolor='white')
axes[1].set_title('Missing Values (%)')
axes[1].set_xlabel('Missing %')
for i, (val, name) in enumerate(zip(missing_pct.values, missing.index)):
    axes[1].text(val + 0.2, i, f'{val}%', va='center', fontsize=9)
axes[1].set_xlim(0, missing_pct.max() * 1.2)

# Data types summary
dtype_counts = {'Binary (0/1)': 7, 'Categorical': 6, 'Numerical': 8, 'Date': 1, 'Target': 1}
wedge_colors = [BLUE, GREEN, ORANGE, PURPLE, RED]
axes[2].pie(dtype_counts.values(), labels=dtype_counts.keys(), colors=wedge_colors,
            autopct='%1.0f%%', startangle=90, pctdistance=0.75,
            wedgeprops={'edgecolor': 'white', 'linewidth': 2})
axes[2].set_title('Feature Types (26 columns)')

plt.tight_layout()
plt.savefig('outputs/eda/01_dataset_overview.png', dpi=150, bbox_inches='tight')
plt.close()
print("   Saved: 01_dataset_overview.png")

# ─── FIG 2: No-Show Rate by Categorical Variables ───────────────────────────
print("[2/8] No-Show Rates by Category...")
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('No-Show Rate by Categorical Variables', fontsize=15, fontweight='bold')

def noshow_rate(group_col, ax, title, rotate=0):
    rates = df.groupby(group_col)['no_show'].apply(
        lambda x: (x == 'yes').mean() * 100).sort_values(ascending=False)
    counts_g = df[group_col].value_counts()
    colors_bar = [RED if r > 35 else ORANGE if r > 28 else GREEN for r in rates.values]
    bars = ax.bar(rates.index, rates.values, color=colors_bar, edgecolor='white', linewidth=1.2)
    ax.axhline(y=31.8, color=GRAY, linestyle='--', linewidth=1.5, label='Overall avg (31.8%)')
    ax.set_title(title)
    ax.set_ylabel('No-Show Rate (%)')
    ax.legend(fontsize=9)
    ax.set_ylim(0, min(rates.max() * 1.25, 70))
    if rotate:
        ax.set_xticklabels(rates.index, rotation=rotate, ha='right', fontsize=9)
    for bar, val in zip(bars, rates.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=9)

noshow_rate('specialty', axes[0, 0], 'By Specialty', rotate=30)
noshow_rate('appointment_shift', axes[0, 1], 'By Appointment Shift')
noshow_rate('gender', axes[0, 2], 'By Gender')
noshow_rate('rain_intensity', axes[1, 0], 'By Rain Intensity')
noshow_rate('heat_intensity', axes[1, 1], 'By Heat Intensity', rotate=20)

# By SMS received
sms_rates = df.groupby('SMS_received')['no_show'].apply(
    lambda x: (x == 'yes').mean() * 100)
axes[1, 2].bar(['No SMS (0)', 'SMS Sent (1)'], sms_rates.values, color=[ORANGE, BLUE],
               edgecolor='white', linewidth=1.2)
axes[1, 2].axhline(y=31.8, color=GRAY, linestyle='--', linewidth=1.5)
axes[1, 2].set_title('By SMS Received')
axes[1, 2].set_ylabel('No-Show Rate (%)')
axes[1, 2].set_ylim(0, 45)
for i, val in enumerate(sms_rates.values):
    axes[1, 2].text(i, val + 0.5, f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('outputs/eda/02_noshow_by_category.png', dpi=150, bbox_inches='tight')
plt.close()
print("   Saved: 02_noshow_by_category.png")

# ─── FIG 3: Age Analysis ────────────────────────────────────────────────────
print("[3/8] Age Analysis...")
fig, axes = plt.subplots(1, 3, figsize=(17, 5))
fig.suptitle('Age Distribution & No-Show Patterns', fontsize=15, fontweight='bold')

age_data = df['age'].dropna()
axes[0].hist(age_data, bins=40, color=BLUE, edgecolor='white', alpha=0.85)
axes[0].axvline(age_data.mean(), color=RED, linestyle='--', linewidth=2,
                label=f'Mean: {age_data.mean():.1f}')
axes[0].axvline(age_data.median(), color=GREEN, linestyle='--', linewidth=2,
                label=f'Median: {age_data.median():.0f}')
axes[0].set_title('Age Distribution')
axes[0].set_xlabel('Age')
axes[0].set_ylabel('Count')
axes[0].legend()

df['age_group'] = pd.cut(df['age'],
    bins=[0, 12, 18, 35, 60, 120],
    labels=['Child\n(0-12)', 'Teen\n(13-18)', 'Adult\n(19-35)', 'Middle\n(36-60)', 'Senior\n(61+)'])
age_noshow = df.groupby('age_group', observed=True)['no_show'].apply(
    lambda x: (x == 'yes').mean() * 100)
age_count = df.groupby('age_group', observed=True).size()
bars = axes[1].bar(age_noshow.index, age_noshow.values,
                   color=[RED if v > 31.8 else GREEN for v in age_noshow.values],
                   edgecolor='white', linewidth=1.2)
axes[1].axhline(31.8, color=GRAY, linestyle='--', linewidth=1.5, label='Overall avg')
axes[1].set_title('No-Show Rate by Age Group')
axes[1].set_ylabel('No-Show Rate (%)')
axes[1].legend()
axes[1].set_ylim(0, 45)
for bar, val, cnt in zip(bars, age_noshow.values, age_count.values):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f'{val:.1f}%\n(n={cnt:,})', ha='center', va='bottom', fontsize=9)

show_ages = df[df['no_show'] == 'no']['age'].dropna()
noshow_ages = df[df['no_show'] == 'yes']['age'].dropna()
axes[2].hist(show_ages, bins=30, alpha=0.6, color=GREEN, label='Show', density=True)
axes[2].hist(noshow_ages, bins=30, alpha=0.6, color=RED, label='No-Show', density=True)
axes[2].set_title('Age Distribution: Show vs No-Show')
axes[2].set_xlabel('Age')
axes[2].set_ylabel('Density')
axes[2].legend()

plt.tight_layout()
plt.savefig('outputs/eda/03_age_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("   Saved: 03_age_analysis.png")

# ─── FIG 4: Health Conditions ───────────────────────────────────────────────
print("[4/8] Health Conditions Analysis...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Health Conditions & No-Show Behavior', fontsize=15, fontweight='bold')

conditions = ['Hipertension', 'Diabetes', 'Alcoholism', 'Handcap', 'Scholarship']
labels      = ['Hypertension', 'Diabetes', 'Alcoholism', 'Handicap', 'Scholarship']
rates_cond  = []
for cond in conditions:
    rate_1 = (df[df[cond] == 1]['no_show'] == 'yes').mean() * 100
    rate_0 = (df[df[cond] == 0]['no_show'] == 'yes').mean() * 100
    rates_cond.append((rate_0, rate_1))

x = np.arange(len(labels))
w = 0.35
bars1 = axes[0].bar(x - w/2, [r[0] for r in rates_cond], w, label='Without', color=GREEN, edgecolor='white')
bars2 = axes[0].bar(x + w/2, [r[1] for r in rates_cond], w, label='With', color=RED, edgecolor='white')
axes[0].set_title('No-Show Rate: With vs Without Condition')
axes[0].set_xticks(x)
axes[0].set_xticklabels(labels)
axes[0].set_ylabel('No-Show Rate (%)')
axes[0].legend()
axes[0].axhline(31.8, color=GRAY, linestyle='--', linewidth=1, alpha=0.7)
for bars in [bars1, bars2]:
    for bar in bars:
        h = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2, h + 0.3,
                     f'{h:.1f}%', ha='center', va='bottom', fontsize=8)

prev_rates = [(df[cond] == 1).mean() * 100 for cond in conditions]
bars = axes[1].bar(labels, prev_rates, color=PALETTE[:5], edgecolor='white', linewidth=1.2)
axes[1].set_title('Prevalence of Each Condition (%)')
axes[1].set_ylabel('Patients with Condition (%)')
for bar, val in zip(bars, prev_rates):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                 f'{val:.1f}%', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('outputs/eda/04_health_conditions.png', dpi=150, bbox_inches='tight')
plt.close()
print("   Saved: 04_health_conditions.png")

# ─── FIG 5: Weather Analysis ────────────────────────────────────────────────
print("[5/8] Weather Analysis...")
fig, axes = plt.subplots(1, 3, figsize=(17, 5))
fig.suptitle('Weather Impact on No-Show Behavior', fontsize=15, fontweight='bold')

df_w = df.dropna(subset=['average_temp_day', 'average_rain_day'])
show_temp   = df_w[df_w['no_show'] == 'no']['average_temp_day']
noshow_temp = df_w[df_w['no_show'] == 'yes']['average_temp_day']
axes[0].hist(show_temp, bins=25, alpha=0.6, color=GREEN, label='Show', density=True)
axes[0].hist(noshow_temp, bins=25, alpha=0.6, color=RED, label='No-Show', density=True)
axes[0].set_title('Avg Temperature by Outcome')
axes[0].set_xlabel('Avg Temp (°C)')
axes[0].set_ylabel('Density')
axes[0].legend()

rain_cats = ['no_rain', 'weak', 'moderate', 'heavy']
rain_ns   = [df[df['rain_intensity'] == r]['no_show'].apply(lambda x: x == 'yes').mean() * 100
             for r in rain_cats]
colors_r  = [GREEN, BLUE, ORANGE, RED]
bars = axes[1].bar(rain_cats, rain_ns, color=colors_r, edgecolor='white', linewidth=1.2)
axes[1].axhline(31.8, color=GRAY, linestyle='--', linewidth=1.5)
axes[1].set_title('No-Show Rate by Rain Intensity')
axes[1].set_ylabel('No-Show Rate (%)')
axes[1].set_ylim(0, 45)
for bar, val in zip(bars, rain_ns):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f'{val:.1f}%', ha='center', va='bottom', fontsize=10)

storm_rates = df.groupby('storm_day_before')['no_show'].apply(
    lambda x: (x == 'yes').mean() * 100)
axes[2].bar(['No Storm Before', 'Storm Day Before'], storm_rates.values,
            color=[GREEN, RED], edgecolor='white', linewidth=1.2)
axes[2].axhline(31.8, color=GRAY, linestyle='--', linewidth=1.5)
axes[2].set_title('No-Show Rate: Storm Day Before')
axes[2].set_ylabel('No-Show Rate (%)')
axes[2].set_ylim(0, 45)
for i, val in enumerate(storm_rates.values):
    axes[2].text(i, val + 0.3, f'{val:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('outputs/eda/05_weather_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("   Saved: 05_weather_analysis.png")

# ─── FIG 6: Time Patterns ───────────────────────────────────────────────────
print("[6/8] Time Patterns...")
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('Temporal Patterns in Appointments', fontsize=15, fontweight='bold')

df['day_of_week'] = df['appointment_date_continuous'].dt.dayofweek
df['month']       = df['appointment_date_continuous'].dt.month
df['week']        = df['appointment_date_continuous'].dt.isocalendar().week.astype(int)

day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
day_vol   = df.groupby('day_of_week').size()
day_ns    = df.groupby('day_of_week')['no_show'].apply(lambda x: (x == 'yes').mean() * 100)
ax1 = axes[0, 0]
ax1b = ax1.twinx()
bars = ax1.bar(day_names[:len(day_vol)], day_vol.values, color=BLUE, alpha=0.7, label='Volume')
ax1b.plot(day_names[:len(day_ns)], day_ns.values, color=RED, marker='o',
          linewidth=2, markersize=6, label='No-Show %')
ax1.set_title('Volume & No-Show Rate by Day of Week')
ax1.set_ylabel('Appointment Count', color=BLUE)
ax1b.set_ylabel('No-Show Rate (%)', color=RED)
ax1.legend(loc='upper left'); ax1b.legend(loc='upper right')

month_names = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
month_vol   = df.groupby('month').size()
month_ns    = df.groupby('month')['no_show'].apply(lambda x: (x == 'yes').mean() * 100)
ax2 = axes[0, 1]
ax2b = ax2.twinx()
ax2.bar([month_names[m-1] for m in month_vol.index], month_vol.values,
        color=GREEN, alpha=0.7)
ax2b.plot([month_names[m-1] for m in month_ns.index], month_ns.values,
          color=RED, marker='o', linewidth=2, markersize=6)
ax2.set_title('Volume & No-Show Rate by Month')
ax2.set_ylabel('Appointment Count', color=GREEN)
ax2b.set_ylabel('No-Show Rate (%)', color=RED)
ax2.set_xticklabels([month_names[m-1] for m in month_vol.index], rotation=45)

daily_counts = df.groupby('appointment_date_continuous').size()
axes[1, 0].plot(daily_counts.index, daily_counts.values, color=BLUE, linewidth=0.8, alpha=0.7)
axes[1, 0].fill_between(daily_counts.index, daily_counts.values, alpha=0.2, color=BLUE)
ma30 = daily_counts.rolling(30).mean()
axes[1, 0].plot(ma30.index, ma30.values, color=RED, linewidth=2, label='30-day MA')
axes[1, 0].set_title('Daily Appointment Volume Over Time')
axes[1, 0].set_xlabel('Date')
axes[1, 0].set_ylabel('Appointments per Day')
axes[1, 0].legend()

hour_vol = df.groupby('appointment_time').size()
hour_ns  = df.groupby('appointment_time')['no_show'].apply(lambda x: (x == 'yes').mean() * 100)
ax3 = axes[1, 1]
ax3b = ax3.twinx()
ax3.bar(hour_vol.index, hour_vol.values, color=ORANGE, alpha=0.7)
ax3b.plot(hour_ns.index, hour_ns.values, color=RED, marker='o', linewidth=2, markersize=4)
ax3.set_title('Volume & No-Show Rate by Appointment Hour')
ax3.set_xlabel('Hour of Day')
ax3.set_ylabel('Count', color=ORANGE)
ax3b.set_ylabel('No-Show Rate (%)', color=RED)

plt.tight_layout()
plt.savefig('outputs/eda/06_temporal_patterns.png', dpi=150, bbox_inches='tight')
plt.close()
print("   Saved: 06_temporal_patterns.png")

# ─── FIG 7: Geographic Analysis ────────────────────────────────────────────
print("[7/8] Geographic Analysis...")
place_data = df.dropna(subset=['place'])
top_places = place_data['place'].value_counts().head(12).index
place_ns   = place_data[place_data['place'].isin(top_places)].groupby('place')['no_show'].apply(
    lambda x: (x == 'yes').mean() * 100).sort_values(ascending=False)
place_vol  = place_data[place_data['place'].isin(top_places)]['place'].value_counts()

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Geographic Distribution & No-Show Rates (Top 12 Cities)', fontsize=15, fontweight='bold')

colors_p = [RED if r > 31.8 else GREEN for r in place_ns.values]
axes[0].barh(place_ns.index, place_ns.values, color=colors_p, edgecolor='white')
axes[0].axvline(31.8, color=GRAY, linestyle='--', linewidth=1.5, label='Overall avg')
axes[0].set_title('No-Show Rate by City')
axes[0].set_xlabel('No-Show Rate (%)')
axes[0].legend()
for i, val in enumerate(place_ns.values):
    axes[0].text(val + 0.3, i, f'{val:.1f}%', va='center', fontsize=9)

axes[1].barh(place_vol.index[:12], place_vol.values[:12], color=BLUE, edgecolor='white', alpha=0.8)
axes[1].set_title('Appointment Volume by City')
axes[1].set_xlabel('Total Appointments')
for i, val in enumerate(place_vol.values[:12]):
    axes[1].text(val + 100, i, f'{val:,}', va='center', fontsize=9)

plt.tight_layout()
plt.savefig('outputs/eda/07_geographic_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("   Saved: 07_geographic_analysis.png")

# ─── FIG 8: Correlation Heatmap ────────────────────────────────────────────
print("[8/8] Correlation Heatmap...")
num_cols = ['age', 'appointment_time', 'Hipertension', 'Diabetes', 'Alcoholism',
            'Handcap', 'Scholarship', 'SMS_received', 'patient_needs_companion',
            'under_12_years_old', 'over_60_years_old',
            'average_temp_day', 'average_rain_day', 'rainy_day_before', 'storm_day_before']
df_corr = df[num_cols].copy()
df_corr['no_show_binary'] = (df['no_show'] == 'yes').astype(int)
corr_matrix = df_corr.corr()

fig, ax = plt.subplots(figsize=(14, 11))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdYlGn',
            center=0, square=True, linewidths=0.5, ax=ax,
            annot_kws={'size': 8}, cbar_kws={'shrink': 0.8})
ax.set_title('Feature Correlation Matrix (including no_show target)', fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig('outputs/eda/08_correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("   Saved: 08_correlation_heatmap.png")

# ─── Summary Stats ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("EDA SUMMARY")
print("=" * 60)
print(f"Total appointments      : {len(df):,}")
print(f"No-show rate            : {(df['no_show']=='yes').mean()*100:.1f}%")
print(f"Date range              : {df['appointment_date_continuous'].min().date()} to {df['appointment_date_continuous'].max().date()}")
print(f"Unique patients (approx): Based on appointments only")
print(f"Specialties             : {df['specialty'].nunique()} (+ Unknown)")
print(f"Cities                  : {df['place'].nunique()} (+ Unknown)")
print(f"Missing - age           : {df['age'].isna().sum():,} ({df['age'].isna().mean()*100:.1f}%)")
print(f"Missing - specialty     : {df['specialty'].isna().sum():,} ({df['specialty'].isna().mean()*100:.1f}%)")
print(f"Highest no-show spec    : sem especialidade (52.8%)")
print(f"Lowest no-show spec     : pedagogo (16.0%)")
print(f"\nAll EDA charts saved to outputs/eda/")
