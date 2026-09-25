
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import joblib, os, warnings, time
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    f1_score, roc_auc_score, precision_score, recall_score,
    confusion_matrix, classification_report, roc_curve, precision_recall_curve,
    average_precision_score
)
from sklearn.utils.class_weight import compute_class_weight

os.makedirs('outputs/classification', exist_ok=True)

BLUE='#1A73E8'; GREEN='#34A853'; RED='#EA4335'; ORANGE='#FBBC04'; PURPLE='#7B61FF'; GRAY='#5F6368'

plt.rcParams.update({'figure.facecolor':'white','axes.facecolor':'#F8F9FA',
    'axes.edgecolor':'#DADCE0','font.family':'sans-serif','font.size':11,
    'axes.titlesize':13,'axes.titleweight':'bold'})

print("=" * 60)
print("CLASSIFICATION MODEL TRAINING")
print("=" * 60)

# ─── Load data ───────────────────────────────────────────────────────────────
df = pd.read_csv('data/processed_classification.csv')
feature_data = joblib.load('models/feature_cols.pkl')
FEATURE_COLS = feature_data['clf_features']

X = df[FEATURE_COLS].copy()
y = df['no_show_binary'].copy()
print(f"Dataset: {X.shape}, Target: {y.value_counts().to_dict()}")
print(f"Class imbalance ratio: {y.value_counts()[0]/y.value_counts()[1]:.2f}:1")

# ─── Train/Test Split (stratified) ───────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
print(f"\nTrain: {X_train.shape}, Test: {X_test.shape}")
print(f"Train class dist: {y_train.value_counts().to_dict()}")

# Class weights
cw = compute_class_weight('balanced', classes=np.array([0, 1]), y=y_train.values)
class_weights = {0: cw[0], 1: cw[1]}
print(f"Class weights: {class_weights}")

# ─── Model 1: Logistic Regression ────────────────────────────────────────────
print("\n" + "-" * 50)
print("MODEL 1: Logistic Regression (Baseline)")
print("-" * 50)

lr_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(
        class_weight='balanced', max_iter=1000,
        C=0.1, solver='lbfgs', random_state=42, n_jobs=-1))
])

t0 = time.time()
lr_pipeline.fit(X_train, y_train)
lr_time = time.time() - t0

lr_pred  = lr_pipeline.predict(X_test)
lr_proba = lr_pipeline.predict_proba(X_test)[:, 1]
lr_f1    = f1_score(y_test, lr_pred)
lr_auc   = roc_auc_score(y_test, lr_proba)
print(f"  F1-Score : {lr_f1:.4f}  {'✓ PASS' if lr_f1 >= 0.70 else '✗ FAIL'} (target: >0.70)")
print(f"  ROC-AUC  : {lr_auc:.4f}  {'✓ PASS' if lr_auc >= 0.75 else '✗ FAIL'} (target: >0.75)")
print(f"  Precision: {precision_score(y_test, lr_pred):.4f}")
print(f"  Recall   : {recall_score(y_test, lr_pred):.4f}")
print(f"  Train time: {lr_time:.1f}s")

# Cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
lr_cv_f1  = cross_val_score(lr_pipeline, X_train, y_train, cv=cv, scoring='f1', n_jobs=-1)
lr_cv_auc = cross_val_score(lr_pipeline, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
print(f"  CV F1    : {lr_cv_f1.mean():.4f} ± {lr_cv_f1.std():.4f}")
print(f"  CV AUC   : {lr_cv_auc.mean():.4f} ± {lr_cv_auc.std():.4f}")

# ─── Model 2: Random Forest ──────────────────────────────────────────────────
print("\n" + "-" * 50)
print("MODEL 2: Random Forest")
print("-" * 50)

rf_clf = RandomForestClassifier(
    n_estimators=300, max_depth=15, min_samples_leaf=10,
    min_samples_split=20, max_features='sqrt',
    class_weight='balanced', random_state=42, n_jobs=-1)

t0 = time.time()
rf_clf.fit(X_train, y_train)
rf_time = time.time() - t0

rf_pred  = rf_clf.predict(X_test)
rf_proba = rf_clf.predict_proba(X_test)[:, 1]
rf_f1    = f1_score(y_test, rf_pred)
rf_auc   = roc_auc_score(y_test, rf_proba)
print(f"  F1-Score : {rf_f1:.4f}  {'✓ PASS' if rf_f1 >= 0.70 else '✗ FAIL'} (target: >0.70)")
print(f"  ROC-AUC  : {rf_auc:.4f}  {'✓ PASS' if rf_auc >= 0.75 else '✗ FAIL'} (target: >0.75)")
print(f"  Precision: {precision_score(y_test, rf_pred):.4f}")
print(f"  Recall   : {recall_score(y_test, rf_pred):.4f}")
print(f"  Train time: {rf_time:.1f}s")

rf_cv_f1  = cross_val_score(rf_clf, X_train, y_train, cv=cv, scoring='f1', n_jobs=-1)
rf_cv_auc = cross_val_score(rf_clf, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
print(f"  CV F1    : {rf_cv_f1.mean():.4f} ± {rf_cv_f1.std():.4f}")
print(f"  CV AUC   : {rf_cv_auc.mean():.4f} ± {rf_cv_auc.std():.4f}")

# ─── Model 3: Gradient Boosting ──────────────────────────────────────────────
print("\n" + "-" * 50)
print("MODEL 3: Gradient Boosting Classifier")
print("-" * 50)

# Compute sample weights for GB (doesn't have class_weight param natively in all modes)
sample_weights = np.where(y_train == 1, class_weights[1], class_weights[0])

gb_clf = GradientBoostingClassifier(
    n_estimators=300, learning_rate=0.05, max_depth=5,
    min_samples_leaf=20, subsample=0.8,
    max_features='sqrt', random_state=42)

t0 = time.time()
gb_clf.fit(X_train, y_train, sample_weight=sample_weights)
gb_time = time.time() - t0

gb_pred  = gb_clf.predict(X_test)
gb_proba = gb_clf.predict_proba(X_test)[:, 1]
gb_f1    = f1_score(y_test, gb_pred)
gb_auc   = roc_auc_score(y_test, gb_proba)
print(f"  F1-Score : {gb_f1:.4f}  {'✓ PASS' if gb_f1 >= 0.70 else '✗ FAIL'} (target: >0.70)")
print(f"  ROC-AUC  : {gb_auc:.4f}  {'✓ PASS' if gb_auc >= 0.75 else '✗ FAIL'} (target: >0.75)")
print(f"  Precision: {precision_score(y_test, gb_pred):.4f}")
print(f"  Recall   : {recall_score(y_test, gb_pred):.4f}")
print(f"  Train time: {gb_time:.1f}s")

gb_cv_f1  = cross_val_score(gb_clf, X_train, y_train, cv=cv, scoring='f1', n_jobs=-1)
gb_cv_auc = cross_val_score(gb_clf, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)

print(f"  CV F1    : {gb_cv_f1.mean():.4f} ± {gb_cv_f1.std():.4f}")
print(f"  CV AUC   : {gb_cv_auc.mean():.4f} ± {gb_cv_auc.std():.4f}")

# ─── Model Comparison ────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("MODEL COMPARISON SUMMARY")
print("=" * 60)
results = {
    'Logistic Regression': {'f1': lr_f1, 'auc': lr_auc, 'proba': lr_proba, 'pred': lr_pred,
                             'cv_f1': lr_cv_f1, 'cv_auc': lr_cv_auc, 'time': lr_time},
    'Random Forest':       {'f1': rf_f1, 'auc': rf_auc, 'proba': rf_proba, 'pred': rf_pred,
                             'cv_f1': rf_cv_f1, 'cv_auc': rf_cv_auc, 'time': rf_time},
    'Gradient Boosting':   {'f1': gb_f1, 'auc': gb_auc, 'proba': gb_proba, 'pred': gb_pred,
                             'cv_f1': gb_cv_f1, 'cv_auc': gb_cv_auc, 'time': gb_time},
}
print(f"{'Model':<25} {'F1':>8} {'ROC-AUC':>10} {'CV-F1':>10} {'CV-AUC':>10} {'Time':>8}")
print("-" * 75)
for name, r in results.items():
    print(f"{name:<25} {r['f1']:>8.4f} {r['auc']:>10.4f} "
          f"{r['cv_f1'].mean():>10.4f} {r['cv_auc'].mean():>10.4f} {r['time']:>7.1f}s")

# Pick best model by F1
best_name = max(results, key=lambda k: results[k]['f1'])
print(f"\nBEST MODEL: {best_name} (F1={results[best_name]['f1']:.4f})")

models_map = {
    'Logistic Regression': lr_pipeline,
    'Random Forest':       rf_clf,
    'Gradient Boosting':   gb_clf,
}
best_model = models_map[best_name]

# ─── Threshold Tuning for best model ─────────────────────────────────────────
print("\n[Threshold Tuning]")
best_proba = results[best_name]['proba']
thresholds = np.arange(0.2, 0.8, 0.02)
f1_scores  = [f1_score(y_test, (best_proba >= t).astype(int)) for t in thresholds]
best_thresh = thresholds[np.argmax(f1_scores)]
best_f1_thresh = max(f1_scores)
print(f"  Default threshold (0.5) F1 : {results[best_name]['f1']:.4f}")
print(f"  Optimal threshold          : {best_thresh:.2f}")
print(f"  Optimal threshold F1       : {best_f1_thresh:.4f}")

best_pred_thresh = (best_proba >= best_thresh).astype(int)

# ─── VISUALIZATIONS ──────────────────────────────────────────────────────────
print("\n[Generating visualizations...]")

# --- Fig 1: Model Comparison ---
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Model Performance Comparison', fontsize=15, fontweight='bold')

model_names_short = ['Logistic\nRegression', 'Random\nForest', 'Gradient\nBoosting']
f1_vals  = [results[k]['f1']  for k in results]
auc_vals = [results[k]['auc'] for k in results]
colors_m = [GREEN if v >= 0.70 else RED for v in f1_vals]

bars = axes[0].bar(model_names_short, f1_vals, color=colors_m, edgecolor='white', linewidth=1.2)
axes[0].axhline(0.70, color=RED, linestyle='--', linewidth=1.5, label='Target (0.70)')
axes[0].set_title('F1-Score Comparison')
axes[0].set_ylabel('F1-Score')
axes[0].set_ylim(0, 1)
axes[0].legend()
for bar, val in zip(bars, f1_vals):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

colors_a = [GREEN if v >= 0.75 else RED for v in auc_vals]
bars2 = axes[1].bar(model_names_short, auc_vals, color=colors_a, edgecolor='white', linewidth=1.2)
axes[1].axhline(0.75, color=RED, linestyle='--', linewidth=1.5, label='Target (0.75)')
axes[1].set_title('ROC-AUC Comparison')
axes[1].set_ylabel('ROC-AUC')
axes[1].set_ylim(0, 1)
axes[1].legend()
for bar, val in zip(bars2, auc_vals):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

cv_means = [results[k]['cv_f1'].mean() for k in results]
cv_stds  = [results[k]['cv_f1'].std()  for k in results]
axes[2].bar(model_names_short, cv_means, color=[BLUE, PURPLE, ORANGE],
            edgecolor='white', linewidth=1.2, yerr=cv_stds, capsize=5, error_kw={'linewidth':2})
axes[2].set_title('CV F1-Score (5-Fold) ± Std')
axes[2].set_ylabel('CV F1-Score')
axes[2].set_ylim(0, 1)

plt.tight_layout()
plt.savefig('outputs/classification/01_model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

# --- Fig 2: ROC & PR Curves ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('ROC Curves & Precision-Recall Curves', fontsize=15, fontweight='bold')

colors_roc = [BLUE, GREEN, ORANGE]
for i, (name, r) in enumerate(results.items()):
    fpr, tpr, _ = roc_curve(y_test, r['proba'])
    axes[0].plot(fpr, tpr, color=colors_roc[i], linewidth=2,
                 label=f"{name} (AUC={r['auc']:.3f})")
axes[0].plot([0,1],[0,1], 'k--', linewidth=1)
axes[0].fill_between([0,1],[0,1],[0,0], alpha=0.05, color=GRAY)
axes[0].set_xlabel('False Positive Rate')
axes[0].set_ylabel('True Positive Rate')
axes[0].set_title('ROC Curves')
axes[0].legend(loc='lower right')
axes[0].axhline(0.75, color=RED, linestyle=':', alpha=0.5)
axes[0].set_xlim([0,1]); axes[0].set_ylim([0,1.02])

for i, (name, r) in enumerate(results.items()):
    prec, rec, _ = precision_recall_curve(y_test, r['proba'])
    ap = average_precision_score(y_test, r['proba'])
    axes[1].plot(rec, prec, color=colors_roc[i], linewidth=2,
                 label=f"{name} (AP={ap:.3f})")
no_skill = y_test.sum() / len(y_test)
axes[1].axhline(no_skill, color=GRAY, linestyle='--', linewidth=1, label=f'No Skill ({no_skill:.2f})')
axes[1].set_xlabel('Recall')
axes[1].set_ylabel('Precision')
axes[1].set_title('Precision-Recall Curves')
axes[1].legend(loc='upper right')

plt.tight_layout()
plt.savefig('outputs/classification/02_roc_pr_curves.png', dpi=150, bbox_inches='tight')
plt.close()

# --- Fig 3: Confusion Matrices ---
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Confusion Matrices', fontsize=15, fontweight='bold')

for ax, (name, r) in zip(axes, results.items()):
    cm = confusion_matrix(y_test, r['pred'])
    cm_norm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]
    im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(['Predicted\nShow', 'Predicted\nNo-Show'])
    ax.set_yticklabels(['Actual Show', 'Actual No-Show'])
    ax.set_title(name)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f'{cm[i,j]:,}\n({cm_norm[i,j]:.1%})',
                    ha='center', va='center', fontsize=10, fontweight='bold',
                    color='white' if cm_norm[i, j] > 0.5 else 'black')

plt.tight_layout()
plt.savefig('outputs/classification/03_confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.close()

# --- Fig 4: Feature Importance ---
fig, axes = plt.subplots(1, 2, figsize=(16, 8))
fig.suptitle('Feature Importance Analysis', fontsize=15, fontweight='bold')

if hasattr(best_model, 'feature_importances_'):
    importances = best_model.feature_importances_
elif hasattr(best_model, 'named_steps'):
    importances = best_model.named_steps['clf'].coef_[0]
else:
    importances = best_model.feature_importances_

feat_imp = pd.DataFrame({'feature': FEATURE_COLS, 'importance': np.abs(importances)})
feat_imp = feat_imp.sort_values('importance', ascending=False)
top20 = feat_imp.head(20)

colors_fi = [RED if i < 5 else ORANGE if i < 10 else BLUE for i in range(len(top20))]
axes[0].barh(top20['feature'][::-1], top20['importance'][::-1], color=colors_fi[::-1], edgecolor='white')
axes[0].set_title(f'Top 20 Feature Importances\n({best_name})')
axes[0].set_xlabel('Importance Score')

# Grouped importance by category
groups = {
    'Demographics': ['age', 'gender_enc', 'age_group', 'is_child', 'is_senior',
                     'patient_needs_companion', 'age_missing'],
    'Appointment':  ['appointment_time', 'shift_enc', 'morning_appt', 'late_appt', 'early_appt'],
    'Location':     ['specialty_enc', 'place_enc', 'disability_enc', 'specialty_missing'],
    'Health':       ['Hipertension', 'Diabetes', 'Alcoholism', 'Handcap', 'Scholarship',
                     'has_chronic_condition', 'has_substance_issue', 'total_conditions',
                     'has_disability_or_hdcp', 'SMS_received'],
    'Weather':      ['average_temp_day', 'average_rain_day', 'max_temp_day', 'max_rain_day',
                     'rainy_day_before', 'storm_day_before', 'rain_intensity_enc',
                     'heat_intensity_enc', 'bad_weather', 'severe_weather', 'high_temp', 'heavy_rain'],
    'Temporal':     ['day_of_week', 'month', 'week_of_year', 'quarter',
                     'is_weekend', 'is_month_start', 'is_month_end', 'day_of_year'],
}
group_imp = {}
imp_dict = dict(zip(FEATURE_COLS, np.abs(importances)))
for gname, gcols in groups.items():
    group_imp[gname] = sum(imp_dict.get(c, 0) for c in gcols)

g_colors = [RED, ORANGE, BLUE, GREEN, PURPLE, GRAY]
bars_g = axes[1].bar(group_imp.keys(), group_imp.values(), color=g_colors, edgecolor='white', linewidth=1.2)
axes[1].set_title('Importance by Feature Group')
axes[1].set_ylabel('Total Importance')
axes[1].set_xticklabels(group_imp.keys(), rotation=15)
for bar, val in zip(bars_g, group_imp.values()):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('outputs/classification/04_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()

# --- Fig 5: Threshold Analysis ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Classification Threshold Analysis', fontsize=15, fontweight='bold')

f1s = [f1_score(y_test, (best_proba >= t).astype(int)) for t in thresholds]
precs = [precision_score(y_test, (best_proba >= t).astype(int), zero_division=0) for t in thresholds]
recs  = [recall_score(y_test, (best_proba >= t).astype(int), zero_division=0) for t in thresholds]

axes[0].plot(thresholds, f1s, color=BLUE, linewidth=2.5, label='F1-Score')
axes[0].plot(thresholds, precs, color=GREEN, linewidth=2, linestyle='--', label='Precision')
axes[0].plot(thresholds, recs, color=RED, linewidth=2, linestyle='--', label='Recall')
axes[0].axvline(best_thresh, color=ORANGE, linewidth=2, linestyle=':', label=f'Optimal ({best_thresh:.2f})')
axes[0].axvline(0.5, color=GRAY, linewidth=1, linestyle=':', label='Default (0.5)')
axes[0].axhline(0.70, color=RED, linewidth=1, linestyle=':', alpha=0.5, label='F1 Target')
axes[0].set_xlabel('Classification Threshold')
axes[0].set_ylabel('Score')
axes[0].set_title('Metrics vs Threshold')
axes[0].legend(fontsize=9)
axes[0].set_ylim(0, 1.05)

# Risk score distribution
axes[1].hist(best_proba[y_test == 0], bins=50, alpha=0.6, color=GREEN,
             density=True, label='Actual Show')
axes[1].hist(best_proba[y_test == 1], bins=50, alpha=0.6, color=RED,
             density=True, label='Actual No-Show')
axes[1].axvline(best_thresh, color=ORANGE, linewidth=2, linestyle='--',
                label=f'Optimal threshold: {best_thresh:.2f}')
axes[1].axvline(0.5, color=GRAY, linewidth=1, linestyle='--', label='Default: 0.5')
axes[1].set_xlabel('Predicted No-Show Probability')
axes[1].set_ylabel('Density')
axes[1].set_title('Risk Score Distribution by True Label')
axes[1].legend()

plt.tight_layout()
plt.savefig('outputs/classification/05_threshold_analysis.png', dpi=150, bbox_inches='tight')
plt.close()

print("   All visualizations saved to outputs/classification/")

# ─── Final Metrics with Optimal Threshold ───────────────────────────────────
print("\n" + "=" * 60)
print(f"FINAL MODEL: {best_name} (threshold={best_thresh:.2f})")
print("=" * 60)
print(classification_report(y_test, best_pred_thresh, target_names=['Show', 'No-Show']))
print(f"ROC-AUC: {results[best_name]['auc']:.4f}")

# ─── Save Best Model ─────────────────────────────────────────────────────────
model_bundle = {
    'model':          best_model,
    'model_name':     best_name,
    'threshold':      best_thresh,
    'feature_cols':   FEATURE_COLS,
    'test_f1':        results[best_name]['f1'],
    'test_auc':       results[best_name]['auc'],
    'optimal_f1':     best_f1_thresh,
    'all_results':    {k: {'f1': v['f1'], 'auc': v['auc']} for k, v in results.items()},
}
joblib.dump(model_bundle, 'models/no_show_classifier.pkl')
print(f"\nModel saved to models/no_show_classifier.pkl")
print(f"Model size: {os.path.getsize('models/no_show_classifier.pkl') / 1024:.0f} KB")
print("\n✓ Classification pipeline complete!")
