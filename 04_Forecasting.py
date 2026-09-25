
import pandas as pd, numpy as np, joblib, os, warnings
warnings.filterwarnings('ignore')
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

os.makedirs('outputs/forecasting', exist_ok=True)
BLUE='#1A73E8'; GREEN='#34A853'; RED='#EA4335'; ORANGE='#FBBC04'; PURPLE='#7B61FF'; GRAY='#5F6368'
plt.rcParams.update({'figure.facecolor':'white','axes.facecolor':'#F8F9FA','font.family':'sans-serif'})

def mape(y_true, y_pred):
    mask = y_true > 0
    return np.mean(np.abs((y_true[mask]-y_pred[mask])/y_true[mask]))*100

print("="*60); print("DEMAND FORECASTING"); print("="*60)
daily_raw = pd.read_csv('data/processed_forecasting.csv', parse_dates=['date'])
daily_raw = daily_raw.sort_values('date').reset_index(drop=True)
daily_raw['exp_smooth']  = daily_raw['total_appointments'].ewm(span=7).mean().shift(1)
daily_raw['expand_mean'] = daily_raw['total_appointments'].expanding().mean().shift(1)
for lag in [1,2,3,7,14,21,30]:
    daily_raw[f'lag_{lag}'] = daily_raw['total_appointments'].shift(lag)
daily_raw['rolling_7_mean']  = daily_raw['total_appointments'].shift(1).rolling(7,  min_periods=1).mean()
daily_raw['rolling_14_mean'] = daily_raw['total_appointments'].shift(1).rolling(14, min_periods=1).mean()
daily_raw['rolling_30_mean'] = daily_raw['total_appointments'].shift(1).rolling(30, min_periods=1).mean()
daily_raw['rolling_7_std']   = daily_raw['total_appointments'].shift(1).rolling(7,  min_periods=2).std().fillna(50)
daily_raw['rolling_7_max']   = daily_raw['total_appointments'].shift(1).rolling(7,  min_periods=1).max()
daily_model = daily_raw.dropna(subset=['lag_30']).reset_index(drop=True)

FORECAST_FEATURES = ['day_of_week','month','week_of_year','quarter','is_weekend','day_of_year',
    'dow_sin','dow_cos','month_sin','month_cos','lag_1','lag_2','lag_3','lag_7','lag_14','lag_21','lag_30',
    'rolling_7_mean','rolling_14_mean','rolling_30_mean','rolling_7_std','rolling_7_max',
    'exp_smooth','expand_mean','avg_temp','avg_rain','pct_rainy','pct_storm']
TARGET = 'total_appointments'

split_idx = int(len(daily_model)*0.80)
train_df = daily_model.iloc[:split_idx].copy(); test_df = daily_model.iloc[split_idx:].copy()
X_train = train_df[FORECAST_FEATURES].fillna(0).values; y_train = train_df[TARGET].values
X_test  = test_df[FORECAST_FEATURES].fillna(0).values;  y_test  = test_df[TARGET].values
y_train_t = np.sqrt(y_train)
print(f"Train: {len(train_df)} | Test: {len(test_df)}")

print("Model 1: Naive Seasonal...")
naive_pred = np.maximum(test_df['lag_7'].fillna(test_df['rolling_7_mean']).values,0)
naive_rmse=np.sqrt(mean_squared_error(y_test,naive_pred)); naive_r2=r2_score(y_test,naive_pred)
naive_mape=mape(y_test,naive_pred); naive_mae=mean_absolute_error(y_test,naive_pred)
print(f"  RMSE={naive_rmse:.1f} MAE={naive_mae:.1f} R2={naive_r2:.3f}")

print("Model 2: Ridge Regression...")
ridge = Pipeline([('scaler',StandardScaler()),('reg',Ridge(alpha=1.0))])
ridge.fit(X_train,y_train_t)
ridge_pred=np.maximum(ridge.predict(X_test)**2,0)
ridge_rmse=np.sqrt(mean_squared_error(y_test,ridge_pred)); ridge_r2=r2_score(y_test,ridge_pred)
ridge_mape=mape(y_test,ridge_pred); ridge_mae=mean_absolute_error(y_test,ridge_pred)
print(f"  RMSE={ridge_rmse:.1f} MAE={ridge_mae:.1f} R2={ridge_r2:.3f}")

print("Model 3: Gradient Boosting...")
gb_reg = GradientBoostingRegressor(n_estimators=400,learning_rate=0.04,max_depth=6,
    min_samples_leaf=3,subsample=0.85,max_features=0.8,random_state=42)
gb_reg.fit(X_train,y_train_t)
gb_pred=np.maximum(gb_reg.predict(X_test)**2,0)
gb_rmse=np.sqrt(mean_squared_error(y_test,gb_pred)); gb_r2=r2_score(y_test,gb_pred)
gb_mape=mape(y_test,gb_pred); gb_mae=mean_absolute_error(y_test,gb_pred)
print(f"  RMSE={gb_rmse:.1f} MAE={gb_mae:.1f} R2={gb_r2:.3f}")

print(f"\nBEST: Gradient Boosting  R2={gb_r2:.3f}  RMSE={gb_rmse:.0f}")

# Charts
fig,axes=plt.subplots(3,1,figsize=(14,13)); fig.suptitle('Demand Forecasting: All Models',fontsize=14,fontweight='bold')
for ax,pred,name,(m,r2,rmse),color in zip(axes,
    [naive_pred,ridge_pred,gb_pred],
    ['Naive Seasonal','Ridge Regression','Gradient Boosting'],
    [(naive_mape,naive_r2,naive_rmse),(ridge_mape,ridge_r2,ridge_rmse),(gb_mape,gb_r2,gb_rmse)],
    [ORANGE,PURPLE,BLUE]):
    ax.plot(test_df['date'].values,y_test,color=GRAY,linewidth=1.5,label='Actual',alpha=0.8)
    ax.plot(test_df['date'].values,pred,color=color,linewidth=2,label='Predicted')
    ax.fill_between(test_df['date'].values,y_test,pred,alpha=0.12,color=color)
    ax.set_title(f'{name}  |  R²:{r2:.3f}  |  RMSE:{rmse:.0f}  |  MAE:{mean_absolute_error(y_test,pred):.0f}')
    ax.set_ylabel('Appointments/Day'); ax.legend(loc='upper left',fontsize=9)
    ax.set_facecolor('#F8F9FA'); ax.grid(True,alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/forecasting/02_actual_vs_predicted.png',dpi=150,bbox_inches='tight'); plt.close()

fi=pd.DataFrame({'feature':FORECAST_FEATURES,'importance':gb_reg.feature_importances_}).sort_values('importance',ascending=False)
fig2,ax2=plt.subplots(figsize=(10,8))
colors_fi=[RED if i<3 else ORANGE if i<8 else BLUE for i in range(len(fi))]
ax2.barh(fi['feature'][::-1],fi['importance'][::-1],color=colors_fi[::-1],edgecolor='white')
ax2.set_title('Feature Importance — Gradient Boosting Regressor',fontsize=13)
ax2.set_xlabel('Importance Score'); ax2.set_facecolor('#F8F9FA'); fig2.patch.set_facecolor('white')
plt.tight_layout(); plt.savefig('outputs/forecasting/04_feature_importance.png',dpi=150,bbox_inches='tight'); plt.close()
print("Charts saved.")

fc_bundle={
    'model':gb_reg,'model_name':'Gradient Boosting','transform':'sqrt',
    'forecast_features':FORECAST_FEATURES,
    'test_mape':gb_mape,'test_r2':gb_r2,'test_rmse':gb_rmse,'test_mae':gb_mae,
    'all_results':{
        'Naive Seasonal':    {'mape':naive_mape,'r2':naive_r2,'rmse':naive_rmse},
        'Ridge Regression':  {'mape':ridge_mape,'r2':ridge_r2,'rmse':ridge_rmse},
        'Gradient Boosting': {'mape':gb_mape,   'r2':gb_r2,   'rmse':gb_rmse},
    },
    'last_values':{
        'last_date':str(train_df['date'].max().date()),
        **{f'lag_{i}':float(y_train[-i]) for i in [1,2,3,7,14,21,30]},
        'rolling_7_mean':float(np.mean(y_train[-7:])),'rolling_14_mean':float(np.mean(y_train[-14:])),
        'rolling_30_mean':float(np.mean(y_train[-30:])),'rolling_7_std':float(np.std(y_train[-7:])),
        'rolling_7_max':float(np.max(y_train[-7:])),'exp_smooth':float(pd.Series(y_train).ewm(span=7).mean().iloc[-1]),
        'expand_mean':float(np.mean(y_train)),
    },
    'daily_data':daily_model,'train_mean':float(y_train.mean()),'train_std':float(y_train.std()),
}
joblib.dump(fc_bundle,'models/demand_forecaster.pkl')
print(f"Saved: {os.path.getsize('models/demand_forecaster.pkl')//1024} KB"); print("Done!")
