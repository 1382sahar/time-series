import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb
import seaborn as sns

# تنظیم استایل حرفه‌ای برای نمودارها
plt.style.use('seaborn')  # استایل حرفه‌ای برای نمودارها
sns.set_context("talk")   # فونت بزرگ‌تر برای ارائه

# -- Step 1: Load data with error handling --
try:
    filename = 'd.txt'
    cols_to_use = ['Unnamed: 0', 'MT_267']
    data = pd.read_csv(filename, sep=';', usecols=cols_to_use, decimal=',')
except FileNotFoundError:
    print(f"خطا: فایل {filename} یافت نشد.")
    exit()

# پیش‌پردازش داده‌ها
data.dropna(inplace=True)
data.rename(columns={'Unnamed: 0': 'Date'}, inplace=True)
data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
if data['Date'].isna().any():
    print("خطا: فرمت برخی تاریخ‌ها نامعتبر است.")
    exit()
data.set_index('Date', inplace=True)

# -- Step 2: Normalize data --
scaler = MinMaxScaler()
values = data['MT_267'].values.reshape(-1, 1)
values_scaled = scaler.fit_transform(values)

# -- Step 3: Prepare sequences for LSTM --
def create_sequences(data, seq_length=48):
    """ایجاد توالی‌های سری زمانی برای مدل LSTM"""
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

seq_length = 48
X, y = create_sequences(values_scaled, seq_length)

# -- Step 4: Train-test split --
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]
dates_test = data.index[-len(y_test):]  # ذخیره تاریخ‌ها برای نمودار

# -- Step 5: LSTM model with early stopping --
lstm_model = Sequential(name="LSTM_Model")
lstm_model.add(LSTM(50, activation='relu', input_shape=(seq_length, 1), name="LSTM_Layer"))
lstm_model.add(Dense(1, name="Output_Layer"))
lstm_model.compile(optimizer='adam', loss='mse')
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
history = lstm_model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1,
                        callbacks=[early_stopping], verbose=1)
y_pred_lstm = lstm_model.predict(X_test)

# -- Step 6: Prepare data for tree-based models --
X_train_2d = X_train.reshape((X_train.shape[0], X_train.shape[1]))
X_test_2d = X_test.reshape((X_test.shape[0], X_test.shape[1]))

# -- Step 7: K-Fold Cross-Validation for tree-based models --
def evaluate_model_with_kfold(model, X, y, model_name, n_splits=5):
    """ارزیابی مدل با اعتبارسنجی متقاطع سری زمانی"""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    r2_scores, mae_scores, rmse_scores = [], [], []
    for train_idx, val_idx in tscv.split(X):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        model.fit(X_tr, y_tr.ravel())
        y_pred = model.predict(X_val)
        y_val_inv = scaler.inverse_transform(y_val)
        y_pred_inv = scaler.inverse_transform(y_pred.reshape(-1, 1))
        r2_scores.append(r2_score(y_val_inv, y_pred_inv))
        mae_scores.append(mean_absolute_error(y_val_inv, y_pred_inv))
        rmse_scores.append(np.sqrt(mean_squared_error(y_val_inv, y_pred_inv)))
    print(f"{model_name} - میانگین R²: {np.mean(r2_scores):.4f} ± {np.std(r2_scores):.4f}")
    print(f"{model_name} - میانگین MAE: {np.mean(mae_scores):.4f} ± {np.std(mae_scores):.4f}")
    print(f"{model_name} - میانگین RMSE: {np.mean(rmse_scores):.4f} ± {np.std(rmse_scores):.4f}")
    return np.mean(r2_scores)

# تعریف مدل‌ها
models = {
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
    "XGBoost": xgb.XGBRegressor(n_estimators=100, random_state=42, verbosity=0),
    "Decision Tree": DecisionTreeRegressor(random_state=42)
}

# آموزش و ارزیابی با K-Fold
kfold_r2_scores = {}
for name, model in models.items():
    kfold_r2_scores[name] = evaluate_model_with_kfold(model, X_train_2d, y_train, name)

# آموزش مدل‌ها روی کل داده‌های آموزشی
y_preds = {}
for name, model in models.items():
    model.fit(X_train_2d, y_train.ravel())
    y_preds[name] = model.predict(X_test_2d)

# پیش‌بینی‌های LSTM
y_preds["LSTM"] = y_pred_lstm.ravel()

# -- Step 8: Ensemble model (weighted by R²) --
r2_scores = {}
y_pred_ensemble = np.zeros_like(y_preds["LSTM"])
weights = [kfold_r2_scores.get("Random Forest", 0.5), r2_scores.get("LSTM", 0.5)]
weights = weights / np.sum(weights)  # نرمال‌سازی وزن‌ها
y_pred_ensemble = weights[0] * y_preds["Random Forest"] + weights[1] * y_preds["LSTM"]
y_preds["LSTM + RF Ensemble"] = y_pred_ensemble

# -- Step 9: Inverse transform predictions --
y_test_inv = scaler.inverse_transform(y_test)
y_pred_invs = {name: scaler.inverse_transform(pred.reshape(-1, 1)) for name, pred in y_preds.items()}

# -- Step 10: Calculate evaluation metrics --
metrics = {"Model": [], "R² Score": [], "MAE": [], "RMSE": []}
for name, y_pred_inv in y_pred_invs.items():
    r2 = r2_score(y_test_inv, y_pred_inv)
    mae = mean_absolute_error(y_test_inv, y_pred_inv)
    rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
    metrics["Model"].append(name)
    metrics["R² Score"].append(r2)
    metrics["MAE"].append(mae)
    metrics["RMSE"].append(rmse)
    r2_scores[name] = r2

# نمایش جدول معیارها
metrics_df = pd.DataFrame(metrics)
print("\nجدول معیارهای ارزیابی:")
print(metrics_df.round(4))

# -- Step 11: Visualization --

# نمودار 1: Loss آموزش و اعتبارسنجی LSTM
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss', color='blue')
plt.plot(history.history['val_loss'], label='Validation Loss', color='orange')
plt.title('روند یادگیری مدل LSTM')
plt.xlabel('دوره (Epoch)')
plt.ylabel('خطای میانگین مربعات (MSE)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('lstm_training_loss.png', dpi=300)  # ذخیره برای رزومه
plt.show()

# نمودار 2: مقایسه R² مدل‌ها
plt.figure(figsize=(12, 6))
sns.barplot(x="Model", y="R² Score", data=metrics_df, palette="viridis")
plt.title('مقایسه امتیاز R² مدل‌ها')
plt.xlabel('مدل')
plt.ylabel('امتیاز R²')
plt.xticks(rotation=45)
plt.grid(True, axis='y')
plt.tight_layout()
plt.savefig('r2_comparison.png', dpi=300)  # ذخیره برای رزومه
plt.show()

# نمودار 3: پیش‌بینی‌ها در مقابل مقادیر واقعی
plt.figure(figsize=(14, 8))
plt.plot(dates_test, y_test_inv, label='مقادیر واقعی', color='black', linewidth=2)
for name, y_pred_inv in y_pred_invs.items():
    linestyle = '--' if name == "LSTM + RF Ensemble" else '-'
    plt.plot(dates_test, y_pred_inv, label=f'پیش‌بینی {name}', linestyle=linestyle)
plt.title('مقایسه پیش‌بینی‌های مصرف انرژی')
plt.xlabel('تاریخ')
plt.ylabel('مصرف انرژی')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.savefig('predictions_comparison.png', dpi=300)  # ذخیره برای رزومه
plt.show()

# نمودار 4: اهمیت ویژگی‌ها برای جنگل تصادفی
rf_model = models["Random Forest"]
feature_importance = pd.DataFrame({
    'Feature': [f'Time Step {i+1}' for i in range(seq_length)],
    'Importance': rf_model.feature_importances_
})
feature_importance = feature_importance.sort_values('Importance', ascending=False)
plt.figure(figsize=(12, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance.head(10), palette='magma')
plt.title('اهمیت 10 مرحله زمانی برتر (جنگل تصادفی)')
plt.xlabel('اهمیت')
plt.ylabel('مرحله زمانی')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300)  # ذخیره برای رزومه
plt.show()