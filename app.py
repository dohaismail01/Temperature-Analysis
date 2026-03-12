import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, max_error
import os

# ── File path — works on any machine ─────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(BASE_DIR, "GLB.Ts+dSST.csv")

st.set_page_config(page_title="Climate Temp Predictor", layout="wide")
st.title("Global Temperature Anomalies Predictor")
st.markdown("**Built by Doha Ismail | AI/ML Engineer**")
st.markdown("---")

monthly_cols = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

# ── Load & clean ──────────────────────────────────────────────
@st.cache_data
def load_and_clean():
    df = pd.read_csv(file_path, na_values="*******")
    df = df.drop_duplicates()
    df = df.dropna(subset=monthly_cols, how='all')
    # Force all monthly columns to numeric (fixes object dtype issues)
    for col in monthly_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    for col in monthly_cols:
        df[col] = df[col].fillna(df[col].mean())
    # Ensure Year is numeric
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    df = df.dropna(subset=['Year'])
    df['Year'] = df['Year'].astype(int)
    df['J-D'] = df[monthly_cols].mean(axis=1)
    df['DJF'] = df[['Dec','Jan','Feb']].mean(axis=1)
    df['MAM'] = df[['Mar','Apr','May']].mean(axis=1)
    df['JJA'] = df[['Jun','Jul','Aug']].mean(axis=1)
    df['SON'] = df[['Sep','Oct','Nov']].mean(axis=1)
    return df

# ── Approach 1: Time Series ───────────────────────────────────
@st.cache_resource
def train_approach1(df):
    X = df[['Year']].values
    y = df['J-D'].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_tr, X_te, y_tr, y_te = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    models = {
        'Linear Regression': LinearRegression(),
        'Ridge': Ridge(alpha=1.0),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'SVM': SVR(kernel='rbf', C=1.0)
    }
    results = []
    for name, m in models.items():
        m.fit(X_tr, y_tr)
        pred = m.predict(X_te)
        mape = np.mean(np.abs((y_te - pred) / np.where(y_te == 0, 1e-10, y_te))) * 100
        r2   = r2_score(y_te, pred)
        n, p = len(y_te), X_tr.shape[1]
        results.append({
            'Model'     : name,
            'Test RMSE' : round(np.sqrt(mean_squared_error(y_te, pred)), 6),
            'Test MAE'  : round(mean_absolute_error(y_te, pred), 6),
            'Test R2'   : round(r2, 4),
            'Adj R2'    : round(1 - (1 - r2) * (n - 1) / (n - p - 1), 4),
            'MAPE (%)'  : round(mape, 4),
            'Max Error' : round(max_error(y_te, pred), 6)
        })

    # Best model for forecasting: Ridge (can extrapolate linearly)
    best = Ridge(alpha=1.0)
    best.fit(X_scaled, y)
    return scaler, best, pd.DataFrame(results).sort_values('Test RMSE'), X_scaled, y

# ── Approach 2: Lag Features ──────────────────────────────────
@st.cache_resource
def train_approach2(df):
    df_lag = df[['Year','J-D']].copy().reset_index(drop=True)
    df_lag['lag_1'] = df_lag['J-D'].shift(1)
    df_lag['lag_2'] = df_lag['J-D'].shift(2)
    df_lag['lag_3'] = df_lag['J-D'].shift(3)
    df_lag = df_lag.dropna().reset_index(drop=True)

    X = df_lag[['Year','lag_1','lag_2','lag_3']].values
    y = df_lag['J-D'].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_tr, X_te, y_tr, y_te = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    models = {
        'Linear Regression': LinearRegression(),
        'Ridge': Ridge(alpha=1.0),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'SVM': SVR(kernel='rbf', C=1.0)
    }
    results = []
    for name, m in models.items():
        m.fit(X_tr, y_tr)
        pred = m.predict(X_te)
        mape = np.mean(np.abs((y_te - pred) / np.where(y_te == 0, 1e-10, y_te))) * 100
        r2   = r2_score(y_te, pred)
        n, p = len(y_te), X_tr.shape[1]
        results.append({
            'Model'     : name,
            'Test RMSE' : round(np.sqrt(mean_squared_error(y_te, pred)), 6),
            'Test MAE'  : round(mean_absolute_error(y_te, pred), 6),
            'Test R2'   : round(r2, 4),
            'Adj R2'    : round(1 - (1 - r2) * (n - 1) / (n - p - 1), 4),
            'MAPE (%)'  : round(mape, 4),
            'Max Error' : round(max_error(y_te, pred), 6)
        })

    best = Ridge(alpha=1.0)
    best.fit(X_scaled, y)
    return scaler, best, pd.DataFrame(results).sort_values('Test RMSE'), X_scaled, y, df_lag

df = load_and_clean()
scaler_ts, model_ts, res_ts, X_ts, y_ts             = train_approach1(df)
scaler_lag, model_lag, res_lag, X_lag, y_lag, df_lag = train_approach2(df)

# ── Sidebar ───────────────────────────────────────────────────
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", [
    "Step 1: EDA",
    "Step 2 & 3: Cleaning & Pre-processing",
    "Step 4: Scaling",
    "Step 5: Data Split",
    "Step 6: Modeling",
    "Approach 1: Time Series Forecast",
    "Approach 2: Lag Features Forecast",
    "Comparison & Custom Prediction",
    "Hyperparameter Tuning",
    "Residual Analysis",
    "Learning Curves",
    "Feature Importance",
    "Confidence Intervals"
])

# ─────────────────────────────────────────────────────────────
# PAGE 1: EDA
# ─────────────────────────────────────────────────────────────
if page == "Step 1: EDA":
    st.header("Step 1: Exploratory Data Analysis")

    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Year Range", f"{df['Year'].min()} - {df['Year'].max()}")

    st.subheader("Dataset Sample")
    st.dataframe(df.head(), use_container_width=True)

    st.subheader("Descriptive Statistics")
    st.dataframe(df[monthly_cols].describe().round(4), use_container_width=True)

    st.subheader("Missing Values (raw)")
    raw = pd.read_csv(file_path, na_values="*******")
    miss = raw.isnull().sum().reset_index()
    miss.columns = ['Column','Missing']
    st.dataframe(miss[miss['Missing'] > 0], use_container_width=True)

    st.subheader("Monthly Distributions")
    fig, axes = plt.subplots(3, 4, figsize=(16, 9))
    for ax, col in zip(axes.flatten(), monthly_cols):
        ax.hist(df[col].dropna(), bins=15, color='steelblue', edgecolor='white')
        ax.set_title(col); ax.set_xlabel('Anomaly (C)')
    plt.tight_layout(); st.pyplot(fig)

    st.subheader("Boxplots")
    fig2, ax2 = plt.subplots(figsize=(14, 5))
    df[monthly_cols].boxplot(ax=ax2)
    ax2.set_title('Monthly Anomaly Boxplots'); ax2.set_ylabel('Anomaly (C)')
    st.pyplot(fig2)

    st.subheader("Correlation Heatmap")
    fig3, ax3 = plt.subplots(figsize=(12, 7))
    sns.heatmap(df[monthly_cols + ['J-D']].corr(), annot=True, fmt='.2f', cmap='coolwarm', ax=ax3)
    plt.tight_layout(); st.pyplot(fig3)

    st.subheader("Annual Trend")
    fig4, ax4 = plt.subplots(figsize=(12, 4))
    ax4.plot(df['Year'], df['J-D'], color='crimson', linewidth=2, marker='o', markersize=3)
    ax4.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax4.fill_between(df['Year'], df['J-D'], 0, where=df['J-D'] > 0, alpha=0.3, color='red', label='Above avg')
    ax4.fill_between(df['Year'], df['J-D'], 0, where=df['J-D'] < 0, alpha=0.3, color='blue', label='Below avg')
    ax4.set_xlabel("Year"); ax4.set_ylabel("Anomaly (C)"); ax4.legend()
    st.pyplot(fig4)

# ─────────────────────────────────────────────────────────────
# PAGE 2: Cleaning & Pre-processing
# ─────────────────────────────────────────────────────────────
elif page == "Step 2 & 3: Cleaning & Pre-processing":
    st.header("Step 2: Data Cleaning")
    raw = pd.read_csv(file_path, na_values="*******")

    col1, col2, col3 = st.columns(3)
    col1.metric("Duplicates Removed", raw.duplicated().sum())
    col2.metric("Missing (raw)", int(raw[monthly_cols].isnull().sum().sum()))
    col3.metric("Missing (after)", int(df[monthly_cols].isnull().sum().sum()))

    st.markdown("**Actions:** Removed duplicates, dropped fully-empty rows, filled remaining NaN with column mean, converted Year to int.")

    st.header("Step 3: Pre-processing")
    st.markdown("All features are numeric — no encoding needed. Target variables computed from monthly averages.")

    st.subheader("Skewness Check")
    skew = df[monthly_cols].skew().reset_index()
    skew.columns = ['Month','Skewness']
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(skew['Month'], skew['Skewness'], color='steelblue')
    ax.axhline(0, color='gray', linestyle='--')
    ax.set_title('Skewness per Month'); ax.set_ylabel('Skewness')
    st.pyplot(fig)

# ─────────────────────────────────────────────────────────────
# PAGE 3: Scaling
# ─────────────────────────────────────────────────────────────
elif page == "Step 4: Scaling":
    st.header("Step 4: Scaling Numeric Features")
    X = df[['Year'] + monthly_cols].copy()
    scalers = {
        'StandardScaler (mean=0, std=1)': StandardScaler(),
        'MinMaxScaler (0 to 1)': MinMaxScaler(),
        'RobustScaler (outlier-resistant)': RobustScaler()
    }

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for ax, (name, s) in zip(axes, scalers.items()):
        Xsc = s.fit_transform(X)
        ax.hist(Xsc[:, 1:].flatten(), bins=30, color='steelblue', edgecolor='white')
        ax.set_title(name, fontsize=9); ax.set_xlabel('Scaled Value')
    plt.suptitle('Feature Distribution After Each Scaler')
    plt.tight_layout(); st.pyplot(fig)
    st.info("StandardScaler selected for both approaches.")

# ─────────────────────────────────────────────────────────────
# PAGE 4: Data Split
# ─────────────────────────────────────────────────────────────
elif page == "Step 5: Data Split":
    st.header("Step 5: Splitting Data")
    y = df['J-D'].values
    X_tr, X_te, y_tr, y_te = train_test_split(X_ts, y, test_size=0.2, random_state=42)
    X_trv, X_val, y_trv, y_val = train_test_split(X_tr, y_tr, test_size=0.2, random_state=42)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total", len(y))
    col2.metric("Train", len(y_trv), f"{len(y_trv)/len(y)*100:.0f}%")
    col3.metric("Validation", len(y_val), f"{len(y_val)/len(y)*100:.0f}%")
    col4.metric("Test", len(y_te), f"{len(y_te)/len(y)*100:.0f}%")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.pie([len(y_trv), len(y_val), len(y_te)],
           labels=['Train','Validation','Test'],
           colors=['#4e79a7','#f28e2b','#e15759'],
           autopct='%1.0f%%', startangle=90)
    ax.set_title('Data Split')
    st.pyplot(fig)

# ─────────────────────────────────────────────────────────────
# PAGE 5: Modeling
# ─────────────────────────────────────────────────────────────
elif page == "Step 6: Modeling":
    st.header("Step 6: Modeling — Both Approaches")

    st.subheader("Approach 1: Time Series (Year only)")
    st.dataframe(res_ts.style.highlight_min(subset=['Test RMSE'], color='#d4edda'), use_container_width=True)

    st.subheader("Approach 2: Lag Features")
    st.dataframe(res_lag.style.highlight_min(subset=['Test RMSE'], color='#d4edda'), use_container_width=True)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    res_ts.set_index('Model')['Test RMSE'].sort_values().plot(kind='barh', ax=axes[0], color='steelblue')
    axes[0].set_title('Approach 1 — Test RMSE')
    res_ts.set_index('Model')['Test MAE'].sort_values().plot(kind='barh', ax=axes[1], color='coral')
    axes[1].set_title('Approach 1 — Test MAE')
    res_ts.set_index('Model')['MAPE (%)'].sort_values().plot(kind='barh', ax=axes[2], color='mediumpurple')
    axes[2].set_title('Approach 1 — MAPE (%)')
    plt.tight_layout(); st.pyplot(fig)

    fig2, axes2 = plt.subplots(1, 3, figsize=(16, 4))
    res_lag.set_index('Model')['Test RMSE'].sort_values().plot(kind='barh', ax=axes2[0], color='seagreen')
    axes2[0].set_title('Approach 2 — Test RMSE')
    res_lag.set_index('Model')['Test MAE'].sort_values().plot(kind='barh', ax=axes2[1], color='coral')
    axes2[1].set_title('Approach 2 — Test MAE')
    res_lag.set_index('Model')['MAPE (%)'].sort_values().plot(kind='barh', ax=axes2[2], color='mediumpurple')
    axes2[2].set_title('Approach 2 — MAPE (%)')
    plt.tight_layout(); st.pyplot(fig2)

# ─────────────────────────────────────────────────────────────
# PAGE 6: Approach 1 Forecast
# ─────────────────────────────────────────────────────────────
elif page == "Approach 1: Time Series Forecast":
    st.header("Approach 1: Time Series Forecast (Ridge)")
    st.info("Uses Year as the only feature. Ridge extrapolates the linear trend into the future.")

    forecast_year = st.slider("Forecast until year", 2026, 2050, 2035)
    future_years  = np.arange(2026, forecast_year + 1).reshape(-1, 1)
    future_scaled = scaler_ts.transform(future_years)
    hist_pred     = model_ts.predict(X_ts)
    future_pred   = model_ts.predict(future_scaled)

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(df['Year'], y_ts, label='Historical', color='steelblue', linewidth=2)
    ax.plot(df['Year'], hist_pred, label='Model Fit', color='orange', linestyle='--', alpha=0.7)
    ax.plot(future_years, future_pred, label=f'Forecast', color='crimson', marker='o', linewidth=2)
    ax.axvline(2025, color='gray', linestyle=':', alpha=0.7, label='Forecast start')
    ax.set_title("Approach 1 — Time Series Annual Forecast")
    ax.set_xlabel("Year"); ax.set_ylabel("Anomaly (C)")
    ax.legend(); ax.grid(True, alpha=0.3)
    st.pyplot(fig)

    st.subheader("Forecast Table")
    st.dataframe(pd.DataFrame({'Year': future_years.flatten(), 'Predicted Anomaly (C)': future_pred.round(4)}),
                 use_container_width=True)

# ─────────────────────────────────────────────────────────────
# PAGE 7: Approach 2 Forecast
# ─────────────────────────────────────────────────────────────
elif page == "Approach 2: Lag Features Forecast":
    st.header("Approach 2: Lag Features Forecast (Ridge + Smoothed Seed)")
    st.info("Uses previous years anomalies as features with Ridge regression. Seed values are smoothed using a 3-year rolling mean to reduce sensitivity to recent spikes.")

    forecast_year    = st.slider("Forecast until year", 2026, 2050, 2035)
    future_years_lst = list(range(2026, forecast_year + 1))

    # Use smoothed seed to avoid oscillation from recent spikes
    smoothed_seed = df_lag['J-D'].rolling(3).mean().dropna().values
    last_vals     = list(smoothed_seed[-3:])
    future_preds  = []
    for yr in future_years_lst:
        row    = np.array([[yr, last_vals[-1], last_vals[-2], last_vals[-3]]])
        row_sc = scaler_lag.transform(row)
        pred   = model_lag.predict(row_sc)[0]
        future_preds.append(pred)
        last_vals.append(pred)

    hist_pred_lag = model_lag.predict(X_lag)

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(df_lag['Year'], y_lag, label='Historical', color='steelblue', linewidth=2)
    ax.plot(df_lag['Year'], hist_pred_lag, label='Model Fit', color='orange', linestyle='--', alpha=0.7)
    ax.plot(future_years_lst, future_preds, label='Forecast', color='crimson', marker='o', linewidth=2)
    ax.axvline(2025, color='gray', linestyle=':', alpha=0.7, label='Forecast start')
    ax.set_title("Approach 2 — Lag Features Annual Forecast")
    ax.set_xlabel("Year"); ax.set_ylabel("Anomaly (C)")
    ax.legend(); ax.grid(True, alpha=0.3)
    st.pyplot(fig)

    st.subheader("Forecast Table")
    st.dataframe(pd.DataFrame({'Year': future_years_lst, 'Predicted Anomaly (C)': np.round(future_preds, 4)}),
                 use_container_width=True)

# ─────────────────────────────────────────────────────────────
# PAGE 8: Comparison & Custom Prediction
# ─────────────────────────────────────────────────────────────
elif page == "Comparison & Custom Prediction":
    st.header("Comparison & Custom Prediction")

    tab1, tab2 = st.tabs(["Approach Comparison", "Custom Prediction"])

    with tab1:
        st.subheader("Forecast Comparison (2026-2035)")
        future_years_arr = np.arange(2026, 2036).reshape(-1, 1)
        pred_ts  = model_ts.predict(scaler_ts.transform(future_years_arr))

        last_vals = list(df_lag['J-D'].rolling(3).mean().dropna().values[-3:])
        pred_lag  = []
        for yr in range(2026, 2036):
            row = np.array([[yr, last_vals[-1], last_vals[-2], last_vals[-3]]])
            p   = model_lag.predict(scaler_lag.transform(row))[0]
            pred_lag.append(p); last_vals.append(p)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].plot(df['Year'], y_ts, color='steelblue', linewidth=2, label='Historical')
        axes[0].plot(future_years_arr, pred_ts, color='crimson', marker='o', label='Forecast')
        axes[0].axvline(2025, color='gray', linestyle=':')
        axes[0].set_title('Approach 1: Time Series (Ridge)')
        axes[0].set_xlabel('Year'); axes[0].set_ylabel('Anomaly (C)'); axes[0].legend()

        axes[1].plot(df_lag['Year'], y_lag, color='steelblue', linewidth=2, label='Historical')
        axes[1].plot(range(2026,2036), pred_lag, color='crimson', marker='o', label='Forecast')
        axes[1].axvline(2025, color='gray', linestyle=':')
        axes[1].set_title('Approach 2: Lag Features (Ridge + Smoothed Seed)')
        axes[1].set_xlabel('Year'); axes[1].set_ylabel('Anomaly (C)'); axes[1].legend()

        plt.tight_layout(); st.pyplot(fig)

        comp = pd.DataFrame({
            'Year'               : list(range(2026, 2036)),
            'Approach 1 (Ridge)' : pred_ts.round(4),
            'Approach 2 (Ridge Lags)': np.round(pred_lag, 4)
        })
        st.dataframe(comp, use_container_width=True)

        st.subheader("Model Performance Summary")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Approach 1 — Time Series**")
            st.dataframe(res_ts, use_container_width=True)
        with col2:
            st.markdown("**Approach 2 — Lag Features**")
            st.dataframe(res_lag, use_container_width=True)

    with tab2:
        st.subheader("Custom Prediction")
        st.markdown("Select approach and enter values to get a prediction.")

        approach = st.radio("Select Approach", ["Approach 1: Time Series", "Approach 2: Lag Features"])
        year = st.number_input("Year", min_value=1880, max_value=2100, value=2030)

        if approach == "Approach 1: Time Series":
            if st.button("Predict", type="primary"):
                row    = scaler_ts.transform(np.array([[year]]))
                pred   = model_ts.predict(row)[0]
                st.success(f"Predicted Annual Anomaly for {int(year)}: **{pred:.4f} C**")
        else:
            st.markdown("Enter the last 3 known anomaly values:")
            c1, c2, c3 = st.columns(3)
            lag1 = c1.number_input("Anomaly (t-1)", value=0.3, step=0.01, format="%.3f")
            lag2 = c2.number_input("Anomaly (t-2)", value=0.25, step=0.01, format="%.3f")
            lag3 = c3.number_input("Anomaly (t-3)", value=0.2, step=0.01, format="%.3f")
            if st.button("Predict", type="primary"):
                row  = scaler_lag.transform(np.array([[year, lag1, lag2, lag3]]))
                pred = model_lag.predict(row)[0]
                st.success(f"Predicted Annual Anomaly for {int(year)}: **{pred:.4f} C**")

# ─────────────────────────────────────────────────────────────
# PAGE: Hyperparameter Tuning
# ─────────────────────────────────────────────────────────────
elif page == "Hyperparameter Tuning":
    st.header("Hyperparameter Tuning (GridSearchCV)")
    st.info("Searching for the best parameters for Ridge, SVM, and Random Forest.")

    from sklearn.model_selection import GridSearchCV

    with st.spinner("Running GridSearchCV — this may take a moment..."):
        # Ridge — Approach 1
        gs_ridge = GridSearchCV(Ridge(), {'alpha': [0.01, 0.1, 1.0, 5.0, 10.0, 50.0, 100.0]},
                                cv=5, scoring='neg_mean_squared_error')
        gs_ridge.fit(X_ts, y_ts)

        # SVM — Approach 2
        gs_svm = GridSearchCV(SVR(), {'C': [0.1, 1.0, 10.0], 'gamma': ['scale', 'auto'], 'kernel': ['rbf', 'linear']},
                              cv=5, scoring='neg_mean_squared_error')
        gs_svm.fit(X_lag, y_lag)

        # Random Forest — Approach 2
        gs_rf = GridSearchCV(RandomForestRegressor(random_state=42),
                             {'n_estimators': [50, 100, 200], 'max_depth': [None, 3, 5], 'min_samples_split': [2, 5]},
                             cv=5, scoring='neg_mean_squared_error')
        gs_rf.fit(X_lag, y_lag)

    col1, col2, col3 = st.columns(3)
    col1.metric("Best Ridge alpha", gs_ridge.best_params_['alpha'])
    col2.metric("Best SVM kernel", gs_svm.best_params_['kernel'])
    col3.metric("Best RF n_estimators", gs_rf.best_params_['n_estimators'])

    st.subheader("Ridge — Alpha Search")
    ridge_results = pd.DataFrame(gs_ridge.cv_results_)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.semilogx(ridge_results['param_alpha'].astype(float),
                np.sqrt(-ridge_results['mean_test_score']), 'o-', color='steelblue')
    ax.set_xlabel('Alpha'); ax.set_ylabel('CV RMSE')
    ax.set_title('Ridge: Alpha vs CV RMSE')
    ax.axvline(gs_ridge.best_params_['alpha'], color='red', linestyle='--', label=f'Best alpha={gs_ridge.best_params_["alpha"]}')
    ax.legend(); st.pyplot(fig)

    st.subheader("Best Parameters Summary")
    summary = pd.DataFrame({
        'Model': ['Ridge (Approach 1)', 'SVM (Approach 2)', 'Random Forest (Approach 2)'],
        'Best Params': [str(gs_ridge.best_params_), str(gs_svm.best_params_), str(gs_rf.best_params_)],
        'Best CV RMSE': [round(np.sqrt(-gs_ridge.best_score_), 6),
                         round(np.sqrt(-gs_svm.best_score_), 6),
                         round(np.sqrt(-gs_rf.best_score_), 6)]
    })
    st.dataframe(summary, use_container_width=True)

# ─────────────────────────────────────────────────────────────
# PAGE: Residual Analysis
# ─────────────────────────────────────────────────────────────
elif page == "Residual Analysis":
    st.header("Residual Analysis")
    st.markdown("Residuals = Actual - Predicted. Ideally they should be random with no pattern.")

    hist_pred_ts_full  = model_ts.predict(X_ts)
    hist_pred_lag_full = model_lag.predict(X_lag)
    residuals_ts  = y_ts  - hist_pred_ts_full
    residuals_lag = y_lag - hist_pred_lag_full

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    axes[0,0].plot(df['Year'], residuals_ts, color='steelblue', marker='o', markersize=4)
    axes[0,0].axhline(0, color='red', linestyle='--')
    axes[0,0].set_title('Approach 1 — Residuals Over Time')
    axes[0,0].set_xlabel('Year'); axes[0,0].set_ylabel('Residual (C)')

    axes[0,1].hist(residuals_ts, bins=12, color='steelblue', edgecolor='white')
    axes[0,1].set_title('Approach 1 — Residual Distribution')
    axes[0,1].set_xlabel('Residual (C)')

    axes[0,2].scatter(hist_pred_ts_full, residuals_ts, color='steelblue', alpha=0.7)
    axes[0,2].axhline(0, color='red', linestyle='--')
    axes[0,2].set_title('Approach 1 — Residuals vs Predicted')
    axes[0,2].set_xlabel('Predicted'); axes[0,2].set_ylabel('Residual')

    axes[1,0].plot(df_lag['Year'], residuals_lag, color='seagreen', marker='o', markersize=4)
    axes[1,0].axhline(0, color='red', linestyle='--')
    axes[1,0].set_title('Approach 2 — Residuals Over Time')
    axes[1,0].set_xlabel('Year'); axes[1,0].set_ylabel('Residual (C)')

    axes[1,1].hist(residuals_lag, bins=12, color='seagreen', edgecolor='white')
    axes[1,1].set_title('Approach 2 — Residual Distribution')
    axes[1,1].set_xlabel('Residual (C)')

    axes[1,2].scatter(hist_pred_lag_full, residuals_lag, color='seagreen', alpha=0.7)
    axes[1,2].axhline(0, color='red', linestyle='--')
    axes[1,2].set_title('Approach 2 — Residuals vs Predicted')
    axes[1,2].set_xlabel('Predicted'); axes[1,2].set_ylabel('Residual')

    plt.suptitle('Residual Analysis', fontsize=14)
    plt.tight_layout(); st.pyplot(fig)

    col1, col2 = st.columns(2)
    col1.metric("Approach 1 — Mean Residual", f"{residuals_ts.mean():.6f}")
    col1.metric("Approach 1 — Std Residual",  f"{residuals_ts.std():.6f}")
    col2.metric("Approach 2 — Mean Residual", f"{residuals_lag.mean():.6f}")
    col2.metric("Approach 2 — Std Residual",  f"{residuals_lag.std():.6f}")

# ─────────────────────────────────────────────────────────────
# PAGE: Learning Curves
# ─────────────────────────────────────────────────────────────
elif page == "Learning Curves":
    st.header("Learning Curves")
    st.markdown("Shows how model performance changes as training data increases. A large gap between train and validation = overfitting.")

    from sklearn.model_selection import learning_curve

    def get_learning_curve(model, X, y):
        train_sizes, train_scores, val_scores = learning_curve(
            model, X, y, cv=5, scoring='neg_mean_squared_error',
            train_sizes=np.linspace(0.2, 1.0, 8), shuffle=True, random_state=42
        )
        return (train_sizes,
                np.sqrt(-train_scores.mean(axis=1)), np.sqrt(-train_scores).std(axis=1),
                np.sqrt(-val_scores.mean(axis=1)),   np.sqrt(-val_scores).std(axis=1))

    with st.spinner("Computing learning curves..."):
        ts1  = get_learning_curve(Ridge(alpha=1.0), X_ts,  y_ts)
        lag1 = get_learning_curve(Ridge(alpha=1.0), X_lag, y_lag)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, (title, color, lc) in zip(axes, [
        ('Approach 1: Ridge (Time Series)', 'steelblue', ts1),
        ('Approach 2: Ridge (Lag Features)', 'seagreen', lag1)
    ]):
        sizes, tr_mean, tr_std, val_mean, val_std = lc
        ax.plot(sizes, tr_mean,  'o-', color=color,   label='Train RMSE')
        ax.plot(sizes, val_mean, 's--', color='crimson', label='Validation RMSE')
        ax.fill_between(sizes, tr_mean - tr_std,  tr_mean + tr_std,  alpha=0.1, color=color)
        ax.fill_between(sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='crimson')
        ax.set_title(title); ax.set_xlabel('Training Size'); ax.set_ylabel('RMSE')
        ax.legend()

    plt.suptitle('Learning Curves', fontsize=14)
    plt.tight_layout(); st.pyplot(fig)

# ─────────────────────────────────────────────────────────────
# PAGE: Feature Importance
# ─────────────────────────────────────────────────────────────
elif page == "Feature Importance":
    st.header("Feature Importance")

    with st.spinner("Computing feature importance..."):
        rf_imp = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_imp.fit(X_lag, y_lag)

        ridge_imp = Ridge(alpha=1.0)
        ridge_imp.fit(X_ts, y_ts)

    feature_names = ['Year', 'Lag-1 (t-1)', 'Lag-2 (t-2)', 'Lag-3 (t-3)']
    importances   = rf_imp.feature_importances_
    indices       = np.argsort(importances)[::-1]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].bar([feature_names[i] for i in indices], importances[indices], color='seagreen', edgecolor='white')
    axes[0].set_title('Random Forest — Feature Importance (Approach 2)')
    axes[0].set_ylabel('Importance Score')

    axes[1].bar(['Year'], np.abs(ridge_imp.coef_), color='steelblue', edgecolor='white')
    axes[1].set_title('Ridge — Coefficient (Approach 1)')
    axes[1].set_ylabel('|Coefficient|')

    plt.suptitle('Feature Importance', fontsize=14)
    plt.tight_layout(); st.pyplot(fig)

    st.subheader("Feature Importance Values")
    fi_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values('Importance', ascending=False)
    st.dataframe(fi_df, use_container_width=True)
    st.markdown("**Interpretation:** Lag-1 (most recent year) typically has the highest importance, confirming that recent anomalies are the strongest predictor of future ones.")

# ─────────────────────────────────────────────────────────────
# PAGE: Confidence Intervals
# ─────────────────────────────────────────────────────────────
elif page == "Confidence Intervals":
    st.header("Confidence Intervals (Bootstrap)")
    st.markdown("Uses bootstrap resampling (200 iterations) to estimate forecast uncertainty.")

    n_bootstrap  = st.slider("Number of bootstrap iterations", 50, 500, 200, step=50)
    forecast_year = st.slider("Forecast until year", 2026, 2050, 2035)

    future_years_arr = np.arange(2026, forecast_year + 1).reshape(-1, 1)
    future_scaled_ci = scaler_ts.transform(future_years_arr)

    with st.spinner(f"Running {n_bootstrap} bootstrap iterations..."):
        np.random.seed(42)
        boot_preds = np.zeros((n_bootstrap, len(future_years_arr)))
        for i in range(n_bootstrap):
            idx = np.random.choice(len(X_ts), len(X_ts), replace=True)
            m   = Ridge(alpha=1.0)
            m.fit(X_ts[idx], y_ts[idx])
            boot_preds[i] = m.predict(future_scaled_ci)

    ci_lower = np.percentile(boot_preds, 5,  axis=0)
    ci_upper = np.percentile(boot_preds, 95, axis=0)
    ci_mean  = boot_preds.mean(axis=0)

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(df['Year'], y_ts, label='Historical', color='steelblue', linewidth=2)
    ax.plot(future_years_arr, ci_mean, label='Forecast (mean)', color='crimson', marker='o', linewidth=2)
    ax.fill_between(future_years_arr.flatten(), ci_lower, ci_upper,
                    alpha=0.25, color='crimson', label='90% Confidence Interval')
    ax.axvline(2025, color='gray', linestyle=':', alpha=0.7)
    ax.set_title('Forecast with 90% Confidence Interval (Bootstrap)')
    ax.set_xlabel('Year'); ax.set_ylabel('Anomaly (C)')
    ax.legend(); ax.grid(True, alpha=0.3)
    st.pyplot(fig)

    ci_df = pd.DataFrame({
        'Year'      : future_years_arr.flatten(),
        'Forecast'  : ci_mean.round(4),
        'Lower 5%'  : ci_lower.round(4),
        'Upper 95%' : ci_upper.round(4),
        'Interval Width': (ci_upper - ci_lower).round(4)
    })
    st.subheader("Confidence Interval Table")
    st.dataframe(ci_df, use_container_width=True)

st.markdown("---")
st.markdown("Built by **Doha Ismail** | AI/ML Engineer | Alexandria, Egypt")
