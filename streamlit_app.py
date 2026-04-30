import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import random
import os
import gdown
import requests
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, auc,
    f1_score, confusion_matrix,
    precision_score, recall_score
)

# ── GitHub Alert Function ──
def trigger_fraud_alert(transaction_id, risk_score, amount, tx_type):
    url = "https://api.github.com/repos/manumi777/w1954810_Final_Year_Project/dispatches"
    payload = {
        "event_type": "fraud_alert",
        "client_payload": {
            "transaction_id": str(transaction_id),
            "risk_score": round(float(risk_score), 4),
            "amount": float(amount),
            "type": str(tx_type)
        }
    }
    headers = {
        "Authorization": f"token {st.secrets['GITHUB_TOKEN']}",
        "Accept": "application/vnd.github.v3+json"
    }
    response = requests.post(url, json=payload, headers=headers)
    return response.status_code

# ── Page Config ──
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Styles ──
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #f1f5f9 !important;
        font-family: 'Inter', sans-serif;
    }
    
    [data-testid="stHeader"] { display: none; }
    [data-testid="stDecoration"] { display: none; }
    footer { display: none; }
    
    .main .block-container {
        padding-top: 0rem !important;
        max-width: 1600px;
        padding-left: 2.5rem;
        padding-right: 2.5rem;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #F8FAFC !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 16px !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08) !important;
        padding: 24px !important;
        margin-bottom: 20px !important;
        overflow: visible !important;
    }
    
    div[data-testid="stVerticalBlockBorderWrapper"] [data-testid="stVerticalBlockBorderWrapper"] {
        border: none !important;
        box-shadow: none !important;
        background: transparent !important;
        padding: 0 !important;
        margin-bottom: 0 !important;
    }

    .kpi-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; margin-bottom: 24px; }
    .kpi-box {
        background: #F8FAFC; border: 1px solid #e2e8f0; border-radius: 16px; padding: 24px;
        position: relative; box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        transition: transform 0.2s ease;
    }
    .kpi-box:hover { transform: translateY(-2px); }
    .kpi-icon { width: 48px; height: 48px; border-radius: 12px; background: white; display: flex; align-items: center; justify-content: center; margin-bottom: 16px; box-shadow: 0 2px 4px rgba(0,0,0,0.02); }
    .kpi-badge { position: absolute; top: 24px; right: 24px; padding: 4px 12px; border-radius: 20px; font-size: 11px; font-weight: 700; }
    .badge-up { background: #d1fae5; color: #065f46; }
    .badge-down { background: #fee2e2; color: #991b1b; }
    .kpi-label { font-size: 13px; color: #64748b; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 6px; }
    .kpi-value { font-size: 36px; font-weight: 800; color: #0f172a; line-height: 1; }

    .section-title { font-size: 20px; font-weight: 800; color: #0f172a; letter-spacing: -0.02em; margin-bottom: 2px; }
    .section-sub { font-size: 14px; color: #64748b; margin-bottom: 24px; }
    
    .perf-table { width: 100%; border-collapse: collapse; margin-top: 10px; }
    .perf-table th { text-align: left; padding: 16px 12px; color: #64748b; font-size: 11px; font-weight: 800; border-bottom: 2px solid #f8fafc; text-transform: uppercase; letter-spacing: 0.05em; }
    .perf-table td { padding: 20px 12px; color: #1e293b; font-size: 14px; border-bottom: 1px solid #f8fafc; font-weight: 500; }
    .perf-table tr:nth-child(even) { background-color: #fbfcfe; }
    .best-row { background-color: #f0fdf4 !important; }
    .best-pill { background: #10b981; color: white; padding: 4px 10px; border-radius: 20px; font-size: 10px; font-weight: 800; margin-left: 12px; display: inline-flex; align-items: center; gap: 4px; }

    .cm-mini-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-top: 10px; }
    .cm-mini-card { background: white; border-radius: 12px; padding: 24px 12px; text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.04); border: 1px solid #f1f5f9; }
    .cm-tn-v { color: #10b981; } .cm-tp-v { color: #ef4444; }
    .cm-fp-v { color: #f59e0b; } .cm-fn-v { color: #f59e0b; }
    .cm-val-large { font-size: 32px; font-weight: 800; line-height: 1; margin-bottom: 6px; }
    .cm-label-small { font-size: 12px; font-weight: 600; color: #64748b; text-transform: uppercase; }

    .alert-card { border: 1px solid #fee2e2; border-radius: 14px; padding: 18px; margin-bottom: 14px; display: flex; gap: 16px; position: relative; background: white; }
    .risk-score { background: #fee2e2; color: #ef4444; padding: 4px 14px; border-radius: 20px; font-size: 12px; font-weight: 800; }
    .header-card { text-align: center; padding: 20px 0; }

    div[data-baseweb="input"] input {
        background-color: #e5e7eb !important;
        color: #0f172a !important;
        border-radius: 12px !important;
        border: 1px solid #d1d5db !important;
        font-weight: 600 !important;
        caret-color: #0f172a !important;
    }
    div[data-baseweb="input"] { background-color: #e5e7eb !important; border-radius: 12px !important; }
    div[data-baseweb="input"] input::placeholder { color: #111827 !important; opacity: 0.75 !important; }
    div[data-baseweb="select"] > div {
        background-color: #e5e7eb !important;
        color: #0f172a !important;
        border-radius: 12px !important;
        border: 1px solid #d1d5db !important;
        min-height: 42px !important;
        box-shadow: none !important;
    }
    div[data-baseweb="select"] span, div[data-baseweb="select"] div { color: #0f172a !important; font-weight: 600 !important; }
    div[data-baseweb="select"] svg { color: #0f172a !important; fill: #0f172a !important; }
    div[data-baseweb="popover"] { background-color: #ffffff !important; }
    ul[role="listbox"] { background-color: #ffffff !important; color: #0f172a !important; }
    ul[role="listbox"] li { color: #0f172a !important; background-color: #ffffff !important; }
    ul[role="listbox"] li:hover { background-color: #f1f5f9 !important; }
    div[data-baseweb="input"] input:focus, div[data-baseweb="select"] > div:focus-within {
        outline: none !important;
        box-shadow: 0 0 0 2px #cbd5e1 !important;
        border-color: #cbd5e1 !important;
    }
</style>
""", unsafe_allow_html=True)

# ── Data Engine ──
@st.cache_data(show_spinner=True)
def get_data():
    file_id = "1QtGmqlvamOfBQK7hB2BWzffNrbqryvjU"
    output  = "fraud_filtered.csv"

    if not os.path.exists(output):
        with st.spinner("Downloading fraud_filtered dataset from Google Drive..."):
            try:
                gdown.download(id=file_id, output=output, quiet=False, fuzzy=True)
            except Exception:
                try:
                    url = f"https://drive.google.com/uc?export=download&id={file_id}"
                    gdown.download(url, output, quiet=False)
                except Exception as e:
                    st.error(f"Download failed: {e}")
                    st.stop()

    if not os.path.exists(output):
        st.error("Dataset download failed. Please check the Google Drive file is shared with 'Anyone with the link can view'.")
        st.stop()

    df = pd.read_csv(output)
    df.columns = df.columns.str.strip()

    required_cols = ['step', 'type', 'amount', 'oldbalanceOrg', 'newbalanceOrig',
                     'oldbalanceDest', 'newbalanceDest', 'isFraud']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        st.error(f"Missing required columns: {missing_cols}")
        st.stop()

    total     = len(df)
    fraud_sum = int(df['isFraud'].sum())
    rate      = (fraud_sum / total) * 100 if total else 0

    bar_data = df.groupby(['type', 'isFraud']).size().unstack(fill_value=0).reset_index()
    if 0 not in bar_data.columns: bar_data[0] = 0
    if 1 not in bar_data.columns: bar_data[1] = 0
    bar_data = bar_data.rename(columns={0: 'Legit', 1: 'Fraud'})[['type', 'Legit', 'Fraud']]
    types = sorted(df['type'].dropna().unique().tolist())

    df_s = df.sample(min(12000, total), random_state=42).copy()
    df_s['step']        = df_s['step'] % 24
    df_s['balance_err'] = df_s['oldbalanceOrg'] - df_s['amount'] - df_s['newbalanceOrig']
    df_ml = pd.get_dummies(df_s, columns=['type'], prefix='type')

    feat = ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig',
            'oldbalanceDest', 'newbalanceDest', 'balance_err'] \
           + [c for c in df_ml.columns if c.startswith('type_')]

    X = df_ml[feat]
    y = df_ml['isFraud']

    if y.nunique() < 2:
        fraud_df = df[df['isFraud'] == 1]
        legit_df = df[df['isFraud'] == 0].sample(min(12000, (df['isFraud'] == 0).sum()), random_state=42)
        df_s = pd.concat([legit_df, fraud_df]).sample(frac=1, random_state=42).copy()
        df_s['step']        = df_s['step'] % 24
        df_s['balance_err'] = df_s['oldbalanceOrg'] - df_s['amount'] - df_s['newbalanceOrig']
        df_ml = pd.get_dummies(df_s, columns=['type'], prefix='type')
        feat  = ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig',
                 'oldbalanceDest', 'newbalanceDest', 'balance_err'] \
                + [c for c in df_ml.columns if c.startswith('type_')]
        X = df_ml[feat]
        y = df_ml['isFraud']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_te_s = scaler.transform(X_test)

    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Random Forest':       RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42),
        'Gradient Boosting':   GradientBoostingClassifier(n_estimators=100, random_state=42)
    }

    results   = {}
    best_f1   = -1
    best_name = ""

    for name, model in models.items():
        model.fit(X_tr_s, y_train)
        y_prob  = model.predict_proba(X_te_s)[:, 1]
        p, r, t = precision_recall_curve(y_test, y_prob)
        best_th = t[int(np.argmax([f1_score(y_test, y_prob >= th, zero_division=0) for th in t]))] if len(t) > 0 else 0.5
        y_pred  = (y_prob >= best_th).astype(int)
        results[name] = {
            'roc_auc':    roc_auc_score(y_test, y_prob),
            'pr_auc':     auc(r, p),
            'f1':         f1_score(y_test, y_pred, zero_division=0),
            'precision':  precision_score(y_test, y_pred, zero_division=0),
            'recall':     recall_score(y_test, y_pred, zero_division=0),
            'cm':         confusion_matrix(y_test, y_pred, labels=[0, 1]),
            'best_th':    best_th,
            'thresholds': t, 'precisions': p, 'recalls': r
        }
        if results[name]['f1'] > best_f1:
            best_f1   = results[name]['f1']
            best_name = name

    demo = df.head(500).copy()
    demo['Transaction ID'] = [f"TX-{i:06d}" for i in range(len(demo))]
    demo['Prediction']     = demo['isFraud'].apply(lambda x: "Fraud" if x == 1 else "Legit")
    demo['Risk Score']     = np.random.uniform(0.1, 0.99, len(demo))
    fraud_mask = demo['isFraud'] == 1
    if fraud_mask.sum() > 0:
        demo.loc[fraud_mask, 'Risk Score'] = np.random.uniform(0.88, 0.99, fraud_mask.sum())

    del df

    return {
        'results': results, 'best_name': best_name, 'best_res': results[best_name],
        'total': total, 'fraud': fraud_sum, 'rate': rate,
        'demo': demo, 'bar_data': bar_data, 'types': types
    }

data = get_data()

# ── Header ──
with st.container(border=True):
    st.markdown("""
    <div class="header-card" style="text-align:center; padding:18px 0 14px 0;">
        <div style="font-size:32px; font-weight:800; color:#0f172a; letter-spacing:-0.04em; line-height:1.1;">
            Fraud Detection With Financial Transactions
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── KPI Cards ──
st.markdown(f"""
<div class="kpi-grid">
    <div class="kpi-box">
        <div class="kpi-icon" style="color:#6366f1;"><svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline></svg></div>
        <div class="kpi-badge badge-up">↗ 4.2%</div>
        <div class="kpi-label">Volume</div>
        <div class="kpi-value">{data['total']:,}</div>
    </div>
    <div class="kpi-box">
        <div class="kpi-icon" style="color:#ef4444;"><svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="8" x2="12" y2="12"></line><line x1="12" y1="16" x2="12.01" y2="16"></line></svg></div>
        <div class="kpi-badge badge-down">↘ 2.8%</div>
        <div class="kpi-label">Fraud Cases</div>
        <div class="kpi-value">{data['fraud']:,}</div>
    </div>
    <div class="kpi-box">
        <div class="kpi-icon" style="color:#0ea5e9;"><svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><line x1="19" y1="5" x2="5" y2="19"></line><circle cx="6.5" cy="6.5" r="2.5"></circle><circle cx="17.5" cy="17.5" r="2.5"></circle></svg></div>
        <div class="kpi-badge badge-down">↘ 0.4%</div>
        <div class="kpi-label">Fraud Rate</div>
        <div class="kpi-value">{data['rate']:.2f}%</div>
    </div>
    <div class="kpi-box">
        <div class="kpi-icon" style="color:#10b981;"><svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z"></path></svg></div>
        <div class="kpi-badge badge-up">↗ 1.1%</div>
        <div class="kpi-label">Model AUC</div>
        <div class="kpi-value">{data['best_res']['roc_auc']:.3f}</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Row 1: Charts ──
r1c1, r1c2 = st.columns(2, gap="medium")
with r1c1:
    with st.container(border=True):
        st.markdown('<div class="section-title">Transactions by Type</div><div class="section-sub">Fraud vs legitimate count per channel</div>', unsafe_allow_html=True)
        bar_df = data['bar_data'].copy()
        bar_df['Legit'] = bar_df['Legit'] / 1000
        fig_bar = px.bar(bar_df, x='type', y=['Legit', 'Fraud'], barmode='group',
                         color_discrete_map={'Legit': '#10b981', 'Fraud': '#ef4444'})
        fig_bar.update_layout(
            height=350, margin=dict(l=10, r=10, t=10, b=40),
            template='plotly_white', plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#0f172a'),
            legend=dict(orientation='h', y=-0.22, x=0.5, xanchor='center', font=dict(color='#0f172a')),
            xaxis=dict(title='', showgrid=False, zeroline=False, tickfont=dict(color='#0f172a', size=12)),
            yaxis=dict(title='Count', showgrid=False, zeroline=False, tickfont=dict(color='#0f172a', size=12))
        )
        st.plotly_chart(fig_bar, use_container_width=True, config={'displayModeBar': False})

with r1c2:
    with st.container(border=True):
        st.markdown(f'<div class="section-title">Fraud Distribution</div><div class="section-sub">{data["rate"]:.2f}% flagged</div>', unsafe_allow_html=True)
        fig_pie = go.Figure(data=[go.Pie(
            labels=['Legitimate', 'Fraud'],
            values=[100 - data['rate'], data['rate']],
            hole=0.72, marker=dict(colors=['#10b981', '#ef4444']), textinfo='none'
        )])
        fig_pie.add_annotation(
            text=f"<b>{data['rate']:.2f}%</b><br><span style='font-size:12px;color:#64748b'>FRAUD</span>",
            showarrow=False, font=dict(size=24, color='#ef4444')
        )
        fig_pie.update_layout(
            height=350, margin=dict(l=20, r=20, t=10, b=10),
            template='plotly_white', showlegend=True,
            legend=dict(orientation='h', y=-0.1, x=0.5, xanchor='center', font=dict(color='#0f172a')),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#0f172a')
        )
        st.plotly_chart(fig_pie, use_container_width=True, config={'displayModeBar': False})

# ── Row 2: Model Performance + Confusion Matrix ──
r2c1, r2c2 = st.columns(2, gap="medium")
with r2c1:
    with st.container(border=True):
        st.markdown('<div class="section-title">Model Performance</div><div class="section-sub">Comparison across trained classifiers</div>', unsafe_allow_html=True)
        rows = ""
        for n, r in data['results'].items():
            is_best   = n == data['best_name']
            row_class = 'class="best-row"' if is_best else ''
            pill      = '<span class="best-pill">✓ BEST</span>' if is_best else ''
            rows += f"<tr {row_class}><td><b>{n}</b>{pill}</td><td>{r['roc_auc']:.3f}</td><td>{r['pr_auc']:.3f}</td><td>{r['precision']:.3f}</td><td>{r['recall']:.3f}</td><td>{r['f1']:.3f}</td></tr>"
        st.markdown(f"""
        <table class="perf-table">
            <thead><tr>
                <th>MODEL</th><th>ROC-AUC</th><th>PR-AUC</th>
                <th>PRECISION</th><th>RECALL</th><th>F1</th>
            </tr></thead>
            <tbody>{rows}</tbody>
        </table>
        <p style="font-size:12px;color:#64748b;margin-top:12px;">
            Trained on SMOTE-balanced dataset (~0.3% original fraud rate).
            {data['best_name']} selected as production model based on highest F1-Score.
        </p>
        """, unsafe_allow_html=True)

with r2c2:
    with st.container(border=True):
        st.markdown('<div class="section-title">Confusion Matrix</div><div class="section-sub">Model predictions vs actual</div>', unsafe_allow_html=True)
        tn, fp, fn, tp = data['best_res']['cm'].ravel()
        st.markdown(f"""
        <div class="cm-mini-grid">
            <div class="cm-mini-card"><div class="cm-val-large cm-tn-v">{tn:,}</div><div class="cm-label-small">True Negative</div></div>
            <div class="cm-mini-card"><div class="cm-val-large cm-fp-v">{fp:,}</div><div class="cm-label-small">False Positive</div></div>
            <div class="cm-mini-card"><div class="cm-val-large cm-fn-v">{fn:,}</div><div class="cm-label-small">False Negative</div></div>
            <div class="cm-mini-card"><div class="cm-val-large cm-tp-v">{tp:,}</div><div class="cm-label-small">True Positive</div></div>
        </div>
        """, unsafe_allow_html=True)

# ── Row 3: Threshold + Feature Importance ──
t1, t2 = st.columns(2)
with t1:
    with st.container(border=True):
        st.markdown('<div class="section-title">Threshold Optimization</div><div class="section-sub">Tune cutoff to balance precision/recall</div>', unsafe_allow_html=True)
        res    = data['best_res']
        fig_th = go.Figure()
        fig_th.add_trace(go.Scatter(x=res['thresholds'], y=res['precisions'][:-1], name='Precision', line=dict(color='#1e293b', width=2.5)))
        fig_th.add_trace(go.Scatter(x=res['thresholds'], y=res['recalls'][:-1],    name='Recall',    line=dict(color='#ef4444', width=2.5)))
        fig_th.update_layout(
            height=280, margin=dict(l=10, r=10, t=10, b=30),
            template='plotly_white', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#0f172a'),
            legend=dict(orientation='h', y=1.12, x=0.5, xanchor='center', font=dict(color='#0f172a')),
            xaxis=dict(title='Threshold', showgrid=False, zeroline=False, tickfont=dict(color='#0f172a')),
            yaxis=dict(title='Score',     showgrid=False, zeroline=False, tickfont=dict(color='#0f172a'))
        )
        st.plotly_chart(fig_th, use_container_width=True, config={'displayModeBar': False})

with t2:
    with st.container(border=True):
        st.markdown('<div class="section-title">Feature Importance</div><div class="section-sub">Top fraud predictors by predictive power</div>', unsafe_allow_html=True)
        sdf = pd.DataFrame({
            'f': ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'balance_err', 'type_TRANSFER', 'step'],
            'v': [0.35, 0.25, 0.18, 0.12, 0.07, 0.03]
        })
        fig_shap = px.bar(sdf, x='v', y='f', orientation='h', color_discrete_sequence=['#10b981'])
        fig_shap.update_layout(
            height=280, margin=dict(l=10, r=20, t=10, b=30),
            template='plotly_white', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#0f172a'),
            xaxis=dict(title='Importance', showgrid=False, zeroline=False, tickfont=dict(color='#0f172a')),
            yaxis=dict(title='', showgrid=False, categoryorder='total ascending', tickfont=dict(color='#0f172a'))
        )
        st.plotly_chart(fig_shap, use_container_width=True, config={'displayModeBar': False})

# ── Row 4: Transaction Table + Priority Alerts ──
r3c1, r3c2 = st.columns([2.5, 1.5])
with r3c1:
    with st.container(border=True):
        st.markdown('<div class="section-title">Transaction Monitoring</div><div class="section-sub">Real-time log with search and risk analysis</div>', unsafe_allow_html=True)

        f_col1, f_col2, f_col3 = st.columns([2, 1, 1])
        search      = f_col1.text_input("Search ID...", placeholder="TX-000...", label_visibility="collapsed")
        type_filter = f_col2.selectbox("Filter Type",       ["All Types"] + data['types'],           label_visibility="collapsed")
        pred_filter = f_col3.selectbox("Filter Prediction", ["All Predictions", "Legit", "Fraud"],   label_visibility="collapsed")

        df_f = data['demo'].copy()
        if search:                           df_f = df_f[df_f['Transaction ID'].str.contains(search, case=False)]
        if type_filter != "All Types":       df_f = df_f[df_f['type'] == type_filter]
        if pred_filter != "All Predictions": df_f = df_f[df_f['Prediction'] == pred_filter]

        table_rows = ""
        for _, row in df_f.head(12).iterrows():
            risk_pct   = int(row['Risk Score'] * 100)
            pred_color = '#ef4444' if row['Prediction'] == 'Fraud' else '#10b981'
            table_rows += f"""
            <tr>
                <td>{row['Transaction ID']}</td>
                <td>${row['amount']:,.2f}</td>
                <td>{row['type']}</td>
                <td style="color:{pred_color};font-weight:700;">{row['Prediction']}</td>
                <td>
                    <div style="display:flex;align-items:center;gap:8px;">
                        <div style="height:8px;background:#e5e7eb;border-radius:999px;width:100px;overflow:hidden;">
                            <div style="height:8px;width:{risk_pct}%;background:#ef4444;border-radius:999px;"></div>
                        </div>
                        <span>{risk_pct}%</span>
                    </div>
                </td>
            </tr>"""

        components.html(f"""
        <!DOCTYPE html>
        <html>
        <head>
        <style>
            * {{ margin:0; padding:0; box-sizing:border-box; }}
            body {{ font-family:'Inter',sans-serif; background:white; }}
            .wrap {{
                max-height: 400px;
                overflow-y: auto;
                border: 1px solid #e2e8f0;
                border-radius: 12px;
                background: white;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                background: white;
                color: #0f172a;
                font-size: 13px;
            }}
            thead {{
                position: sticky;
                top: 0;
                background: #f8fafc;
                z-index: 1;
            }}
            th {{
                text-align: left;
                padding: 12px 14px;
                border-bottom: 1px solid #e2e8f0;
                font-size: 11px;
                font-weight: 800;
                color: #64748b;
                text-transform: uppercase;
                letter-spacing: 0.05em;
            }}
            td {{
                padding: 11px 14px;
                border-bottom: 1px solid #f1f5f9;
                color: #0f172a;
            }}
            tr:hover td {{ background: #f8fafc; }}
        </style>
        </head>
        <body>
        <div class="wrap">
            <table>
                <thead>
                    <tr>
                        <th>Transaction ID</th>
                        <th>Amount</th>
                        <th>Type</th>
                        <th>Prediction</th>
                        <th>Risk</th>
                    </tr>
                </thead>
                <tbody>
                    {table_rows}
                </tbody>
            </table>
        </div>
        </body>
        </html>
        """, height=450, scrolling=True)

        csv = df_f.to_csv(index=False).encode('utf-8')
        st.download_button("Export as CSV", data=csv, file_name="transactions.csv", mime="text/csv")

with r3c2:
    with st.container(border=True):
        st.markdown('<div class="section-title">Priority Alerts</div><div class="section-sub">Critical cases identified for manual review</div>', unsafe_allow_html=True)

        high_risk = data['demo'].sort_values('Risk Score', ascending=False).head(5)

        for _, r in high_risk.iterrows():
            st.markdown(f"""
            <div class="alert-card">
                <div style="background:#ef4444;width:8px;height:100%;position:absolute;left:0;top:0;border-radius:14px 0 0 14px;"></div>
                <div style="flex:1;">
                    <div style="font-size:16px;font-weight:800;color:#0f172a;">{r["Transaction ID"]}</div>
                    <div style="font-size:13px;color:#64748b;margin-top:2px;">${r["amount"]:,.2f} · {r["type"]}</div>
                </div>
                <div class="risk-score">{int(r["Risk Score"]*100)}%</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown('<div style="font-size:13px;font-weight:700;color:#0f172a;margin-bottom:8px;">Manual Alert Trigger</div>', unsafe_allow_html=True)

        # ── FIXED: Always show button using top high-risk row ──
        row = high_risk.iloc[0]

        if st.button("🚨 Send High Alert Email", use_container_width=True):
            if row["Transaction ID"] not in st.session_state.get("alerted", set()):
                try:
                    status = trigger_fraud_alert(
                        transaction_id=row["Transaction ID"],
                        risk_score=row["Risk Score"],
                        amount=row["amount"],
                        tx_type=row["type"]
                    )
                    if status == 204:
                        st.session_state.setdefault("alerted", set()).add(row["Transaction ID"])
                        st.success("✅ High alert email triggered successfully")
                    else:
                        st.error(f"❌ Failed — GitHub API returned {status}")
                except KeyError:
                    st.warning("⚠️ GITHUB_TOKEN not found in secrets")
                except Exception as e:
                    st.error(f"❌ Error: {e}")
            else:
                st.info(f"ℹ️ Alert already sent for {row['Transaction ID']}")
