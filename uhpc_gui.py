"""
UHPC Tensile Properties Prediction GUI
=======================================
A Streamlit web app for predicting tensile strength and strain capacity
of Ultra-High Performance Concrete using trained ML models.

Usage:
    streamlit run uhpc_gui.py

Requirements:
    pip install streamlit joblib numpy scikit-learn lightgbm catboost xgboost ngboost

File structure expected (same folder as this script):
    models/
        Tensile_strength_LightGBM_all.joblib
        Tensile_strength_NGBoost_all.joblib
        Strain_capacity_CatBoost_all.joblib
        Strain_capacity_NGBoost_all.joblib
        Tensile_strength_XGBoost_steel_static.joblib
        Tensile_strength_NGBoost_steel_static.joblib
        Strain_capacity_RF_steel_static.joblib
        Strain_capacity_NGBoost_steel_static.joblib
        Tensile_strength_imputer_all.joblib
        Tensile_strength_imputer_steel_static.joblib
        Strain_capacity_imputer_all.joblib
        Strain_capacity_imputer_steel_static.joblib
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import joblib
import streamlit as st
from pathlib import Path

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title='UHPC ML Predictor',
    page_icon='🏗️',
    layout='wide',
    initial_sidebar_state='expanded',
)

# ── Paths ──────────────────────────────────────────────────────────────────────
MODEL_DIR = Path(__file__).parent

MODELS = {
    'all': {
        'TS':  {'best': 'Tensile_strength_LightGBM_all.joblib',   'name': 'LightGBM',  'ng': 'Tensile_strength_NGBoost_all.joblib'},
        'SC':  {'best': 'Strain_capacity_CatBoost_all.joblib',    'name': 'CatBoost',  'ng': 'Strain_capacity_NGBoost_all.joblib'},
        'imp': {'TS': 'Tensile_strength_imputer_all.joblib', 'SC': 'Strain_capacity_imputer_all.joblib'},
        'n_features': 14,
    },
    'steel': {
        'TS':  {'best': 'Tensile_strength_XGBoost_steel_static.joblib', 'name': 'XGBoost', 'ng': 'Tensile_strength_NGBoost_steel_static.joblib'},
        'SC':  {'best': 'Strain_capacity_RF_steel_static.joblib',       'name': 'RF',      'ng': 'Strain_capacity_NGBoost_steel_static.joblib'},
        'imp': {'TS': 'Tensile_strength_imputer_steel_static.joblib', 'SC': 'Strain_capacity_imputer_steel_static.joblib'},
        'n_features': 13,
    },
}

# ── Feature definitions ────────────────────────────────────────────────────────
# (label, min, max, default, step, help)
FEATURES_ALL = [
    ('Fiber Volume Fraction (%)',        0.5,    6.4,   2.0,   0.1,  'Total fiber volume fraction'),
    ('Straight Fraction',                0.0,    1.0,   1.0,   0.05, 'Fraction of straight fibers (0–1, must sum to 1 with other fractions)'),
    ('Hooked Fraction',                  0.0,    1.0,   0.0,   0.05, 'Fraction of hooked fibers'),
    ('Twisted Fraction',                 0.0,    1.0,   0.0,   0.05, 'Fraction of twisted fibers'),
    ('Wavy Fraction',                    0.0,    1.0,   0.0,   0.05, 'Fraction of wavy/crimped fibers'),
    ('PE Fraction',                      0.0,    1.0,   0.0,   0.05, 'Fraction of PE (polyethylene) fibers'),
    ('Surface Condition',                0,      1,     1,     1,    '0 = Smooth, 1 = Rough/Striated/Treated'),
    ('Fiber Tensile Strength (MPa)',     1100.0, 3800.0,2500.0,50.0, 'Tensile strength of the fiber material'),
    ('Fiber Length (mm)',                6.0,    62.0,  13.0,  0.5,  'Fiber length'),
    ('Fiber Diameter (mm)',              0.019,  0.775, 0.20,  0.005,'Fiber diameter'),
    ('Cross-Sectional Area (mm²)',       381.0,  10000.0,625.0,10.0, 'Specimen cross-sectional area'),
    ('Compressive Strength (MPa)',       84.8,   231.0, 150.0, 1.0,  'Matrix/composite compressive strength (leave blank if unknown — will be KNN imputed)'),
    ('Gauge Length (mm)',                50.0,   350.0, 80.0,  5.0,  'Test specimen gauge length'),
    ('Strain Rate (1/s)',                4.76e-6,37.0,  0.000125,0.0001,'Loading strain rate'),
]

FEATURES_STEEL = [f for f in FEATURES_ALL if f[0] != 'PE Fraction']

# ── Model cache ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    loaded = {}
    for scope, cfg in MODELS.items():
        loaded[scope] = {}
        for target in ['TS', 'SC']:
            loaded[scope][target] = {
                'best': joblib.load(MODEL_DIR / cfg[target]['best']),
                'ng':   joblib.load(MODEL_DIR / cfg[target]['ng']),
                'name': cfg[target]['name'],
            }
            loaded[scope][f'imp_{target}'] = joblib.load(MODEL_DIR / cfg['imp'][target])
    return loaded


def predict(models, scope, features_arr):
    """Run prediction for both targets using best model + NGBoost."""
    results = {}
    for target in ['TS', 'SC']:
        imp   = models[scope][f'imp_{target}']
        X_imp = imp.transform(features_arr)

        # Best model point prediction
        best_model = models[scope][target]['best']
        best_name  = models[scope][target]['name']
        point_pred = best_model.predict(X_imp)[0]

        # NGBoost uncertainty
        ng_model = models[scope][target]['ng']
        dist     = ng_model.pred_dist(X_imp)
        ng_mean  = float(dist.loc[0])
        ng_std   = float(dist.scale[0])

        results[target] = {
            'point':     point_pred,
            'best_name': best_name,
            'ng_mean':   ng_mean,
            'ng_std':    ng_std,
            'lower1':    ng_mean - ng_std,
            'upper1':    ng_mean + ng_std,
            'lower2':    ng_mean - 2 * ng_std,
            'upper2':    ng_mean + 2 * ng_std,
        }
    return results


def draw_uncertainty_bar(label, mean, std, unit, color):
    """Draw a horizontal confidence interval diagram using Streamlit."""
    lower2 = mean - 2 * std
    upper2 = mean + 2 * std
    lower1 = mean - std
    upper1 = mean + std

    st.markdown(f"""
    <div style='margin: 8px 0 4px 0; font-size: 13px; color: #555;'>
        <b>{label}</b> — 68% confidence interval (±1σ) &nbsp;|&nbsp; 95% confidence interval (±2σ)
    </div>
    <div style='position: relative; height: 36px; background: #f0f0f0;
                border-radius: 6px; margin-bottom: 4px; overflow: hidden;'>
        <!-- 2σ band -->
        <div style='position: absolute;
                    left: calc({max(lower2/upper2*100,0):.1f}%);
                    width: calc({min((upper2-lower2)/upper2*100,100):.1f}%);
                    height: 100%; background: {color}33; border-radius: 4px;'></div>
        <!-- 1σ band -->
        <div style='position: absolute;
                    left: calc({max(lower1/upper2*100,0):.1f}%);
                    width: calc({min((upper1-lower1)/upper2*100,100):.1f}%);
                    height: 100%; background: {color}88; border-radius: 4px;'></div>
        <!-- Mean marker -->
        <div style='position: absolute;
                    left: calc({mean/upper2*100:.1f}% - 2px);
                    width: 4px; height: 100%;
                    background: {color}; border-radius: 2px;'></div>
    </div>
    <div style='display: flex; justify-content: space-between;
                font-size: 11px; color: #777; margin-bottom: 12px;'>
        <span>{lower2:.3f} {unit}</span>
        <span>{lower1:.3f}</span>
        <span><b>{mean:.3f} {unit}</b></span>
        <span>{upper1:.3f}</span>
        <span>{upper2:.3f} {unit}</span>
    </div>
    """, unsafe_allow_html=True)


# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2rem; font-weight: 700;
        color: #1a3a5c; margin-bottom: 0;
    }
    .sub-header {
        font-size: 1rem; color: #666; margin-top: 0; margin-bottom: 1.5rem;
    }
    .result-card {
        background: #f8f9fa; border-radius: 10px;
        padding: 20px; margin: 10px 0;
        border-left: 5px solid #1a3a5c;
    }
    .result-value {
        font-size: 2.2rem; font-weight: 700; color: #1a3a5c;
    }
    .result-label {
        font-size: 0.9rem; color: #888; margin-bottom: 4px;
    }
    .model-badge {
        display: inline-block; background: #1a3a5c; color: white;
        border-radius: 4px; padding: 2px 8px; font-size: 0.8rem;
        margin-bottom: 8px;
    }
    .warning-box {
        background: #fff3cd; border-left: 4px solid #ffc107;
        padding: 10px 14px; border-radius: 4px; font-size: 0.9rem;
    }
    .stTabs [data-baseweb='tab'] { font-size: 1rem; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


# ── Main App ───────────────────────────────────────────────────────────────────
def main():
    # Header
    st.markdown('<p class="main-header">🏗️ UHPC Tensile Properties Predictor</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Predict tensile strength and strain capacity of Ultra-High Performance Concrete using ML models</p>', unsafe_allow_html=True)

    # Load models
    try:
        models = load_models()
    except Exception as e:
        st.error(f'❌ Could not load model files. Make sure all .joblib files are in the `models/` folder.\n\nError: {e}')
        st.stop()

    # Tabs
    tab_all, tab_steel = st.tabs([
        'UHPC with Steel and PE fibers at static and Dynamic strain rate',
        'UHPC with Steel fibers at static strain rate',
    ])

    # ── TAB 1: All data ────────────────────────────────────────────────────────
    with tab_all:
        st.markdown('### Input Parameters')
        st.caption('Dataset includes steel fibers, PE fibers, static and dynamic strain rates. Best models: **LightGBM** (tensile strength) and **CatBoost** (strain capacity).')

        col_left, col_right = st.columns([1, 1])
        inputs_all = {}

        with col_left:
            st.markdown('**Fiber Properties**')
            inputs_all['Fiber Volume Fraction (%)']    = st.number_input('Volume Fraction (%)',    0.5, 6.4,   2.0,   0.1,  key='all_vf',  help='Total fiber volume fraction')
            inputs_all['Straight Fraction']            = st.number_input('Straight Fraction',      0.0, 1.0,   1.0,   0.05, key='all_sf',  help='Must sum to 1 with other geometry fractions')
            inputs_all['Hooked Fraction']              = st.number_input('Hooked Fraction',        0.0, 1.0,   0.0,   0.05, key='all_hf',  help='Fraction of hooked fibers')
            inputs_all['Twisted Fraction']             = st.number_input('Twisted Fraction',       0.0, 1.0,   0.0,   0.05, key='all_tf',  help='Fraction of twisted fibers')
            inputs_all['Wavy Fraction']                = st.number_input('Wavy Fraction',          0.0, 1.0,   0.0,   0.05, key='all_wf',  help='Fraction of wavy/crimped fibers')
            inputs_all['PE Fraction']                  = st.number_input('PE Fraction',            0.0, 1.0,   0.0,   0.05, key='all_pef', help='Fraction of PE (polyethylene) fibers')
            inputs_all['Surface Condition']            = st.selectbox('Surface Condition', [0, 1], format_func=lambda x: '0 — Smooth' if x == 0 else '1 — Rough/Striated/Treated', key='all_sc')
            inputs_all['Fiber Tensile Strength (MPa)'] = st.number_input('Fiber Tensile Strength (MPa)', 1100.0, 3800.0, 2500.0, 50.0, key='all_fts')
            inputs_all['Fiber Length (mm)']            = st.number_input('Fiber Length (mm)',      6.0,  62.0,  13.0,  0.5,  key='all_fl')
            inputs_all['Fiber Diameter (mm)']          = st.number_input('Fiber Diameter (mm)',    0.019,0.775, 0.20,  0.005,key='all_fd')

        with col_right:
            st.markdown('**Matrix, Specimen & Loading Properties**')
            inputs_all['Cross-Sectional Area (mm²)']   = st.number_input('Cross-Sectional Area (mm²)', 381.0, 10000.0, 625.0, 10.0, key='all_csa')
            inputs_all['Compressive Strength (MPa)']   = st.number_input('Compressive Strength (MPa)', 84.8, 231.0, 150.0, 1.0,  key='all_cs',  help='If unknown, enter 0 — will be KNN imputed')
            inputs_all['Gauge Length (mm)']            = st.number_input('Gauge Length (mm)',           50.0, 350.0, 80.0,  5.0,  key='all_gl')
            inputs_all['Strain Rate (1/s)']            = st.number_input('Strain Rate (1/s)',           4.76e-6, 37.0, 0.000125, 0.0001, key='all_sr', format='%f')

            # Fraction sum check
            frac_sum = (inputs_all['Straight Fraction'] + inputs_all['Hooked Fraction'] +
                        inputs_all['Twisted Fraction']  + inputs_all['Wavy Fraction'] +
                        inputs_all['PE Fraction'])
            if abs(frac_sum - 1.0) > 0.01:
                st.markdown(f'<div class="warning-box">⚠️ Geometry fractions sum to <b>{frac_sum:.2f}</b> — they should sum to <b>1.0</b></div>', unsafe_allow_html=True)
            else:
                st.success(f'✅ Geometry fractions sum to {frac_sum:.2f}')

        if st.button('🔮 Predict', key='btn_all', type='primary', use_container_width=True):
            X = np.array([[
                inputs_all['Fiber Volume Fraction (%)'],
                inputs_all['Straight Fraction'],
                inputs_all['Hooked Fraction'],
                inputs_all['Twisted Fraction'],
                inputs_all['Wavy Fraction'],
                inputs_all['PE Fraction'],
                inputs_all['Surface Condition'],
                inputs_all['Fiber Tensile Strength (MPa)'],
                inputs_all['Fiber Length (mm)'],
                inputs_all['Fiber Diameter (mm)'],
                inputs_all['Cross-Sectional Area (mm²)'],
                inputs_all['Compressive Strength (MPa)'] if inputs_all['Compressive Strength (MPa)'] > 0 else np.nan,
                inputs_all['Gauge Length (mm)'],
                inputs_all['Strain Rate (1/s)'],
            ]])

            with st.spinner('Running predictions...'):
                res = predict(models, 'all', X)

            st.markdown('---')
            st.markdown('### Predictions')

            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"""
                <div class="result-card">
                    <div class="result-label">Tensile Strength</div>
                    <div class="model-badge">{res['TS']['best_name']} (Best Model)</div>
                    <div class="result-value">{res['TS']['point']:.2f} <span style='font-size:1rem;color:#888'>MPa</span></div>
                </div>
                """, unsafe_allow_html=True)
                st.markdown('**NGBoost Uncertainty Estimate**')
                draw_uncertainty_bar('Tensile Strength', res['TS']['ng_mean'], res['TS']['ng_std'], 'MPa', '#1a3a5c')

            with c2:
                st.markdown(f"""
                <div class="result-card">
                    <div class="result-label">Strain Capacity</div>
                    <div class="model-badge">{res['SC']['best_name']} (Best Model)</div>
                    <div class="result-value">{res['SC']['point']:.4f} <span style='font-size:1rem;color:#888'>%</span></div>
                </div>
                """, unsafe_allow_html=True)
                st.markdown('**NGBoost Uncertainty Estimate**')
                draw_uncertainty_bar('Strain Capacity', res['SC']['ng_mean'], res['SC']['ng_std'], '%', '#2e7d32')

            # Summary table
            st.markdown('#### Summary')
            st.table({
                'Property':        ['Tensile Strength (MPa)', 'Strain Capacity (%)'],
                'Best Model':      [res['TS']['best_name'], res['SC']['best_name']],
                'Prediction':      [f"{res['TS']['point']:.3f}", f"{res['SC']['point']:.4f}"],
                'NGBoost Mean':    [f"{res['TS']['ng_mean']:.3f}", f"{res['SC']['ng_mean']:.4f}"],
                '±1σ Range':       [f"{res['TS']['lower1']:.3f} – {res['TS']['upper1']:.3f}", f"{res['SC']['lower1']:.4f} – {res['SC']['upper1']:.4f}"],
                '±2σ Range':       [f"{res['TS']['lower2']:.3f} – {res['TS']['upper2']:.3f}", f"{res['SC']['lower2']:.4f} – {res['SC']['upper2']:.4f}"],
            })

    # ── TAB 2: Steel + Static ──────────────────────────────────────────────────
    with tab_steel:
        st.markdown('### Input Parameters')
        st.caption('Steel fiber UHPC under static loading only. Best models: **XGBoost** (tensile strength) and **RF** (strain capacity). Note: PE Fraction not applicable here.')

        col_left2, col_right2 = st.columns([1, 1])
        inputs_steel = {}

        with col_left2:
            st.markdown('**Fiber Properties**')
            inputs_steel['Fiber Volume Fraction (%)']    = st.number_input('Volume Fraction (%)',    0.5,   6.4,   2.0,   0.1,  key='ss_vf')
            inputs_steel['Straight Fraction']            = st.number_input('Straight Fraction',      0.0,   1.0,   1.0,   0.05, key='ss_sf',  help='Must sum to 1 with other geometry fractions')
            inputs_steel['Hooked Fraction']              = st.number_input('Hooked Fraction',        0.0,   1.0,   0.0,   0.05, key='ss_hf')
            inputs_steel['Twisted Fraction']             = st.number_input('Twisted Fraction',       0.0,   1.0,   0.0,   0.05, key='ss_tf')
            inputs_steel['Wavy Fraction']                = st.number_input('Wavy Fraction',          0.0,   1.0,   0.0,   0.05, key='ss_wf')
            inputs_steel['Surface Condition']            = st.selectbox('Surface Condition', [0, 1], format_func=lambda x: '0 — Smooth' if x == 0 else '1 — Rough/Striated/Treated', key='ss_sc')
            inputs_steel['Fiber Tensile Strength (MPa)'] = st.number_input('Fiber Tensile Strength (MPa)', 1100.0, 3800.0, 2500.0, 50.0, key='ss_fts')
            inputs_steel['Fiber Length (mm)']            = st.number_input('Fiber Length (mm)',      6.0,   62.0,  13.0,  0.5,  key='ss_fl')
            inputs_steel['Fiber Diameter (mm)']          = st.number_input('Fiber Diameter (mm)',    0.019, 0.775, 0.20,  0.005,key='ss_fd')

        with col_right2:
            st.markdown('**Matrix, Specimen & Loading Properties**')
            inputs_steel['Cross-Sectional Area (mm²)']  = st.number_input('Cross-Sectional Area (mm²)', 381.0, 10000.0, 625.0, 10.0, key='ss_csa')
            inputs_steel['Compressive Strength (MPa)']  = st.number_input('Compressive Strength (MPa)', 84.8, 231.0, 150.0, 1.0, key='ss_cs', help='If unknown, enter 0 — will be KNN imputed')
            inputs_steel['Gauge Length (mm)']           = st.number_input('Gauge Length (mm)',  50.0, 350.0, 80.0,  5.0,  key='ss_gl')
            inputs_steel['Strain Rate (1/s)']           = st.number_input('Strain Rate (1/s)', 4.76e-6, 0.001, 0.000125, 0.0001, key='ss_sr', format='%f', help='Static range only: < 0.001 s⁻¹')

            frac_sum2 = (inputs_steel['Straight Fraction'] + inputs_steel['Hooked Fraction'] +
                         inputs_steel['Twisted Fraction']  + inputs_steel['Wavy Fraction'])
            if abs(frac_sum2 - 1.0) > 0.01:
                st.markdown(f'<div class="warning-box">⚠️ Geometry fractions sum to <b>{frac_sum2:.2f}</b> — they should sum to <b>1.0</b></div>', unsafe_allow_html=True)
            else:
                st.success(f'✅ Geometry fractions sum to {frac_sum2:.2f}')

        if st.button('🔮 Predict', key='btn_steel', type='primary', use_container_width=True):
            X2 = np.array([[
                inputs_steel['Fiber Volume Fraction (%)'],
                inputs_steel['Straight Fraction'],
                inputs_steel['Hooked Fraction'],
                inputs_steel['Twisted Fraction'],
                inputs_steel['Wavy Fraction'],
                inputs_steel['Surface Condition'],
                inputs_steel['Fiber Tensile Strength (MPa)'],
                inputs_steel['Fiber Length (mm)'],
                inputs_steel['Fiber Diameter (mm)'],
                inputs_steel['Cross-Sectional Area (mm²)'],
                inputs_steel['Compressive Strength (MPa)'] if inputs_steel['Compressive Strength (MPa)'] > 0 else np.nan,
                inputs_steel['Gauge Length (mm)'],
                inputs_steel['Strain Rate (1/s)'],
            ]])

            with st.spinner('Running predictions...'):
                res2 = predict(models, 'steel', X2)

            st.markdown('---')
            st.markdown('### Predictions')

            c3, c4 = st.columns(2)
            with c3:
                st.markdown(f"""
                <div class="result-card">
                    <div class="result-label">Tensile Strength</div>
                    <div class="model-badge">{res2['TS']['best_name']} (Best Model)</div>
                    <div class="result-value">{res2['TS']['point']:.2f} <span style='font-size:1rem;color:#888'>MPa</span></div>
                </div>
                """, unsafe_allow_html=True)
                st.markdown('**NGBoost Uncertainty Estimate**')
                draw_uncertainty_bar('Tensile Strength', res2['TS']['ng_mean'], res2['TS']['ng_std'], 'MPa', '#1a3a5c')

            with c4:
                st.markdown(f"""
                <div class="result-card">
                    <div class="result-label">Strain Capacity</div>
                    <div class="model-badge">{res2['SC']['best_name']} (Best Model)</div>
                    <div class="result-value">{res2['SC']['point']:.4f} <span style='font-size:1rem;color:#888'>%</span></div>
                </div>
                """, unsafe_allow_html=True)
                st.markdown('**NGBoost Uncertainty Estimate**')
                draw_uncertainty_bar('Strain Capacity', res2['SC']['ng_mean'], res2['SC']['ng_std'], '%', '#2e7d32')

            st.markdown('#### Summary')
            st.table({
                'Property':        ['Tensile Strength (MPa)', 'Strain Capacity (%)'],
                'Best Model':      [res2['TS']['best_name'], res2['SC']['best_name']],
                'Prediction':      [f"{res2['TS']['point']:.3f}", f"{res2['SC']['point']:.4f}"],
                'NGBoost Mean':    [f"{res2['TS']['ng_mean']:.3f}", f"{res2['SC']['ng_mean']:.4f}"],
                '±1σ Range':       [f"{res2['TS']['lower1']:.3f} – {res2['TS']['upper1']:.3f}", f"{res2['SC']['lower1']:.4f} – {res2['SC']['upper1']:.4f}"],
                '±2σ Range':       [f"{res2['TS']['lower2']:.3f} – {res2['TS']['upper2']:.3f}", f"{res2['SC']['lower2']:.4f} – {res2['SC']['upper2']:.4f}"],
            })


    st.caption('Developed as part of PhD research — University of Connecticut | Advisor: Dr. Kay Wille')


if __name__ == '__main__':
    main()
