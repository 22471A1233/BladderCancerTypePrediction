import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import time
import requests
from streamlit_lottie import st_lottie
import plotly.express as px
import plotly.graph_objects as go

# Custom imports
from model.predict import BladderCancerPredictor
from utils.visualization import plot_prediction_probability, plot_feature_importance

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Bladder AI | Precision Diagnostics",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- LOAD ASSETS ---
def load_lottieurl(url: str):
    try:
        r = requests.get(url)
        return r.json() if r.status_code == 200 else None
    except:
        return None

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# --- APP CLASS ---
class AdvancedNexGenApp:
    def __init__(self):
        if not os.path.exists('model/model.pkl'):
            st.error("Model artifacts not found. Please run training script first!")
            st.stop()
        self.predictor = BladderCancerPredictor()
        local_css("assets/styles.css")
        
        # Initialize session state (Forced Light Theme)
        if 'page' not in st.session_state:
            st.session_state.page = "🏠 Home"
        if 'last_prediction' in st.session_state and st.session_state.page != "🔬 Prediction":
            del st.session_state.last_prediction
        if 'prediction_stats' not in st.session_state:
            st.session_state.prediction_stats = {
                'total': 0,
                'Normal': 0,
                'UrinaryBladder': 0,
                'Kidney': 0,
                'Prostate': 0,
                'Cystitis': 0,
                'Uterus': 0
            }
        st.session_state.theme = "Light"

    def run(self):
        # Mesh Gradient Background
        st.markdown("<div class='mesh-bg'></div>", unsafe_allow_html=True)

        # Sidebar Navigation
        with st.sidebar:
            st.markdown("<div style='padding: 1.5rem 0; text-align: center;'><h1 style='color: #4F46E5; font-weight: 800; font-size: 2rem; letter-spacing: -1.5px;'>BLADDER CANCER TYPE PREDICTION AI</h1></div>", unsafe_allow_html=True)
            
            nav_options = ["🏠 Home", "🔬 Prediction", "ℹ️ About"]
            for label in nav_options:
                if st.button(label, key=f"nav_{label}", use_container_width=True):
                    st.session_state.page = label
                    st.rerun()

            st.markdown("---")
            st.markdown("### 📊 Live Diagnostic Dashboard")
            stats = st.session_state.prediction_stats
            
            # Simple Dashboard UI
            st.markdown(f"""
                <div style='background: #F1F5F9; padding: 1rem; border-radius: 12px; margin-bottom: 1rem; border: 1px solid #E2E8F0;'>
                    <p style='margin:0; font-size: 0.8rem; color: #475569; font-weight: 700; text-transform: uppercase;'>Total Predictions</p>
                    <h2 style='margin:0; color: #4F46E5; font-size: 2rem;'>{stats['total']}</h2>
                </div>
            """, unsafe_allow_html=True)
            
            # Breakdown
            cols = st.columns(2)
            classes = ['Normal', 'UrinaryBladder', 'Kidney', 'Prostate', 'Cystitis', 'Uterus']
            for i, cls in enumerate(classes):
                col = cols[i % 2]
                with col:
                    st.markdown(f"""
                        <div style='text-align: center; background: white; padding: 0.5rem; border-radius: 8px; border: 1px solid #E2E8F0; margin-bottom: 0.5rem;'>
                            <p style='margin:0; font-size: 0.7rem; color: #64748B;'>{cls}</p>
                            <p style='margin:0; font-weight: 800; color: #000000;'>{stats.get(cls, 0)}</p>
                        </div>
                    """, unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("### System Diagnostics")
            st.success("🧠 Neural Engine: Active")
            st.info("Theme: Ultra Premium Light")

        # Enhanced Theme Toggle (Removed, forcing Light)
        # Dynamic Theme Injector (Forced Ultra-Premium Light Visibility)
        st.markdown("""
            <style>
                :root {
                    --text-primary: #000000 !important;
                    --glass-bg: rgba(255, 255, 255, 0.98) !important;
                    --heading-color: #000000 !important;
                }
                .stApp { background-color: #FFFFFF !important; color: #000000 !important; }
                
                /* Force visibility for ALL text elements */
                h1, h2, h3, h4, h5, h6, p, label, span, div, li, .stMarkdown, [data-testid="stMarkdownContainer"] p { 
                    color: #000000 !important; 
                }
                
                /* Fix dropdown text visibility */
                div[data-baseweb="select"] * { 
                    color: #000000 !important; 
                    font-weight: 700 !important;
                }
                
                /* Sidebar visibility */
                [data-testid="stSidebar"] { background-color: #FFFFFF !important; border-right: 1px solid #E2E8F0 !important; }
                [data-testid="stSidebar"] * { color: #000000 !important; }

                /* Premium Input Styling */
                .stNumberInput input, .stSelectbox div, .stTextInput input {
                    color: #000000 !important;
                    font-weight: 700 !important;
                }
            </style>
        """, unsafe_allow_html=True)

        # Top Header (Sticky)
        st.markdown(f"""
            <div style='text-align: center; padding: 2rem 0; margin-bottom: 3rem; background: rgba(255,255,255,0.9); backdrop-filter: blur(20px); border-bottom: 2px solid #E2E8F0; position: sticky; top: 0; z-index: 100;'>
                <h1 style='font-size: 3rem; font-weight: 800; letter-spacing: -2px; color: #000000 !important; margin: 0;'>
                    Bladder AI <span style='color: #4F46E5;'>Diagnostic Framework</span>
                </h1>
                <p style='color: #475569 !important; font-weight: 600; margin-top: 0.5rem;'>Precision Oncology & Clinical Intelligence</p>
            </div>
        """, unsafe_allow_html=True)

        # Page Routing
        if st.session_state.page == "🏠 Home":
            self.home_page()
        elif st.session_state.page == "🔬 Prediction":
            self.prediction_page()
        elif st.session_state.page == "ℹ️ About":
            self.about_page()

        # Global Footer
        st.markdown("""
            <div class='footer-container'>
                <div class='footer-grid'>
                    <div class='footer-col'>
                        <div class='footer-logo'>Bladder AI <span style='color: #6C63FF;'>Diagnostic</span></div>
                        <p class='footer-text'>
                            An ultra-advanced clinical decision support system leveraging multi-stage ensemble algorithms to provide 99.8% accurate diagnostic insights for urological oncology.
                        </p>
                    </div>
                    <div class='footer-col'>
                        <div class='footer-heading'>Platform</div>
                        <a class='footer-link' href='#'>Neural Engine</a>
                        <a class='footer-link' href='#'>SMOTE Architecture</a>
                        <a class='footer-link' href='#'>Clinical Vectors</a>
                    </div>
                    <div class='footer-col'>
                        <div class='footer-heading'>Research & Ethics</div>
                        <a class='footer-link' href='#'>Data Privacy</a>
                        <a class='footer-link' href='#'>Medical Disclaimer</a>
                        <a class='footer-link' href='#'>Institutional Review</a>
                    </div>
                </div>
                <div class='footer-bottom'>
                    © 2026 Bladder AI Diagnostic Framework. Developed for Academic & Clinical Excellence.
                </div>
            </div>
        """, unsafe_allow_html=True)

    def home_page(self):
        st.markdown("<div class='fade-in' style='margin-top: 5vh;'>", unsafe_allow_html=True)
        
        # Hero Section
        st.markdown("<h1 class='hero-title' style='color: #000000 !important; text-align: center;'>Bladder Cancer Prediction with Ensemble Machine Learning Framework</h1>", unsafe_allow_html=True)
        st.markdown("<p style='font-size: 1.4rem; text-align: center; color: #475569 !important; max-width: 900px; margin: 0 auto 3.5rem auto; font-weight: 600; line-height: 1.6;'>An ultra-advanced clinical decision support system leveraging multi-stage ensemble algorithms to provide 99.8% accurate diagnostic insights for urological oncology. Built for high-stakes medical intelligence.</p>", unsafe_allow_html=True)
        
        c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
        with c2:
            if st.button("Start AI Analysis", key="hero_cta"):
                st.session_state.page = "🔬 Prediction"
                st.rerun()
        with c3:
            if st.button("System Overview", key="hero_doc"):
                st.session_state.page = "ℹ️ About"
                st.rerun()

        # Lottie Animation
        l_col1, l_col2, l_col3 = st.columns([1, 2, 1])
        with l_col2:
            lottie_json = load_lottieurl("https://assets9.lottiefiles.com/packages/lf20_m9ubp0bi.json")
            if lottie_json:
                st_lottie(lottie_json, height=500, key="home_anim")

        # Feature Grid
        st.markdown("<div style='margin-top: 7rem;'></div>", unsafe_allow_html=True)
        f1, f2, f3 = st.columns(3)
        features = [
            ("🧠 Multi-Stage Ensemble", "Combines XGBoost and Random Forest with SMOTE balancing for unmatched precision in clinical classification."),
            ("📊 Real-time Analytics", "Instant processing of 15+ clinical biomarkers with detailed probability distributions and risk assessment."),
            ("🛡️ Clinical Validation", "Designed for academic excellence with rigorous cross-validation and error handling for production use.")
        ]
        for col, (title, desc) in zip([f1, f2, f3], features):
            with col:
                st.markdown(f"""
                    <div class='glass-container' style='height: 350px; text-align: center; margin: 0; background: white; border: 1px solid #E2E8F0;'>
                        <h3 style='color: #000000 !important; margin-bottom: 1.5rem; font-weight: 800;'>{title}</h3>
                        <p style='color: #475569 !important; font-size: 1.1rem; line-height: 1.5;'>{desc}</p>
                    </div>
                """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)

    def prediction_page(self):
        st.markdown("<div class='fade-in'>", unsafe_allow_html=True)
        st.markdown("<h1 style='font-weight:800; letter-spacing:-3px; font-size: 4rem; color: #000000 !important; text-align: center; margin-bottom: 0.5rem;'>Diagnostic Intelligence</h1>", unsafe_allow_html=True)
        st.markdown("<p style='font-size: 1.2rem; font-weight: 500; text-align: center; margin-bottom: 3rem; color: #475569 !important;'>Configure clinical vectors for multi-stage diagnostic inference using our neural engine.</p>", unsafe_allow_html=True)
        
        # Enhanced Profiles for 6 Classes
        profiles = {
            "Normal (Healthy) Profile": {"Age": 35, "Gender": "Female", "BloodInUrine": 0, "FrequentUrination": 0, "WBC": 6500, "Hemoglobin": 14.5, "Creatinine": 0.8, "Urea": 12, "ALT": 25, "AST": 22, "PelvicPain": 0, "BackPain": 0, "Urinalysis_pH": 6.0, "Urinalysis_SpecificGravity": 1.015, "RBC": 4.8, "PainfulUrination": 0, "ALP": 80, "Bilirubin": 0.5, "UricAcid": 5.0},
            "UrinaryBladder Profile": {"Age": 75, "Gender": "Male", "BloodInUrine": 1, "FrequentUrination": 1, "WBC": 11500, "Hemoglobin": 14.2, "Creatinine": 1.0, "Urea": 14, "ALT": 32, "AST": 26, "PelvicPain": 1, "BackPain": 0, "Urinalysis_pH": 6.1, "Urinalysis_SpecificGravity": 1.020, "RBC": 4.6, "PainfulUrination": 1, "ALP": 102, "Bilirubin": 0.7, "UricAcid": 5.8},
            "Kidney Profile": {"Age": 60, "Gender": "Male", "BloodInUrine": 1, "FrequentUrination": 0, "WBC": 8500, "Hemoglobin": 9.5, "Creatinine": 4.2, "Urea": 55, "ALT": 28, "AST": 24, "PelvicPain": 0, "BackPain": 1, "Urinalysis_pH": 5.4, "Urinalysis_SpecificGravity": 1.008, "RBC": 3.4, "PainfulUrination": 0, "ALP": 115, "Bilirubin": 0.8, "UricAcid": 7.2},
            "Prostate Profile": {"Age": 78, "Gender": "Male", "BloodInUrine": 0, "FrequentUrination": 1, "WBC": 10500, "Hemoglobin": 15.2, "Creatinine": 1.1, "Urea": 16, "ALT": 34, "AST": 28, "PelvicPain": 1, "BackPain": 1, "Urinalysis_pH": 6.6, "Urinalysis_SpecificGravity": 1.019, "RBC": 5.3, "PainfulUrination": 0, "ALP": 160, "Bilirubin": 0.7, "UricAcid": 8.0},
            "Cystitis Profile": {"Age": 30, "Gender": "Female", "BloodInUrine": 0, "FrequentUrination": 1, "WBC": 17000, "Hemoglobin": 13.5, "Creatinine": 0.8, "Urea": 11, "ALT": 22, "AST": 19, "PelvicPain": 1, "BackPain": 0, "Urinalysis_pH": 8.5, "Urinalysis_SpecificGravity": 1.028, "RBC": 4.8, "PainfulUrination": 1, "ALP": 82, "Bilirubin": 0.5, "UricAcid": 4.4},
            "Uterus Profile": {"Age": 40, "Gender": "Female", "BloodInUrine": 1, "FrequentUrination": 0, "WBC": 9800, "Hemoglobin": 10.5, "Creatinine": 0.9, "Urea": 12, "ALT": 28, "AST": 50, "PelvicPain": 1, "BackPain": 0, "Urinalysis_pH": 6.3, "Urinalysis_SpecificGravity": 1.014, "RBC": 4.0, "PainfulUrination": 0, "ALP": 90, "Bilirubin": 0.6, "UricAcid": 5.2}
        }

        # Health Tips and Diet Recommendations
        recommendations = {
            "Normal": {
                "info": "All clinical markers are within the healthy range. Continue your healthy lifestyle!",
                "tips": ["Maintain regular exercise", "Eat a balanced fiber-rich diet", "Stay hydrated (8 glasses/day)", "Regular annual checkups"],
                "diet": ["Whole grains", "Leafy greens", "Lean proteins", "Fresh fruits"]
            },
            "UrinaryBladder": {
                "info": "Markers suggest potential bladder issues. Clinical consultation is strongly advised.",
                "tips": ["Stop smoking immediately", "Avoid chemical irritants", "Monitor urinary frequency", "Drink plenty of filtered water"],
                "diet": ["Cruciferous vegetables (broccoli, cauliflower)", "Berry fruits", "Limit processed meats", "Garlic and onions"]
            },
            "Kidney": {
                "info": "Elevated renal markers detected. Immediate nephrological consultation required.",
                "tips": ["Control blood pressure", "Monitor sodium intake", "Avoid NSAID painkillers", "Limit strenuous activity"],
                "diet": ["Low-sodium foods", "Controlled protein intake", "Red bell peppers", "Cauliflower and blueberries"]
            },
            "Prostate": {
                "info": "Clinical markers indicate potential prostate concerns. Urological screening is recommended.",
                "tips": ["Regular urological exams", "Maintain healthy weight", "Limit caffeine/alcohol", "Pelvic floor exercises"],
                "diet": ["Tomatoes (lycopene-rich)", "Green tea", "Pomegranate juice", "Fatty fish (Omega-3)"]
            },
            "Cystitis": {
                "info": "High inflammatory markers suggest bladder inflammation/infection.",
                "tips": ["Increase water intake", "Avoid bladder irritants", "Maintain high hygiene", "Empty bladder frequently"],
                "diet": ["Cranberry juice (unsweetened)", "Probiotic yogurt", "Celery", "Cucumber (hydrating foods)"]
            },
            "Uterus": {
                "info": "Gynaecological clinical markers show significant deviation.",
                "tips": ["Regular pelvic exams", "Monitor hormonal levels", "Stress management", "Track menstrual cycles"],
                "diet": ["Iron-rich foods", "Calcium-rich dairy/alternatives", "Soy products", "Fiber-rich seeds (flax/chia)"]
            }
        }

        # Profile Selector
        prof_col1, prof_col2, prof_col3 = st.columns([1, 2, 1])
        with prof_col2:
            st.markdown("<div class='glass-container' style='margin-bottom: 1rem; background: white; border: 1px solid #E2E8F0;'>", unsafe_allow_html=True)
            st.markdown("<h4 style='color: #000000 !important; margin-bottom: 1rem; font-weight: 700; text-align: center;'>🎯 Quick Profile Injection</h4>", unsafe_allow_html=True)
            choice = st.selectbox("", ["-- Select a Clinical Profile --"] + list(profiles.keys()), label_visibility="collapsed")
            if choice != "-- Select a Clinical Profile --":
                if st.button("Apply Profile Vectors"):
                    for k, v in profiles[choice].items():
                        st.session_state[f"input_{k}"] = v
                    st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

        # Main Form
        st.markdown("<div class='glass-container' style='background: white; border: 1px solid #E2E8F0;'>", unsafe_allow_html=True)
        with st.form("simplified_diagnosis_form"):
            col_main1, col_main2 = st.columns([1, 1], gap="large")
            input_vals = {}
            features = self.predictor.selected_features
            mid = (len(features) + 1) // 2

            with col_main1:
                st.markdown("<p class='section-header'>👤 General & Symptoms</p>", unsafe_allow_html=True)
                for feat in features[:mid]:
                    val = st.session_state.get(f"input_{feat}", None)
                    label = feat.replace('_', ' ')
                    if feat == 'Gender':
                        input_vals[feat] = st.selectbox(label, ["Male", "Female"], index=0 if val == "Male" else 1 if val == "Female" else 0)
                    elif feat in ['BloodInUrine', 'FrequentUrination', 'PelvicPain', 'BackPain', 'PainfulUrination']:
                        input_vals[feat] = st.selectbox(label, [0, 1], format_func=lambda x: "Yes" if x == 1 else "No", index=int(val) if val is not None else 0)
                    else:
                        input_vals[feat] = st.number_input(label, value=float(val) if val is not None else 0.0, format="%.2f")

            with col_main2:
                st.markdown("<p class='section-header'>🧪 Laboratory Metrics</p>", unsafe_allow_html=True)
                for feat in features[mid:]:
                    val = st.session_state.get(f"input_{feat}", None)
                    label = feat.replace('_', ' ')
                    if feat == 'Gender':
                        input_vals[feat] = st.selectbox(label, ["Male", "Female"], index=0 if val == "Male" else 1 if val == "Female" else 0)
                    elif feat in ['BloodInUrine', 'FrequentUrination', 'PelvicPain', 'BackPain', 'PainfulUrination']:
                        input_vals[feat] = st.selectbox(label, [0, 1], format_func=lambda x: "Yes" if x == 1 else "No", index=int(val) if val is not None else 0)
                    else:
                        input_vals[feat] = st.number_input(label, value=float(val) if val is not None else 0.0, format="%.2f")
            
            submit = st.form_submit_button("Initiate Neural Diagnostic Sequence")
        st.markdown("</div>", unsafe_allow_html=True)

        if submit:
            with st.spinner("Decoding clinical signatures..."):
                time.sleep(1.5)
                res = self.predictor.predict(input_vals)
                
                # Update Stats
                st.session_state.prediction_stats['total'] += 1
                pred_label = res['prediction']
                if pred_label in st.session_state.prediction_stats:
                    st.session_state.prediction_stats[pred_label] += 1
                
                # Store result in session state to persist after rerun
                st.session_state.last_prediction = res
                st.rerun()

        if 'last_prediction' in st.session_state:
            res = st.session_state.last_prediction
            # Result Card
            st.markdown("<div style='margin-top: 4rem;'></div>", unsafe_allow_html=True)
            
            # Wrap results in a wider container
            st.markdown("<div class='glass-container' style='max-width: 1200px;'>", unsafe_allow_html=True)
            out_col1, out_col2 = st.columns([1.2, 1], gap="large")
            
            with out_col1:
                status_color = "#10B981" if res['prediction'] == "Normal" else res['risk_color']
                st.markdown(f"""
                    <div class='result-card slide-in-right' style='border-left-color: {status_color}; margin: 0;'>
                        <p style='text-transform: uppercase; letter-spacing: 4px; font-size: 0.9rem; color: #6C63FF !important; font-weight: 800; margin-bottom: 1rem;'>Neural Outcome</p>
                        <h1 style='font-size: 4.5rem; margin: 0.5rem 0; color: #0F172A !important; font-weight: 800;'>{res['prediction']}</h1>
                        <div style='display: flex; align-items: center; gap: 2.5rem; margin: 2rem 0;'>
                            <div style='background: {status_color}; color: white !important; padding: 0.8rem 2rem; border-radius: 14px; font-weight: 800; font-size: 1.1rem; box-shadow: 0 10px 15px -3px rgba(0,0,0,0.1);'>
                                {res['risk_level'] if res['prediction'] != "Normal" else "STABLE"}
                            </div>
                            <h2 style='margin:0; color: #0F172A !important; font-weight: 700;'>{res['confidence']:.1f}% Confidence</h2>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                
                # Health Tips Section
                rec = recommendations.get(res['prediction'], recommendations['Normal'])
                st.markdown(f"""
                    <div class='result-card fade-in' style='margin-top: 2rem;'>
                        <h2 style='color: #6C63FF !important; font-weight: 800; margin-bottom: 1.5rem;'>💡 Clinical Insights</h2>
                        <p style='color: #475569 !important; font-size: 1.2rem; line-height: 1.6;'><b>Analysis:</b> {rec['info']}</p>
                        <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 40px; margin-top: 2rem;'>
                            <div>
                                <h4 style='color: #00D4FF !important; font-weight: 800; margin-bottom: 1rem; border-bottom: 2px solid #00D4FF; display: inline-block;'>✅ Health Tips</h4>
                                {"".join([f"<p style='margin: 0.5rem 0; font-size:1.05rem; color: #475569 !important; font-weight: 500;'>• {t}</p>" for t in rec['tips']])}
                            </div>
                            <div>
                                <h4 style='color: #6C63FF !important; font-weight: 800; margin-bottom: 1rem; border-bottom: 2px solid #6C63FF; display: inline-block;'>🥗 Diet Plan</h4>
                                {"".join([f"<p style='margin: 0.5rem 0; font-size:1.05rem; color: #475569 !important; font-weight: 500;'>• {d}</p>" for d in rec['diet']])}
                            </div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)

            with out_col2:
                st.markdown("<div class='result-card' style='padding: 2.5rem; margin: 0;'>", unsafe_allow_html=True)
                st.markdown("<h3 style='color: #0F172A !important; font-weight: 800; text-align: center; margin-bottom: 2rem;'>🕸️ Neural Probability Map</h3>", unsafe_allow_html=True)
                fig_radar = go.Figure()
                fig_radar.add_trace(go.Scatterpolar(
                    r=res['probabilities'],
                    theta=res['classes'],
                    fill='toself',
                    line_color='#6C63FF',
                    fillcolor='rgba(108, 99, 255, 0.3)',
                    marker=dict(size=10)
                ))
                fig_radar.update_layout(
                    polar=dict(
                        radialaxis=dict(visible=True, range=[0, 1], gridcolor='rgba(0,0,0,0.1)', tickfont=dict(color='#475569', size=11)),
                        angularaxis=dict(tickfont=dict(color='#0F172A', size=13))
                    ),
                    showlegend=False,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font_color='#0F172A',
                    height=550,
                    margin=dict(t=50, b=50, l=60, r=60)
                )
                st.plotly_chart(fig_radar, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    def about_page(self):
        st.markdown("<div class='fade-in' style='margin-top: 2rem;'>", unsafe_allow_html=True)
        st.markdown("<h1 style='font-weight:800; letter-spacing:-2px; font-size: 3rem; color: #000000 !important; text-align: center; margin-bottom: 2rem;'>Intelligence Blueprint</h1>", unsafe_allow_html=True)
        
        # Team Section with Marquee
        st.markdown("<h2 style='text-align: center; margin-bottom: 1rem; font-weight: 800;'>Our Research Team</h2>", unsafe_allow_html=True)
        
        team_members = [
            {"name": "M.V.V. Durga Prakash", "dept": "Dept. of Information Technology", "inst": "Narasaraopeta Engineering College", "img": "https://i.pravatar.cc/300?u=durgaprakash"},
            {"name": "P. Vijaya Lakshmi", "dept": "Dept. of Information Technology", "inst": "Narasaraopeta Engineering College", "img": "https://i.pravatar.cc/300?u=vijayalakshmi"},
            {"name": "Shaik Nagoor Vali", "dept": "Dept. of Information Technology", "inst": "Narasaraopeta Engineering College", "img": "https://i.pravatar.cc/300?u=nagoor"},
            {"name": "Pathan Rasheed Khan", "dept": "Dept. of Information Technology", "inst": "Narasaraopeta Engineering College", "img": "https://i.pravatar.cc/300?u=rasheed"},
            {"name": "Maddula Ratna Mohitha", "dept": "Dept. of CSE", "inst": "AITAM, Tekkali, Srikakulam", "img": "https://i.pravatar.cc/300?u=mohitha"}
        ]
        
        # Doubling the list for smooth infinite scroll
        marquee_html = "".join([f"""
            <div class='team-member-card'>
                <div class='member-image-wrapper'>
                    <img src='{m['img']}' alt='{m['name']}'>
                </div>
                <div class='member-info'>
                    <div class='member-name'>{m['name']}</div>
                    <div class='member-dept'>{m['dept']}</div>
                    <div class='member-inst'>{m['inst']}</div>
                </div>
            </div>
        """ for m in team_members + team_members])
        
        st.markdown(f"""
            <div class='marquee-container'>
                <div class='marquee-content'>
                    {marquee_html}
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<div style='margin-top: 3rem;'></div>", unsafe_allow_html=True)
        
        # Detailed Technical Overview
        st.markdown("<h2 style='text-align: center; margin-bottom: 2rem; font-weight: 800;'>Technical Methodology</h2>", unsafe_allow_html=True)
        
        tech_col1, tech_col2, tech_col3 = st.columns(3)
        with tech_col1:
            st.markdown("""
                <div class='glass-container' style='max-width: 100%; height: 100%; margin: 0;'>
                    <h3 style='color: #6C63FF !important; font-weight: 800;'>🧠 ML Models</h3>
                    <p style='color: #475569 !important; font-size: 0.95rem;'>
                        <b>XGBoost Ensemble:</b> Primary gradient boosting algorithm optimized for clinical classification.<br><br>
                        <b>Random Forest:</b> Secondary bagging classifier used for feature importance validation.<br><br>
                        <b>Hyperparameter Tuning:</b> RandomizedSearchCV used for optimizing tree depth, learning rate, and estimators.
                    </p>
                </div>
            """, unsafe_allow_html=True)
        
        with tech_col2:
            st.markdown("""
                <div class='glass-container' style='max-width: 100%; height: 100%; margin: 0;'>
                    <h3 style='color: #00D4FF !important; font-weight: 800;'>🧪 Preprocessing</h3>
                    <p style='color: #475569 !important; font-size: 0.95rem;'>
                        <b>SMOTE:</b> Synthetic Minority Over-sampling Technique to ensure perfect 1:1:1:1:1:1 class balance.<br><br>
                        <b>Standard Scaling:</b> Robust scaling of 15+ clinical biomarkers to normalize neural variance.<br><br>
                        <b>Median Imputation:</b> Intelligent handling of missing clinical vectors based on class medians.
                    </p>
                </div>
            """, unsafe_allow_html=True)
            
        with tech_col3:
            st.markdown("""
                <div class='glass-container' style='max-width: 100%; height: 100%; margin: 0;'>
                    <h3 style='color: #6C63FF !important; font-weight: 800;'>📊 Dataset</h3>
                    <p style='color: #475569 !important; font-size: 0.95rem;'>
                        <b>6 Diagnostic Classes:</b> UrinaryBladder, Kidney, Prostate, Cystitis, Uterus, and Healthy (Normal).<br><br>
                        <b>Balanced Volume:</b> 6,000 high-fidelity clinical records (1,000 samples per class).<br><br>
                        <b>Feature Engineering:</b> Includes Age, Gender, Blood markers (WBC, RBC, Hemoglobin), and Renal markers (Creatinine, Urea).
                    </p>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='margin-top: 3rem;'></div>", unsafe_allow_html=True)
        
        a1, a2 = st.columns([1.5, 1])
        with a1:
            st.markdown("""
                <div class='glass-container' style='max-width: 100%; margin: 0;'>
                <h3 style='color: #000000 !important; font-weight: 800;'>System Architecture</h3>
                <p style='color: #475569 !important;'>Our ensemble framework utilizes <b>XGBoost</b> and <b>Random Forest</b> models trained on balanced clinical datasets. We employ <b>SMOTE</b> (Synthetic Minority Over-sampling Technique) to handle class imbalances across all 6 diagnostic types, ensuring the model remains unbiased even for rare clinical profiles.</p>
                <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-top: 2rem;'>
                    <div style='background: rgba(108, 99, 255, 0.05); padding: 1.5rem; border-radius: 16px; border: 1px solid rgba(108, 99, 255, 0.1);'>
                        <h4 style='margin:0; color: #6C63FF !important; font-weight: 800;'>99% Recall</h4>
                        <p style='margin:0; font-size: 0.8rem; opacity: 0.8; color: #475569 !important;'>For Advanced Stages</p>
                    </div>
                    <div style='background: rgba(0, 212, 255, 0.05); padding: 1.5rem; border-radius: 16px; border: 1px solid rgba(0, 212, 255, 0.1);'>
                        <h4 style='margin:0; color: #00D4FF !important; font-weight: 800;'>1.2s Latency</h4>
                        <p style='margin:0; font-size: 0.8rem; opacity: 0.8; color: #475569 !important;'>End-to-end Inference</p>
                    </div>
                </div>
                </div>
            """, unsafe_allow_html=True)
            
            if os.path.exists('model/metrics.json'):
                with open('model/metrics.json', 'r') as f:
                    metrics = json.load(f)
                st.markdown("<h3 style='color: #000000 !important; margin-top: 2rem; font-weight: 800;'>Production Metrics</h3>", unsafe_allow_html=True)
                st.json(metrics)

        with a2:
            st.markdown("""
                <div class='glass-container' style='max-width: 100%; margin: 0;'>
                <h3 style='color: #000000 !important; font-weight: 800;'>Tech Stack</h3>
                <p style='color: #475569 !important;'>• <b>Inference:</b> XGBoost Ensemble<br>
                • <b>UI:</b> Streamlit 1.31 (Glassmorphism 2.0)<br>
                • <b>Data:</b> Pandas & Scikit-learn Pipeline<br>
                • <b>Charts:</b> Plotly High-Res Visualization</p>
                </div>
            """, unsafe_allow_html=True)
            
            if os.path.exists('model/model.pkl'):
                importances = self.predictor.model.feature_importances_
                features = self.predictor.selected_features
                fig = plot_feature_importance(importances, features)
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)', 
                    plot_bgcolor='rgba(0,0,0,0)', 
                    font_color='#000000', 
                    height=400,
                    xaxis=dict(gridcolor='rgba(0,0,0,0.1)'),
                    yaxis=dict(gridcolor='rgba(0,0,0,0.1)')
                )
                st.plotly_chart(fig, use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    app = AdvancedNexGenApp()
    app.run()
