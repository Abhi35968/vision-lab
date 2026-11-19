"""
Modern Client-Focused Dashboard Theme for 3D Reconstruction Studio.
Custom UI/UX styled with the client-dashboard palette:
- Canvas: #FDFDFE (Pale Grey)
- Surfaces / Sidebars: #F0F3F5 (Anti-Flash White)
- Borders / Outlines: #D6D9DF (Hidden Creek)
- Secondary Greys: #BDC2C7 (Metallic)
- Body Muted Text: #8F9192 (Alloy)
- Primary Accent: #3D766D (Greensleeves Deep Teal)
"""

import streamlit as st

# Color Palette Constants
COLOR_CANVAS = "#FDFDFE"
COLOR_SURFACE = "#F0F3F5"
COLOR_CARD = "#FFFFFF"
COLOR_BORDER = "#D6D9DF"
COLOR_METALLIC = "#BDC2C7"
COLOR_ALLOY = "#8F9192"
COLOR_TEAL = "#3D766D"
COLOR_TEAL_DARK = "#29534D"
COLOR_TEAL_LIGHT = "#EAF2F0"
COLOR_TEAL_HOVER = "#34665E"
COLOR_TEXT_MAIN = "#142220"
COLOR_CORAL = "#D06B4C"

CUSTOM_CSS = """
<style>
/* Import Google Fonts */
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

/* Root Variables */
:root {
    --bg-canvas: #FDFDFE;
    --bg-surface: #F0F3F5;
    --bg-card: #FFFFFF;
    --border-subtle: #D6D9DF;
    --text-metallic: #BDC2C7;
    --text-alloy: #8F9192;
    --text-main: #142220;
    --teal-primary: #3D766D;
    --teal-dark: #29534D;
    --teal-light: #EAF2F0;
    --teal-hover: #34665E;
    --accent-coral: #D06B4C;
}

/* Base resets & typography */
*, *:before, *:after {
    box-sizing: border-box !important;
}

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    background-color: var(--bg-canvas) !important;
    color: var(--text-main) !important;
    letter-spacing: -0.01em;
}

/* Streamlit Chrome / Branding Removals */
#MainMenu { visibility: hidden !important; }
footer { visibility: hidden !important; }
.stDeployButton { display: none !important; }
div[data-testid="stDecoration"] { display: none !important; }
div[data-testid="stStatusWidget"] { display: none !important; }

/* Streamlit Header Bar Clean-up: Make it transparent and borderless */
header[data-testid="stHeader"] {
    background: transparent !important;
    border-bottom: none !important;
    height: 1.5rem !important;
}

/* Main Container Padding: eliminate clunky default gaps */
.main .block-container {
    padding-top: 1rem !important;
    padding-bottom: 3rem !important;
    max-width: 1420px !important;
}

/* Headings */
h1, h2, h3, h4, h5, h6 {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    color: var(--text-main) !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em !important;
}

h1 {
    font-size: 2.1rem !important;
    margin-bottom: 0.5rem !important;
}

h2 {
    font-size: 1.55rem !important;
    margin-top: 1.5rem !important;
    margin-bottom: 0.75rem !important;
}

h3 {
    font-size: 1.25rem !important;
    margin-top: 1.2rem !important;
    color: var(--teal-dark) !important;
}

p, span, label {
    color: var(--text-main);
    font-size: 0.95rem;
}

/* Sidebar Overhaul */
section[data-testid="stSidebar"] {
    background-color: var(--bg-surface) !important;
    border-right: 1px solid var(--border-subtle) !important;
    box-shadow: 2px 0 12px rgba(0, 0, 0, 0.02) !important;
}

section[data-testid="stSidebar"] .block-container {
    padding-top: 2rem !important;
    padding-left: 1.25rem !important;
    padding-right: 1.25rem !important;
}

/* Sidebar Radio Buttons -> Sleek Nav Pills */
div[data-testid="stRadio"] > div {
    gap: 6px !important;
    display: flex !important;
    flex-direction: column !important;
}

div[data-testid="stRadio"] label {
    background-color: transparent !important;
    border-radius: 10px !important;
    padding: 10px 14px !important;
    margin-bottom: 2px !important;
    border: 1px solid transparent !important;
    cursor: pointer !important;
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
    display: flex !important;
    align-items: center !important;
}

div[data-testid="stRadio"] label:hover {
    background-color: #E8EDF0 !important;
    color: var(--teal-dark) !important;
    border-color: #D6D9DF !important;
    transform: translateX(3px);
}

/* Selected Radio Pill */
div[data-testid="stRadio"] label:has(input:checked) {
    background: linear-gradient(135deg, var(--teal-primary) 0%, var(--teal-dark) 100%) !important;
    color: #FFFFFF !important;
    font-weight: 600 !important;
    border-color: var(--teal-dark) !important;
    box-shadow: 0 4px 12px rgba(61, 118, 109, 0.25) !important;
}

div[data-testid="stRadio"] label:has(input:checked) p,
div[data-testid="stRadio"] label:has(input:checked) span {
    color: #FFFFFF !important;
    font-weight: 600 !important;
}

/* Hide standard radio dot circles */
div[data-testid="stRadio"] input[type="radio"] {
    display: none !important;
}

/* Metrics Cards Overhaul */
div[data-testid="stMetric"] {
    background-color: var(--bg-card) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 14px !important;
    padding: 18px 20px !important;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.03), 0 1px 2px rgba(0, 0, 0, 0.02) !important;
    transition: all 0.22s ease !important;
    overflow: hidden !important;
    box-sizing: border-box !important;
}

div[data-testid="stMetric"]:hover {
    border-color: var(--teal-primary) !important;
    box-shadow: 0 6px 18px rgba(61, 118, 109, 0.12) !important;
    transform: translateY(-2px);
}

div[data-testid="stMetricLabel"] {
    color: var(--text-alloy) !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
    margin-bottom: 4px !important;
}

div[data-testid="stMetricValue"] {
    color: var(--text-main) !important;
    font-size: 1.75rem !important;
    font-weight: 750 !important;
    font-feature-settings: "cv02", "cv03", "cv04", "cv11";
}

div[data-testid="stMetricDelta"] {
    font-weight: 600 !important;
    font-size: 0.82rem !important;
}

/* Tabs Transformation */
div[data-baseweb="tab-list"] {
    background-color: var(--bg-surface) !important;
    border-radius: 14px !important;
    padding: 6px !important;
    border: 1px solid var(--border-subtle) !important;
    gap: 6px !important;
    margin-bottom: 1.5rem !important;
}

button[data-baseweb="tab"] {
    border-radius: 10px !important;
    padding: 10px 22px !important;
    font-weight: 600 !important;
    font-size: 0.92rem !important;
    color: var(--text-alloy) !important;
    background-color: transparent !important;
    border: 1px solid transparent !important;
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
}

button[data-baseweb="tab"]:hover {
    color: var(--teal-primary) !important;
    background-color: rgba(255, 255, 255, 0.7) !important;
}

button[data-baseweb="tab"][aria-selected="true"] {
    background-color: var(--bg-card) !important;
    color: var(--teal-primary) !important;
    font-weight: 700 !important;
    border-color: var(--border-subtle) !important;
    box-shadow: 0 3px 10px rgba(0, 0, 0, 0.06) !important;
}

div[data-baseweb="tab-highlight"], div[data-baseweb="tab-border"] {
    display: none !important;
}

/* Button Overhaul */
button[kind="primary"], .stButton > button[kind="primary"] {
    background: linear-gradient(135deg, var(--teal-primary) 0%, var(--teal-dark) 100%) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 10px 24px !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    box-shadow: 0 4px 14px rgba(61, 118, 109, 0.28) !important;
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
}

button[kind="primary"]:hover, .stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, var(--teal-hover) 0%, #20413C 100%) !important;
    box-shadow: 0 6px 20px rgba(61, 118, 109, 0.4) !important;
    transform: translateY(-1.5px) !important;
}

button[kind="secondary"], .stButton > button[kind="secondary"] {
    background-color: var(--bg-card) !important;
    color: var(--teal-primary) !important;
    border: 1.5px solid var(--border-subtle) !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
}

button[kind="secondary"]:hover, .stButton > button[kind="secondary"]:hover {
    border-color: var(--teal-primary) !important;
    background-color: var(--teal-light) !important;
    color: var(--teal-dark) !important;
}

/* Selectbox, Inputs & Sliders */
div[data-baseweb="select"] > div {
    background-color: var(--bg-card) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 10px !important;
    color: var(--text-main) !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.02) !important;
}

div[data-baseweb="select"] > div:focus-within {
    border-color: var(--teal-primary) !important;
    box-shadow: 0 0 0 3px rgba(61, 118, 109, 0.16) !important;
}

/* Slider Track & Thumb */
div[data-baseweb="slider"] div[role="slider"] {
    background-color: var(--teal-primary) !important;
    border: 2px solid #FFFFFF !important;
    box-shadow: 0 2px 6px rgba(61, 118, 109, 0.35) !important;
}

/* Checkbox and Radio styling */
div[data-testid="stCheckbox"] span[aria-checked="true"] {
    background-color: var(--teal-primary) !important;
    border-color: var(--teal-primary) !important;
}

/* Clean Custom Scrollbar */
::-webkit-scrollbar {
    width: 7px;
    height: 7px;
}
::-webkit-scrollbar-track {
    background: var(--bg-surface);
}
::-webkit-scrollbar-thumb {
    background: var(--border-subtle);
    border-radius: 4px;
}
::-webkit-scrollbar-thumb:hover {
    background: var(--text-metallic);
}

/* Custom Alert Boxes */
div[data-testid="stAlert"] {
    border-radius: 12px !important;
    border: 1px solid var(--border-subtle) !important;
    padding: 14px 18px !important;
}

div[data-testid="stAlert"]:has([data-testid="stNotificationContentInfo"]) {
    background-color: var(--teal-light) !important;
    border-left: 4px solid var(--teal-primary) !important;
}

/* Expanders */
div[data-testid="stExpander"] {
    background-color: var(--bg-card) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 12px !important;
    box-shadow: 0 1px 4px rgba(0, 0, 0, 0.02) !important;
    overflow: hidden !important;
}

div[data-testid="stExpander"] summary {
    font-weight: 600 !important;
    color: var(--text-main) !important;
    padding: 12px 16px !important;
}

div[data-testid="stExpander"] summary:hover {
    color: var(--teal-primary) !important;
}

/* Dataframe styling */
div[data-testid="stDataFrame"] {
    border: 1px solid var(--border-subtle) !important;
    border-radius: 12px !important;
    overflow: hidden !important;
}

/* Custom Pill and Card Classes */
.client-header-bar {
    background: linear-gradient(180deg, #FFFFFF 0%, #F9FBFA 100%);
    border: 1px solid #D6D9DF;
    border-radius: 16px;
    padding: 18px 24px;
    margin-bottom: 22px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.03);
}

.client-header-title {
    font-size: 1.35rem;
    font-weight: 800;
    color: #142220;
    margin: 0;
    display: flex;
    align-items: center;
    gap: 10px;
}

.client-header-subtitle {
    font-size: 0.85rem;
    color: #8F9192;
    margin-top: 3px;
}

.status-pill-teal {
    background-color: #EAF2F0;
    color: #3D766D;
    font-weight: 600;
    font-size: 0.78rem;
    padding: 5px 12px;
    border-radius: 9999px;
    border: 1px solid #C5DAD5;
    display: inline-flex;
    align-items: center;
    gap: 6px;
}

.status-pill-coral {
    background-color: #FAEEE9;
    color: #D06B4C;
    font-weight: 600;
    font-size: 0.78rem;
    padding: 5px 12px;
    border-radius: 9999px;
    border: 1px solid #F0CFC3;
    display: inline-flex;
    align-items: center;
    gap: 6px;
}

.client-card {
    background-color: #FFFFFF;
    border: 1px solid #D6D9DF;
    border-radius: 14px;
    padding: 18px 18px;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.03);
    margin-bottom: 16px;
    box-sizing: border-box !important;
    display: flex;
    flex-direction: column;
    justify-content: flex-start;
    height: 100%;
}

.client-card-title {
    font-size: 0.95rem;
    font-weight: 700;
    color: #29534D;
    margin-bottom: 8px;
    display: flex;
    align-items: center;
    gap: 8px;
    line-height: 1.25;
}
</style>
"""


def apply_theme():
    """Injects the custom CSS theme into the active Streamlit app."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def render_client_header(
    title: str = "3D Reconstruction Studio",
    subtitle: str = "Multi-View Structure from Motion & Neural Surface Analysis",
    badge_label: str = "10/10 Videos Reconstructed",
    engine_label: str = "COLMAP + Dense Fusion",
):
    """Renders the client-dashboard header bar inspired by Hireus."""
    html = f"""
    <div class="client-header-bar">
        <div>
            <div class="client-header-title">
                <span style="display:inline-block; width:12px; height:12px; background:#3D766D; border-radius:3px;"></span>
                {title}
            </div>
            <div class="client-header-subtitle">{subtitle}</div>
        </div>
        <div style="display:flex; gap:10px; align-items:center;">
            <span class="status-pill-teal">
                <span style="width:6px; height:6px; background:#3D766D; border-radius:50%; display:inline-block;"></span>
                {badge_label}
            </span>
            <span class="status-pill-coral">
                <span style="width:6px; height:6px; background:#D06B4C; border-radius:50%; display:inline-block;"></span>
                {engine_label}
            </span>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_sidebar_brand():
    """Renders custom logo branding at the top of the sidebar."""
    html = """
    <div style="padding:10px 4px 18px 4px; border-bottom:1px solid #D6D9DF; margin-bottom:14px;">
        <div style="display:flex; align-items:center; gap:9px;">
            <div style="width:34px; height:34px; border-radius:9px; background:linear-gradient(135deg, #3D766D 0%, #29534D 100%); display:flex; align-items:center; justify-content:center; color:#FFFFFF; font-weight:800; font-size:1.1rem; box-shadow:0 3px 8px rgba(61,118,109,0.3);">
                3D
            </div>
            <div>
                <div style="font-weight:800; font-size:1.05rem; color:#142220; line-height:1.1;">VisionLab</div>
                <div style="font-size:0.75rem; color:#8F9192; font-weight:500;">Reconstruction Studio</div>
            </div>
        </div>
    </div>
    """
    st.sidebar.markdown(html, unsafe_allow_html=True)


def get_plotly_theme():
    """Returns a unified Plotly layout theme dictionary matching the client palette."""
    return dict(
        paper_bgcolor="#FDFDFE",
        plot_bgcolor="#F0F3F5",
        font=dict(family="Plus Jakarta Sans, sans-serif", color="#142220"),
        xaxis=dict(gridcolor="#D6D9DF", zerolinecolor="#D6D9DF"),
        yaxis=dict(gridcolor="#D6D9DF", zerolinecolor="#D6D9DF"),
    )
