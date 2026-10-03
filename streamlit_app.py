"""
================================================================================
SOLID-STATE LIGHT ENGINE & CIE 1931 COLORIMETRY INTERACTIVE SIMULATOR
================================================================================
An industrial-grade Streamlit web application designed for optical engineers
and display architects. Models hybrid Laser-Phosphor projector light engines,
calculates CIE 1931 chromaticity coordinates, solves D65/DCI white balance
calibration via linear algebra, and evaluates color gamut coverage.
================================================================================
"""
import os
import numpy as np
import plotly.graph_objects as go
import streamlit as st

# ==============================================================================
# STREAMLIT PAGE CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="Light Engine & Colorimetry Simulator",
    page_icon="🎛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject custom CSS for clean padding and centered chart containers
st.markdown("""
<style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }
    .stPlotlyChart {
        display: flex;
        justify-content: center;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# STEP 1: NUMPY 2.0+ BACKWARD COMPATIBILITY MONKEYPATCH
# ==============================================================================
if not hasattr(np, "asfarray"):
    np.asfarray = lambda a, dtype=np.float64: np.asarray(a, dtype=dtype)
if not hasattr(np, "float_"):
    np.float_ = np.float64
if not hasattr(np, "int_"):
    np.int_ = np.int64
if not hasattr(np, "bool_"):
    np.bool_ = bool


# ==============================================================================
# STEP 2: HELPER FUNCTIONS (PHYSICAL MODELING & COLORIMETRY)
# ==============================================================================

@st.cache_data
def get_cie1931_color_matching_functions(wavelengths):
    wl = wavelengths
    x1 = 1.056 * np.exp(-0.5 * ((wl - 599.8) / 37.9)**2)
    x2 = 0.362 * np.exp(-0.5 * ((wl - 442.0) / 16.0)**2)
    x3 = -0.065 * np.exp(-0.5 * ((wl - 501.1) / 20.4)**2)
    x_bar = np.maximum(0, x1 + x2 + x3)

    y1 = 0.821 * np.exp(-0.5 * ((wl - 568.8) / 46.9)**2)
    y2 = 0.286 * np.exp(-0.5 * ((wl - 530.9) / 22.7)**2)
    y_bar = np.maximum(0, y1 + y2)

    z1 = 1.217 * np.exp(-0.5 * ((wl - 437.0) / 11.8)**2)
    z2 = 0.681 * np.exp(-0.5 * ((wl - 459.0) / 26.0)**2)
    z_bar = np.maximum(0, z1 + z2)

    return x_bar, y_bar, z_bar


def gaussian_spd(wavelengths, peak, fwhm, amplitude=1.0):
    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    return amplitude * np.exp(-0.5 * ((wavelengths - peak) / sigma)**2)


def spd_to_xyz(wavelengths, spd, x_bar, y_bar, z_bar):
    d_lambda = wavelengths[1] - wavelengths[0]
    X = np.sum(spd * x_bar) * d_lambda
    Y = np.sum(spd * y_bar) * d_lambda
    Z = np.sum(spd * z_bar) * d_lambda
    total = X + Y + Z + 1e-12
    x = X / total
    y = Y / total
    return X, Y, Z, x, y


def polygon_area(x_coords, y_coords):
    return 0.5 * np.abs(np.dot(x_coords, np.roll(y_coords, 1)) - np.dot(y_coords, np.roll(x_coords, 1)))


# ==============================================================================
# STREAMLIT USER INTERFACE & SIDEBAR CONTROLS
# ==============================================================================

st.title("🎛️Solid-State Light Engine & Colorimetry Simulator")
st.markdown("""
**Interactive Optical Engineering Dashboard** | Simulate hybrid **Blue Laser + Yellow Phosphor + Red Laser/LED** 
architectures for DLP projection systems. Tune source wavelengths, dichroic filter bandwidths, and calibrate 
white balance to target CIE color points in real time.
""")

st.sidebar.header("🔬 Light Engine Architecture Parameters")

# Sidebar Group 1: Blue Laser Diode Settings
st.sidebar.subheader("1. Blue Laser Diode Primary")
blue_peak = st.sidebar.slider("Blue Wavelength (nm)", 440.0, 470.0, 455.0, 1.0, help="Center wavelength of excitation laser diode.")
blue_fwhm = st.sidebar.slider("Blue FWHM Line-width (nm)", 1.0, 10.0, 3.5, 0.5)

# Sidebar Group 2: Yellow Phosphor & Green Dichroic Extraction Filter
st.sidebar.subheader("2. Phosphor Wheel & Green Dichroic Filter")
phosphor_peak = st.sidebar.slider("Phosphor Peak Emission (nm)", 520.0, 580.0, 550.0, 2.0)
phosphor_fwhm = st.sidebar.slider("Phosphor Bandwidth FWHM (nm)", 80.0, 140.0, 110.0, 5.0)
green_filter_fwhm = st.sidebar.slider("Green Dichroic Filter Bandwidth (nm)", 15.0, 60.0, 35.0, 1.0, 
                                     help="Wider filter = higher brightness, but reduced green color purity.")

# Sidebar Group 3: Red Primary Source
st.sidebar.subheader("3. Red Primary Source")
red_peak = st.sidebar.slider("Red Peak Wavelength (nm)", 615.0, 660.0, 638.0, 1.0, 
                            help="625 nm = Red LED / 638 nm = Direct Red Laser Diode.")
red_fwhm = st.sidebar.slider("Red FWHM Line-width (nm)", 2.0, 25.0, 12.0, 1.0)

# Sidebar Group 4: Target White Point & Gamut Overlays
st.sidebar.subheader("4. Display White Point & Standards")
white_option = st.sidebar.selectbox(
    "Target Calibration White Point",
    ["D65 (Daylight - x=0.3127, y=0.3290)", 
     "DCI White (Cinema - x=0.3140, y=0.3510)", 
     "D50 (Graphic Arts - x=0.3457, y=0.3585)", 
     "Custom (User Defined)"]
)

if "D65" in white_option:
    target_x_val, target_y_val = 0.3127, 0.3290
elif "DCI" in white_option:
    target_x_val, target_y_val = 0.3140, 0.3510
elif "D50" in white_option:
    target_x_val, target_y_val = 0.3457, 0.3585
else:
    target_x_val = st.sidebar.number_input("Custom Target x", 0.100, 0.600, 0.3127, 0.005)
    target_y_val = st.sidebar.number_input("Custom Target y", 0.100, 0.600, 0.3290, 0.005)

st.sidebar.markdown("---")
st.sidebar.subheader("Display Gamut Overlays")
show_rec709 = st.sidebar.checkbox("Overlay Rec. 709 (HDTV)", value=True)
show_dcip3 = st.sidebar.checkbox("Overlay DCI-P3 (Cinema)", value=True)
show_rec2020 = st.sidebar.checkbox("Overlay Rec. 2020 (UHD)", value=False)

st.sidebar.markdown("---")
st.sidebar.subheader("📐 Chart Dimensions & Window Controls")
chart_width = st.sidebar.slider("Graph Width (px)", 350, 1100, 600, 25, 
                               help="Adjust exact width of the figure window.")
chart_height = st.sidebar.slider("Graph Height (px)", 300, 800, 500, 25, 
                                help="Adjust exact height of the figure window.")
lock_aspect = st.sidebar.checkbox("Lock 1:1 Aspect Ratio (Prevents Distortion)", value=True,
                                 help="Keeps x and y axes at equal physical scale so color geometry isn't stretched.")


# ==============================================================================
# COMPUTATION ENGINE
# ==============================================================================
wavelengths = np.linspace(380, 780, 401)
x_bar, y_bar, z_bar = get_cie1931_color_matching_functions(wavelengths)

# Generate Light Spectra
spd_blue_laser = gaussian_spd(wavelengths, peak=blue_peak, fwhm=blue_fwhm, amplitude=1.0)
spd_yellow_phosphor = gaussian_spd(wavelengths, peak=phosphor_peak, fwhm=phosphor_fwhm, amplitude=0.85)
spd_red_laser = gaussian_spd(wavelengths, peak=red_peak, fwhm=red_fwhm, amplitude=0.90)

# Dichroic Filter Extraction for Green Primary
green_filter = gaussian_spd(wavelengths, peak=532.0, fwhm=green_filter_fwhm, amplitude=1.0)
spd_green_channel = spd_yellow_phosphor * green_filter

# Calculate Individual Primary Tristimulus Values
X_b, Y_b, Z_b, x_blue, y_blue = spd_to_xyz(wavelengths, spd_blue_laser, x_bar, y_bar, z_bar)
X_g, Y_g, Z_g, x_green, y_green = spd_to_xyz(wavelengths, spd_green_channel, x_bar, y_bar, z_bar)
X_r, Y_r, Z_r, x_red, y_red = spd_to_xyz(wavelengths, spd_red_laser, x_bar, y_bar, z_bar)

# Solve Linear Matrix System for Target White Point Calibration
z_target_val = 1.0 - target_x_val - target_y_val
M = np.array([
    [X_r, X_g, X_b],
    [Y_r, Y_g, Y_b],
    [Z_r, Z_g, Z_b]
])
target_xyz = np.array([target_x_val, target_y_val, z_target_val])

try:
    weights = np.linalg.solve(M, target_xyz)
    weights = weights / np.max(weights)
    w_r, w_g, w_b = weights[0], weights[1], weights[2]
except np.linalg.LinAlgError:
    w_r, w_g, w_b = 1.0, 1.0, 1.0

# Total Calibrated Engine Spectrum
spd_engine_total = w_b * spd_blue_laser + w_g * spd_green_channel + w_r * spd_red_laser
_, _, _, x_white, y_white = spd_to_xyz(wavelengths, spd_engine_total, x_bar, y_bar, z_bar)

# Gamut Area Calculations
engine_area = polygon_area([x_red, x_green, x_blue], [y_red, y_green, y_blue])
rec709_area = polygon_area([0.64, 0.30, 0.15], [0.33, 0.60, 0.06])
dcip3_area = polygon_area([0.68, 0.265, 0.15], [0.32, 0.690, 0.06])

percent_rec709 = (engine_area / rec709_area) * 100.0
percent_dcip3 = (engine_area / dcip3_area) * 100.0


# ==============================================================================
# DASHBOARD DISPLAY & METRIC CARDS
# ==============================================================================
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Red Primary (x, y)", f"{x_red:.3f}, {y_red:.3f}")
col2.metric("Green Primary (x, y)", f"{x_green:.3f}, {y_green:.3f}")
col3.metric("Blue Primary (x, y)", f"{x_blue:.3f}, {y_blue:.3f}")
col4.metric("Calibrated White (x, y)", f"{x_white:.3f}, {y_white:.3f}")
col5.metric("DCI-P3 Gamut Coverage", f"{percent_dcip3:.1f}%")

st.markdown("---")

# Tab Layout
tab1, tab2, tab3 = st.tabs(["📊 CIE 1931 Chromaticity Diagram", "📈 Spectral Power Distributions", "📋 Numerical Data & Calibration"])

with tab1:
    st.caption("💡 Adjust Graph Width & Height sliders in the sidebar to resize this window box. Toggle 'Lock 1:1 Aspect Ratio' to preserve true color geometry.")

    fig_cie = go.Figure()

    # Calculate outer spectral locus
    locus_x, locus_y, locus_wl = [], [], []
    for wl in np.linspace(380, 700, 200):
        spd_mono = gaussian_spd(wavelengths, peak=wl, fwhm=1.0)
        _, _, _, lx, ly = spd_to_xyz(wavelengths, spd_mono, x_bar, y_bar, z_bar)
        locus_x.append(lx)
        locus_y.append(ly)
        locus_wl.append(wl)

    # 1. CIE 1931 Spectral Locus
    fig_cie.add_trace(go.Scatter(
        x=locus_x, y=locus_y,
        mode='lines',
        name='CIE 1931 Spectral Locus',
        line=dict(color='#888888', width=2),
        customdata=locus_wl,
        hovertemplate='Wavelength: %{customdata:.1f} nm<br>x: %{x:.4f}<br>y: %{y:.4f}<extra></extra>'
    ))

    # Line of purples
    fig_cie.add_trace(go.Scatter(
        x=[locus_x[0], locus_x[-1]], y=[locus_y[0], locus_y[-1]],
        mode='lines',
        name='Line of Purples',
        line=dict(color='#aa55ff', width=1.5, dash='dot'),
        hoverinfo='skip'
    ))

    # 2. Reference Display Gamut Overlays
    if show_rec709:
        fig_cie.add_trace(go.Scatter(
            x=[0.64, 0.30, 0.15, 0.64],
            y=[0.33, 0.60, 0.06, 0.33],
            mode='lines',
            name='Rec. 709 (HDTV)',
            line=dict(color='#aaaaaa', width=1.5, dash='dash'),
            hovertemplate='Rec. 709<br>x: %{x:.3f}, y: %{y:.3f}<extra></extra>'
        ))

    if show_dcip3:
        fig_cie.add_trace(go.Scatter(
            x=[0.68, 0.265, 0.15, 0.68],
            y=[0.32, 0.690, 0.06, 0.32],
            mode='lines',
            name='DCI-P3 (Cinema)',
            line=dict(color='#ff00ff', width=1.5, dash='dashdot'),
            hovertemplate='DCI-P3<br>x: %{x:.3f}, y: %{y:.3f}<extra></extra>'
        ))

    if show_rec2020:
        fig_cie.add_trace(go.Scatter(
            x=[0.708, 0.170, 0.131, 0.708],
            y=[0.292, 0.797, 0.046, 0.292],
            mode='lines',
            name='Rec. 2020 (UHD)',
            line=dict(color='#00ffaa', width=1.5, dash='dot'),
            hovertemplate='Rec. 2020<br>x: %{x:.3f}, y: %{y:.3f}<extra></extra>'
        ))

    # 3. Simulated Engine Gamut Polygon
    engine_x = [x_red, x_green, x_blue, x_red]
    engine_y = [y_red, y_green, y_blue, y_red]
    fig_cie.add_trace(go.Scatter(
        x=engine_x, y=engine_y,
        mode='lines',
        fill='toself',
        fillcolor='rgba(30, 144, 255, 0.20)',
        name='Simulated Engine Gamut',
        line=dict(color='#1e90ff', width=2.5),
        hovertemplate='Engine Gamut Corner<br>x: %{x:.4f}<br>y: %{y:.4f}<extra></extra>'
    ))

    # 4. Primary Markers & White Point
    fig_cie.add_trace(go.Scatter(
        x=[x_red], y=[y_red],
        mode='markers',
        name=f'Red Primary ({x_red:.3f}, {y_red:.3f})',
        marker=dict(color='#ff3333', size=12, symbol='circle', line=dict(color='#ffffff', width=1.5)),
        hovertemplate='<b>Red Primary</b><br>Peak: ' + f'{red_peak:.1f}' + ' nm<br>x: %{x:.4f}<br>y: %{y:.4f}<extra></extra>'
    ))

    fig_cie.add_trace(go.Scatter(
        x=[x_green], y=[y_green],
        mode='markers',
        name=f'Green Primary ({x_green:.3f}, {y_green:.3f})',
        marker=dict(color='#33cc33', size=12, symbol='circle', line=dict(color='#ffffff', width=1.5)),
        hovertemplate='<b>Green Primary</b><br>Extracted Green<br>x: %{x:.4f}<br>y: %{y:.4f}<extra></extra>'
    ))

    fig_cie.add_trace(go.Scatter(
        x=[x_blue], y=[y_blue],
        mode='markers',
        name=f'Blue Primary ({x_blue:.3f}, {y_blue:.3f})',
        marker=dict(color='#3388ff', size=12, symbol='circle', line=dict(color='#ffffff', width=1.5)),
        hovertemplate='<b>Blue Primary</b><br>Peak: ' + f'{blue_peak:.1f}' + ' nm<br>x: %{x:.4f}<br>y: %{y:.4f}<extra></extra>'
    ))

    fig_cie.add_trace(go.Scatter(
        x=[x_white], y=[y_white],
        mode='markers',
        name=f'Engine White ({x_white:.3f}, {y_white:.3f})',
        marker=dict(color='#ffd700', size=16, symbol='star', line=dict(color='#ffffff', width=1.5)),
        hovertemplate='<b>Engine Calibrated White</b><br>x: %{x:.4f}<br>y: %{y:.4f}<extra></extra>'
    ))

    fig_cie.add_trace(go.Scatter(
        x=[target_x_val], y=[target_y_val],
        mode='markers',
        name=f'Target ({target_x_val:.3f}, {target_y_val:.3f})',
        marker=dict(color='#ff9900', size=12, symbol='x', line=dict(color='#ff9900', width=2.5)),
        hovertemplate='<b>Target White Point</b><br>x: %{x:.4f}<br>y: %{y:.4f}<extra></extra>'
    ))

    # Configure axes dict with optional 1:1 scale locking
    yaxis_dict = dict(title="CIE y Chromaticity", range=[0.0, 0.90], gridcolor="rgba(128,128,128,0.25)", zeroline=False)
    if lock_aspect:
        yaxis_dict["scaleanchor"] = "x"
        yaxis_dict["scaleratio"] = 1.0

    fig_cie.update_layout(
        title=dict(text="<b>CIE 1931 Chromaticity Diagram & Display Gamut Comparison</b>", font=dict(size=15)),
        xaxis=dict(title="CIE x Chromaticity", range=[0.0, 0.82], gridcolor="rgba(128,128,128,0.25)", zeroline=False),
        yaxis=yaxis_dict,
        width=chart_width,
        height=chart_height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(x=0.62, y=0.98, bgcolor="rgba(128,128,128,0.18)", bordercolor="rgba(128,128,128,0.3)", borderwidth=1),
        hovermode="closest"
    )

    # Centered container display
    c_left, c_mid, c_right = st.columns([0.05, 0.9, 0.05])
    with c_mid:
        st.plotly_chart(fig_cie, use_container_width=False)


with tab2:
    st.caption("💡 Adjust Graph Width & Height sliders in the sidebar to resize this window box.")

    fig_spd = go.Figure()

    fig_spd.add_trace(go.Scatter(
        x=wavelengths, y=w_b * spd_blue_laser,
        mode='lines',
        name='Blue Laser Primary',
        line=dict(color='#3388ff', width=2.5),
        hovertemplate='Blue Laser<br>Wavelength: %{x:.1f} nm<br>Intensity: %{y:.4f}<extra></extra>'
    ))

    fig_spd.add_trace(go.Scatter(
        x=wavelengths, y=spd_yellow_phosphor * 0.5,
        mode='lines',
        name='Yellow Phosphor Emission',
        line=dict(color='#ffd700', width=2, dash='dash'),
        hovertemplate='Yellow Phosphor<br>Wavelength: %{x:.1f} nm<br>Intensity: %{y:.4f}<extra></extra>'
    ))

    fig_spd.add_trace(go.Scatter(
        x=wavelengths, y=w_g * spd_green_channel,
        mode='lines',
        name='Filtered Green Primary',
        line=dict(color='#33cc33', width=2.5),
        hovertemplate='Filtered Green<br>Wavelength: %{x:.1f} nm<br>Intensity: %{y:.4f}<extra></extra>'
    ))

    fig_spd.add_trace(go.Scatter(
        x=wavelengths, y=w_r * spd_red_laser,
        mode='lines',
        name='Red Laser/LED Primary',
        line=dict(color='#ff3333', width=2.5),
        hovertemplate='Red Primary<br>Wavelength: %{x:.1f} nm<br>Intensity: %{y:.4f}<extra></extra>'
    ))

    fig_spd.add_trace(go.Scatter(
        x=wavelengths, y=spd_engine_total,
        mode='lines',
        name='D65 Balanced Engine Output',
        line=dict(color='#17becf', width=3),
        hovertemplate='Calibrated Engine Output<br>Wavelength: %{x:.1f} nm<br>Intensity: %{y:.4f}<extra></extra>'
    ))

    fig_spd.update_layout(
        title=dict(text="<b>Solid-State Light Engine Spectral Power Distributions (SPDs)</b>", font=dict(size=15)),
        xaxis=dict(title="Wavelength (nm)", range=[380, 750], gridcolor="rgba(128,128,128,0.25)"),
        yaxis=dict(title="Normalized Spectral Intensity (a.u.)", range=[0, 1.25], gridcolor="rgba(128,128,128,0.25)"),
        width=chart_width,
        height=int(chart_height * 0.85),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(x=0.62, y=0.98, bgcolor="rgba(128,128,128,0.18)", bordercolor="rgba(128,128,128,0.3)", borderwidth=1),
        hovermode="x unified"
    )

    c_left, c_mid, c_right = st.columns([0.05, 0.9, 0.05])
    with c_mid:
        st.plotly_chart(fig_spd, use_container_width=False)


with tab3:
    st.subheader("📋 Engineering Parameters & Matrix Solution Summary")
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.markdown("**Calibrated Channel Weighting Ratios:**")
        st.write(f"- **Red Power Weight (w_R)**: `{w_r:.4f}`")
        st.write(f"- **Green Power Weight (w_G)**: `{w_g:.4f}`")
        st.write(f"- **Blue Power Weight (w_B)**: `{w_b:.4f}`")
        
        st.markdown("**Color Gamut Area Ratios:**")
        st.write(f"- **Engine Gamut Area**: `{engine_area:.4f}` CIE x-y units²")
        st.write(f"- **vs Rec. 709 Standard**: `{percent_rec709:.1f}%`")
        st.write(f"- **vs DCI-P3 Cinema Standard**: `{percent_dcip3:.1f}%`")

    with col_b:
        st.markdown("**Tristimulus Matrix [M] (Primary Stimulus Vectors):**")
        st.code(f"""
[ X_R  X_G  X_B ] = [{X_r:.3f}  {X_g:.3f}  {X_b:.3f}]
[ Y_R  Y_G  Y_B ] = [{Y_r:.3f}  {Y_g:.3f}  {Y_b:.3f}]
[ Z_R  Z_G  Z_B ] = [{Z_r:.3f}  {Z_g:.3f}  {Z_b:.3f}]
        """)
        st.markdown("**Target Tristimulus Vector:**")
        st.code(f"[X_target, Y_target, Z_target] = [{target_xyz[0]:.3f}, {target_xyz[1]:.3f}, {target_xyz[2]:.3f}]")

st.markdown("---")
st.caption("Optical Engineer Portfolio Project | Developed in Python & Streamlit")
