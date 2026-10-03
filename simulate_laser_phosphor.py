import os
import numpy as np
import matplotlib
# Set matplotlib backend to 'Agg' for non-interactive headless image saving
# (This ensures the script runs without requiring a GUI window or display server)
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ==============================================================================
# STEP 1: NUMPY 2.0+ BACKWARD COMPATIBILITY
# ==============================================================================
# Legacy numerical packages or third-party optics tools sometimes call deprecated 
# NumPy 1.x aliases. Checking and adding these attributes guarantees the script runs 
# smoothly regardless of which NumPy version is installed on your machine.
if not hasattr(np, "asfarray"):
    np.asfarray = lambda a, dtype=np.float64: np.asarray(a, dtype=dtype)
if not hasattr(np, "float_"):
    np.float_ = np.float64
if not hasattr(np, "int_"):
    np.int_ = np.int64
if not hasattr(np, "bool_"):
    np.bool_ = bool


# ==============================================================================
# STEP 2: CIE 1931 COLOR MATCHING FUNCTIONS (2-DEGREE STANDARD OBSERVER)
# ==============================================================================
def cie1931_color_matching_functions(wavelengths):
    """
    Computes the CIE 1931 Color Matching Functions: x_bar(λ), y_bar(λ), z_bar(λ).
    
    Background Physics:
    -------------------
    The human eye perceives color through three types of cone photoreceptors in the retina:
    - Long-wavelength cones (L) -> sensitive primarily to Red
    - Medium-wavelength cones (M) -> sensitive primarily to Green
    - Short-wavelength cones (S) -> sensitive primarily to Blue
    
    In 1931, the International Commission on Illumination (CIE) standardized these
    curves mathematically. Any light spectrum S(λ) can be multiplied by these three 
    curves to determine how a human observer perceives that light.
    
    Parameters:
    -----------
    wavelengths : numpy array
        Array of optical wavelengths in nanometers (e.g., 380 nm to 780 nm).
        
    Returns:
    --------
    x_bar, y_bar, z_bar : numpy arrays
        Spectral sensitivity values for Red (x_bar), Green/Luminance (y_bar), 
        and Blue (z_bar) at each wavelength.
    """
    wl = wavelengths
    
    # Analytical Gaussian approximations for the x_bar (Red response) curve.
    # Note: x_bar has two peaks—a primary peak in the red region and a smaller lobe in the blue.
    x1 = 1.056 * np.exp(-0.5 * ((wl - 599.8) / 37.9)**2)
    x2 = 0.362 * np.exp(-0.5 * ((wl - 442.0) / 16.0)**2)
    x3 = -0.065 * np.exp(-0.5 * ((wl - 501.1) / 20.4)**2)
    x_bar = np.maximum(0, x1 + x2 + x3)  # Ensure sensitivity is non-negative

    # Analytical Gaussian approximation for the y_bar (Green / Luminance response) curve.
    # Note: y_bar matches the photopic luminous efficiency of the human eye (peak ~ 555 nm).
    y1 = 0.821 * np.exp(-0.5 * ((wl - 568.8) / 46.9)**2)
    y2 = 0.286 * np.exp(-0.5 * ((wl - 530.9) / 22.7)**2)
    y_bar = np.maximum(0, y1 + y2)

    # Analytical Gaussian approximation for the z_bar (Blue response) curve.
    z1 = 1.217 * np.exp(-0.5 * ((wl - 437.0) / 11.8)**2)
    z2 = 0.681 * np.exp(-0.5 * ((wl - 459.0) / 26.0)**2)
    z_bar = np.maximum(0, z1 + z2)

    return x_bar, y_bar, z_bar


# ==============================================================================
# STEP 3: GAUSSIAN SPECTRAL POWER DISTRIBUTION (SPD) GENERATOR
# ==============================================================================
def gaussian_spd(wavelengths, peak, fwhm, amplitude=1.0):
    """
    Generates a Gaussian spectral emission profile for a light source.
    
    Light Source Physics:
    ---------------------
    - Laser Diodes: Have very narrow spectral emission (small FWHM, e.g., 2 to 5 nm).
    - Phosphors (e.g., Ce:YAG): Have broad spectral emission (large FWHM, e.g., 100+ nm) 
      due to electronic energy band transitions.
      
    Parameters:
    -----------
    wavelengths : numpy array
        Target wavelength array in nanometers.
    peak : float
        Center emission wavelength in nanometers (e.g., 455 nm for blue laser).
    fwhm : float
        Full-Width at Half-Maximum of the spectral peak in nanometers.
    amplitude : float
        Peak intensity scale factor.
        
    Returns:
    --------
    spd : numpy array
        Spectral Power Distribution array across all wavelengths.
    """
    # Convert FWHM to Gaussian standard deviation (sigma): fwhm = 2 * sqrt(2 * ln(2)) * sigma
    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    
    # Calculate Gaussian bell curve
    return amplitude * np.exp(-0.5 * ((wavelengths - peak) / sigma)**2)


# ==============================================================================
# STEP 4: SPECTRAL INTEGRATION TO TRISTIMULUS VALUES (X, Y, Z) AND CHROMATICITY (x, y)
# ==============================================================================
def spd_to_xyz(wavelengths, spd, x_bar, y_bar, z_bar):
    """
    Converts a continuous Spectral Power Distribution (SPD) into CIE 1931 Tristimulus
    Values (X, Y, Z) and normalized 2D Chromaticity Coordinates (x, y).
    
    Mathematical Integration:
    -------------------------
    X = ∫ S(λ) * x_bar(λ) dλ  (Amount of Red stimulus)
    Y = ∫ S(λ) * y_bar(λ) dλ  (Amount of Green/Luminance stimulus)
    Z = ∫ S(λ) * z_bar(λ) dλ  (Amount of Blue stimulus)
    
    2D Chromaticity Normalization:
    ------------------------------
    x = X / (X + Y + Z)  (Fraction of Red color perception)
    y = Y / (X + Y + Z)  (Fraction of Green color perception)
    (Note: z = Z / (X + Y + Z) is implicit since x + y + z = 1.0)
    """
    # Step size dλ between discrete wavelength points in nanometers
    d_lambda = wavelengths[1] - wavelengths[0]
    
    # Riemann sum approximation of the spectral integrals
    X = np.sum(spd * x_bar) * d_lambda
    Y = np.sum(spd * y_bar) * d_lambda
    Z = np.sum(spd * z_bar) * d_lambda
    
    # Total stimulus intensity (add small epsilon 1e-12 to prevent division by zero)
    total = X + Y + Z + 1e-12
    
    # Calculate normalized 2D chromaticity coordinates (x, y)
    x = X / total
    y = Y / total
    
    return X, Y, Z, x, y


# ==============================================================================
# STEP 5: MAIN LIGHT ENGINE SIMULATION & CALIBRATION ENGINE
# ==============================================================================
def generate_colorimetry_simulation():
    print("Executing Light Engine Colorimetry Simulation...")

    # 1. Define Wavelength Axis
    # The human visual spectrum spans approximately 380 nm (violet) to 780 nm (deep red).
    # We sample 401 points -> 1.0 nm wavelength resolution.
    wavelengths = np.linspace(380, 780, 401)
    
    # Get standard CIE 1931 color matching curves for this wavelength axis
    x_bar, y_bar, z_bar = cie1931_color_matching_functions(wavelengths)

    # 2. Define Solid-State Light Sources
    # Blue Laser Diode: High-power excitation source at 455 nm with narrow 3.5 nm line
    spd_blue_laser = gaussian_spd(wavelengths, peak=455.0, fwhm=3.5, amplitude=1.0)
    
    # Yellow Phosphor (Ce:YAG): Excited by blue laser light, emits broad yellow spectrum (550 nm peak, 110 nm FWHM)
    spd_yellow_phosphor = gaussian_spd(wavelengths, peak=550.0, fwhm=110.0, amplitude=0.85)
    
    # Red Laser / LED: Pure red primary at 638 nm (12 nm FWHM) for rich color gamut extension
    spd_red_laser = gaussian_spd(wavelengths, peak=638.0, fwhm=12.0, amplitude=0.90)

    # Green Channel Extraction:
    # A dichroic optical filter isolates the green spectrum (~532 nm peak) from the yellow phosphor emission
    green_filter = gaussian_spd(wavelengths, peak=532.0, fwhm=35.0, amplitude=1.0)
    spd_green_channel = spd_yellow_phosphor * green_filter

    # 3. Calculate Individual Tristimulus Values (X, Y, Z) for Primaries
    X_b, Y_b, Z_b, x_blue, y_blue = spd_to_xyz(wavelengths, spd_blue_laser, x_bar, y_bar, z_bar)
    X_g, Y_g, Z_g, x_green, y_green = spd_to_xyz(wavelengths, spd_green_channel, x_bar, y_bar, z_bar)
    X_r, Y_r, Z_r, x_red, y_red = spd_to_xyz(wavelengths, spd_red_laser, x_bar, y_bar, z_bar)

    # 4. Calibrate Power Weights to Target D65 Daylight White Point
    # D65 is the standard white point for displays/cinema (x = 0.3127, y = 0.3290).
    # We solve a 3x3 linear matrix equation:
    #   [ X_r  X_g  X_b ] [ w_r ]   [ X_d65 ]
    #   [ Y_r  Y_g  Y_b ] [ w_g ] = [ Y_d65 ]
    #   [ Z_r  Z_g  Z_b ] [ w_b ]   [ Z_d65 ]
    x_target, y_target = 0.3127, 0.3290
    z_target = 1.0 - x_target - y_target

    M = np.array([
        [X_r, X_g, X_b],
        [Y_r, Y_g, Y_b],
        [Z_r, Z_g, Z_b]
    ])
    target_xyz = np.array([x_target, y_target, z_target])
    
    # Linear algebra solve to find exact power ratios (w_r, w_g, w_b)
    weights = np.linalg.solve(M, target_xyz)
    weights = weights / np.max(weights)  # Normalize highest weight to 1.0

    w_r, w_g, w_b = weights[0], weights[1], weights[2]

    # Synthesize Total Calibrated Light Engine Output Spectrum
    spd_engine_total = w_b * spd_blue_laser + w_g * spd_green_channel + w_r * spd_red_laser
    _, _, _, x_white, y_white = spd_to_xyz(wavelengths, spd_engine_total, x_bar, y_bar, z_bar)

    # 5. Standard Reference Color Gamuts for Comparison
    rec709_r, rec709_g, rec709_b = (0.64, 0.33), (0.30, 0.60), (0.15, 0.06)      # HDTV Standard
    dci_p3_r, dci_p3_g, dci_p3_b = (0.68, 0.32), (0.265, 0.690), (0.15, 0.06)   # Cinema Standard
    d65_white = (0.3127, 0.3290)                                                # Target White Point

    # ==========================================================================
    # STEP 6: PLOTTING VISUALIZATIONS
    # ==========================================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=150)

    # --------------------------------------------------------------------------
    # SUBPLOT 1: Spectral Power Distribution (SPD) Plot
    # --------------------------------------------------------------------------
    ax1.plot(wavelengths, w_b * spd_blue_laser, 'b-', label='Blue Laser (455 nm)', linewidth=2)
    ax1.plot(wavelengths, spd_yellow_phosphor * 0.5, 'y--', label='Yellow Phosphor (Ce:YAG)', linewidth=1.8, alpha=0.8)
    ax1.plot(wavelengths, w_g * spd_green_channel, 'g-', label='Extracted Green Channel (532 nm)', linewidth=2)
    ax1.plot(wavelengths, w_r * spd_red_laser, 'r-', label='Red Laser (638 nm)', linewidth=2)
    ax1.plot(wavelengths, spd_engine_total, 'k-', label='D65 Balanced Engine Output', linewidth=2.5)

    ax1.set_title("Light Engine: Solid-State Spectral Power Distributions", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Wavelength (nm)", fontsize=10)
    ax1.set_ylabel("Normalized Intensity (a.u.)", fontsize=10)
    ax1.set_xlim([380, 750])
    ax1.set_ylim([0, 1.15])
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper right", fontsize=8)

    # --------------------------------------------------------------------------
    # SUBPLOT 2: CIE 1931 Chromaticity Diagram & Gamut Comparison
    # --------------------------------------------------------------------------
    # Calculate the outer curved spectral locus (monochromatic light from 380 nm to 700 nm)
    locus_x, locus_y = [], []
    for wl in np.linspace(380, 700, 200):
        spd_mono = gaussian_spd(wavelengths, peak=wl, fwhm=1.0)
        _, _, _, lx, ly = spd_to_xyz(wavelengths, spd_mono, x_bar, y_bar, z_bar)
        locus_x.append(lx)
        locus_y.append(ly)

    # Draw outer tongue-shaped CIE spectral locus
    ax2.plot(locus_x, locus_y, 'k-', linewidth=2, label="CIE 1931 Spectral Locus")
    # Draw purple line of purples connecting 380 nm (violet) to 700 nm (red)
    ax2.plot([locus_x[0], locus_x[-1]], [locus_y[0], locus_y[-1]], 'k:', linewidth=1)

    # Draw Rec. 709 Gamut Triangle (HDTV Standard)
    rec709_x = [rec709_r[0], rec709_g[0], rec709_b[0], rec709_r[0]]
    rec709_y = [rec709_r[1], rec709_g[1], rec709_b[1], rec709_r[1]]
    ax2.plot(rec709_x, rec709_y, 'grey', linestyle='--', label='Rec. 709 Standard', linewidth=1.5)

    # Draw DCI-P3 Gamut Triangle (Digital Cinema Standard)
    dcip3_x = [dci_p3_r[0], dci_p3_g[0], dci_p3_b[0], dci_p3_r[0]]
    dcip3_y = [dci_p3_r[1], dci_p3_g[1], dci_p3_b[1], dci_p3_r[1]]
    ax2.plot(dcip3_x, dcip3_y, 'magenta', linestyle='-.', label='DCI-P3 Cinema Standard', linewidth=1.5)

    # Draw Simulated Light Engine Gamut Triangle
    engine_x = [x_red, x_green, x_blue, x_red]
    engine_y = [y_red, y_green, y_blue, y_red]
    ax2.plot(engine_x, engine_y, 'b-', linewidth=2.5, label='Simulated Light Engine Gamut')
    ax2.fill(engine_x, engine_y, color='blue', alpha=0.1)

    # Scatter plot primary colors and white points
    ax2.scatter([x_red], [y_red], color='red', s=70, zorder=5, label=f'Red ({x_red:.3f}, {y_red:.3f})')
    ax2.scatter([x_green], [y_green], color='green', s=70, zorder=5, label=f'Green ({x_green:.3f}, {y_green:.3f})')
    ax2.scatter([x_blue], [y_blue], color='blue', s=70, zorder=5, label=f'Blue ({x_blue:.3f}, {y_blue:.3f})')
    ax2.scatter([x_white], [y_white], color='black', marker='*', s=130, zorder=6, label=f'Engine White ({x_white:.3f}, {y_white:.3f})')
    ax2.scatter([d65_white[0]], [d65_white[1]], color='orange', marker='x', s=90, zorder=6, label='D65 Target (0.313, 0.329)')

    ax2.set_title("CIE 1931 Chromaticity Diagram & Gamut Coverage", fontsize=11, fontweight="bold")
    ax2.set_xlabel("CIE x Chromaticity", fontsize=10)
    ax2.set_ylabel("CIE y Chromaticity", fontsize=10)
    ax2.set_xlim([0, 0.8])
    ax2.set_ylim([0, 0.9])
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper right", fontsize=7.5)

    plt.tight_layout()

    # ==========================================================================
    # STEP 7: CROSS-PLATFORM SAFE FILE SAVING
    # ==========================================================================
    # Use relative pathing (`./output_colorimetry`) relative to the directory where 
    # this script is executed. This prevents hardcoded path errors on Mac, Windows, or Linux!
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "output_colorimetry")
    os.makedirs(output_dir, exist_ok=True)
    
    out_img = os.path.join(output_dir, "cie1931_laser_phosphor_gamut.png")
    plt.savefig(out_img, bbox_inches="tight")
    plt.close()

    print(f"\nColorimetry Engine Simulation Complete!")
    print(f"----------------------------------------")
    print(f"Red Primary   : (x={x_red:.4f}, y={y_red:.4f})")
    print(f"Green Primary : (x={x_green:.4f}, y={y_green:.4f})")
    print(f"Blue Primary  : (x={x_blue:.4f}, y={y_blue:.4f})")
    print(f"Engine White  : (x={x_white:.4f}, y={y_white:.4f})")
    print(f"Calibrated Weights (R, G, B): {w_r:.3f}, {w_g:.3f}, {w_b:.3f}")
    print(f"Output Plot Saved To        : {out_img}\n")

    return out_img

if __name__ == "__main__":
    generate_colorimetry_simulation()
