import numpy as np
import scipy.optimize as opt
from scipy.integrate import trapezoid
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# -------------------------------------------------
# L3 Physique - Univ. Rennes - Winter 2026
# Author : Yveline Lebreton

# -------------------------------------------------
# Constants
plt.rcParams['figure.figsize'] = (8, 5)
ln2 = np.log(2.0)

# --------------------------------------------------
# Gaussian fitting

def gaussian(x, a0, a1, a2):
    """Gaussian profile (FWHM parameterization)."""
    return a0 * np.exp(-ln2 * ((x - a1) / a2) ** 2)

def split_gaussian(x, a0, a1, a2, a3):
    """Gaussian with different left/right widths."""
    return np.where(
        x <= a1,
        gaussian(x, a0, a1, a2),
        gaussian(x, a0, a1, a3)
    )

# --------------------------------------------------
# Coefficient of determination R²

def rsq(y, f):
    stot = np.sum((y - np.mean(y)) ** 2)
    sres = np.sum((y - f) ** 2)
    return 1.0 - sres / stot

# --------------------------------------------------
# Peak detection

def detect_peak_auto_width(wavelength, intensity, approx_lower, approx_upper, fraction=0.5):
    mask = (wavelength >= approx_lower) & (wavelength <= approx_upper)
    x_window = wavelength[mask]
    y_window = intensity[mask]

    if len(x_window) == 0:
        print("Error: No data in the specified window. Exit.")
        sys.exit(1)

    peak_idx = np.argmax(y_window)
    peak_wavelength = x_window[peak_idx]
    peak_intensity = y_window[peak_idx]

    # Find points above fraction*peak for FWHM estimate
    significant_idx = np.where(y_window >= fraction * peak_intensity)[0]
    if len(significant_idx) < 2:
        # fallback: use quarter-width of the window
        fit_lower = peak_wavelength - (x_window[-1]-x_window[0])/4
        fit_upper = peak_wavelength + (x_window[-1]-x_window[0])/4
    else:
        fit_lower = x_window[significant_idx[0]]
        fit_upper = x_window[significant_idx[-1]]

    fwhm_guess = fit_upper - fit_lower

    return peak_wavelength, fit_lower, fit_upper, fwhm_guess

# --------------------------------------------------
# Fit the selected wavelength range with a given fitting function

def fit_profile(fitfun, wavelength, intensity,
                lower_limit, upper_limit,
                initial_guess):

    mask = (wavelength >= lower_limit) & (wavelength <= upper_limit)
    x = wavelength[mask]
    y = intensity[mask]

    if len(x) < 5:
        raise ValueError("Not enough data points in selected range.")

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    popt, pcov = opt.curve_fit(fitfun, x, y, p0=initial_guess)
    perr = np.sqrt(np.diag(pcov))
    yfit = fitfun(x, *popt)

    print("Fit parameters:", *popt)
    print("Parameter uncertainties:", *perr)
    print(f"R²: {rsq(y, yfit):.6f}")

    auc_value = trapezoid(yfit, x)
    print(f"AUC (scipy.trapezoid): {auc_value:.6f}")
    print()

    # Plot
    fig, ax = plt.subplots()
    ax.plot(x, y, label="Data")
    ax.plot(x, yfit, 'r--', label="Fit")
    ax.minorticks_on()
    ax.set_xlabel(r'$\lambda\ (\AA)$', fontsize=14)
    ax.set_ylabel(r'I (ADU)', fontsize=14)
    ax.legend()
    plt.tight_layout()
    plt.show()
    plt.close(fig)

    return popt, pcov

# --------------------------------------------------
# Main programme

if __name__ == '__main__':

    # Ensure file path
    base_dir = Path(__file__).resolve().parent

    # Choose spectrum
    spectrum = ['Spectres/2026/_alcyone_20250930_92.dat',
               'Spectres/2026/_alp_cep_20250930_856.dat',
               'Spectres/2019/ngc6543_20191021_784.dat',
               'Spectres/2019/ngc6543_20191021_784_continuum_off.dat',
               'Spectres/2019/agdra_20191021_877',
               'Spectres/2019/hd145454_20180924_918'
                ]

    print("Available spectra : alcyone (0), alp cep (1), ngc6543 (2), ngc6543 continuum (3), agdra (4), hd145454 (5)")
    i = int(input("Choose your spectrum: "))
    #i=0
    file_path = base_dir/spectrum[i]

    if not file_path.exists():
        print(f"Error: File not found: {file_path}")
        sys.exit(1)

    data = np.genfromtxt(file_path)
    wavelength = data[:, 0]
    intensity = data[:, 1]

    # Full spectrum plot
    fig, ax = plt.subplots()
    ax.plot(wavelength, intensity)
    ax.minorticks_on()
    ax.set_xlabel(r'$\lambda\ (\AA)$', fontsize=14)
    ax.set_ylabel(r'I (ADU)', fontsize=14)
    plt.tight_layout()
    plt.show()
    plt.close(fig)

    proceed = input("Do you want to perform the Gaussian fit? (y/n): ").strip().lower()
    if proceed not in ('y', 'yes', 'o', 'oui'):
      print("Fitting skipped by user. Exiting program.")
      sys.exit(0)  

    # Approximate expected line window
    print("Fitting of a specified line:")
    approx_lower = float(input("Enter approximate lower wavelength: "))
    approx_upper = float(input("Enter approximate upper wavelength: "))

    # Automatic peak detection and FWHM
    peak, lower_limit, upper_limit, a2_guess = detect_peak_auto_width(
        wavelength, intensity, approx_lower, approx_upper
    )
    print(f"Detected peak at λ = {peak:.2f} Å")
    print(f"Fitting window: {lower_limit:.2f} Å → {upper_limit:.2f} Å")
    print(f"Initial FWHM guess: {a2_guess:.2f} Å\n")

    # Initial amplitude guess
    idx = (wavelength >= lower_limit) & (wavelength <= upper_limit)
    a0_guess = intensity[idx].max()

    # Gaussian fit
    print("Gaussian fit:")
    initial_guess_gauss = (a0_guess, peak, a2_guess)
    fit_profile(gaussian, wavelength, intensity, lower_limit, upper_limit, initial_guess_gauss)

    # Split-Gaussian fit
    print("Split-Gaussian fit:")
    initial_guess_split = (a0_guess, peak, a2_guess, a2_guess)
    fit_profile(split_gaussian, wavelength, intensity, lower_limit, upper_limit, initial_guess_split)
