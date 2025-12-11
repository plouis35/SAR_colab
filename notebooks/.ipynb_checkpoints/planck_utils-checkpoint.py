from astropy.io import fits
from astropy import units as u
import numpy as np
import matplotlib.pyplot as plt
from specutils import Spectrum
from specutils.analysis import equivalent_width
from specutils.spectra import SpectralRegion

# --- FONCTION D'ESTIMATION LOG(G) ---
def estimer_logg_balmer(ew_angstrom, teff_k):
    """
    Estime log(g) pour une étoile de type A
    basé sur la largeur équivalente.
    """
    if teff_k < 7000 or teff_k > 13000:
        return np.nan, "Hors domaine de température"

    # Point pivot pour A0V : Teff=9600 K, logg=4.1 => EW ~ 16.0 Å
    # Sensibilité : +1.0 log(g) ajoute environ 4.5 Å à la largeur
    delta_ew = ew_angstrom - 16.0 
    logg_estime = 4.1 + (delta_ew / 4.5)
    
    # Correction de température : 
    # Si l'étoile est plus froide que 9600K, la raie grandit naturellement.
    # Il faut donc réduire le log(g) estimé pour compenser cet effet thermique.
    delta_t = teff_k - 9600
    correction_t = - (delta_t / 1000) * 0.15
    
    return (logg_estime + correction_t, "")

"""
# --- CHARGEMENT DES DONNÉES ---
fichier_fits = 'data/plouis/c_Vega.fit' # Votre fichier
temperature_etoile = 9600   # Température mesurée précédemment

with fits.open(fichier_fits) as hdul:
    flux = hdul[0].data
    header = hdul[0].header
    # Reconstruction de l'axe des longueurs d'onde
    wvl = header['CRVAL1'] + header['CDELT1'] * np.arange(len(flux))
    try:
        flux_unit = u.Unit(header['BUNIT'])
    except:
        flux_unit = u.dimensionless_unscaled
"""

# --- DÉFINITION DES ZONES ---
centre = 4340.5   #(H-gamma) 

# Zones de continuum (loin des ailes de la raie)
zones_cont = [
    (centre - 120, centre - 100),
    (centre + 100, centre + 120)
]

# --- FIT DU CONTINUUM ---
# On récupère (mask de booléens) les points qui sont DANS les zones de la raie
mask_cont = np.zeros_like(wvl, dtype=bool)
for (start, end) in zones_cont:
    mask_cont |= (wvl >= start) & (wvl <= end)

# on masque les données
x_points = wvl[mask_cont]
y_points = flux[mask_cont]

# On ajuste une droite (deg=1) sur ces points
coefs = np.polyfit(x_points, y_points, 1) 
model_continuum = np.poly1d(coefs)

# On génère la courbe du continuum pour tout le spectre
flux_cont = model_continuum(wvl)

# --- NORMALISATION ET CALCUL EW ---
flux_norm = flux / flux_cont

# Préparation pour Specutils
spec_norm_obj = Spectrum(spectral_axis=wvl * u.AA, flux=flux_norm * u.dimensionless_unscaled)
region_raie = SpectralRegion((centre - 30)*u.AA, (centre + 30)*u.AA)

# Calcul EW officiel en Ang
ew = equivalent_width(spec_norm_obj, regions=region_raie)
val_ew = ew.to(u.AA).value

# --- LOG(G) ---
logg_calc, commentaire = estimer_logg_balmer(val_ew, temperature_etoile)

print(f"--- RÉSULTATS POUR {fichier_fits} ---")
print(f"Température réf : {temperature_etoile} K")
print(f"EW (H-gamma)    : {val_ew:.3f} Å")
print(f"Log(g) estimé   : {logg_calc:.2f} ({commentaire})")


"""
# --- VISUALISATION ---
# Spectre Normalisé + Aire EW
plt.figure(figsize=(10, 8))
plt.plot(wvl, flux_norm, color='black', linewidth=1, label='Spectre Normalisé')
plt.axhline(1.0, color='blue', linestyle='--', alpha=0.5) # Le continuum est à 1.0 par définition ici

# Colorier la zone EW
mask_ew = (wvl > centre - 30) & (wvl < centre + 30)
plt.fill_between(wvl[mask_ew], flux_norm[mask_ew], 1.0, color='red', alpha=0.4, label=f'Aire EW = {val_ew:.2f} Å')

plt.xlim(centre - 150, centre + 150)
plt.ylim(0, 1.2)
plt.xlabel("Longueur d'onde (Å)")
plt.ylabel("Flux Normalisé")
plt.title(f"2. Mesure : log(g) ~ {logg_calc:.2f}")
plt.legend()
plt.grid(True, alpha=0.2)

plt.tight_layout()
plt.show()
"""