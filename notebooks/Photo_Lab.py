import os, glob, time, functools
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb 
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.visualization import AsinhStretch
from photutils.segmentation import detect_threshold, detect_sources
from scipy import ndimage
from scipy.fft import fft2, ifft2
from PIL import Image
import ipywidgets as widgets
from IPython.display import display, clear_output
from ipyfilechooser import FileChooser
import imageio.v3 as imageio

class PhotoLab:
    def __init__(self, logger_callback=None):
        self.calibrated = None
        self.masks = {}
        self.log = logger_callback if logger_callback else print

    # =========================================================================
    # MODULES DE CHARGEMENT D'IMAGES 
    # =========================================================================

    def _load_single_raw(self, path):
        ext = os.path.splitext(path)[1].lower()
        if ext in ['.fit', '.fits', '.fts']:
            with fits.open(path) as h: 
                return h[0].data.astype(np.float32)
                
        return np.array(Image.open(path)).astype(np.float32)

    def _get_shift(self, img1, img2):
        r = np.mean(ref, axis=2) if ref.ndim == 3 else ref
        t = np.mean(target, axis=2) if target.ndim == 3 else target
        prod = fft2(r) * fft2(t).conj()
        cc = ifft2(prod / (np.abs(prod) + 1e-10))
        shift = np.unravel_index(np.argmax(np.abs(cc)), r.shape)
        return [s if s <= r.shape[i]//2 else s - r.shape[i] for i, s in enumerate(shift)]

    def get_shift_fft(self, ref, target):
        i1 = np.log1p(img1)
        i2 = np.log1p(img2)
        # On force la moyenne à 0 et l'écart-type à 1
        i1 = (i1 - np.mean(i1)) / np.std(i1)
        i2 = (i2 - np.mean(i2)) / np.std(i2)
        return self._get_shift_fft(i1, i2)
        
           
    def load_files(self, file_paths):
        if not file_paths: return
            
        self.dark, self.flat = 0, 1

        # charge les master s'ils existent
        for f in glob.glob("master*"):
            try:
                data = self._load_single_raw(f)
                norm = data / (65535.0 if data.max() > 255 else 255.0)
                if "dark" in f.lower(): 
                    self.dark = norm
                    self.log(f"masterdark loaded")
                if "flat" in f.lower(): 
                    self.flat = norm / (np.mean(norm) + 1e-10)
                    self.log(f"masterflat loaded")

            except: pass

        all_aligned = []
        ref_data, target_shape = None, None
        for i, p in enumerate(sorted(file_paths)):
            if "master" in os.path.basename(p).lower(): continue
                
            try:
                raw = self._load_single_raw(p) / (65535.0 if self._load_single_raw(p).max() > 255 else 255.0)
                self.log(f"Reducing image {i+1}/{len(file_paths)}...")
                calib = (raw - self.dark) / (self.flat + 1e-6)

                if target_shape is None: 
                    target_shape, ref_data = calib.shape, calib
                    all_aligned.append(calib)
                else:
                    if calib.shape != target_shape:
                        if calib.shape[0] == target_shape[1]: calib = np.rot90(calib)
                        else: continue
                    shift = self.get_shift_fft(ref_data, calib)
                    #shift = get_shift_fft(np.log1p(ref_data), np.log1p(calib))
                    #print(shift)
                    aligned = np.stack([ndimage.shift(calib[:,:,c], shift, order=1) for c in range(3)], axis=-1) if calib.ndim == 3 else ndimage.shift(calib, shift, order=1)
                    all_aligned.append(aligned)
                self.log(f"Aligning image {i+1}/{len(file_paths)}...")
            except: pass
            
        if all_aligned:
            stacked = np.mean(all_aligned, axis=0)
            if stacked.ndim == 3:
                for c in range(3): stacked[:,:,c] -= np.percentile(stacked[:,:,c], 25)
            p_high = np.percentile(stacked, 99.9)
            self.calibrated = np.clip(stacked / (p_high + 1e-10), 0, 1)
            if self.calibrated.ndim == 2: self.calibrated = self.calibrated[:,:,None]
            self.log(f"Load and stacking complete ({len(all_aligned)} images)")

    # =========================================================================
    # MODULES DE TRAITEMENT D'IMAGE 
    # =========================================================================

    # --- 0. GESTION DU GRADIENT ---

    def remove_gradient(self, img):
        self.log(f"Removing gradient...")

        try:
            h, w = img.shape[:2]
            out = img.copy()
            
            # Paramètres de la grille (8x8 zones pour M31)
            grid_size = 8
            step_y, step_x = h // grid_size, w // grid_size
            
            for i in range(img.shape[2]):
                chan = img[:,:,i]
                # Création d'une carte de fond basse résolution
                bg_map_small = np.zeros((grid_size, grid_size))
                
                for gy in range(grid_size):
                    for gx in range(grid_size):
                        # Extraction d'une zone (tile)
                        tile = chan[gy*step_y:(gy+1)*step_y, gx*step_x:(gx+1)*step_x]
                        # On prend le 10ème percentile pour éviter de mesurer la galaxie ou les étoiles
                        bg_map_small[gy, gx] = np.percentile(tile, 10)
                
                # On ré-étire cette carte à la taille d'origine avec un lissage cubique
                # pour créer un modèle de gradient très doux
                bg_model = ndimage.zoom(bg_map_small, (h/grid_size, w/grid_size), order=3)
                bg_model = bg_model[:h, :w] # Recalage dimensions
                
                # Soustraction souple
                out[:,:,i] = np.clip(chan - bg_model, 0, 1)
            
            # Rééquilibrage final pour éviter les zones "trouées"
            return out - np.min(out)
        except Exception as e:
            return img

    
    # --- 1. GESTION DES MASQUES (V3 : Morphologie & Protection) ---

    def _make_star_mask_v3_dilated(self, gray_img, sensitivity=3.0):
        """Détecte les étoiles et dilate la sélection pour couvrir les halos."""
        # High Pass pour isoler les pics
        low_freq = ndimage.gaussian_filter(gray_img, sigma=8)
        high_freq = gray_img - low_freq
        
        # Seuillage
        limit = np.std(high_freq) * sensitivity
        binary_mask = (high_freq > limit)
        
        # Dilation (vrai secret pour bien protéger les étoiles)
        return ndimage.binary_dilation(binary_mask, iterations=2).astype(float)

    def _make_galaxy_mask_v3_morpho(self, gray_img, galaxy_sigma):
        """Isole la nébuleuse en supprimant d'abord les étoiles (Opening)."""
        if galaxy_sigma <= 0.1: return np.zeros_like(gray_img)

        # Suppression des étoiles par morphologie (carré 5x5)
        starless = ndimage.grey_opening(gray_img, size=(5,5))
        smooth_starless = ndimage.gaussian_filter(starless, sigma=4)
        
        # Seuillage adaptatif (Moyenne + 1.5 Sigma)
        bg_mean = np.mean(smooth_starless)
        bg_std = np.std(smooth_starless)
        threshold = bg_mean + (1.5 * bg_std)
        
        return (smooth_starless > threshold).astype(float)

    def extract_masks(self, img, stars_sigma=15, galaxy_sigma=20):
        """Coordonne la création des masques avec la 'Bulle de Protection'."""
        gray = np.mean(img, axis=2)
        
        # 1. Masque Etoiles
        raw_star = self._make_star_mask_v3_dilated(gray, sensitivity=3.0)
        s_mask = ndimage.gaussian_filter(raw_star, sigma=stars_sigma)
        
        # 2. Masque Galaxie
        raw_gal = self._make_galaxy_mask_v3_morpho(gray, galaxy_sigma)
        
        # 3. Bulle de Protection : On soustrait les étoiles gonflées du masque galaxie
        if np.max(raw_gal) > 0:
            star_shield = ndimage.binary_dilation(raw_star, iterations=4).astype(float)
            raw_gal = np.clip(raw_gal - star_shield, 0, 1)
            
        g_mask = ndimage.gaussian_filter(raw_gal, sigma=galaxy_sigma)
        
        # 4. Stockage
        self.masks = {
            'stars': np.clip(s_mask, 0, 1),
            'galaxy': np.clip(g_mask, 0, 1),
            'background': np.clip(1.0 - g_mask - s_mask, 0, 1)
        }

    # --- 2. MODULES COSMETIQUES (V2 : clarté & staturation couleurs) ---

    def _step_denoise_harmonized(self, img, amount, galaxy_sigma):
        """Denoise RGB 'Safe'. Pas de HSV (évite le vert). Mixe Fond/Objet."""
        if amount <= 0: return img
        
        # Sigma (y,x,0) pour ne pas mélanger les couleurs
        img_strong = ndimage.gaussian_filter(img, sigma=(amount, amount, 0))
        
        if galaxy_sigma > 0.0 and 'background' in self.masks:
            # Objet lissé à 40% seulement
            img_light = ndimage.gaussian_filter(img, sigma=(amount * 0.4, amount * 0.4, 0))
            m_back = self.masks['background'][..., None]
            return (img_strong * m_back) + (img_light * (1.0 - m_back))
        else:
            return img_strong

    def _step_clarity_multiscale(self, img, strength, galaxy_sigma):
        """Clarity Volume : Mélange détails fins et moyens."""
        if strength <= 0: return img
        
        fine = img - ndimage.gaussian_filter(img, sigma=(2, 2, 0))
        medium = img - ndimage.gaussian_filter(img, sigma=(8, 8, 0))
        structure = (fine * 0.5) + (medium * 1.5)
        
        if galaxy_sigma > 0.1 and 'galaxy' in self.masks:
            return img + (structure * strength * self.masks['galaxy'][..., None])
        else:
            return img + (structure * strength)

    def _step_dynamic_compression(self, img, bp_strength, galaxy_sigma):
        """
        Compression Dynamique V2 (pour le Black Point).
        - Clip préventif (fix crash Clarity).
        - Ancrage du noir (soustraction du plancher de bruit).
        - Gamma sélectif sur le fond.
        """
        if self.calibrated is None: return img
        
        # Sécurité Clarity
        img = np.clip(img, 0, 1)
        
        # Ancrage (Auto-Pedestal)
        if galaxy_sigma > 0.1 and 'background' in self.masks:
             luma = np.mean(img, axis=2)
             bg_vals = luma[self.masks['background'] > 0.5]
             if len(bg_vals) > 0:
                 pedestal = np.percentile(bg_vals, 5) # On enlève le socle
                 img = np.clip(img - pedestal, 0, 1)

        # Gamma
        gamma_strength = 1.0 + ((bp_strength - 1.0) * 2.0)
        
        if galaxy_sigma > 0.1 and 'background' in self.masks:
            gamma_map = 1.0 + ((gamma_strength - 1.0) * self.masks['background'])
            gamma_map = gamma_map[..., None]
            return np.power(img, gamma_map)
        else:
            return np.power(img, gamma_strength)

    def _step_saturation_smart(self, img, sat_amount, galaxy_sigma):
        """Saturation HSV sélective (protège le fond)."""
        if sat_amount == 1.0: return img
        
        hsv = rgb_to_hsv(np.clip(img, 0, 1))
        
        if galaxy_sigma > 0.0 and 'galaxy' in self.masks:
            # Fond -> 0.5, Objet -> sat_amount
            target_sat = 0.5 + (sat_amount - 0.5) * self.masks['galaxy']
            hsv[:,:,1] *= target_sat
        else:
            hsv[:,:,1] *= sat_amount
            
        return hsv_to_rgb(np.clip(hsv, 0, 1))

    # =========================================================================
    # FONCTION PRINCIPALE
    # =========================================================================

    def apply_cosmetics(self, stretch, bp_ratio, clarity, denoise, saturation, galaxy_sigma, stars_sigma, rgb=(1,1,1), do_grad=True):
        if self.calibrated is None: return None
        img = self.calibrated.copy()
        
        # 1. Extraction Masques
        self.log(f"Generating masks...")
        self.extract_masks(img, stars_sigma=stars_sigma, galaxy_sigma=galaxy_sigma)
        
        # 2. Gradient & Balance RGB
        if do_grad and hasattr(self, 'remove_gradient'): 
            self.log(f"Removing gradient...")
            img = self.remove_gradient(img)
            
        self.log(f"Balancing colors...")
        for i in range(3): img[:,:,i] *= rgb[i]

        # 3. Pipeline de Traitement
        # A. Denoise (Texture propre)
        self.log(f"Denoise...")
        img = self._step_denoise_harmonized(img, denoise, galaxy_sigma)
        
        # B. Clarity (Volume & Détails)
        self.log(f"Clarity...")
        img = self._step_clarity_multiscale(img, clarity, galaxy_sigma)
        
        # C. Compression (Gestion du fond noir & Contraste)
        self.log(f"Black point...")
        img = self._step_dynamic_compression(img, bp_ratio, galaxy_sigma)
        
        # D. Saturation (Couleur finale)
        self.log(f"Color saturation...")
        img = self._step_saturation_smart(img, saturation, galaxy_sigma)

        # 4. Stretch Final
        self.log(f"Stretching...")

        return np.clip(AsinhStretch(a=stretch)(img), 0, 1)

        