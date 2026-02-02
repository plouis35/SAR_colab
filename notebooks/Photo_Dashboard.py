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
from PIL import Image
import ipywidgets as widgets
from IPython.display import display, clear_output
from ipyfilechooser import FileChooser
import imageio.v3 as imageio

# --- CONFIGURATION ---
plt.close('all')
plt.ioff()
plt.style.use('dark_background')

class PhotoDashboard:
    def __init__(self, lab):
        self.lab = lab
        self.output_plot = widgets.Output()
        self.output_hist = widgets.Output()
        
        self.status_dot = widgets.HTML(value="<div style='width: 12px; height: 12px; border-radius: 50%; background-color: #2ecc71; margin-right: 8px;'></div>")
        self.log_text = widgets.Label(value="Ready")
        self.log_box = widgets.HBox([self.status_dot, self.log_text], layout={'align_items': 'center', 'margin': '0 0 5px 10px'})
        self.lab.log = lambda m: setattr(self.log_text, 'value', f"{m}")
        #plt.ioff()  # Désactive l'affichage auto
        
        with self.output_plot:
            self.fig, self.ax = plt.subplots(figsize=(9, 7), facecolor='black')
            plt.subplots_adjust(0, 0, 1, 1)
            self.fig.canvas.header_visible = False
            self.fig.canvas.toolbar_position = 'right'
            #self.fig.canvas.layout.width, self.fig.canvas.layout.height = '99%', '99%'
            #self.fig.canvas.layout.min_height = '500px'
            display(self.fig.canvas)

        with self.output_hist:
            self.fig_h, self.ax_h = plt.subplots(figsize=(3.2, 2), facecolor='black')
            plt.subplots_adjust(0, 0, 1, 1); 
            self.fig_h.canvas.header_visible = False
            self.fig_h.canvas.toolbar_visible = False
            display(self.fig_h.canvas)

        #plt.ion() # On réactive le mode normal 
        
        self.fc = FileChooser(os.getcwd(), title='<b>Select a directory : </b>', dir_only=True, show_only_dirs=True)
        self.fl = widgets.SelectMultiple(options=[])#, layout={'height': '120px', 'width': '99%'})
        self.btn_load = widgets.Button(description="Load", button_style='info')#, layout={'width':'49%'})
        self.btn_exp = widgets.Button(description="Export", button_style='success')#, layout={'width':'49%'})
        self.btn_mask_toggle = widgets.ToggleButton(value=False, description='Show Masks', button_style='warning', icon='eye')#, layout={'width':'99%'})
        self.chk_grad = widgets.Checkbox(value=False, description='Gradient removal', indent=False)
        
        #s_sty, s_lay = {'description_width': '99'}, {'width': '99%'}
        s_sty, s_lay = {}, {}
        
        self.sliders = {
            'Stars σ': widgets.FloatSlider(value=0.0, min=0.0, max=20, step=0.5, description='Stars σ', style=s_sty, layout=s_lay, continuous_update=False),
            'Galaxy σ': widgets.FloatSlider(value=0.0, min=0.0, max=80, step=0.1, description='Galaxy σ', style=s_sty, layout=s_lay, continuous_update=False),
            'Stretch': widgets.FloatSlider(value=0.01, min=0.01, max=0.1, step=0.001, description='Stretch', style=s_sty, layout=s_lay, continuous_update=False),
            'BlackPt': widgets.FloatSlider(value=1.0, min=0.0, max=2, step=0.01, description='BlackPt', style=s_sty, layout=s_lay, continuous_update=False),
            'Clarity': widgets.FloatSlider(value=0.0, min=0.0, max=2, step=0.1, description='Clarity', style=s_sty, layout=s_lay, continuous_update=False),
            'Denoise': widgets.FloatSlider(value=0.0, min=0.0, max=2, step=0.1, description='Denoise', style=s_sty, layout=s_lay, continuous_update=False),
            'Red': widgets.FloatSlider(value=1.0, min=0.5, max=2.0, step=0.05, description='Red', style=s_sty | {'handle_color': 'red'}, layout=s_lay, continuous_update=False),
            'Green': widgets.FloatSlider(value=1.0, min=0.5, max=2.0, step=0.05, description='Green', style=s_sty | {'handle_color': 'green'}, layout=s_lay, continuous_update=False),
            'Blue': widgets.FloatSlider(value=1.0, min=0.5, max=2.0, step=0.05, description='Blue', style=s_sty | {'handle_color': 'blue'}, layout=s_lay, continuous_update=False),
            'Saturation': widgets.FloatSlider(value=0.0, min=0.0, max=3.0, step=0.1, description='Saturation',style=s_sty, layout=s_lay, continuous_update=False),
        }

        self.fc.register_callback(self._update_list)
        self.btn_load.on_click(self._on_load_click)
        self.btn_exp.on_click(lambda b: self._export())
        self.btn_mask_toggle.observe(lambda x: self._refresh(False, f"Show masks : {'Masques' if x['new'] else 'Image'}"), 'value')
        self.chk_grad.observe(lambda x: self._refresh(False, f"Remove gradient : {'ON' if x['new'] else 'OFF'}"), 'value')
        
        for name, s in self.sliders.items():
            is_mask = name in ['Stars σ', 'Galaxy σ']
            s.observe(lambda x, n=name, m=is_mask: self._refresh(m, f"Settings change : {n}"), 'value')

    def busy_indicator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            self._set_status(True)
            try:
                return func(self, *args, **kwargs)
            finally:
                self._set_status(False)
        return wrapper

    def _set_status(self, busy=True):
        color = "#e74c3c" if busy else "#2ecc71"
        self.status_dot.value = f"<div style='width: 12px; height: 12px; border-radius: 50%; background-color: {color}; margin-right: 8px;'></div>"

    def _update_list(self, c):
        path = c.selected_path
        f = sorted(glob.glob(os.path.join(path, "*.png")) + glob.glob(os.path.join(path, "*.tif*")) + glob.glob(os.path.join(path, "*.fit*")), key=os.path.getmtime, reverse=True)
        self.fl.options = [(os.path.basename(x), x) for x in f]

    @busy_indicator
    def _on_load_click(self, b):
        self.lab.log("Load in progress...")
        # On force un petit temps mort pour laisser le thread UI passer au rouge
        time.sleep(0.05) 
        self.lab.load_files(self.fl.value)
        self._refresh(True, "Refreshing image...")

    @busy_indicator
    def _refresh(self, update_masks=False, log_msg=None):
        if self.lab.calibrated is None: return
        if log_msg: self.lab.log(log_msg)
        
        xlim, ylim = self.ax.get_xlim(), self.ax.get_ylim()
        has_zoom = xlim != (0.0, 1.0) and xlim != (0, 1)
        
        if update_masks: 
            self.lab.extract_masks(self.lab.calibrated, 
                           self.sliders['Stars σ'].value, 
                           self.sliders['Galaxy σ'].value)
            
        self.ax.clear()

        if self.btn_mask_toggle.value and self.lab.masks:
            self.lab.log(f"Rendering masks...")
            m = np.zeros((*self.lab.calibrated.shape[:2], 3))
            
            # 1. Étoiles -> Toujours en Rouge
            m += self.lab.masks['stars'][:,:,None] * [1, 0, 0]
            
            # 2. Galaxie -> Vert UNIQUEMENT si le masque est actif (sigma > 0)
            # Si galaxy_sigma est à 0, on ne rajoute pas de vert.
            g_sigma = self.sliders['Galaxy σ'].value
            if g_sigma > 0:
                m += self.lab.masks['galaxy'][:,:,None] * [0, 1, 0]
            
            # 3. Fond -> Bleu nuit (on peut aussi le masquer si sigma est à 0)
            if g_sigma > 0:
                m += self.lab.masks['background'][:,:,None] * [0, 0, 0.2]
            else:
                # En mode "étoiles/amas", on peut laisser le fond noir ou gris très sombre
                pass 
            
            self.ax.imshow(m)
            
        else:
            # Traitement normal de l'image
            self.lab.log(f"Applying cosmetics...")
            rgb = (self.sliders['Red'].value, self.sliders['Green'].value, self.sliders['Blue'].value)
            
            img = self.lab.apply_cosmetics(
                    stretch=self.sliders['Stretch'].value,     
                    bp_ratio=self.sliders['BlackPt'].value,     
                    clarity=self.sliders['Clarity'].value,     
                    denoise=self.sliders['Denoise'].value,     
                    saturation=self.sliders['Saturation'].value,
                    galaxy_sigma=self.sliders['Galaxy σ'].value, 
                    stars_sigma=self.sliders['Stars σ'].value,  
                    rgb=rgb, 
                    do_grad=self.chk_grad.value
                )        
            # Affichage de l'image traitée
            # np.squeeze permet d'enlever les dimensions inutiles pour imshow
            self.ax.imshow(np.squeeze(img), cmap='gray' if img.ndim==2 or img.shape[2]==1 else None) 
        
        if has_zoom: self.ax.set_xlim(xlim); self.ax.set_ylim(ylim)
        self.ax.axis('off'); self.fig.canvas.draw_idle()

        # --- MISE À JOUR DE L'HISTOGRAMME ---
        self.ax_h.clear()
        
        # On étire les données pour l'affichage de l'histogramme
        # Cela permet de voir la "forme" du signal après stretch
        raw_stretch = AsinhStretch(a=self.sliders['Stretch'].value)(self.lab.calibrated)
        data = raw_stretch.ravel()
        
        # On filtre les pixels noirs ou saturés pour un histogramme plus lisible
        useful = data[(data > 0.0001) & (data < 0.9999)]
        
        if len(useful) > 0:
            # On définit les bords du graphique pour zoomer sur le signal utile
            h_min, h_max = np.percentile(useful, [0.5, 99.5])
            
            self.ax_h.hist(useful, bins=100, range=(h_min, h_max), color='cyan', alpha=0.4)
            
            # --- SYNCHRONISATION DU BLACK POINT ---
            # On reproduit exactement le calcul de apply_cosmetics
            # (Mais appliqué sur la version stretchée pour l'affichage graphique)
            p25 = np.percentile(raw_stretch, 25)
            p50 = np.percentile(raw_stretch, 50)
            std_fond = p50 - p25
            
            # Position de la ligne rouge (Black Point) sur l'histogramme
            cutoff = p25 + (std_fond * (self.sliders['BlackPt'].value - 1.0) * 5)
            
            self.ax_h.axvline(cutoff, color='red', linestyle='-', linewidth=2, alpha=0.8)
            
        self.ax_h.axis('off')
        self.ax.axis('off')
        #self.fig.canvas.draw()      # Force le redessin immédiat (plot principal)
        #self.fig_h.canvas.draw()    # Force le redessin immédiat (histogramme)
        self.fig_h.canvas.draw_idle()
        self.fig.canvas.draw_idle()
        self.lab.log(f"Ready")

    @busy_indicator
    def _export(self):
        if not self.fc.selected_path: return

        self.lab.log(f"Applying cosmetics...")
        rgb = (self.sliders['Red'].value, self.sliders['Green'].value, self.sliders['Blue'].value)
        
        img = self.lab.apply_cosmetics(
                stretch=self.sliders['Stretch'].value,     
                bp_ratio=self.sliders['BlackPt'].value,     
                clarity=self.sliders['Clarity'].value,     
                denoise=self.sliders['Denoise'].value,     
                saturation=self.sliders['Saturation'].value,
                galaxy_sigma=self.sliders['Galaxy σ'].value,  
                stars_sigma=self.sliders['Stars σ'].value,  
                rgb=rgb, 
                do_grad=self.chk_grad.value
            )

        p = os.path.join(self.fc.selected_path, f"photo_{int(time.time())}")
        img = (img * 65535).astype(np.uint16)
        
        # Sauvegarde directe en TIFF (sans compression)
        imageio.imwrite(p + '.tiff', img, compression=None)        
        self.lab.log(f"Export complete : {p}.tiff")

 
    @busy_indicator    
    def display(self):
        self.lab.log(f"Render image...")
        clear_output()
        rgb_box = widgets.VBox([widgets.HTML("<b style='color:white; font-size:11px;'>COLOR BALANCE</b>"), 
                        self.sliders['Red'], self.sliders['Green'], self.sliders['Blue']], 
                               layout={'border': '1px solid #444', 'padding': '5px', 'margin': '5px 0'})

        mask_box = widgets.VBox([widgets.HTML("<b style='color:white; font-size:11px;'>MASKS</b>"), 
                        self.sliders['Stars σ'], self.sliders['Galaxy σ']], 
                               layout={'border': '1px solid #444', 'padding': '5px', 'margin': '5px 0'})
        
        cosm_box = widgets.VBox([widgets.HTML("<b style='color:white; font-size:11px;'>COSMETICS</b>"), 
                        self.sliders['Stretch'], self.sliders['BlackPt'], 
                        self.sliders['Clarity'], self.sliders['Denoise'], self.sliders['Saturation']],
                               layout={'border': '1px solid #444', 'padding': '5px', 'margin': '5px 0'})

        left = widgets.VBox([
            self.fc, self.fl, 
            widgets.HBox([self.btn_load, self.btn_exp]), 
            self.btn_mask_toggle, self.chk_grad, 
            widgets.HTML("<b style='color:cyan; font-size:11px;'>Histogram (Red=Cutoff)</b>"), 
            self.output_hist, 
            rgb_box, mask_box, cosm_box
        ], layout={'width':'25%'})#, 'padding':'15px'})
        right = widgets.VBox([self.log_box, self.output_plot], layout={'width':'70%'})
        display(widgets.HBox([left, right]))#, layout={'width':'99%', 'background-color':'#000'}))
        self.lab.log(f"Ready")

        

