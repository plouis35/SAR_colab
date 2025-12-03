import ipywidgets as widgets
from IPython.display import display
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import matplotlib.ticker as ticker

import numpy as np
import os

class SpectroDashboard:
    def __init__(self):
        """
        initialisation du dashboard
        """
        # données publiques
        self.original_data = None   # données brutes (pour les stats)
        self.display_data = None    # données réduites (pour l'affichage)
        self.spectra_lines = []     # 
        self.fill_obj = None        
        self.lines_markers = [] 
        self.last_x, self.last_y = None, None    # dernier spectre (x, y) chargé pour la colorisation
        self.reference_doppler = 6562.8      # H_Alpha par défaut
        self.lines_file = "lines.txt"

        #  création des objects de l'interface 
        self._create_widgets()
        
        # widgets de sortie 
        self.out_image = widgets.Output(layout=widgets.Layout(width='100%'))
        self.out_spectrum = widgets.Output(layout=widgets.Layout(width='100%'))
        
        # création des figures (image + spectre)
        with plt.style.context('dark_background'):
            
            # image...
            with self.out_image:
                self.fig_img, self.ax_img = plt.subplots(figsize=(10, 4), constrained_layout=True)
                self.fig_img.canvas.layout.width = '100%'  
                self.fig_img.canvas.layout.height = 'auto'
                self.fig_img.set_label(' ')
                self.ax_img.axis("off")
                
                def format_coord(x,y):
                    return "(x, y) [adu]: ({:.0f}, {:.0f})".format(x,y)
                self.ax_img.format_coord=format_coord
                
                self.fig_img.patch.set_facecolor('black')
                self.ax_img.set_facecolor('black')
                
#                self.im_obj = self.ax_img.imshow(np.zeros((10,10)), cmap='inferno', origin='lower')
                self.im_obj = self.ax_img.imshow(plt.imread('./logo_SAR.png'), cmap='inferno', origin='upper')
                
                self.cbar = plt.colorbar(self.im_obj, ax=self.ax_img)
                plt.show()
                
            # Spectre...
            with self.out_spectrum:
                self.fig_spec, self.ax_spec = plt.subplots(figsize=(10, 4), constrained_layout=True)
                self.fig_spec.canvas.layout.width = '100%' 
                self.fig_spec.canvas.layout.height = 'auto'
                self.fig_spec.set_label(' ')
                
                def format_coord(x,y):
                    return "Lambda: {:.2f}, Intensity: {:.2f}".format(x,y)
                self.ax_spec.format_coord=format_coord

                self.fig_spec.patch.set_facecolor('black')
                self.ax_spec.set_facecolor('black')
                
                self.ax_spec.set_xlabel(r"Longueur d'onde ($\AA$)", color='white')
                self.ax_spec.set_ylabel("Intensité relative", color='white')
                self.ax_spec.grid(True, alpha=0.2, linestyle='--', color='white')
                self.ax_spec.xaxis.set_major_locator(ticker.AutoLocator())
                self.ax_spec.xaxis.set_minor_locator(ticker.AutoMinorLocator())

                
                # axes en blanc
                for spine in self.ax_spec.spines.values():
                    spine.set_color('white')
                self.ax_spec.tick_params(axis='x', colors='white')
                self.ax_spec.tick_params(axis='y', colors='white')
                
                # axe doppler (vitesse)
                c = 299792.458         # en dur pour ne pas importer astropy
                self.forward = lambda lam: c * (lam - self.reference_doppler) / self.reference_doppler
                self.inverse = lambda v: v * self.reference_doppler / c + self.reference_doppler
                
                self.secax = self.ax_spec.secondary_xaxis('top', functions=(self.forward, self.inverse))
                self.secax.set_xlabel(f"Doppler (km/s @ {self.reference_doppler})", color='white') #'#ff6666')
                self.secax.tick_params(axis='x', colors='white') #'#ff6666')
                self.secax.xaxis.set_major_locator(ticker.AutoLocator())
                self.secax.xaxis.set_minor_locator(ticker.AutoMinorLocator())

                self.secax.set_visible(False)

                plt.show()

        self._build_layout()    # affiche le tout
        
        # définitions des callbacks (actions) pour chaque widget
        self.slider_cuts.observe(self._on_cuts_change, names='value')
        self.dropdown_cmap.observe(self._on_cmap_change, names='value')
        self.btn_clear.on_click(self._on_clear_click)
        self.btn_velo.on_click(self._on_velocity_click)
        self.btn_color.on_click(self._on_color_click)
        self.btn_lines.on_click(self._on_lines_click)

    def _create_widgets(self):
        """
        crée tous les widgets (ipywidgets)
        """
        # nom de l'image
        self.label_name = widgets.Label(value="Aucune")

        # choix colormap
        self.dropdown_cmap = widgets.Dropdown(
            options=['viridis', 'plasma', 'inferno', 'gray', 'seismic_r', 'gist_heat', 'nipy_spectral_r'], 
            value='inferno', 
            description='Colormap:', 
            layout=widgets.Layout(width='200px')
        )
        
        # seuils en pourcentage
        self.slider_cuts = widgets.FloatRangeSlider(
            value=[5, 99.5], 
            min=0, max=100, step=0.1, 
            description='Cuts (%):', 
            continuous_update=False, 
            layout=widgets.Layout(width='700px'),
            readout_format='.4f'
        )
        
        # Labels pour afficher les valeurs réelles ADU
        self.label_cuts_values = widgets.Label(value="ADU: [ - , - ]")

        # labels des stats
        #self.stats_html = widgets.HTML(value="Waiting for data...")

        # boutons
        btn_layout = widgets.Layout(width='120px')
        self.btn_clear = widgets.Button(description="CLEAR", button_style='warning', icon='eraser', layout=btn_layout)
        self.btn_color = widgets.Button(description="COLOR", icon='toggle-off', layout=btn_layout)
        self.btn_lines = widgets.Button(description="LINES", icon='toggle-off', layout=btn_layout)
        self.btn_velo = widgets.Button(description="DOPPLER", icon='toggle-off', layout=btn_layout)

    def _build_layout(self):
        """
        assemble et organise tous les widgets
        """
        # ligne des controles de l'image
        row_image = widgets.HBox(
            #[self.label_name, self.dropdown_cmap, self.slider_cuts], #, self.label_cuts_values], 
            [self.dropdown_cmap, self.slider_cuts, self.label_cuts_values], #, self.label_cuts_values], 
            layout=widgets.Layout(align_items='center', justify_content='space-between', border='1px solid #555', padding='5px', width='100%')
        )

        # ligne des boutons
        row_buttons = widgets.HBox(
            [self.btn_clear, self.btn_color, self.btn_lines, self.btn_velo], 
            layout=widgets.Layout(justify_content='space-around', margin='10px 0', width='100%')
        )

        # cadres + empilement vertical / horizontal
        self.main_widget = widgets.VBox([
            row_image, 
            widgets.Box([self.out_image], layout=widgets.Layout(justify_content='center', width='100%')),
            #widgets.HBox([self.stats_html], layout=widgets.Layout(justify_content='center', margin='5px 0', border='1px solid #555', width='100%')), 
            widgets.Box([self.out_spectrum], layout=widgets.Layout(justify_content='center', width='100%')),
            row_buttons
        ], layout=widgets.Layout(border='2px solid #333', padding='10px', width='98%'))


    def _read_lines_file(self):
        """Lecture du fichier contenant les raies à afficher"""
        lines_list = []
        if not os.path.exists(self.lines_file):
            return []
            
        try:
            with open(self.lines_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'): continue
                    parts = line.split(',')
                    if len(parts) >= 2:
                        try:
                            lines_list.append((float(parts[0].strip()), parts[1].strip()))
                        except ValueError: continue
        except Exception as e:
            print(f"Erreur lecture fichier: {e}")
            
        return lines_list
        
    def show(self):
        """
        Affiche le dashboard dans la cellule de sortie (output widget)
        cette cellule peut être isolée et déplacée en haut ou à droite avec le menu "create new view for cell output"
        de sorte qu'elle soit toujours visible même si on scrolle le code :-) 
        """
        display(self.main_widget)

    # callbacks

    def load_image(self, data, name="Sans nom", binning=1):
        """
        Charge l'image
        data : np.array, données de l'image
        name : str, nom de l'image à afficher
        binning : int (défaut 1). Si > 1, réduit l'image affichée pour accélérer le zoom/pan
        """
        # ne pas utiliser les unités définies avec Astropy.u (inconnues de matplotlib)
        if hasattr(data, 'value'): data = data.value
        data = np.array(data, dtype=float)
        
        # on applique un binning si demandé (pour accelerer l'affichage)
        self.original_data = data 
        
        if binning > 1:
            self.display_data = data[::binning, ::binning]
        else:
            self.display_data = data

        # Reset Slider à une position standard (5% - 99.5%)
        self.slider_cuts.unobserve(self._on_cuts_change, names='value')
        self.slider_cuts.value = [5, 99.5] 
        self.slider_cuts.observe(self._on_cuts_change, names='value')
        
        # Initialisation Affichage
        self.im_obj.set_data(self.display_data)
        
        # Gestion correcte des coordonnées (Extent) même avec le binning
        h, w = data.shape
        self.im_obj.set_extent([0, w, 0, h])
        
        # Application des cuts initiaux
        self._apply_cuts([5, 99.5])
        
        # on affiche qq stats sur l'image chargée
        print(f"{name} : bin={binning}, shape={data.shape}, min={np.nanmin(data):.0f}, avg={np.nanmean(data):.1f}, max={np.nanmax(data):.0f}, std={np.std(data):.1f}")

        self.fig_img.canvas.draw_idle()

    
    def load_spectrum(self, x, y, label=None):
        """
        chargement d'un spectre
        x : np.array, données des abscisses
        y : np.array, données des ordonnées
        label : str, légende 
        fwhm_neon : int, largeur en px d'une raie du neon (pour calculer R max théorique)
        """
        # si c'est un CCDData, ne pas prendre les unités définies par Astropy (inconnu de matplotlib)
        if hasattr(x, 'value'): x = x.value
        if hasattr(y, 'value'): y = y.value
        
        if label is None: label = f"Spec {len(self.spectra_lines)+1}"
        
        line, = self.ax_spec.plot(x, y, linewidth=1, label=label)
        self.spectra_lines.append(line)
        self.ax_spec.relim()
        self.ax_spec.autoscale_view()
        
        # Légendes
        leg = self.ax_spec.legend(loc='upper right', fontsize='small', facecolor='black', edgecolor='white')
        for text in leg.get_texts(): text.set_color("white")

        # on sauve le dernier spectre chargé (pour ne coloriser que ce dernier)
        self.last_x, self.last_y = x, y

        # on affiche des stats sur le spectre chargé
        dispersion = np.abs(x[1] - x[0])
        
        print(f"{label} : {dispersion=:.4f} Å/px")

        self.fig_spec.canvas.draw_idle()


    def _apply_cuts(self, range_percent):
        """
        calcule et applique les nouveaux cuts à partir de pourcentages
        range_percent: (int, int), valeurs passés par le widget
        """
        if self.original_data is None: return
        low_pct, high_pct = range_percent
        
        # On calcule les seuils sur un sous-ensemble pour etre plus rapide
        sample = self.original_data[::10, ::10]
        vmin, vmax = np.nanpercentile(sample, [low_pct, high_pct])

        # refuser le cas ou on réduit à zéro le range
        if vmin == vmax: vmax = vmin + 1

        # met à jour l'image
        self.im_obj.set_clim(vmin, vmax)
        self.label_cuts_values.value = f"ADU (min, max): {vmin:.0f}, {vmax:.0f}"
        self.fig_img.canvas.draw_idle()

    def _on_cuts_change(self, change):
        """
        callback pour le slider des cuts
        change : widgets selection info
        """
        self._apply_cuts(change['new'])


    def _on_cmap_change(self, change):
        """
        choix de la colormap
        change : widgets selection info
        """
        self.im_obj.set_cmap(change['new'])
        self.fig_img.canvas.draw_idle()

    def _on_clear_click(self, b):
        """
        bouton CLEAR : nettoie les spectres, légendes, raies et colorisation
        b : widgets selection info
        """
        # spectres
        for line in self.spectra_lines: line.remove()
        self.spectra_lines = []

        # legendes
        if self.ax_spec.get_legend(): self.ax_spec.get_legend().remove()
        
        # colorisation
        if self.fill_obj: 
            self.fill_obj.remove()
            self.fill_obj = None
            self.btn_color.icon = 'toggle-off'
            
        # raies
        for m in self.lines_markers: 
            m.remove()
        self.lines_markers = []
        self.btn_lines.icon = 'toggle-off'
        
        # velocité
        self.secax.set_visible(False)
        self.btn_velo.icon = 'toggle-off'
        
        self.fig_spec.canvas.draw_idle()

    def _on_color_click(self, b):
        """
        bouton COLOR : affiche la couleur de la longeur d'onde sous le spectre
        b : widgets selection info (inutilisé)       
        """
        if self.fill_obj:
            self.fill_obj.remove()
            self.fill_obj = None
            self.btn_color.icon = 'toggle-off'
        else:
            if not self.spectra_lines: return

            # on ne travaille que sur le dernier spectre chargé
            x, y = self.last_x, self.last_y
            extent = [x.min(), x.max(), y.min(), y.max()]
            
            # Gradient physique calé sur x (longueur d'onde)
            gradient = np.linspace(x.min(), x.max(), 500).reshape(1, -1)

            # creation de l'image 'arc en ciel' corresponsante aux lambda affichés
            im = self.ax_spec.imshow(
                gradient, aspect='auto', extent=extent, 
                cmap= 'nipy_spectral',  
                vmin=3800, vmax=7500, # attention il faudra adapter pour le NIR ...
                origin='lower', alpha=0.6, zorder=0
            )

            # création des polygones qui ne vont remplir que sous le spectre
            y_min = self.ax_spec.get_ylim()[0]
            verts = list(zip(x, y)) + [(x[-1], y_min), (x[0], y_min)]
            poly = Polygon(verts, transform=self.ax_spec.transData)
            im.set_clip_path(poly)

            # remplissage 
            self.fill_obj = im
            
            self.btn_color.icon = 'toggle-on'
            
        self.fig_spec.canvas.draw_idle()

    def _on_lines_click(self, b):
        """
        bouton COLOR : affiche la couleur de la longeur d'onde sous le spectre
        b : widgets selection info (inutilisé)       
        """
        if self.lines_markers:
            # --- OFF ---
            for m in self.lines_markers: m.remove()
            self.lines_markers = []
            self.btn_lines.icon = 'toggle-off'
        else:
            # --- ON ---
            lines_data = self._read_lines_file()
            
            if not lines_data: 
                print(f"Aucune raie trouvée dans {self.lines_file}")
                return
            
            #  on regarde ce qui est affiché AVANT de tracer
            xmin, xmax = self.ax_spec.get_xlim()
            ymin, ymax = self.ax_spec.get_ylim()
            y_range = ymax - ymin
            
            # Compteur pour gérer le zig-zag des textes uniquement sur les raies visibles
            displayed_count = 0 
            
            for wvl, name in lines_data:
                # on ignore tout ce qui est hors champ
                if wvl < xmin or wvl > xmax:
                    continue
                
                # Tracé de la ligne
                line = self.ax_spec.axvline(x=wvl, color='#ff6666', linestyle='--', alpha=0.6, linewidth=0.8)
                
                # Calcul hauteur zig-zag (basé sur les raies affichées seulement)
                level = 0.95 - (displayed_count % 3) * 0.07 
                y_text_pos = ymin + y_range * level
                
                text = self.ax_spec.text(wvl, y_text_pos, f" {name}", color='#ff6666', fontsize=8, rotation=90, verticalalignment='top')
                
                self.lines_markers.extend([line, text])
                displayed_count += 1
                
            self.btn_lines.icon = 'toggle-on'
            
            # Petit message si aucune raie n'est dans la zone
            if displayed_count == 0:
                print(f"Aucune raie du fichier n'est visible dans la zone {xmin:.0f}-{xmax:.0f} A")
                
        self.fig_spec.canvas.draw_idle()   
        
    def _on_velocity_click(self, b):
        """
        bouton DOPPLER : affiche l'axe montrant les vitesses par effet doppler 
        b : widgets selection info (inutilisé)             
        """
        current_state = self.secax.get_visible()
        self.secax.set_xlabel(f"Doppler (km/s @ {self.reference_doppler})", color='white') #'#ff6666')
        self.secax.set_visible(not current_state)
        self.btn_velo.icon = 'toggle-on' if not current_state else 'toggle-off'
        self.fig_spec.canvas.draw_idle()