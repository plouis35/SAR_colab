import numpy as np
import os
import traceback

import ipywidgets as widgets
from IPython.display import display

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import matplotlib.ticker as ticker


class ImageWidget:
    def __init__(self):
        """
        initialisation du dashboard
        """
        # données publiques
        self.original_data = None   # données brutes (pour les stats)
        self.display_data = None    # données réduites (pour l'affichage)
        self.fill_obj = None        

        #  création des objects de l'interface 
        self._create_widgets()
        
        # widgets de sortie 
        self.out_image = widgets.Output(layout=widgets.Layout(width='100%'))
        
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
                
                self.im_obj = self.ax_img.imshow(np.zeros((10,10)), cmap='inferno', origin='lower')
                #self.im_obj = self.ax_img.imshow(plt.imread('./LOGO_SAR.png'), cmap='inferno', origin='upper')
                
                self.cbar = plt.colorbar(self.im_obj, ax=self.ax_img, location='right', shrink=0.6)
                plt.show()
                
        self._build_layout()    # affiche le tout
        
        # définitions des callbacks (actions) pour chaque widget
        self.slider_cuts.observe(self._on_cuts_change, names='value')
        self.dropdown_cmap.observe(self._on_cmap_change, names='value')

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


        # cadres + empilement vertical / horizontal
        self.main_widget = widgets.VBox([
            row_image, 
            widgets.Box([self.out_image], layout=widgets.Layout(justify_content='center', width='100%')),
        ], layout=widgets.Layout(border='2px solid #333', padding='10px', width='98%'))


    def show(self):
        """
        Affiche le dashboard dans la cellule de sortie (output widget)
        cette cellule peut être isolée et déplacée en haut ou à droite avec le menu "create new view for cell output"
        de sorte qu'elle soit toujours visible même si on scrolle le code :-) 
        """
        display(self.main_widget)

    # callbacks

    def show_image(self, data, name="Sans nom", binning=1, **kwargs):
        """
        Charge l'image
        data : np.array, données de l'image
        name : str, nom de l'image à afficher
        binning : int (défaut 1). Si > 1, réduit l'image affichée pour accélérer le zoom/pan
        kwargs : parametres supplémetaire (non utilisés pour l'instant)
        """
        # ne pas utiliser les unités définies avec Astropy.u (inconnues de matplotlib)
        if hasattr(data, 'value'): data = data.value
        data = np.asarray(data, dtype=float)  # gère PIL Image ET np.array
        
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
        
        # Gestion des coordonnées (Extent) même avec le binning
        h, w = data.shape
        self.im_obj.set_extent([0, w, 0, h])
        
        # Application des cuts initiaux
        self._apply_cuts([5, 99.5])

        # on affiche un label (on les enleve tous d'abord)
        for txt in self.ax_img.texts: txt.remove()
        self.ax_img.text(0.02, 0.02, name, fontsize=10, verticalalignment='top', bbox=dict(boxstyle="round", facecolor="white", alpha=0.5))
        
        # on affiche qq stats sur l'image chargée
        print(f"INFO: affichage de l'image {name} : bin={binning}, shape={data.shape}, min={np.nanmin(data):.0f}, avg={np.nanmean(data):.1f}, max={np.nanmax(data):.0f}, stddev={np.std(data):.1f}")

        self.fig_img.canvas.draw_idle()

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
