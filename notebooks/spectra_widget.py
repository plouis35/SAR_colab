import numpy as np
import os
import traceback

import ipywidgets as widgets
from IPython.display import display

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import matplotlib.ticker as ticker


class SpectraWidget:
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
        self.out_spectrum = widgets.Output(layout=widgets.Layout(width='100%'))
        
        # création des figures (image + spectre)
        with plt.style.context('dark_background'):
                            
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
        self.btn_clear.on_click(self._on_clear_click)
        self.btn_velo.on_click(self._on_velocity_click)
        self.btn_color.on_click(self._on_color_click)
        self.btn_lines.on_click(self._on_lines_click)

    def _create_widgets(self):
        """
        crée tous les widgets (ipywidgets)
        """

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

        # ligne des boutons
        row_buttons = widgets.HBox(
            [self.btn_clear, self.btn_color, self.btn_lines, self.btn_velo], 
            layout=widgets.Layout(justify_content='space-around', margin='10px 0', width='100%')
        )

        # cadres + empilement vertical / horizontal
        self.main_widget = widgets.VBox([
            #widgets.Box([self.out_image], layout=widgets.Layout(justify_content='center', width='100%')),
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

    ### callbacks

    def show_spectrum(self, x, y, label=None, **kwargs):
        """
        chargement d'un spectre
        x : np.array, données des abscisses
        y : np.array, données des ordonnées
        label : str, légende 
        kwargs : paramètres suplémentaires pour le plot (optionel)
        """
        # Nettoyage des unités et conversion numpy
        if hasattr(x, 'value'): x = x.value
        if hasattr(y, 'value'): y = y.value
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        
        # Sécurité : on retire les points invalides qui cassent l'autoscale
        mask = np.isfinite(x) & np.isfinite(y)
        x, y = x[mask], y[mask]

        if label is None: label = f"Spec {len(self.spectra_lines)+1}"

        # On travaille dans le widget de sortie dédié au spectre
        with self.out_spectrum:
            # On trace le spectre
            line, = self.ax_spec.plot(x, y, label=label, linewidth=1, **kwargs)
            self.spectra_lines.append(line)
            
            # On force Matplotlib à oublier les anciennes limites
            self.ax_spec.relim()
            
            # Si un axe secondaire existe, il peut bloquer l'autoscale. 
            # On force donc les limites manuellement pour être sûr.
            if len(x) > 0:
                padding_y = (y.max() - y.min()) * 0.05 if y.max() != y.min() else 1.0
                self.ax_spec.set_xlim(x.min(), x.max())
                self.ax_spec.set_ylim(y.min() - padding_y, y.max() + padding_y)

            # Mise à jour de la légende
            leg = self.ax_spec.legend(fontsize='small', facecolor='black', edgecolor='white')
            if leg:
                for text in leg.get_texts(): text.set_color("white")

            # Rafraîchissement forcé du canvas
            self.fig_spec.canvas.draw_idle() 
            self.fig_spec.canvas.draw() # On double pour certains backends
            
        self.last_x, self.last_y = x, y
 
        print(f"INFO: affichage du spectre '{label}' : {len(x)} pts, X:[{x.min():.1f}:{x.max():.1f}]")


    def clear_spectra(self):
        self._on_clear_click(self)
        
    def _on_clear_click(self, b):
        """
        bouton CLEAR : nettoie les spectres, légendes, raies et colorisation
        b : widgets selection info
        """
        try:
            with self.out_spectrum:
                # 1. Supprimer les spectres
                for line in self.spectra_lines:
                    line.remove()
                self.spectra_lines = []

                # 2. Supprimer les raies et leurs labels
                for m in self.lines_markers:
                    m.remove()
                self.lines_markers = []

                # 3. Supprimer les textes sur l'axe
                for txt in list(self.ax_spec.texts):
                    txt.remove()

                # 4. Supprimer la colorisation
                if self.fill_obj: 
                    self.fill_obj.remove()
                    self.fill_obj = None
                    self.btn_color.icon = 'toggle-off'

                # 5. Reset interface et Doppler
                self.btn_lines.icon = 'toggle-off'
                self.secax.set_visible(False)
                self.btn_velo.icon = 'toggle-off'
                
                # Supprimer la légende
                if self.ax_spec.get_legend():
                    self.ax_spec.get_legend().remove()

                # Rafraîchir
                self.fig_spec.canvas.draw()
                
        except Exception as e:
            print(f"Erreur lors du clear: {e}")
            
            
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
        with self.out_spectrum:
            if self.lines_markers:
                # --- ACTION OFF ---
                for m in self.lines_markers:
                    try:
                        m.remove()
                    except:
                        pass
                self.lines_markers = []
                
                # On nettoie les textes au cas où
                for txt in list(self.ax_spec.texts):
                    txt.remove()
                    
                self.btn_lines.icon = 'toggle-off'
            else:
                # --- ACTION ON ---
                lines_data = self._read_lines_file()
                if not lines_data: return
                
                xmin, xmax = self.ax_spec.get_xlim()
                ymin, ymax = self.ax_spec.get_ylim()
                y_range = ymax - ymin
                displayed_count = 0 
                
                for wvl, name in lines_data:
                    if wvl < xmin or wvl > xmax: continue
                    
                    # Ligne
                    line = self.ax_spec.axvline(x=wvl, color='#ff6666', linestyle='--', alpha=0.6, linewidth=0.8)
                    # Texte
                    level = 0.95 - (displayed_count % 3) * 0.07 
                    y_text_pos = ymin + y_range * level
                    text = self.ax_spec.text(wvl, y_text_pos, f" {name}", color='#ff6666', fontsize=8, rotation=90, verticalalignment='top')
                    
                    self.lines_markers.extend([line, text])
                    displayed_count += 1
                
                self.btn_lines.icon = 'toggle-on'
            
            self.fig_spec.canvas.draw()
            
        
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