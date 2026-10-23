"""
ui_widgets.py
=============
Couche interface ipywidgets pour JupyterLab.
Dépend de astro_utils et plot_utils, mais pas l'inverse.
"""

import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display, IFrame
from astropy.time import Time

from astro_utils import get_simbad_info, find_similar_stars, moon_illumination, moon_status
from astropy.utils.iers import conf
from plot_utils import build_dashboard_figure


# ---------------------------------------------------------------------------
# Panneau d'informations HTML
# ---------------------------------------------------------------------------

def _fmt_float(value, fmt: str, fallback: str = "N/A") -> str:
    """Formate un float ou retourne N/A si None."""
    if value is None:
        return fallback
    return format(value, fmt)


def build_info_panel(target_info: dict, moon_illum: float, similar_stars: list) -> widgets.HTML:
    """
    Construit le panneau HTML d'informations sur la cible.

    Parameters
    ----------
    target_info : dict
        Infos SIMBAD de la cible.
    moon_illum : float
        Illumination lunaire (0-1).
    similar_stars : list[dict]
        Étoiles similaires.

    Returns
    -------
    widgets.HTML
    """
    if target_info is None:
        return widgets.HTML("<p style='color:red;'>⚠️ Infos non disponibles</p>")

    moon_label, moon_color = moon_status(moon_illum)

    rows = [
        ("Type spectral", target_info.get("sptype") or "N/A"),
        ("Magnitude V",   _fmt_float(target_info.get("mag_v"), ".2f")),
        ("Distance",      _fmt_float(target_info.get("distance_pc"), ".1f", "N/A") + " pc"
                          if target_info.get("distance_pc") else "N/A"),
        ("Parallaxe",     _fmt_float(target_info.get("parallax_mas"), ".2f", "N/A") + " mas"
                          if target_info.get("parallax_mas") else "N/A"),
        ("Vit. radiale",  _fmt_float(target_info.get("rv_kms"), ".1f", "N/A") + " km/s"
                          if target_info.get("rv_kms") else "N/A"),
        ("RA / DEC",      f"{target_info.get('ra', '?')} / {target_info.get('dec', '?')}"),
    ]

    """
    # Couleurs fixes : fond blanc, texte noir — indépendant du thème JupyterLab
    BG    = "#ffffff"
    FG    = "#111111"
    FG2   = "#444444"
    BORDER= "#cccccc"
    BG2   = "#f0f0f0"
    """
    # Couleurs fixes : fond blanc, texte noir — indépendant du thème JupyterLab
    BG    = "#111111"
    FG    = "#aaaaaa"
    FG2   = "#666666"
    BORDER= "#cccccc"
    BG2   = "#222222"

    table_rows = "".join(
        f"<tr>"
        f"<td style='padding:3px 8px; color:{FG2};'><b>{k}</b></td>"
        f"<td style='padding:3px 8px; color:{FG};'>{v}</td>"
        f"</tr>"
        for k, v in rows
    )

    moon_row = (
        f"<tr>"
        f"<td style='padding:3px 8px; color:{FG2};'><b>Lune</b></td>"
        f"<td style='padding:3px 8px; color:{moon_color};'>{moon_label} ({moon_illum:.0%})</td>"
        f"</tr>"
    )

    html = f"""
    <div style='font-family:monospace; padding:12px; background:{BG}; color:{FG};
                border-radius:6px; border:1px solid {BORDER}; display:inline-block; min-width:280px;'>
      <h3 style='margin:0 0 10px; color:{FG};'>📍 {target_info['name']}</h3>
      <table style='width:100%; border-collapse:collapse; color:{FG};'>
        {table_rows}
        {moon_row}
      </table>
    </div>
    """

    # Tableau étoiles de reférence
    if similar_stars:
        similar_rows = "".join(
            f"<td style='padding:4px 6px; color:{FG};'>{s['name']}</td>"
            f"<td style='padding:4px 6px; text-align:center; color:{FG};'>{s['sptype']}</td>"
            f"<td style='padding:4px 6px; text-align:center; color:{FG};'>{s['mag_v']:.2f}</td>"
            f"<td style='padding:4px 6px; text-align:center; color:{FG};'>"
            #f"{_fmt_float(s.get('distance_pc'), '.1f')} pc</td>"
            f"</tr>"
            for s in similar_stars
        )
        html += f"""
        <div style='margin-top:12px; font-family:monospace; background:{BG}; color:{FG};
                    padding:8px; border-radius:6px; border:1px solid {BORDER};'>
          <h4 style='margin:0 0 6px; color:{FG};'>⭐ Étoiles de référence de type A</h4> 
          <table style='width:100%; font-size:0.88em; border-collapse:collapse; color:{FG};'>
            <thead>
              <tr style='background:{BG2};'>
                <th style='padding:4px 6px; text-align:left; border-top:none; border-bottom:none; border-right:none; color:{FG};'>Nom</th>
                <th style='padding:4px 6px; border-right:none; border-top:none; border-bottom:none; color:{FG};'>Type</th>
                <th style='padding:4px 6px; border-right:none;border-top:none; border-bottom:none;  color:{FG};'>Mag V</th>
                <!-- <th style='padding:4px 6px; border-right:none;border-top:none; border-bottom:none;  color:{FG};'>Distance</th> -->
              </tr>
            </thead>
            <tbody>{similar_rows}</tbody>
          </table>
        </div>
        """
    else:
        html += f"<p style='color:orange; margin-top:10px;'>ℹ️ Pas d'étoiles de référence trouvées</p>"

    return widgets.HTML(html)


# ---------------------------------------------------------------------------
# Dashboard complet (widget composite)
# ---------------------------------------------------------------------------

def build_dashboard(
    target_name: str,
    obs_date: str,
    latitude: float,
    longitude: float,
    height: float,
    utc_offset_hours: float = 2.0,
) -> widgets.Widget:
    """
    Construit le dashboard complet pour une cible.

    Disposition :
    ┌─────────────────┬────────────────────────┐
    │  Info panel     │                        │
    ├─────────────────┤   Figure matplotlib    │
    │  (vide/note)    │   (alt + airmass + sky)│
    └─────────────────┴────────────────────────┘

    Parameters
    ----------
    target_name : str
        Nom de la cible (résolu via SIMBAD / CDS).
    obs_date : str
        Date 'YYYY-MM-DD'.
    latitude, longitude, height : float
        Coordonnées de l'observatoire.
    utc_offset_hours : float
        Décalage UTC local.

    Returns
    -------
    widgets.Widget
        Layout ipywidgets prêt à être affiché.
    """
    # --- Infos SIMBAD --------------------------------------------------
    target_info = get_simbad_info(target_name)
    if target_info is None:
        return widgets.HTML(
            f"<p style='color:red; font-size:16px;'>"
            f"❌ '{target_name}' introuvable dans SIMBAD</p>"
        )

    similar_stars = find_similar_stars(target_info)
    midnight = Time(obs_date + " 00:00:00") - utc_offset_hours
    moon_illum = moon_illumination(midnight)

    # --- Panneau infos (gauche) ----------------------------------------
    info_panel = build_info_panel(target_info, moon_illum, similar_stars)

    simbad_url = (
        f"https://simbad.cds.unistra.fr/simbad/sim-basic"
        f"?Ident={target_name}&submit=SIMBAD+search"
    )
    simbad_link = widgets.HTML(
        f"<a href='{simbad_url}' target='_blank' "
        f"style='font-family:monospace; font-size:0.85em;'>🔗 Ouvrir SIMBAD</a>"
    )

    left_col = widgets.VBox(
        [info_panel, simbad_link],
        layout=widgets.Layout(width="340px", padding="4px"),
    )

    # --- Figure matplotlib (droite) ------------------------------------
    fig_out = widgets.Output(layout=widgets.Layout(flex="1"))
    with fig_out:
        fig = build_dashboard_figure(
            target_name, obs_date, latitude, longitude, height, utc_offset_hours
        )
        plt.show()
        plt.close(fig)

    # --- Layout final --------------------------------------------------
    return widgets.HBox(
        [left_col, fig_out],
        layout=widgets.Layout(width="100%", align_items="flex-start"),
    )


# ---------------------------------------------------------------------------
# Contrôles de sélection (date + cible + bouton)
# ---------------------------------------------------------------------------

def build_controls(targets: list[str]) -> tuple[widgets.DatePicker, widgets.Dropdown, widgets.Button]:
    """
    Construit les widgets de contrôle du notebook.

    Parameters
    ----------
    targets : list[str]
        Liste de cibles (peut contenir des séparateurs commençant par '--').

    Returns
    -------
    tuple[DatePicker, Dropdown, Button]
    """
    date_picker = widgets.DatePicker(
        description="Date :",
        disabled=False,
        style={"description_width": "60px"},
    )

    target_dropdown = widgets.Dropdown(
        options=targets,
        description="Cible :",
        value="--",
        style={"description_width": "60px"},
        layout=widgets.Layout(width="320px"),
    )

    run_button = widgets.Button(
        description="Afficher",
        button_style="info",
        tooltip="Affiche le dashboard pour la cible sélectionnée",
        icon="telescope",
    )

    return date_picker, target_dropdown, run_button


def build_ui(
    targets: list[str],
    latitude: float,
    longitude: float,
    height: float,
    utc_offset_hours: float = 2.0,
) -> None:
    """
    Point d'entrée principal : construit et affiche l'interface complète.

    Parameters
    ----------
    targets : list[str]
        Liste des cibles.
    latitude, longitude, height : float
        Coordonnées de l'observatoire.
    utc_offset_hours : float
        Décalage UTC local.
    """
    # Disable automatic astropy skycoords downloads
    conf.auto_download = False
    conf.auto_max_age = None

    date_picker, target_dropdown, run_button = build_controls(targets)
    output = widgets.Output()

    controls = widgets.HBox(
        [date_picker, target_dropdown, run_button],
        layout=widgets.Layout(gap="12px", align_items="center", margin="8px 0"),
    )

    display(controls, output)

    def on_click(_):
        with output:
            output.clear_output(wait=True)

            target = target_dropdown.value
            if not target or target.startswith("--"):
                print("⚠️  Sélectionnez une cible valide (pas un séparateur).")
                return

            if date_picker.value is None:
                print("⚠️  Sélectionnez une date.")
                return

            obs_date = str(date_picker.value)

            moon_illum = moon_illumination(Time(obs_date + " 00:00:00"))
            #print(f"Illumination lunaire : {moon_illum:.0%}")

            try:
                dashboard = build_dashboard(
                    target_name=target,
                    obs_date=obs_date,
                    latitude=latitude,
                    longitude=longitude,
                    height=height,
                    utc_offset_hours=utc_offset_hours,
                )
                display(dashboard)
            except Exception as exc:
                print(f"❌ Erreur dashboard : {exc}")
                print(f"   Vérifiez que '{target}' existe dans SIMBAD.")

    run_button.on_click(on_click)
    