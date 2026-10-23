"""
plot_utils.py
=============
Fonctions de visualisation matplotlib pour préparer une observation.
MOntre le graphiques d'altitude, airmass et sky plot.
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.coordinates import get_sun, get_body, SkyCoord, AltAz, EarthLocation
from astropy.time import Time
from astropy import units as u
from astroplan import Observer, FixedTarget
from astroplan.plots import plot_sky, plot_airmass
from astroplan.plots import dark_style_sheet


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_observer(latitude: float, longitude: float, height: float) -> tuple[Observer, EarthLocation]:
    """
    Construit un Observer pour astroplan et EarthLocation.

    Parameters
    ----------
    latitude, longitude : float
        Coordonnées géographiques en degrés.
    height : float
        Altitude en mètres.

    Returns
    -------
    tuple[Observer, EarthLocation]
    """
    location = EarthLocation(lat=latitude * u.deg, lon=longitude * u.deg, height=height * u.m)
    observer = Observer(location=location)
    return observer, location


def _build_time_grid(obs_date: str, utc_offset_hours: float = 2.0, n_points: int = 1000):
    """
    Construit la grille temporelle centrée sur minuit local.

    Parameters
    ----------
    obs_date : str
        Date au format 'YYYY-MM-DD'.
    utc_offset_hours : float
        Décalage UTC (positif = est). Défaut : 2 (CEST).
    n_points : int
        Nombre de points sur la grille.

    Returns
    -------
    tuple[Time, np.ndarray, Time]
        (grille de temps, delta en heures, minuit UTC)
    """
    midnight_utc = Time(obs_date + " 00:00:00") - utc_offset_hours * u.hour
    delta = np.linspace(-12, 12, n_points) * u.hour
    times = midnight_utc + delta
    return times, delta, midnight_utc


# ---------------------------------------------------------------------------
# Graphiques individuels
# ---------------------------------------------------------------------------

def plot_altitude(
    ax: plt.Axes,
    target_name: str,
    obs_date: str,
    latitude: float,
    longitude: float,
    height: float,
    utc_offset_hours: float = 2.0,
) -> None:
    """
    Trace la courbe d'altitude (Soleil, Lune, cible) en fonction du temps.

    Parameters
    ----------
    ax : plt.Axes
        Axes matplotlib où tracer.
    target_name : str
        Nom de la cible (résolu via CDS).
    obs_date : str
        Date 'YYYY-MM-DD'.
    latitude, longitude, height : float
        Coordonnées de l'observatoire.
    utc_offset_hours : float
        Décalage UTC local.
    """
    _, location = _build_observer(latitude, longitude, height)
    times, delta, _ = _build_time_grid(obs_date, utc_offset_hours)
    frame = AltAz(obstime=times, location=location)

    sun_altaz = get_sun(times).transform_to(frame)
    moon_altaz = get_body("moon", times).transform_to(frame)
    target_coord = SkyCoord.from_name(target_name)
    target_altaz = target_coord.transform_to(frame)

    # Extraire les valeurs numériques (degrés / heures) pour matplotlib
    delta_h        = delta.to(u.hour).value
    sun_alt        = sun_altaz.alt.deg
    moon_alt       = moon_altaz.alt.deg
    target_alt     = target_altaz.alt.deg
    target_az      = target_altaz.az.deg

    ax.plot(delta_h, sun_alt,  color="orange",    ls="--", lw=2, label="Soleil")
    ax.plot(delta_h, moon_alt, color=[0.75] * 3,  ls="--", lw=2, label="Lune")

    scatter = ax.scatter(
        delta_h, target_alt,
        c=target_az,
        #label=target_name, 
        lw=0, s=20, cmap="viridis", zorder=5,
    )
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Azimut [deg]")

    ax.fill_between(
        delta_h, 0, 90,
        sun_alt < 0,
        color="0.5", zorder=0, alpha=0.3, #label="Crépuscule",
    )
    ax.fill_between(
        delta_h, 0, 90,
        sun_alt < -18,
        color="red", zorder=0, alpha=0.5, #label="Nuit",
    )

    ax.set_xticks(np.arange(-12, 13, 2))
    ax.set_xlim(-7, 10)
    ax.set_ylim(0, 90)
    ax.set_xlabel("Heures depuis minuit")
    ax.set_ylabel("Altitude [deg]")
    #ax.set_title(f"Altitude", fontsize=12, fontweight="bold")
    ax.legend()#loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)


def plot_airmass_panel(
    ax: plt.Axes,
    target_name: str,
    obs_date: str,
    latitude: float,
    longitude: float,
    height: float,
    utc_offset_hours: float = 2.0,
) -> None:
    """
    Trace la courbe d'airmass pour la cible.

    Parameters
    ----------
    ax : plt.Axes
        Axes matplotlib où tracer.
    target_name : str
        Nom de la cible.
    obs_date : str
        Date 'YYYY-MM-DD'.
    latitude, longitude, height : float
        Coordonnées de l'observatoire.
    utc_offset_hours : float
        Décalage UTC local.
    """
    observer, _ = _build_observer(latitude, longitude, height)
    _, _, midnight_utc = _build_time_grid(obs_date, utc_offset_hours)

    plot_airmass(
        FixedTarget.from_name(target_name),
        observer,
        midnight_utc,
        max_airmass=3.0,
        altitude_yaxis=True,
        style_sheet=dark_style_sheet,
        ax=ax,
    )
    ax.set_title(f"Airmass — {target_name}", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)


def plot_sky_panel(
    ax,
    target_name: str,
    obs_date: str,
    latitude: float,
    longitude: float,
    height: float,
    utc_offset_hours: float = 2.0,
    n_points: int = 10,
) -> None:
    """
    Trace le sky plot (projection polaire) pour la cible.

    Parameters
    ----------
    ax : plt.Axes
        Axes polaires matplotlib.
    target_name : str
        Nom de la cible.
    obs_date : str
        Date 'YYYY-MM-DD'.
    latitude, longitude, height : float
        Coordonnées de l'observatoire.
    utc_offset_hours : float
        Décalage UTC local.
    n_points : int
        Nombre de points de la trajectoire.
    """
    observer, _ = _build_observer(latitude, longitude, height)
    _, _, midnight_utc = _build_time_grid(obs_date, utc_offset_hours)

    observe_times = midnight_utc + np.linspace(-4, 6, n_points) * u.hour
    plot_sky(
        target=FixedTarget.from_name(target_name),
        observer=observer,
        time=observe_times,
        ax=ax,
        grid=True,
        style_sheet=dark_style_sheet
    )


# ---------------------------------------------------------------------------
# Dashboard 
# ---------------------------------------------------------------------------

def build_dashboard_figure(
    target_name: str,
    obs_date: str,
    latitude: float,
    longitude: float,
    height: float,
    utc_offset_hours: float = 2.0,
) -> plt.Figure:
    """
    Construit la figure matplotlib complète du dashboard (altitude + sky).

    Parameters
    ----------
    target_name : str
        Nom de la cible.
    obs_date : str
        Date 'YYYY-MM-DD'.
    latitude, longitude, height : float
        Coordonnées de l'observatoire.
    utc_offset_hours : float
        Décalage UTC local.

    Returns
    -------
    plt.Figure
        Figure prête à être affichée ou intégrée dans un widget Output.
    """
    
    plt.style.use('dark_background')

    #fig = plt.figure(figsize=(16, 10))
    fig = plt.figure(figsize=(16, 4))
    # Grille : 2 lignes × 3 colonnes
    # Altitude occupe toute la ligne du bas (colonnes 0-1)
    # Airmass : haut gauche (col 0-1), Sky : haut droite (col 2)
    gs = fig.add_gridspec(2, 3, hspace=0.4, wspace=0.35)

    #ax_alt = fig.add_subplot(gs[1, :2])       # bas gauche-centre
    ax_alt = fig.add_subplot(gs[:, 1])       # bas gauche-centre
    #ax_airmass = fig.add_subplot(gs[0, :2])   # haut gauche-centre
    ax_sky = fig.add_subplot(gs[:, 2], projection="polar")  # droite, pleine hauteur

    plot_altitude(ax_alt, target_name, obs_date, latitude, longitude, height, utc_offset_hours)
    #plot_airmass_panel(ax_airmass, target_name, obs_date, latitude, longitude, height, utc_offset_hours)
    plot_sky_panel(ax_sky, target_name, obs_date, latitude, longitude, height, utc_offset_hours)

    fig.suptitle(
        f"{target_name}  —  {obs_date}",
        fontsize=10, fontweight="bold", y=1.01,
    )
    return fig
    