"""
astro_utils.py
==============
Fonctions utilitaires astro.
Calculs lunaires, interrogation SIMBAD, recherche d'étoiles de référence.
"""

import numpy as np
from astropy.coordinates import get_sun, get_body
from astropy.time import Time
from astroquery.simbad import Simbad

# ---------------------------------------------------------------------------
# Lune
# ---------------------------------------------------------------------------

def moon_phase_angle(time: Time, ephemeris=None) -> float:
    """
    Calcule l'angle de phase lunaire.

    Parameters
    ----------
    time : Time
        Instant de calcul.
    ephemeris : optional
        Source des éphémérides (par défaut : builtin).

    Returns
    -------
    float
        Angle de phase en radians.
    """
    sun = get_sun(time)
    moon = get_body("moon", time, ephemeris=ephemeris)
    elongation = sun.separation(moon)
    return np.arctan2(
        sun.distance * np.sin(elongation),
        moon.distance - sun.distance * np.cos(elongation),
    )


def moon_illumination(time: Time, ephemeris=None) -> float:
    """
    Calcule l'illumination lunaire (0 à 1).

    Parameters
    ----------
    time : Time
        Instant de calcul.
    ephemeris : optional
        Source des éphémérides (par défaut : builtin).

    Returns
    -------
    float
        0 = nouvelle lune, 1 = pleine lune.
    """
    i = moon_phase_angle(time, ephemeris=ephemeris)
    k = (1 + np.cos(i)) / 2.0
    return float(k.value)


def moon_status(illumination: float) -> tuple[str, str]:
    """
    Retourne le label et la couleur CSS selon l'illumination lunaire.

    Parameters
    ----------
    illumination : float
        Valeur entre 0 et 1.

    Returns
    -------
    tuple[str, str]
        (label, couleur_css)
    """
    if illumination > 0.8:
        return "🌕 Pleine Lune", "red"
    if illumination > 0.5:
        return "🌖 Gibbeuse", "orange"
    if illumination > 0.2:
        return "🌙 Croissant", "gold"
    return "🌑 Nouvelle Lune", "green"


# ---------------------------------------------------------------------------
# SIMBAD
# ---------------------------------------------------------------------------

def get_simbad_info(target_name: str) -> dict | None:
    """
    Récupère les informations SIMBAD pour un objet céleste.

    Parameters
    ----------
    target_name : str
        Nom de l'objet (ex : "Vega").

    Returns
    -------
    dict | None
        Dictionnaire avec les infos, ou None si l'objet est introuvable.
    """
    try:
        simbad = Simbad()
        # astroquery < 0.4.8  : flux(V) -> colonne FLUX_V
        # astroquery >= 0.4.8 : V      -> colonne V
        try:
            simbad.add_votable_fields("sptype", "flux(V)", "plx", "rvz_radvel")
            flux_col = "FLUX_V"
        except Exception:
            simbad.add_votable_fields("sptype", "V", "plx", "rvz_radvel")
            flux_col = "V"
        result = simbad.query_object(target_name)

        if result is None:
            return None

        star = result[0]

        def decode(val):
            return val.decode() if isinstance(val, bytes) else val

        cols = result.colnames
        # Noms de colonnes en minuscules depuis astroquery 0.4.8+
        # RA/DEC sont des floats (degrés décimaux), plus des strings
        def _col(name):
            """Résout le nom de colonne indépendamment de la casse."""
            lname = name.lower()
            for c in cols:
                if c.lower() == lname:
                    return c
            return None

        def safe_float(name):
            """Retourne None si colonne absente, masquée ou non numérique."""
            col = _col(name)
            if col is None:
                return None
            try:
                val = star[col]
                if np.ma.is_masked(val):
                    return None
                f = float(val)
                return f if np.isfinite(f) else None
            except (TypeError, ValueError):
                return None

        def safe_str(name, fallback="N/A"):
            """Retourne fallback si colonne absente ou masquée."""
            col = _col(name)
            if col is None:
                return fallback
            try:
                val = star[col]
                if np.ma.is_masked(val):
                    return fallback
                return decode(val) if isinstance(val, (str, bytes)) else str(val)
            except Exception:
                return fallback

        def safe_coord(name):
            """Retourne la coordonnée formatée (float deg -> string HH MM SS ou DD MM SS)."""
            col = _col(name)
            if col is None:
                return "N/A"
            try:
                val = star[col]
                if np.ma.is_masked(val):
                    return "N/A"
                return f"{float(val):.6f}"
            except Exception:
                return "N/A"

        plx_float   = safe_float("plx_value")
        distance_pc = (1000.0 / plx_float) if (plx_float and plx_float > 0) else None

        return {
            "name":         safe_str("main_id", target_name),
            "sptype":       safe_str("sp_type"),
            "mag_v":        safe_float("V"),
            "parallax_mas": plx_float,
            "distance_pc":  distance_pc,
            "rv_kms":       safe_float("rvz_radvel"),
            "ra":           safe_coord("ra"),
            "dec":          safe_coord("dec"),
        }

    except Exception as exc:
        print(f"[SIMBAD] Erreur pour '{target_name}': {exc}")
        return None


# ---------------------------------------------------------------------------
# Étoiles similaires
# ---------------------------------------------------------------------------

def _extract_spectral_type(sptype_str: str) -> str | None:
    """Extrait le type spectral principal (ex : 'A0' depuis 'A0V')."""
    if not sptype_str:
        return None
    return sptype_str[:1].strip()
    #return sptype_str[:2].strip()


def find_similar_stars(target_info: dict, max_results: int = 5) -> list[dict]:
    """
    Trouve des étoiles de même type spectral et de magnitude proche.

    Parameters
    ----------
    target_info : dict
        Infos de la cible (doit contenir 'sptype' et 'mag_v').
    max_results : int
        Nombre maximum de résultats.

    Returns
    -------
    list[dict]
        Liste triée par écart de magnitude.
    """
    if not target_info or not target_info.get("sptype") or target_info.get("mag_v") is None:
        return []

    sptype = _extract_spectral_type(target_info["sptype"])
    mag_v = target_info["mag_v"]

    try:
        simbad = Simbad()
        #simbad.add_votable_fields("sptype", "flux(V)", "plx")
        #results = simbad.query_criteria(f"sptype = {sptype}*")
        Simbad.add_votable_fields('flux(V)')
        #results = Simbad.query_tap(f"SELECT TOP 10 main_id, sp_type FROM basic WHERE sp_type LIKE '{sptype}%'")   
        query = f"""
        SELECT TOP 10 b.main_id, b.ra, b.dec, f.flux, f.filter, b.sp_type
        FROM basic AS b
        JOIN flux AS f ON b.oid = f.oidref
        WHERE f.filter = 'V'
          AND CONTAINS(POINT('ICRS', b.ra, b.dec),
                       CIRCLE('ICRS', {target_info.get('ra', '?')}, {target_info.get('dec', '?')}, 10.0)) = 1
          AND f.flux BETWEEN 0.0 AND 8.0
          AND (sp_type LIKE 'A%' OR sp_type LIKE 'B%')
        """
          #AND f.flux BETWEEN {target_info.get("mag_v") - 1} AND {target_info.get("mag_v") + 1}
          #AND sp_type LIKE '{target_info.get("sptype")[:2].strip()}%'
        results = Simbad.query_tap(query)   
        #print(results)
        """
        main_id          ra               dec        flux filter sp_type
        """
        
        if results is None or len(results) == 0:
            return []

        similar = []
        for star in results:
            try:
                star_mag = float(star["flux"]) if star["flux"] else None
                #if star_mag is None or abs(star_mag - mag_v) > 2.0:
                 #   continue

                def decode(val):
                    return val.decode() if isinstance(val, bytes) else val

                #plx = star["PLX_VALUE"]
                plx_float = 0.0 #float(plx) if plx and plx > 0 else None

                similar.append({
                    "name": decode(star["main_id"]),
                    "sptype": decode(star["sp_type"]),
                    "mag_v": star_mag,
                    "distance_pc": (1000.0 / plx_float) if plx_float else None,
                    "mag_diff": abs(star_mag - mag_v),
                })
            except Exception:
                continue

        similar.sort(key=lambda x: x["mag_v"])

        return similar[:max_results]

    except Exception as exc:
        print(f"[SIMBAD] Erreur recherche similaires : {exc}")
        return []