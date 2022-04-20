#
# Lightkurve Extension to interface with Pyaneti Modeling Library
# - https://github.com/oscaribv/pyaneti
#

from collections import OrderedDict
from collections.abc import Iterable
import os
from os import path
from pathlib import Path
import re
import shutil
import sys
import warnings

# memoization would not introduce additional dependency
# as lightkurve has already depends on it.
from memoization import cached

import astropy.constants
from astropy.table import Table
from astropy.time import Time
from astropy import units as u
import numpy as np
import lightkurve as lk

import logging

logger = logging.getLogger(__name__)
log_stdout_handler = logging.StreamHandler(stream=sys.stdout)

def log_to_stdout_at_level(level):
    """Set the module logger to the given level, and make it prints to `stdout`.
        (In Jupyter notebook, info level logging is not printed as there is no handler)
    """
    # ensure we add the handler only once, even if the function
    # is called multiple times, to avoid repeated logging.
    # We set an attribute to the logger instance as the state, because in the event
    # the module is reloaded, the logger will not be reloaded, so the state
    # needs to be associated with the logger instance itself.
    if not getattr(logger, "_stdout_handler_added", False):
        logger.addHandler(log_stdout_handler)
        logger._stdout_handler_added = True
    logger.setLevel(level)


class PyanetiEnv:
    """Define the directories used for 1 modeling session"""

    def __init__(self, home_dir, alias, sector):
        self.home_dir = Path(home_dir)
        self.alias = alias
        self.sector = sector

    @property
    def base_in_dir(self):
        """The base directory for all modeling input of the Pyaneti installation"""
        return Path(self.home_dir, "inpy")

    @property
    def base_out_dir(self):
        """The base directory for all modeling output of the Pyaneti installation"""
        return Path(self.home_dir, "outpy")

    @property
    def target_in_dir(self):
        return Path(self.base_in_dir, self.alias)

    @property
    def target_out_dir(self):
        return Path(self.base_out_dir, f"{self.alias}_out")

    @property
    def lc_dat_filename(self):
        sector_str = to_sector_str(self.sector)
        if sector_str is not None:
            return f"{self.alias}_lc_s{sector_str}.dat"
        else:
            # no sector is specified, i.e., all available ones
            # OPEN: if we want to know the sectors, we'd need to change
            # the API so that it it takes in the actual lc objects
            return f"{self.alias}_lc.dat"

    @property
    def lc_dat_filepath(self):
        return Path(self.target_in_dir, self.lc_dat_filename)


class Fraction:
    """Represent a fraction. It is used to specify `window_` parameters, as a fraction of some other value."""

    def __init__(self, value):
        self.value = value


notebook_location = dict()

def _init_notebook_location_if_needed(force=False):
    from IPython.display import display, Javascript
    # Get the URL of the running notebook to construct URLs for modeling files later
    if force or len(notebook_location) < 1:
        display(Javascript("""
IPython.notebook.kernel.execute(`lkep.notebook_location["href"] = "${window.location.href}"`);
IPython.notebook.kernel.execute(`lkep.notebook_location["origin"] = "${window.location.origin}"`);
IPython.notebook.kernel.execute(`lkep.notebook_location["pathname"] = "${window.location.pathname}"`);
// basedir of the URL, i.e., without the .ipynb filename
IPython.notebook.kernel.execute(`lkep.notebook_location["pathdir"] = "${window.location.pathname.replace(/[/][^/]+$/, '')}"`);
"""))


def init_notebook_js_utils():
    """Define Javascript helper functions used in a notebook UI."""
    from IPython.display import display, HTML
    _init_notebook_location_if_needed(force=True)

    display(
        HTML(
            """
<script>
function show_message(msg) {
    document.body.insertAdjacentHTML('beforeend',
        '<div id="msg_popin" style="word-break: break-all; font-size: 18px; position: fixed; top: 10%; left: 35vw; width: 30vw; padding: 1em; background-color: #feefc3; color: #333; z-index: 999999;">' +
        msg + '</div>');
    const ctr = document.getElementById("msg_popin");
    const remove_msg = () => {
        ctr.remove();
    };
    ctr.onclick = remove_msg;
    setTimeout(remove_msg, 5000);
}

async function copyTextToClipboard(text) {
    const res = await navigator.clipboard.writeText(text);
    show_message(`Copied to clipboard:<br>${text}`);
}
</script>
"""
        )
    )


def html_a_of_file(file_url, a_text, target="_blank", is_dir=False):
    """Create an HTML `<a>` link for the given file url.
    It will create a link that make sense within a Jupyter notebook context.
    """
    # needed if users have reloaded the module (and notebook_location object has been reset)
    _init_notebook_location_if_needed()  # TODO: the lazy re-init `notebook_location` does not really work
    # TODO: only works for relative url for now
    base_dir = notebook_location["pathdir"]
    if is_dir:
        base_dir = re.sub(r"^[/]notebooks[/]", "/tree/", base_dir)
    elif re.search(r"[.](py|dat)$", str(file_url)) is not None:
        # For .py files:
        # - use /edit link to have syntax highlighting, plus
        #   it lets user edit generated `input_fit.py`
        # For .dat files:
        # - use /edit link as it's the only one that let you view them in-place
        #   (/view link does not work: no content is shown)
        base_dir = re.sub(r"^[/]notebooks[/]", "/edit/", base_dir)
    elif re.search(r"[.]ipynb$", str(file_url)) is not None:
        # no-op for .ipynb
        pass
    else:
        base_dir = re.sub(r"^[/]notebooks[/]", "/view/", base_dir)

    file_url_to_show = base_dir + f"/{file_url}"
    return f"""<a href="{file_url_to_show}" target="{target}">{a_text}</a>"""


#
# Download helper
#


def _map_cadence_type(cadence_in_days):
    long_minimum = 6 / 60 / 24  # 6 minutes cutoff is somewhat arbitrary.
    short_minimum = 0.9 / 60 / 24  # 1 minute in days, with some margin of error
    if cadence_in_days is None:
        return None
    if cadence_in_days >= long_minimum:
        return "long"
    if cadence_in_days >= short_minimum:
        return "short"
    return "fast"


def _filter_by_priority(
    sr,
    author_priority=["SPOC", "TESS-SPOC", "QLP"],
    exptime_priority=["short", "long", "fast"],
):
    author_sort_keys = {}
    for idx, author in enumerate(author_priority):
        author_sort_keys[author] = idx + 1

    exptime_sort_keys = {}
    for idx, exptime in enumerate(exptime_priority):
        exptime_sort_keys[exptime] = idx + 1

    def calc_filter_priority(row):
        # Overall priority key is in the form of <author_key><exptime_key>, e.g., 101
        # - "01" is the exptime_key
        # - the leading "1" is the author_key, given it is the primary one
        author_default = max(dict(author_sort_keys).values()) + 1
        author_key = author_sort_keys.get(row["author"], author_default) * 100

        # secondary priority
        exptime_default = max(dict(exptime_sort_keys).values()) + 1
        exptime_key = exptime_sort_keys.get(_map_cadence_type(row["exptime"] / 60 / 60 / 24), exptime_default)
        return author_key + exptime_key

    sr.table["_filter_priority"] = [calc_filter_priority(r) for r in sr.table]

    # A temporary table that sorts the table by the priority
    sorted_t = sr.table.copy()
    sorted_t.sort(["mission", "_filter_priority"])

    # create an empty table for results, with the same set of columns
    res_t = sr.table[np.zeros(len(sr), dtype=bool)].copy()

    # for each mission (e.g., TESS Sector 01), select a row based on specified priority
    # - select the first row given the table has been sorted by priority
    uniq_missions = list(OrderedDict.fromkeys(sorted_t["mission"]))
    for m in uniq_missions:
        mission_t = sorted_t[sorted_t["mission"] == m]
        # OPEN: if for a given mission, the only row available is not listed in the priorities,
        # the logic still add a row to the result.
        # We might want it to be an option specified by the user.
        res_t.add_row(mission_t[0])

    return lk.SearchResult(table=res_t)


def _stitch_lc_collection(lcc, warn_if_multiple_authors=True):
    lc = lcc.stitch()
    lc.meta["SECTORS"] = [lc.meta.get("SECTOR") for lc in lcc]
    lc.meta["AUTHORS"] = [lc.meta.get("AUTHOR") for lc in lcc]
    if warn_if_multiple_authors:
        unique_authors = np.unique(lc.meta["AUTHORS"])
        if len(unique_authors) > 1:
            warnings.warn(
                f"Multiple authors in the collection. The stitched lightcurve might not be suitable for modeling: {lcc}"
            )
    return lc


def download_lightcurves_by_cadence_type(
    tic, sector, cadence="short", author_priority=["SPOC", "TESS-SPOC", "QLP"], download_dir=None, return_sr=False
):
    """Download the lightcurves of the given TIC - sector combination.
    The downloaded lightcurves are partitioned by cadence type (long / short),
    so they could be fed to `Pyaneti` separately (one band for each cadence type).
    """
    # for not-yet-released query cache in https://github.com/lightkurve/lightkurve/pull/1039
    if hasattr(lk.search, "sr_cache"):
        lk.search.sr_cache.cache_dir = download_dir

    sr_all = lk.search_lightcurve(f"TIC{tic}", mission="TESS")

    sr = _filter_by_priority(sr_all, author_priority=author_priority, exptime_priority=["short", "long"])

    # filter by sector and cadence
    if sector is not None:
        sr = sr[np.in1d(sr.table["sequence_number"], sector)]

    lc_by_cadence_type = dict()
    if cadence == "short" or cadence == "short_long":
        lcc_short = sr[sr.exptime == 120 * u.s].download_all(download_dir=download_dir)
        lc_short = _stitch_lc_collection(lcc_short, warn_if_multiple_authors=True)
        lc_by_cadence_type["SC"] = lc_short

    if cadence == "long" or cadence == "short_long":
        lcc_long = sr[sr.exptime == 1800 * u.s].download_all(download_dir=download_dir)
        lc_long = _stitch_lc_collection(lcc_long, warn_if_multiple_authors=True)
        lc_by_cadence_type["LC"] = lc_long

    if return_sr:
        return lc_by_cadence_type, sr_all
    else:
        return lc_by_cadence_type


def display_sr(sr, header):
    from IPython.display import display, HTML

    html = "<details open>"
    html += f"\n<summary>{header}</summary>"
    html += sr.__repr__(html=True)
    html += "\n</details>"

    return display(HTML(html))


def display_lc_by_band_summary(lc_by_band, header):
    from IPython.display import display, HTML

    html = ""
    if header is not None:
        html = f"<h5>{header}</h5>\n"
    html += "<pre>"
    for band, lc in lc_by_band.items():
        authors = np.unique(lc.meta.get("AUTHORS"))
        sectors = lc.meta.get("SECTORS")
        html += f"Band {band}  : {lc.label} ; Sectors: {sectors} ; Author(s): {authors}\n"
    html += "</pre>"
    return display(HTML(html))


#
# Prepare and export `LightCurve` to Pyaneti input data
#


def to_sector_str(sector):
    def format(a_sector):
        return f"{a_sector:02d}"

    if sector is None:
        return None
    elif isinstance(sector, Iterable):
        return "_".join([format(i) for i in sector])
    else:
        return format(sector)


def _create_dir_if_needed(path):
    basedir = os.path.dirname(path)
    if not os.path.isdir(basedir):
        os.makedirs(basedir)


def create_transit_mask(lc, transit_specs, include_surround_time=False, default_surround_time_func=None):
    """Create a mask for the transits specified in the `transit_spec` for the lightcurve."""

    def calc_duration_to_use(spec):
        """Calc a duration for the purpose of masking,
        by adding transit duration with an additional `surround_time`,
        with a default of (`max(transit duration * 2, 1 day)`
        """
        duration = spec["duration_hr"] / 24
        if include_surround_time:
            surround_time = spec.get("surround_time", None)
            if surround_time is None and default_surround_time_func is not None:
                surround_time = default_surround_time_func(duration)
            if surround_time is not None:
                duration = duration + surround_time
        return duration

    def calc_period_to_use(spec):
        period = spec.get("period", None)
        if period is not None and period > 0:
            return period
        # else period is not really specified,
        # for the use case that a single dip is observed with noted
        # use an arbitrary large period as a filler
        return 9999999999

    period = [calc_period_to_use(t) for t in transit_specs]
    duration = [calc_duration_to_use(t) for t in transit_specs]
    transit_time = [t["epoch"] for t in transit_specs]

    # a mask to include the transits,
    # and optionally their surrounding (for out of transit observations)
    mask = lc.create_transit_mask(period=period, duration=duration, transit_time=transit_time)

    return mask


def _truncate_lc_to_around_transits(lc, transit_specs):
    if transit_specs is None:
        return lc

    lc = lc.remove_nans().normalize()

    # a mask to include the transits and their surrounding (out of transit observations)
    mask = create_transit_mask(
        lc, transit_specs,
        include_surround_time=True,
        default_surround_time_func=lambda duration: duration * 2
        )

    return lc[mask]


def _merge_and_truncate_lcs(lc_or_lc_by_band, transit_specs):
    if isinstance(lc_or_lc_by_band, lk.LightCurve):
        return _truncate_lc_to_around_transits(lc_or_lc_by_band, transit_specs)
    # if it's a dictionary of lcs,
    # - truncate and stitch with an added `band` column indicating the source for each row
    #   (short cadence vs long cadence in typical TESS Transit model case)
    lc_trunc_by_band = dict()
    for band, lc_trunc in lc_or_lc_by_band.items():
        lc_trunc = _truncate_lc_to_around_transits(lc_trunc, transit_specs)
        lc_trunc["band"] = [band] * len(lc_trunc)
        lc_trunc_by_band[band] = lc_trunc

    lc_trunc = lk.LightCurveCollection(lc_trunc_by_band.values()).stitch(corrector_func=None)
    lc_trunc.sort("time")
    return lc_trunc, lc_trunc_by_band


def to_pyaneti_dat(lc_or_lc_by_band, transit_specs, pyaneti_env, return_processed_lc=False):
    "Output lc data to a file readable by Pyaneti, with lc pre-processed to be suitable for Pyaneti modeling"
    lc_trunc, lc_trunc_by_band = _merge_and_truncate_lcs(lc_or_lc_by_band, transit_specs)

    out_path = pyaneti_env.lc_dat_filepath

    # finally write to output
    # lc column subset does not work due to bug: https://github.com/lightkurve/lightkurve/issues/1194
    #  lc_trunc["time", "flux", "flux_err"]
    lc1 = type(lc_trunc)(time=lc_trunc.time.copy(), flux=lc_trunc.flux, flux_err=lc_trunc.flux_err)
    if "band" in lc_trunc.colnames:
        lc1["band"] = lc_trunc["band"]
    _create_dir_if_needed(out_path)
    lc1.write(out_path, format="ascii.commented_header", overwrite=True)

    if return_processed_lc:
        return out_path, lc_trunc, lc_trunc_by_band
    else:
        return out_path


def scatter_by_band(lc, **kwargs):
    """Do scatter plot of the given lightcurve, with each band labelled separately."""
    if "band" not in lc.colnames:
        return lc.scatter(**kwargs)
    band_names = np.unique(lc["band"])

    ax = None
    for band_name in band_names:
        ax = lc[lc["band"] == band_name].scatter(ax=ax, label=band_name, **kwargs)
    if lc.label is not None:
        ax.set_title(lc.label)
    return ax


RHO_SUN_CGS = (astropy.constants.M_sun / (4 / 3 * np.pi * astropy.constants.R_sun**3)).to(u.g / u.cm**3).value


@cached
def catalog_info_TIC(tic_id):
    """Takes TIC_ID, returns stellar information from TIC Catalog at MAST"""
    if type(tic_id) is not int:
        raise TypeError('tic_id must be of type "int"')
    try:
        from astroquery.mast import Catalogs
    except:
        raise ImportError("Package astroquery required but failed to import")

    result_tab = Catalogs.query_criteria(catalog="Tic", ID=tic_id)
    result = {c: result_tab[0][c] for c in result_tab[0].colnames}
    # In MAST result, rho is in the unit of solar density. we prefer g/cm^3 (ExoFOP UI also uses g/cm^3)
    # Reference: Appendix A, Notes on the individual columns 75, 76, Stassun et al., The TESS Input Catalog and Candidate Target List
    # https://ui.adsabs.harvard.edu/abs/2018AJ....156..102S/
    rho_in_solar = result.get("rho")
    if rho_in_solar is not None:
        result["rho_in_solar"] = rho_in_solar  # keep original data
        result["rho"] = rho_in_solar * RHO_SUN_CGS  # in g/cm^3
    e_rho_in_solar = result.get("e_rho")
    if e_rho_in_solar is not None:
        result["e_rho_in_solar"] = e_rho_in_solar  # keep original data
        result["e_rho"] = e_rho_in_solar * RHO_SUN_CGS  # in g/cm^3

    # convert Gaia ID from str to preferred int
    gaia_dr2_id_str = result.get("GAIA")
    if gaia_dr2_id_str is not None:
        result["GAIA"] = int(gaia_dr2_id_str)

    return result


@cached
def stellar_parameters_from_gaia(gaia_dr2_id):
    try:
        from astroquery.gaia import Gaia
    except:
        raise ImportError("Package astroquery required but failed to import")

    if gaia_dr2_id is None:
        return {}

    if type(gaia_dr2_id) is not int:
        raise TypeError('gaia_dr2_id must be of type "int"')

    def val_and_error_of_param(row, name):
        key_val = f"{name}_val"
        key_p_upper = f"{name}_percentile_upper"
        key_p_lower = f"{name}_percentile_lower"

        val = row[key_val]
        if val is not None:
            # Gaia DR2 gives more precise lower/upper bound of 68% CI, we convert them to a single one error
            e_val = max(row[key_val] - row[key_p_lower], row[key_p_upper] - row[key_val])
            return val, e_val
        else:
            return None, None

    query = (
        """SELECT
source_id,
ra,
dec,
teff_val,
teff_percentile_lower,
teff_percentile_upper,
radius_val,
radius_percentile_lower,
radius_percentile_upper
FROM gaiadr2.gaia_source
WHERE source_id=%d"""
        % gaia_dr2_id
    )

    result_tab = Gaia.launch_job(query).get_results()
    if len(result_tab) < 1:
        return None

    result = {}

    row = result_tab[0]

    teff, e_teff = val_and_error_of_param(row, "teff")
    rad, e_rad = val_and_error_of_param(row, "radius")

    if teff is not None:
        result["Teff"] = teff
        result["e_Teff"] = e_teff

    if rad is not None:
        result["rad"] = rad
        result["e_rad"] = e_rad

    return result


def stellar_parameters_of_tic(tic, also_use_gaia=True, diff_warning_threshold_percent=10):
    """Obtain stellar parameters from MAST, and optionally from Gaia as well."""

    def warn_if_significant_diff(meta_mast, meta_gaia, param_name):
        val_mast, val_gaia = meta_mast[param_name], meta_gaia[param_name]
        if val_mast is not None and val_gaia is not None:
            if abs(val_mast - val_gaia) / val_mast > diff_warning_threshold_percent / 100:
                warnings.warn(
                    f"Significant difference (> {diff_warning_threshold_percent}%) in {param_name} . MAST: {val_mast} ; Gaia DR2: {val_gaia}"
                )

    meta = catalog_info_TIC(tic)
    gaia_dr2_id = meta.get("GAIA")
    if also_use_gaia and gaia_dr2_id is not None:
        meta_gaia = stellar_parameters_from_gaia(gaia_dr2_id)
        warn_if_significant_diff(meta, meta_gaia, "rad")
        warn_if_significant_diff(meta, meta_gaia, "Teff")
        meta.update(meta_gaia)

    return meta


def get_limb_darkening_params(tic_meta):
    """Estimate Limb Darkening Quadratic Coefficients for TESS.
    The data is from
    [Claret et al. (2017)](https://ui.adsabs.harvard.edu/abs/2017A%26A...600A..30C/abstract),
    specifically, the subset of model `PHOENIX-COND`, 	quasi-spherical type `q`.
    The original data is hosted at:
    https://vizier.cds.unistra.fr/viz-bin/VizieR-3?-source=J/A%2bA/600/A30/tableab
    """
    # Logic derived from:
    # https://github.com/hippke/tls/blob/v1.0.31/transitleastsquares/catalog.py
    logg, Teff, = (
        tic_meta.get("logg"),
        tic_meta.get("Teff"),
    )
    if logg is None:
        logg = 4
        warnings.warn("No logg in metadata. Proceeding with logg=4")

    if Teff is None:
        Teff = 6000
        warnings.warn("No Teff in metadata Proceeding with Teff=6000")

    ld = np.genfromtxt(
        path.join("catalogs", "ld_claret_tess.csv"),
        skip_header=1,
        delimiter=",",
        dtype="f8, int32, f8, f8",
        names=["logg", "Teff", "a", "b"],
    )

    """
        - Take Teff from star catalog and find nearest entry in LD catalog
        - Same for logg, but only for the Teff values returned before
        - Return  best-match LD
    """
    nearest_Teff = ld["Teff"][(np.abs(ld["Teff"] - Teff)).argmin()]
    idx_all_Teffs = np.where(ld["Teff"] == nearest_Teff)
    relevant_lds = np.copy(ld[idx_all_Teffs])
    idx_nearest = np.abs(relevant_lds["logg"] - logg).argmin()
    # the `a`, `b` columns in the csv are the u1, u2 in Pyaneti,
    # the coefficients in the quadratic model:
    #   I(μ) = 1 − u1(1−μ) − u2(1−μ)^2
    u1 = relevant_lds["a"][idx_nearest]
    u2 = relevant_lds["b"][idx_nearest]

    # Pyaneti prefers parametrization in q1 / q2, an optimal way to sample the parameter space
    # see: https://github.com/oscaribv/pyaneti/wiki/Parametrizations#limb-darkening-coefficients
    q1 = (u1 + u2) ** 2
    q2 = u1 / (2 * (u1 + u2))

    # Provide a rough guess on error for u1/u2/q1/q2
    # it's a rough heuristics that tried to be conservation (i.e, erred to be larger than actual)
    e_u1 = np.ceil(u1 * 0.33 * 100) / 100
    e_u2 = np.ceil(u2 * 0.33 * 100) / 100
    e_q1 = np.ceil(q1 * 0.33 * 100) / 100
    e_q2 = np.ceil(q2 * 0.33 * 100) / 100

    return dict(q1=q1, e_q1=e_q1, q2=q2, e_q2=e_q2, u1=u1, e_u1=e_u1, u2=u2, e_u2=e_u2)


def _round_n_decimals(num, num_decimals):
    factor = np.power(10, num_decimals)
    return np.round(num * factor) / factor


def estimate_planet_radius_in_r_star(r_star, depth):
    """Return a back of envelope estimate of a planet object's radius,
    based on the simple model of a planet with circular orbit,
    transiting across the center of the host star (impact parameter `b` = 0)
    """
    depth = np.asarray(depth)
    if r_star is None or r_star < 0 or depth is None: # TODO: handle depth <= 0:
        return None  # cannot estimate

    R_JUPITER_IN_R_SUN = 71492 / 695700

    r_planet = np.sqrt(r_star * r_star * depth)

    r_planet_in_r_star = r_planet / r_star

    # Provide some rough min / max estimate
    min_r_planet_in_r_star = np.zeros_like(depth)
    # a rough guess for max: 2 times of the above estimate, capped to the size of 2.5 R_jupiter
    max_r_planet_in_r_star = np.full_like(depth, 2.5 * R_JUPITER_IN_R_SUN / r_star)
    max_r_planet_in_r_star = np.minimum(r_planet_in_r_star * 2, max_r_planet_in_r_star)

    # somehow the number in full precision causes pyaneti to behave strangely
    # (getting invalid numeric number in calculation, results in `nan` in `T_full`, etc.
    # capped at 4 decimal, taking a cue from pyaneti output
    return dict(
        r_planet_in_r_star=_round_n_decimals(r_planet_in_r_star, 4),
        min_r_planet_in_r_star=_round_n_decimals(min_r_planet_in_r_star, 4),
        max_r_planet_in_r_star=_round_n_decimals(max_r_planet_in_r_star, 4),
    )


def estimate_orbital_distance_in_r_star(tic_meta, transit_specs):
    # TODO: possibly use Kepler third law for better constraints
    return dict(
        # the arbitrary min/max is inspired by what Pyaneti chooses in a
        # (somewhat)similar scenario
        # https://github.com/oscaribv/pyaneti/blob/c6b6eb66854b8e079a7dbe8057df4cb809f10764/src/prepare_data.py#L243-L244
        min_a=np.full(np.shape(transit_specs), 1.1),
        max_a=np.full(np.shape(transit_specs), 1000.0)
        )


def define_impact_parameter():
    return dict(min_b=0.0, max_b=1.0)


def define_mcmc_controls(thin_factor=1, niter=500, nchains=100):
    return dict(mcmc_thin_factor=thin_factor, mcmc_niter=niter, mcmc_nchains=nchains)


def display_stellar_meta_links(meta, header=None):
    from IPython.display import display, HTML

    if header is not None:
        display(HTML(header))
    tic = meta["ID"]
    exofop_html = f'<a target="_exofop" href="https://exofop.ipac.caltech.edu/tess/target.php?id={tic}">ExoFOP</a>'

    gaia_id = meta.get("GAIA")
    gaia_html = ""
    if gaia_id is not None:
        gaia_html = f"""
<a target="_gaia_dr2" href="https://vizier.u-strasbg.fr/viz-bin/VizieR-S?Gaia%20DR2%20{gaia_id}">Gaia DR2 at Vizier</a>
&emsp;(<a target="_gaia_esa" href="https://gea.esac.esa.int/archive/" style="font-size: 85%;">Official Archive at ESA</a>)<br>
"""
    display(HTML(f"{exofop_html}<br>{gaia_html}"))


class ModelTemplate:
    FIT_TYPES = ["orbital_distance", "rho", "single_transit"]
    _FIT_TYPES_ABBREV = {"orbital_distance": "fit_a", "rho": "fit_rho", "single_transit": "single_transit"}
    ORBIT_TYPES = ["circular", "eccentric"]
    _ORBIT_TYPES_ABBREV = {"circular": "circular", "eccentric": "eccentric"}
    _ORBIT_TYPES_ABBREV2 = {"circular": "c", "eccentric": "e"}

    def __init__(self, transit_specs, orbit_type, fit_type) -> None:
        num_planets = len(transit_specs)
        self._validate_and_set(num_planets, np.arange(1, 20), "num_planets")
        self._validate_and_set(orbit_type, self.ORBIT_TYPES, "orbit_type")
        self._validate_and_set(fit_type, self.FIT_TYPES, "fit_type")
        self.template_filename = None  # i.e., uses the default

        if self.num_planets > 1:
            self.default_alias_suffix = f"{self.num_planets}planets"
        else:
            self.default_alias_suffix = transit_specs[0].get("label", None)

    def _validate_and_set(self, val, allowed_values, val_name):
        self._validate(val, allowed_values, val_name)
        setattr(self, val_name, val)

    @property
    def abbrev(self) -> str:
        """A textual shorthand describing the type of template"""
        orbit_abbrev = self._ORBIT_TYPES_ABBREV[self.orbit_type]
        fit_abbrev = self._FIT_TYPES_ABBREV[self.fit_type]

        return f"1planet_{orbit_abbrev}_orbit_{fit_abbrev}"

    def default_alias(self, tic) -> str:
        orbit_abbrev2 = self._ORBIT_TYPES_ABBREV2[self.orbit_type]
        fit_abbrev = self._FIT_TYPES_ABBREV[self.fit_type]
        res = f"TIC{tic}_{orbit_abbrev2}_{fit_abbrev}"
        if self.default_alias_suffix:
            res += f"_{self.default_alias_suffix}"
        return res

    @staticmethod
    def _validate(val, allowed_values, val_name):
        if val not in allowed_values:
            raise ValueError(f"{val_name} 's value {val} is invalid. Options: {allowed_values}")


def create_input_fit(
    template,
    tic,
    alias,
    pti_env,
    lc_or_lc_by_band,
    transit_specs,
    impact_parameter,
    meta,
    q1_q2,
    r_planet_dict,
    a_planet_dict,
    mcmc_controls,
    write_to_file=True,
    return_content=False,
):
    """Output parts of Pyaneti `input_fit.py` based on the specification included"""

    def set_if_None(map, key, value):
        if map.get(key) is None:
            map[key] = value
            return True
        else:
            return False

    def repeat(val, repeat_n):
        if repeat_n is None or repeat_n < 1:
            # use case: `val` is already a list
            return val
        elif repeat_n == 1:
            return [val]
        else:
            return [val] * repeat_n

    def process_priors(map, key_prior, src, key_prior_src=None, fraction_base_func=None, repeat_n=1, default=None, repeat_default=None):

        def do_repeat(val, repeat_n_to_use=repeat_n):
            return repeat(val, repeat_n_to_use)

        def get_len(val):
            if isinstance(val, Iterable):
                return len(val)
            else:
                return 1

        if key_prior_src is None:
            key_prior_src = key_prior

        # keys for accessing the input in `src`
        key_prior_src_error = f"e_{key_prior_src}"
        key_prior_src_window = f"window_{key_prior_src}"
        key_prior_src_min = f"min_{key_prior_src}"
        key_prior_src_max = f"max_{key_prior_src}"

        # keys for the output to the `map`
        key_prior_type = f"type_{key_prior}"
        key_prior_val1 = f"val1_{key_prior}"
        key_prior_val2 = f"val2_{key_prior}"
        if src.get(key_prior_src) is not None and src.get(key_prior_src_error) is not None:
            logger.info(f"Prior {key_prior}: resolved to Gaussian")
            map[key_prior_val1] = do_repeat(src.get(key_prior_src))  # Mean
            map[key_prior_val2] = do_repeat(src.get(key_prior_src_error))  # Standard Deviation
            num_to_repeat = get_len(map[key_prior_val1]) # needed for case the value is already an list (thus `repeat_n` is None)
            map[key_prior_type] = do_repeat("g", num_to_repeat)  # Gaussian Prior
        elif src.get(key_prior_src_min) is not None and src.get(key_prior_src_max) is not None:
            logger.info(f"Prior {key_prior}: resolved to Uniform")
            map[key_prior_val1] = do_repeat(src.get(key_prior_src_min))  # Minimum
            map[key_prior_val2] = do_repeat(src.get(key_prior_src_max))  # Maximum
            num_to_repeat = get_len(map[key_prior_val1]) # needed for case the value is already an list (thus `repeat_n` is None)
            map[key_prior_type] = do_repeat("u", num_to_repeat)  # Uniform Prior
        elif src.get(key_prior_src) is not None and src.get(key_prior_src_window) is not None:
            logger.info(f"Prior {key_prior}: resolved to Uniform (by mean and window)")
            window = src.get(key_prior_src_window)
            # check for `value` attribute rather than testing against Fraction instance
            # so that the codes would work even if users have reloaded the module after
            # fraction is defined initially in `transit_specs`.
            # if isinstance(window, Fraction):
            if hasattr(window[0], "value"):  # i.e, a Fraction type # OPEN: it does not work when `src` returns scalars
                fraction_base = src.get(key_prior_src) if fraction_base_func is None else fraction_base_func(src)
                window = fraction_base * np.asarray([i.value for i in window])
            map[key_prior_val1] = do_repeat(src.get(key_prior_src) - window / 2)  # Minimum
            map[key_prior_val2] = do_repeat(src.get(key_prior_src) + window / 2)  # Maximum
            num_to_repeat = get_len(map[key_prior_val1]) # needed for case the value is already an list (thus `repeat_n` is None)
            map[key_prior_type] = do_repeat("u", num_to_repeat)  # Uniform Prior
        elif src.get(key_prior_src) is not None:
            logger.info(f"Prior {key_prior}: resolved to Fixed")
            map[key_prior_val1] = do_repeat(src.get(key_prior_src))  # Fixed value
            map[key_prior_val2] = do_repeat(src.get(key_prior_src))  # does not matter for fixed value
            num_to_repeat = get_len(map[key_prior_val1]) # needed for case the value is already an list (thus `repeat_n` is None)
            map[key_prior_type] = do_repeat("f", num_to_repeat)  # Fixed Prior
        elif default is not None:
            prior_type = default[key_prior_type]
            if prior_type is not None:
                logger.info(f"Prior {key_prior}: not defined. Default is used. Type: '{prior_type}'")
                for key, value in default.items():
                    map[key] = repeat(value, repeat_default)
            else:
                raise AssertionError(f"Prior {key_prior} is not defined, but the supplied default is incomplete: {default}")
        else:
            raise ValueError(f"Prior {key_prior} is not defined or only partly defined.")

    def process_orbit_type(map, num_planets):
        if template.orbit_type == "circular":
            map["type_ew"] = repeat("f", num_planets)  # Fixed
            map["val1_ew1"] = repeat(0.0, num_planets)
            map["val2_ew1"] = repeat(0.0, num_planets)
            map["val1_ew2"] = repeat(0.0, num_planets)
            map["val2_ew2"] = repeat(0.0, num_planets)
        elif template.orbit_type == "eccentric":
            map["type_ew"] = repeat("u", num_planets)  # Uniform
            map["val1_ew1"] = repeat(-1.0, num_planets)
            map["val2_ew1"] = repeat(1.0, num_planets)
            map["val1_ew2"] = repeat(-1.0, num_planets)
            map["val2_ew2"] = repeat(1.0, num_planets)
        else:
            raise ValueError(f"Unsupported orbit type: {template.orbit_type}")

    def process_fit_type(map, num_planets):
        if template.fit_type == "orbital_distance":
            # TODO: validate the supplied `a` does have `num_planets` elements
            process_priors(map, "a", map, repeat_n=None)
            map["comment_a"] = "a/R*"
            map["sample_stellar_density"] = False
            map["is_single_transit"] = False
        elif template.fit_type == "rho":
            process_priors(map, "a", map, key_prior_src="rho", repeat_n=num_planets)
            map["comment_a"] = "rho*"
            map["sample_stellar_density"] = True
            map["is_single_transit"] = False
        elif template.fit_type == "single_transit":
            # currently, single transit implies fitting orbital distance rather than rho;
            # as it is Pyaneti's actual behavior
            # TODO: validate the supplied `a` does have `num_planets` elements
            process_priors(map, "a", map, repeat_n=None)
            map["comment_a"] = "a/R*"
            map["sample_stellar_density"] = False
            map["is_single_transit"] = True
        else:
            raise ValueError(f"Unsupported fit_type: {template.fit_type}")

    def process_cadence(map, lc_or_lc_by_band):
        if isinstance(lc_or_lc_by_band, lk.LightCurve):
            lc_or_lc_by_band = {"default-band": lc_or_lc_by_band}

        def calc_cadence(lc):
            # deduce the cadence
            cadence_in_min = np.round(np.nanmedian(np.asarray([t.to(u.min).value for t in np.diff(lc.time)])), decimals=1)
            # For n_cad, we use the following reference:
            # - t_cad_in_min == 30 (Kepler) ==> n_cad = 10
            # - t_cad_in_min == 2 (TESS Short cadence) ==> n_cad = 1
            # and scale it accordingly
            n_cad = np.ceil(10.0 * cadence_in_min / 30.0)
            return n_cad, cadence_in_min

        num_bands = len(lc_or_lc_by_band)
        if num_bands <= 1:
            # the default value of single band. see:
            # https://github.com/oscaribv/pyaneti/blob/ff570e7f92120ee4ef36683105fa709871382e50/src/default.py#L180
            map["bands"] = [""]
        else:
            map["bands"] = list(lc_or_lc_by_band.keys())

        cad_n_cad_in_min_pairs = [calc_cadence(lc) for lc in lc_or_lc_by_band.values()]

        map["n_cad"] = [pair[0] for pair in cad_n_cad_in_min_pairs]
        map["t_cad_in_min"] = [pair[1] for pair in cad_n_cad_in_min_pairs]

        # OPEN: should it defaulted to True or False for multiband case?
        is_multi_radius = True if len(lc_or_lc_by_band) > 1 else False
        # if users have specified is_multi_radius a priori, we honor their choice
        set_if_None(map, "is_multi_radius", is_multi_radius)

        return num_bands

    def add_dummy_rvs_params(map, num_planets):
        """Add dummy RV fitting params.
           They are necessary for multi-planet case, as Pyaneti cannot process the input file otherwise.
           """
        map["fit_rv"] = repeat(False, num_planets)
        map["fit_k"] = repeat('f', num_planets)
        map["min_k"] = repeat(0.0, num_planets)
        map["max_k"] = repeat(1.0, num_planets)


    def transit_specs_to_columns(transit_specs):
        """Convert row-oriented free form transit_specs to a set of columns.
           Use case: the set of columns is used by `process_priors`()`.
        """
        result = dict()

        def add_to_result_if_exist(param_name):
            spec0 = transit_specs[0]
            values = np.asarray([i.get(param_name) for i in transit_specs])
            values_are_none = (None == values)

            if values_are_none.all():
                # the param is not in transit specs
                return False
            elif values_are_none.any():
                # the param is in some specs, but not all
                warnings.warn(f"In transit_specs, parameter {param_name} does not exist for all entries. It is ignored in the mapping due to implementation limitation.")
                return False
            else:
                result[param_name] = values
                return True

        # Note: the implementation is not fully generic, but is sufficient for our use case here.
        # TODO: handle cases such that transit_specs[0] uses `window_epoch`, but transit_specs[1] uses `min_epoch` / `max_epoch`
        for param_name in [
            "epoch", "window_epoch", "min_epoch", "max_epoch", "e_epoch",
            "period", "window_period", "min_period", "max_period", "e_period",
            "duration_hr",
            ]:
            add_to_result_if_exist(param_name)

        return result


    # First process and combine all the given parameters
    # into a mapping table, which will be used to instantiate
    # the actual `input_fit.py`

    mapping = meta.copy()
    mapping["template_type"] = template.abbrev
    mapping.update(q1_q2)
    mapping.update(r_planet_dict)
    mapping.update(a_planet_dict)
    mapping.update(impact_parameter)
    mapping.update(mcmc_controls)
    mapping["tic"] = tic
    mapping["alias"] = alias
    mapping["fname_tr"] = pti_env.lc_dat_filename

    # Per-planet processing
    num_planets = len(transit_specs)
    mapping["nplanets"] = num_planets
    mapping["fit_tr"] = repeat(True, repeat_n=num_planets)
    add_dummy_rvs_params(mapping, num_planets)

    process_orbit_type(mapping, num_planets=num_planets)
    transit_spec_cols = transit_specs_to_columns(transit_specs)
    process_priors(mapping, "epoch", transit_spec_cols, fraction_base_func=lambda specs: specs["duration_hr"] / 24, repeat_n=None)
    process_priors(mapping, "period", transit_spec_cols, repeat_n=None,
        # if users do not specify period (single transit case), we give a large uniform prior
        # note: for default, the keys refers to the ones in the template file, not transit specs
        default=dict(type_period='u', val1_period=0.01, val2_period=9999),
        repeat_default=len(transit_specs),
        )
    process_priors(mapping, "b", mapping, repeat_n=num_planets)
    process_fit_type(mapping, num_planets=num_planets)
    # TODO: validate the supplied `r_planet_in_r_star` does have `num_planets` elements
    process_priors(mapping, "rp", mapping, "r_planet_in_r_star", repeat_n=None)

    # Per-band / cadence type processing
    num_bands = process_cadence(mapping, lc_or_lc_by_band)
    process_priors(mapping, "q1", mapping, repeat_n=num_bands)
    process_priors(mapping, "q2", mapping, repeat_n=num_bands)

    if isinstance(lc_or_lc_by_band, lk.LightCurve):
        time_col = lc_or_lc_by_band.time
    else:
        time_col = list(lc_or_lc_by_band.values())[0].time
    lc_time_label = time_col.format.upper()
    if time_col.format == "btjd":
        lc_time_label = "BJD - 2457000 (BTJD days)"
    elif time_col.format == "bkjd":
        lc_time_label = "BJD - 2454833 (BKJD days)"
    set_if_None(mapping, "lc_time_label", lc_time_label)
    set_if_None(mapping, "time_format", time_col.format)

    # Now all parameters are assembled in `mapping``, create the actual `input_fit.py`
    #
    template_filename = "template_input_tr_fit.py"
    if template.template_filename is not None:
        template_filename = template.template_filename
    result = Path("templates", template_filename).read_text()
    for key, value in mapping.items():
        value_str = str(value)
        if isinstance(value, np.ndarray) and value.ndim >= 1:
            # str(ndarray) has no comma, we need to provide them to be proper python list
            # do not use `np.array2string()`, because
            # - its output  can be affected by various numpy printoptions settings
            # - requires numpy > 1.11 (not as important)
            value_str = '[' + ', '.join([str(v) for v in value]) + ']'
        result = result.replace("{" + key + "}", value_str)

    if re.search(r"{[^}]+}", result):
        warnings.warn("create_input_fit(): the created `input_fit.py` still has values not yet defined.")

    input_fit_filepath = Path(pti_env.target_in_dir, "input_fit.py")
    if write_to_file:
        input_fit_filepath.write_text(result)

    if return_content:
        return input_fit_filepath, result
    else:
        return input_fit_filepath


def display_pyaneti_input_py_location(input_fit_filepath):
    from IPython.display import display, HTML

    display(
        HTML(
            f"""
    {html_a_of_file(input_fit_filepath, input_fit_filepath, "_input_fit")}
    """
        )
    )


def display_pyaneti_instructions(pti_env):
    from IPython.display import display, Markdown

    display(
        Markdown(
            f"""
To do the modeling, just run next cell. Or run the following in a separate terminal:
```
!cd {pti_env.home_dir.resolve()}; python pyaneti.py  {pti_env.alias}
```
    """
        )
    )


def beep():
    """Emits a beep sound. It works only in IPython / Jupyter environment only"""
    # a beep to remind the users that the data has been downloaded
    # css tweak to hide beep
    import IPython
    from IPython.display import display, HTML, Audio

    display(
        HTML(
            """<script>
function tweakCSS() {
  if (document.getElementById("hide-beep-css")) {
      return;
  }
  document.head.insertAdjacentHTML('beforeend', `<style id="hide-beep-css" type="text/css">
  #beep { /* hide the audio control for the beep, generated from tplt.beep() */
    width: 1px;
    height: 1px;
  }
</style>`);
}
tweakCSS();
</script>
"""
        )
    )
    # the actual beep
    ## source: https://upload.wikimedia.org/wikipedia/commons/f/fb/NEC_PC-9801VX_ITF_beep_sound.ogg
    beep_url = "beep_sound.ogg"
    if int(re.sub(r"[.].+", "", IPython.__version__)) < 7:
        # compatibility with older older IPython (e.g., google colab)
        audio = Audio(filename=beep_url, autoplay=True, embed=True)
    else:
        audio = Audio(filename=beep_url, autoplay=True, embed=True, element_id="beep")
    display(audio)

#
# Read Pyaneti model output, lightcurve files, etc.
#


def save_params_as_txt_file(pti_env):
    "Save the params `.dat` as `.txt` so that it can be easily viewed on Google Drive."
    target_out_dir = pti_env.target_out_dir
    alias = pti_env.alias
    file_params = Path(target_out_dir, f"{alias}_params.dat")
    file_params_txt = Path(target_out_dir, f"{alias}_params.txt")
    shutil.copyfile(file_params, file_params_txt)


def copy_input_fit_py_to_out_dir(pti_env):
    "Copy `input_fit.py` to output directory so that it can be easily shared (on Google Drive)."
    input_fit_filepath = Path(pti_env.target_in_dir, "input_fit.py")
    destination = Path(pti_env.target_out_dir, "input.py")
    shutil.copyfile(input_fit_filepath, destination)


def _char_list_inclusive(c1, c2):
    """Return the list of characters from `c1` to `c2`, inclusive."""
    return [chr(c) for c in range(ord(c1), ord(c2) + 1)]


def display_model(
    pti_env,
    template,
    show_params=True,
    show_posterior=True,
    show_correlations=False,
    show_transits=True,
    show_lightcurve=True,
    show_chains=False,
):
    from IPython.display import display, Image, HTML

    target_out_dir = pti_env.target_out_dir
    alias = pti_env.alias
    display(
        HTML(
            f"""
<h3>Model for {html_a_of_file(target_out_dir, alias, target="_out_dir", is_dir=True)}&nbsp;
<a href="https://github.com/oscaribv/pyaneti/wiki/Output-files" target="_doc_pti_out"
   style="font-size: 75%; font-weight: normal;">(documentation)</a>
</h3>"""
        )
    )

    if show_params:
        file_params = Path(target_out_dir, f"{alias}_params.dat")
        file_init = Path(target_out_dir, f"{alias}_init.dat")
        display(
            HTML(
                f"""<ul>
    <li>{html_a_of_file(file_params, "Model params", target="_params")}: {file_params}</li>
    <li>{html_a_of_file(file_init, "Init params", target="_init")}: {file_init}</li>
</ul>
"""
            )
        )

    if show_posterior:
        display(Image(Path(target_out_dir, f"{alias}_posterior.png")))
    if show_correlations:
        display(Image(Path(target_out_dir, f"{alias}_correlations.png")))
    if show_transits:
        planets_suffix = _char_list_inclusive('b', 'z')
        planets_suffix = planets_suffix[:template.num_planets]
        for suffix in planets_suffix:
            display(HTML(f"""<h5 style="text-align: center;">Planet {suffix}:</h5>"""))
            display(Image(Path(target_out_dir, f"{alias}{suffix}_tr.png")))
    if show_lightcurve:
        display(Image(Path(target_out_dir, f"{alias}_lightcurve.png")))
    if show_chains:
        display(Image(Path(target_out_dir, f"{alias}_chains.png")))


def read_pyaneti_lc_dat(filename, time_format="btjd", time_converter_func=None):
    """Read Pyaneti lightcurve files as `LightCurve` objects, e.g.,
    `inpy/.../<starname>.dat`, `outpy/.../<starname>-trdata_lightcurve.txt`."""
    # format="ascii.commented_header" does not work
    tab = Table.read(filename, format="ascii")
    if len(tab.colnames) >= 3:
        (n_time, n_flux, n_flux_err, *rest) = tab.colnames
        flux = tab[n_flux]
        flux_err = tab[n_flux_err]
    else:
        n_time, n_flux = tab.colnames  # for trmodel files
        flux = tab[n_flux]
        flux_err = np.zeros_like(flux)

    if time_converter_func is not None:
        time = time_converter_func(tab[n_time])
    else:
        time = Time(tab[n_time], format=time_format)

    lc_cls = lk.LightCurve
    if time.format == "btjd":
        lc_cls = lk.TessLightCurve
    return lc_cls(time=time, flux=flux, flux_err=flux_err)
