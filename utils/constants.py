from __future__ import annotations

CANONICAL_VARIABLES = {
    "GPP",
    "LAI",
    "VOD",
    "tree_diversity",
    "structure_diversity",
    "MAT",
    "MAP",
    "VPD",
    "SM",
    "SSRD",
    "DEM",
    "slope",
    "aspect",
}

VARIABLE_ALIAS_MAP = {
    "PRE": "MAP",
}

RESPONSE_VARIABLES = [
    "GPP",
    "LAI",
    "VOD",
]

DEFAULT_PREDICTOR_FEATURES = [
    "tree_diversity",
    "structure_diversity",
    "MAT",
    "MAP",
    "VPD",
    "SM",
    "SSRD",
    "DEM",
    "slope",
    "aspect",
]

DISPLAY_UNITS = {
    "GPP": "gC m-2 y-1",
    "LAI": "",
    "VOD": "",
    "tree_diversity": "",
    "structure_diversity": "",
    "MAT": "degC",
    "MAP": "mm",
    "VPD": "hPa",
    "SM": "m3 m-3",
    "SSRD": "W/m-2",
    "DEM": "m",
    "slope": "deg",
    "aspect": "deg",
}

PREVIEW_STYLE = {
    "GPP": {"cmap": "viridis", "fixed_range": None},
    "LAI": {"cmap": "viridis", "fixed_range": None},
    "VOD": {"cmap": "viridis", "fixed_range": None},
    "tree_diversity": {"cmap": "viridis", "fixed_range": None},
    "structure_diversity": {"cmap": "viridis", "fixed_range": None},
    "MAT": {"cmap": "viridis", "fixed_range": None},
    "MAP": {"cmap": "viridis", "fixed_range": None},
    "VPD": {"cmap": "viridis", "fixed_range": None},
    "SM": {"cmap": "viridis", "fixed_range": None},
    "SSRD": {"cmap": "viridis", "fixed_range": None},
    "DEM": {"cmap": "viridis", "fixed_range": None},
    "slope": {"cmap": "viridis", "fixed_range": None},
    "aspect": {"cmap": "viridis", "fixed_range": (0.0, 360.0)},
}
