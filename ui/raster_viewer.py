from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
from matplotlib import font_manager, rcParams
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from rasterio.transform import Affine
from PySide6.QtWidgets import QVBoxLayout, QWidget


def _configure_matplotlib_font() -> None:
    preferred_fonts = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "Arial Unicode MS",
    ]
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    selected_fonts = [font for font in preferred_fonts if font in available_fonts]
    if selected_fonts:
        rcParams["font.sans-serif"] = selected_fonts + list(rcParams.get("font.sans-serif", []))
    rcParams["axes.unicode_minus"] = False


_configure_matplotlib_font()

ABSOLUTE_GAIN_CMAP = LinearSegmentedColormap.from_list(
    "absolute_gain",
    ["#1b1f3b", "#136f63", "#f4d35e"],
)
PERCENT_GAIN_CMAP = LinearSegmentedColormap.from_list(
    "percent_gain",
    ["#3d0c02", "#275dad", "#8ac926"],
)


class RasterViewer(QWidget):
    """Matplotlib-backed single-band raster preview."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._figure = Figure(figsize=(5, 4))
        self._canvas = FigureCanvasQTAgg(self._figure)
        self._axes = self._figure.add_subplot(111)

        layout = QVBoxLayout(self)
        layout.addWidget(self._canvas)

        self.clear()

    def clear(self) -> None:
        self._figure.clear()
        self._axes = self._figure.add_subplot(111)
        self._axes.set_title("Raster Preview")
        self._axes.set_axis_off()
        self._canvas.draw_idle()

    def show_single_band(
        self,
        array: np.ndarray,
        title: str,
        nodata: Optional[float] = None,
        transform: Optional[Affine] = None,
        overlay_geometries: Optional[Iterable[dict]] = None,
        unit_label: str = "",
        cmap_name: Optional[str] = None,
        fixed_range: Optional[tuple[float, float]] = None,
        signed_mode: str = "auto",
        percentile_range: tuple[float, float] = (2.0, 98.0),
    ) -> None:
        if array.ndim != 2:
            raise ValueError("RasterViewer only supports single-band 2D arrays.")

        self._figure.clear()
        self._axes = self._figure.add_subplot(111)
        masked = np.ma.masked_invalid(array)
        if nodata is not None:
            masked = np.ma.masked_where(np.isclose(masked, nodata), masked)

        compressed = masked.compressed()
        cmap = cmap_name or ("viridis")
        image_kwargs = {"cmap": cmap, "origin": "upper"}
        if compressed.size:
            if fixed_range is not None:
                vmin, vmax = fixed_range
                image_kwargs.update(vmin=vmin, vmax=vmax)
            else:
                vmin = float(np.percentile(compressed, percentile_range[0]))
                vmax = float(np.percentile(compressed, percentile_range[1]))
            if (
                fixed_range is None
                and signed_mode != "off"
                and np.any(compressed < 0)
                and np.any(compressed > 0)
            ):
                limit = max(abs(vmin), abs(vmax))
                diverging_cmap = ABSOLUTE_GAIN_CMAP if signed_mode == "absolute_gain" else PERCENT_GAIN_CMAP
                if signed_mode == "auto":
                    diverging_cmap = ABSOLUTE_GAIN_CMAP
                image_kwargs.update(vmin=-limit, vmax=limit, cmap=diverging_cmap)
            elif vmax > vmin:
                image_kwargs.update(vmin=vmin, vmax=vmax)

        if transform is not None:
            left = transform.c
            top = transform.f
            right = left + transform.a * array.shape[1]
            bottom = top + transform.e * array.shape[0]
            image_kwargs["extent"] = (left, right, bottom, top)

        image = self._axes.imshow(masked, **image_kwargs)
        self._axes.set_title(title)
        if overlay_geometries:
            self._draw_overlays(overlay_geometries)
        if transform is None:
            self._axes.set_axis_off()
        else:
            self._axes.set_xlabel("Longitude")
            self._axes.set_ylabel("Latitude")
        colorbar = self._figure.colorbar(image, ax=self._axes, fraction=0.046, pad=0.04)
        if unit_label:
            colorbar.set_label(unit_label)
        self._canvas.draw_idle()

    def _draw_overlays(self, geometries: Iterable[dict]) -> None:
        for geometry in geometries:
            self._draw_geometry(geometry)

    def _draw_geometry(self, geometry: dict) -> None:
        geometry_type = geometry.get("type")
        coordinates = geometry.get("coordinates", [])
        if geometry_type == "Polygon":
            self._draw_polygon(coordinates)
            return
        if geometry_type == "MultiPolygon":
            for polygon in coordinates:
                self._draw_polygon(polygon)

    def _draw_polygon(self, polygon: Iterable) -> None:
        for ring in polygon:
            xs = [point[0] for point in ring]
            ys = [point[1] for point in ring]
            self._axes.plot(xs, ys, color="white", linewidth=1.3, alpha=0.95)
            self._axes.plot(xs, ys, color="black", linewidth=0.5, alpha=0.7)
