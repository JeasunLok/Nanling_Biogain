from __future__ import annotations

from pathlib import Path

from core.temporal_aggregation import compute_multi_year_mean, write_mean_manifest


def main() -> int:
    project_root = Path(__file__).resolve().parent
    response_root = project_root / "data" / "prepared" / "wgs84_1km" / "responses"
    output_dir = project_root / "data" / "prepared" / "wgs84_1km" / "response_means"
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "GPP": compute_multi_year_mean(
            response_root / "GPP",
            output_dir / "GPP_mean.tif",
        ),
        "LAI": compute_multi_year_mean(
            response_root / "LAI",
            output_dir / "LAI_mean.tif",
        ),
        "VOD": compute_multi_year_mean(
            response_root / "VOD",
            output_dir / "VOD_mean.tif",
        ),
    }
    manifest_path = write_mean_manifest(output_dir, manifest)
    print("Multi-year means written to:", output_dir)
    print("Manifest:", manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
