from __future__ import annotations

from pathlib import Path

from core.data_preparation import default_preparation_paths, prepare_nanling_wgs84_dataset


def main() -> int:
    project_root = Path(__file__).resolve().parent
    paths = default_preparation_paths(project_root)
    manifest = prepare_nanling_wgs84_dataset(paths)
    print("Prepared data written to:", paths.output_dir)
    print("Manifest:", manifest["manifest_path"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
