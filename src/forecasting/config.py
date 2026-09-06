from pathlib import Path
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def load_config(path: str | Path | None = None) -> dict:
    cfg_path = Path(path) if path else PROJECT_ROOT / "config.yaml"
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


def ensure_dirs(cfg: dict) -> None:
    for key in ("raw_path", "processed_path"):
        Path(PROJECT_ROOT / cfg["data"][key]).mkdir(parents=True, exist_ok=True)
    for key in ("results", "plots", "models"):
        Path(PROJECT_ROOT / cfg["paths"][key]).mkdir(parents=True, exist_ok=True)
