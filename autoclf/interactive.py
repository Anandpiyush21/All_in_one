"""The guided prompt flow, preserved from the original tool.

Running ``python main.py`` with no arguments still walks the user through the
same four questions; every answer simply fills a field of :class:`RunConfig`
instead of constructing an estimator on the spot, so the interactive and
scripted paths share one execution engine.
"""

from __future__ import annotations

from autoclf.config import RunConfig
from autoclf.cross_val import SPLITTERS
from autoclf.feature_selection import SELECTORS
from autoclf.models import MODELS
from autoclf.preprocessing import SCALERS

BOLD, DIM, RESET = "\033[1m", "\033[2m", "\033[0m"


def _menu(title: str, registry: dict, description_index: int, multi: bool = False) -> list[str]:
    keys = list(registry)
    print(f"\n{BOLD}{title}{RESET}")
    for i, key in enumerate(keys, start=1):
        entry = registry[key]
        print(f"  {i:>2}. {entry[0]:<34} {DIM}{entry[description_index]}{RESET}")
    suffix = " (comma-separated for several, blank = 1)" if multi else " (blank = 1)"
    while True:
        raw = input(f"Enter option{suffix}: ").strip()
        if not raw:
            return [keys[0]]
        try:
            picked = [keys[int(part) - 1] for part in raw.split(",") if part.strip()]
        except (ValueError, IndexError):
            print("  Please enter number(s) from the list above.")
            continue
        if picked and (multi or len(picked) == 1):
            return picked
        print("  Only one option is allowed here.")


def _ask_int(prompt: str, default: int) -> int:
    raw = input(f"{prompt} [{default}]: ").strip()
    try:
        return int(raw) if raw else default
    except ValueError:
        return default


def configure_interactively(base: RunConfig | None = None) -> RunConfig:
    """Fill a :class:`RunConfig` by asking the questions the original tool asked."""
    config = base or RunConfig()
    print(f"\n{BOLD}autoclf — interactive setup{RESET}")
    print(f"{DIM}Press Enter at any prompt to accept the default.{RESET}")

    dataset = input(f"\nDataset CSV [{config.dataset}]: ").strip()
    if dataset:
        config.dataset = dataset

    config.scaler = _menu("Scaling strategy", SCALERS, 2)[0]
    config.selector = _menu("Feature selection", SELECTORS, 2)[0]
    if config.selector not in {"variance", "none"}:
        config.n_features = _ask_int("Number of features to keep", config.n_features)
    config.cv = _menu("Cross-validation", SPLITTERS, 0)[0]
    config.n_splits = _ask_int("Number of folds", config.n_splits)
    config.models = _menu("Classifier(s)", MODELS, 3, multi=True)
    config.tune = input("Tune hyper-parameters? (y/N): ").strip().lower().startswith("y")

    config.__post_init__()  # re-validate after mutation
    return config
