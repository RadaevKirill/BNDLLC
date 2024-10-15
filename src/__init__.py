from pathlib import Path

from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).parent.resolve()
ASSETS_DIR = REPO_ROOT.parent / 'assets'

OmegaConf.register_new_resolver(name='ASSETS_DIR', resolver=lambda x: ASSETS_DIR / x, replace=True)
