import yaml

from .sky import AltAzBoxGenerator, DataMatchingGenerator
from .fixed import FixedObsGenerator


def generator(config):
    if isinstance(config, str):
        cfg = yaml.load(open(config, "r"), Loader=yaml.FullLoader)
    else:
        cfg = config

    runs = []

    if cfg['type'] == "altaz_box":
        runs = AltAzBoxGenerator.get_runs_from_config(cfg)
    elif cfg['type'] == "fixed_altaz":
        FixedObsGenerator.get_runs_from_config(cfg)
    elif cfg['type'] == "data_matching":
        DataMatchingGenerator.get_runs_from_config(cfg)
    else:
        raise ValueError(f"Unknown run generator type '{cfg['type']}'")

    return runs