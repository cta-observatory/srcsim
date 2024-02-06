import yaml

from .sky import AltAzBoxGenerator, DataMatchingGenerator
# from .fixed import FixedObsGenerator


def generator(config):
    """
    Returns runs automatically selecting 
    the generator appropriate to config.

    Parameters
    ----------
    config: dict
        Configuration dictionary

    Returns
    -------
    runs: tuple
        Generated runs
    """

    if isinstance(config, str):
        cfg = yaml.load(open(config, "r"), Loader=yaml.FullLoader)
    else:
        cfg = config

    runs = []

    if cfg['type'] == "altaz_box":
        runs = AltAzBoxGenerator.get_runs_from_config(cfg)
    # elif cfg['type'] == "fixed_altaz":
    #     runs = FixedObsGenerator.get_runs_from_config(cfg)
    elif cfg['type'] == "data_matching":
        runs = DataMatchingGenerator.get_runs_from_config(cfg)
    else:
        raise ValueError(f"Unknown run generator type '{cfg['type']}'")

    return runs