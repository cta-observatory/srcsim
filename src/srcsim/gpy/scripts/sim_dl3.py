import os
import yaml
import datetime
import argparse
import random
import pandas as pd
import astropy.units as u

from gammapy.modeling.models import Models

from srcsim.gpy.irf import IRFCollection
from srcsim.gpy.rungen import generator


def info_message(text):
    """
    This function prints the specified text with the prefix of the current date

    Parameters
    ----------
    text: str

    Returns
    -------
    None

    """

    date_str = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    print("{date:s}: {message:s}".format(date=date_str, message=text))


def main():
    arg_parser = argparse.ArgumentParser(
        description="""
        LST event simulator.
        """
    )

    arg_parser.add_argument(
        "--config", 
        default="config.yaml",
        help='Configuration file to steer the code execution.'
    )
    args = arg_parser.parse_args()

    cfg = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)

    info_message('Loading IRFs')
    irfs = IRFCollection(
        cfg['irf']['files']
    )
    print(irfs)

    info_message('Preparing sources')
    source_models = Models.from_dict(cfg['model'])
    print(source_models)

    info_message('Preparing the data run')
    runs = generator(cfg['rungen'])
    info_message(f'{len(runs)} to be simulated')

    info_message('Starting simulation')
    observations = [
        run.predict(irfs, source_models, cfg['irf']['search_radius'])
        for run in runs
    ]

    for obs in observations:
        obs.write(
            os.path.join(cfg['io']['out'], f'run{obs.obs_id}.fits'),
            overwrite=True
        )

    info_message('Simulation complete')


if __name__ == '__main__':
    main()
