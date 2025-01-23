import os
import gc
import yaml
import datetime
import argparse
import random
import pandas as pd
import astropy.units as u

from progressbar import ProgressBar
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
    arg_parser.add_argument(
        "--id",
        default=-1,
        type=int,
        help='Obs ID to simulate'
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

    info_message('Preparing the data runs')
    runs = generator(cfg['rungen'])
    info_message(f'{len(runs)} runs generated')

    if args.id >= 0:
        runs = runs[args.id:args.id+1]

    info_message('Starting simulation')

    with ProgressBar(max_value=len(runs), prefix="simulation: ") as progress:
        for ri, run in enumerate(runs):
            obs = run.predict(
                irfs,
                source_models,
                cfg['irf']['search_radius']
            )
            obs.write(
                os.path.join(cfg['io']['out'], f'run{obs.obs_id}.fits'),
                overwrite=True
            )
            del obs
            gc.collect()
            progress.update(ri)

    info_message('Simulation complete')


if __name__ == '__main__':
    main()
