import yaml
import datetime
import argparse
import pandas as pd
import astropy.units as u
import progressbar

from srcsim.mc import MCCollection
from srcsim.src import generator as srcgen
from srcsim.rungen import AltAzBoxGenerator


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


if __name__ == '__main__':
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

    info_message('Loading MCs')
    mc = {
        emission_type: MCCollection(cfg['mc']['files'][emission_type])
        for emission_type in cfg['mc']['files']
    }

    if cfg['mc']['max_samples'] is not None:
        for emission_type in mc:
            mc[emission_type].samples = mc[emission_type].samples[:cfg['mc']['max_samples']]

    if cfg['mc']['search_radius'] is not None:
        search_radius = u.Quantity(cfg['mc']['search_radius'])
    else:
        search_radius = None

    info_message('Preparing sources')
    srcs = srcgen(cfg['sources'])

    info_message('Preparing data runs')
    runs = AltAzBoxGenerator.get_runs_from_config(cfg['rungen'])

    info_message('Starting event sampling')
    with progressbar.ProgressBar(max_value=len(runs), prefix="Event sampling: ") as progress:
        for run_id, run in enumerate(runs):
            evt = [
                run.predict(
                    mc,
                    src,
                    tel_pos_tolerance=search_radius
                )
                for src in srcs
            ]
            events = pd.concat(evt)
            
            events.to_hdf(cfg['io']['out'] + f'run{run_id}.h5', 'dl2/event/telescope/parameters/LST_LSTCam')

            progress.update(run_id)

    info_message('Simulation complete')
