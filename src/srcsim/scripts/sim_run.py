import yaml
import datetime
import argparse
import random
import pandas as pd
import astropy.units as u

from srcsim.mc import MCCollection
from srcsim.src import generator as srcgen
from srcsim.run import SkyDataRun


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

    info_message('Loading MCs')
    mc = {
        emission_type: MCCollection(cfg['mc'][emission_type]['files'])
        for emission_type in cfg['mc']
    }

    for emission_type in mc:
        if cfg['mc'][emission_type]['max_samples'] is not None:
            mc[emission_type].samples = tuple(
                random.sample(
                    mc[emission_type].samples,
                    cfg['mc'][emission_type]['max_samples']
                )
            )

    print(mc)

    search_radius = {
        emission_type: cfg['mc'][emission_type]['search_radius']
        for emission_type in mc
    }

    for emission_type in search_radius:
        if search_radius[emission_type] is not None:
            if isinstance(search_radius[emission_type], str):
                search_radius[emission_type] = u.Quantity(search_radius[emission_type])
            else:
                search_radius[emission_type] = [u.Quantity(s) for s in search_radius[emission_type]]
                
    cfg['sampling']['time_step'] = u.Quantity(cfg['sampling']['time_step'])

    info_message('Preparing sources')
    srcs = srcgen(cfg['sources'])
    print(srcs)

    info_message('Preparing the data run')
    run = SkyDataRun.from_config(cfg['run'])
    print(run)

    info_message('Starting event sampling')
    evt = [
        run.predict(
            mc,
            src,
            search_radius[src.emission_type],
            cfg['sampling']['time_step']
        )
        for src in srcs
    ]
    events = pd.concat(evt)

    events = run.time_sort(events)
    events = run.update_time_delta(events)

    events.to_hdf(cfg['io']['out'] + f'run{run.id}.h5', 'dl2/event/telescope/parameters/LST_LSTCam')

    info_message('Simulation complete')


if __name__ == '__main__':
    main()
