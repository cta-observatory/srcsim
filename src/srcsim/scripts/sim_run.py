import yaml
import argparse
import logging
import random
import pandas as pd
import astropy.units as u

from srcsim.mc import MCCollection
from srcsim.src import generator as srcgen
from srcsim.run import DataRun


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
        '-v',
        "--verbose",
        action='store_true',
        help='extra verbosity'
    )
    args = arg_parser.parse_args()

    logging.basicConfig(
        format='%(asctime)s %(name)-10s : %(levelname)-8s %(message)s',
        datefmt='%Y-%m-%dT%H:%M:%S',
    )

    log = logging.getLogger(__name__)

    if args.verbose:
        log.setLevel(logging.DEBUG)
    else:
        log.setLevel(logging.INFO)

    cfg = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)

    log.info('loading MCs')
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

    log.info('preparing sources')
    srcs = srcgen(cfg['sources'])
    print(srcs)

    log.info('preparing the data run')
    run = DataRun.from_config(cfg['run'])
    print(run)

    log.info('starting event sampling')
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

    log.info('simulation complete')


if __name__ == '__main__':
    main()
