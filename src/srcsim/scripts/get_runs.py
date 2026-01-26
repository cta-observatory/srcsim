import yaml
import argparse
import logging

from srcsim.rungen import generator as rungen


def main():
    arg_parser = argparse.ArgumentParser(
        description="""
        LST simulated runs generator.
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

    log.info('Preparing data runs')
    runs = rungen(cfg['rungen'])

    for run_id, run in enumerate(runs):
        output_name = cfg['io']['out'] + f'run{run_id}.yaml'
        run_cfg = dict(
            io = cfg['io'],
            mc = cfg['mc'],
            sampling = cfg['sampling'],
            sources = cfg['sources'],
            run = run.to_dict()
        )

        with open(output_name, 'w') as output:
            yaml.dump(run_cfg, output)

    log.info('Generation complete')


if __name__ == '__main__':
    main()
