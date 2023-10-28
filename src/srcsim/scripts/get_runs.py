import yaml
import datetime
import argparse

from srcsim.rungen.generator import generator as rungen


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
        LST simulated runs generator.
        """
    )

    arg_parser.add_argument(
        "--config", 
        default="config.yaml",
        help='Configuration file to steer the code execution.'
    )
    args = arg_parser.parse_args()

    cfg = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)

    info_message('Preparing data runs')
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

    info_message('Generation complete')


if __name__ == '__main__':
    main()
