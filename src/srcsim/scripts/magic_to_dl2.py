import argparse
import logging
import healpy

from srcsim.magic.mc import MagicMcFile


def main():
    placeholder = "[HEALPY]"

    arg_parser = argparse.ArgumentParser(
        description="""
        Convert MAGIC Melibea/Superstar MC files to the DL2 (HDF5) format.
        """
    )

    arg_parser.add_argument(
        "-i",
        "--input-file", 
        help='MAGIC MC file in Root format'
    )
    arg_parser.add_argument(
        "-n",
        "--nside",
        type=int,
        default=0,
        help='NSIDE to use in HEALPix split of the input file.' \
            ' If nside=0, no spliting is applied. Defaults to 0.'
    )
    arg_parser.add_argument(
        "-o",
        "--output-file", 
        help='Output HDF5 file name.' \
            f' If nside > 0 is given, must containt a "{placeholder}" placeholder,' \
            ' that will be replaced with "healpy-nside-pixid" when writing the file.'
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

    log.info(f'reading MC file')
    mcfile = MagicMcFile.from_file(args.input_file)
    log.debug(str(mcfile))

    if args.nside > 0:
        if not placeholder in args.output_file:
            raise ValueError(
                f'nside > 0 specified, but no "{placeholder}" given in output name "{args.output_file}"'
            )
        
        log.info('running HEALPix spliting')
        files = mcfile.healpy_split(args.nside)

        npix = healpy.nside2npix(args.nside)
        ids = range(npix)
        with_events = tuple(
            filter(lambda x: x[1].events_simulated.n_events > 0, zip(ids, files))
        )
        log.info(f'split complete: {len(with_events)} non-empty files (npix={npix})')

        log.info(f'writing the DL2 files')
        for pix_id, mfile in with_events:
            output_name = args.output_file.replace(placeholder, f'healpy-{args.nside}-{pix_id}')
            log.debug(f'writing {output_name}')
            mfile.write(output_name)
    else:
        log.info(f'writing the DL2 file')
        mcfile.write(args.output_file)

    log.info('conversion complete')


if __name__ == '__main__':
    main()
