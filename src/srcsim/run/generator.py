from srcsim.run import SkyDataRun, FixedPointingDataRun


def generator(cfg):
    if 'alt' in cfg['pointing']:
        run = FixedPointingDataRun.from_config(cfg)
    else:
        run = SkyDataRun.from_config(cfg)

    return run