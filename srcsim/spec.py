import yaml
import numpy as np
import astropy.units as u


def generator(config):
    if isinstance(config, str):
        cfg = yaml.load(open(config, "r"), Loader=yaml.FullLoader)
    else:
        cfg = config

    for par in ('norm', 'index', 'ecut', 'beta'):
        if par in cfg:
            cfg[par] = u.Quantity(cfg[par])
        
    if cfg['type'] == 'pwl':
        spec = lambda e: power_law(e, norm=cfg['norm'], e0=cfg['e0'], index=cfg['index'])
    elif cfg['type'] == 'pwlec':
        spec = lambda e: power_law_ecut(e, norm=cfg['norm'], e0=cfg['e0'], index=cfg['index'], ecut=cfg['ecut'])
    elif cfg['type'] == 'lp':
        spec = lambda e: logparabola(e, norm=cfg['norm'], e0=cfg['e0'], index=cfg['index'], beta=cfg['beta'])
    else:
        raise ValueError(f"Unknown spectrum type '{cfg['type']}'")
    
    return spec


def power_law(e, e0, norm, index):
    return norm * (e/e0).decompose()**index


def power_law_ecut(e, e0, norm, index, ecut):
    return norm * (e/e0).decompose()**index * np.exp(-(e/ecut))


def logparabola(e, e0, norm, index, beta):
    return norm * (e/e0).decompose()**(index + beta * np.log10((e/e0).decompose()))
