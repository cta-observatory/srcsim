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
    elif cfg['type'] == 'sbpwl':
        spec = lambda e: smoothly_broken_power_law(
            e,
            norm=cfg['norm'],
            e0=cfg['e0'],
            ebr=cfg['ebr'],
            index=cfg['index'],
            index_delta=cfg['index_delta'],
            smoothing=cfg['smoothing']
        )
    elif cfg['type'] == 'lp':
        spec = lambda e: logparabola(e, norm=cfg['norm'], e0=cfg['e0'], index=cfg['index'], beta=cfg['beta'])
    else:
        raise ValueError(f"Unknown spectrum type '{cfg['type']}'")
    
    return spec


def power_law(e, e0, norm, index):
    return norm * (e/e0).decompose()**index


def power_law_ecut(e, e0, norm, index, ecut):
    return norm * (e/e0).decompose()**index * np.exp(-(e/ecut))


def smoothly_broken_power_law(e, e0, norm, index, index_delta, ebr, smoothing):
    """
    From Eq.9 in https://ui.adsabs.harvard.edu/abs/2021PhRvL.126t1102A/abstract
    """
    return norm * (e/e0).decompose()**index * (1 + (e/ebr).decompose()**smoothing)**(index_delta / smoothing)


def logparabola(e, e0, norm, index, beta):
    return norm * (e/e0).decompose()**(index + beta * np.log10((e/e0).decompose()))
