import yaml
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord

from .spec import generator as specgen


def generator(config):
    if isinstance(config, str):
        cfg = yaml.load(open(config, "r"), Loader=yaml.FullLoader)
    else:
        cfg = config

    for par in ('ra', 'dec', 'radius', 'sigma'):
        if par in cfg['spatial']:
            cfg['spatial'][par] = u.Quantity(cfg['spatial'][par])
            
    if cfg['spatial']['type'] == 'disk':
        src = DiskSource(
            SkyCoord(ra=cfg['spatial']['ra'], dec=cfg['spatial']['dec'], frame='icrs'),
            cfg['spatial']['radius'],
            specgen(cfg['spectral'])
        )
    elif cfg['spatial']['type'] == 'gauss':
        src = GaussSource(
            SkyCoord(ra=cfg['spatial']['ra'], dec=cfg['spatial']['dec'], frame='icrs'),
            cfg['spatial']['sigma'],
            specgen(cfg['spectral'])
        )
    elif cfg['spatial']['type'] == 'iso':
        src = IsotropicSource(
            SkyCoord(ra=cfg['spatial']['ra'], dec=cfg['spatial']['dec'], frame='icrs'),
            specgen(cfg['spectral'])
        )
    else:
        raise ValueError(f"Unknown source type '{cfg['type']}'")
    
    return src


class Source:
    def __init__(self, pos, dnde):
        self.pos = pos
        self.dnde = dnde
        
    def dndo(self, x, y):
        pass


class DiskSource(Source):
    def __init__(self, pos, rad, dnde):
        super().__init__(pos, dnde)
        self.rad = rad

    def __repr__(self):
        print(
f"""{type(self).__name__} instance
    {'Position':.<20s}: {self.pos}
    {'Radius':.<20s}: {self.rad}
"""
        )

        return super().__repr__()

    def dndo(self, coord):
        sky_area = 2 * np.pi * (1 - np.cos(self.rad)) * u.sr
        norm = 1 / sky_area
        
        return norm * (self.pos.separation(coord) < self.rad)


class GaussSource(Source):
    def __init__(self, pos, sigma, dnde):
        super().__init__(pos, dnde)
        self.sigma = sigma

    def __repr__(self):
        print(
f"""{type(self).__name__} instance
    {'Position':.<20s}: {self.pos}
    {'Sigma':.<20s}: {self.sigma}
"""
        )

        return super().__repr__()

    def dndo(self, coord):
        norm = 1 / (2 * np.pi * self.sigma**2)
        
        r = self.pos.separation(coord)
        
        return norm * np.exp( -(r**2 / (2 * self.sigma**2)).decompose() )


class IsotropicSource(Source):
    def __init__(self, pos, dnde):
        super().__init__(pos, dnde)

    def __repr__(self):
        print(
f"""{type(self).__name__} instance
    {'Position':.<20s}: {self.pos}
"""
        )

        return super().__repr__()

    def dndo(self, coord):
        return 1 / (4 * np.pi * u.sr)
