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

    sources = []

    for scfg in cfg:
        for par in ('ra', 'dec', 'radius', 'sigma'):
            if par in scfg['spatial']:
                scfg['spatial'][par] = u.Quantity(scfg['spatial'][par])

        if scfg['spatial']['type'] == 'disk':
            src = DiskSource(
                SkyCoord(ra=scfg['spatial']['ra'], dec=scfg['spatial']['dec'], frame='icrs'),
                scfg['spatial']['radius'],
                specgen(scfg['spectral'])
            )
        elif scfg['spatial']['type'] == 'gauss':
            src = GaussSource(
                SkyCoord(ra=scfg['spatial']['ra'], dec=scfg['spatial']['dec'], frame='icrs'),
                scfg['spatial']['sigma'],
                specgen(scfg['spectral'])
            )
        elif scfg['spatial']['type'] == 'iso':
            src = IsotropicSource(
                SkyCoord(ra=scfg['spatial']['ra'], dec=scfg['spatial']['dec'], frame='icrs'),
                specgen(scfg['spectral'])
            )
        else:
            raise ValueError(f"Unknown source type '{scfg['type']}'")

        sources.append(src)
    
    return sources


class Source:
    def __init__(self, pos, dnde, name='source'):
        self.pos = pos
        self.dnde = dnde
        self.name = name
        
    def dndo(self, coord):
        pass

    def dndedo(self, energy, coord):
        return self.dnde(energy) * self.dndo(coord)


class DiskSource(Source):
    def __init__(self, pos, rad, dnde):
        super().__init__(pos, dnde)
        self.rad = rad

    def __repr__(self):
        print(
f"""{type(self).__name__} instance
    {'Name':.<20s}: {self.name}
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
    {'Name':.<20s}: {self.name}
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
    {'Name':.<20s}: {self.name}
    {'Position':.<20s}: {self.pos}
"""
        )

        return super().__repr__()

    def dndo(self, coord):
        return 1 / (4 * np.pi * u.sr)
