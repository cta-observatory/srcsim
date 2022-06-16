import yaml
import numpy as np
import scipy.interpolate
import astropy.units as u
from astropy.io import fits
from astropy.wcs import WCS
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
                emission_type=scfg['emission_type'],
                pos=SkyCoord(ra=scfg['spatial']['ra'], dec=scfg['spatial']['dec'], frame='icrs'),
                rad=scfg['spatial']['radius'],
                dnde=specgen(scfg['spectral']),
                name=scfg['name']
            )
        elif scfg['spatial']['type'] == 'gauss':
            src = GaussSource(
                emission_type=scfg['emission_type'],
                pos=SkyCoord(ra=scfg['spatial']['ra'], dec=scfg['spatial']['dec'], frame='icrs'),
                sigma=scfg['spatial']['sigma'],
                dnde=specgen(scfg['spectral']),
                name=scfg['name']
            )
        elif scfg['spatial']['type'] == 'iso':
            src = IsotropicSource(
                emission_type=scfg['emission_type'],
                pos=SkyCoord(ra=scfg['spatial']['ra'], dec=scfg['spatial']['dec'], frame='icrs'),
                dnde=specgen(scfg['spectral']),
                name=scfg['name']
            )
        elif scfg['spatial']['type'] == 'fitscube':
            src = FitsCubeSource(
                emission_type=scfg['emission_type'],
                file_name=scfg['spatial']['file_name'],
                name=scfg['name']
            )
        else:
            raise ValueError(f"Unknown source type '{scfg['type']}'")

        sources.append(src)
    
    return sources


class Source:
    def __init__(self, emission_type, pos, dnde, name='source'):
        self.pos = pos
        self.dnde = dnde
        self.name = name
        self.emission_type = emission_type
        
    def dndo(self, coord):
        pass

    def dndedo(self, energy, coord):
        return self.dnde(energy) * self.dndo(coord)


class DiskSource(Source):
    def __init__(self, emission_type, pos, rad, dnde, name='disk_source'):
        super().__init__(emission_type, pos, dnde, name)
        self.rad = rad

    def __repr__(self):
        print(
f"""{type(self).__name__} instance
    {'Name':.<20s}: {self.name}
    {'Emission type':.<20s}: {self.emission_type}
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
    def __init__(self, emission_type, pos, sigma, dnde, name='gauss_source'):
        super().__init__(emission_type, pos, dnde, name)
        self.sigma = sigma

    def __repr__(self):
        print(
f"""{type(self).__name__} instance
    {'Name':.<20s}: {self.name}
    {'Emission type':.<20s}: {self.emission_type}
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
    def __init__(self, emission_type, pos, dnde, name='iso_source'):
        super().__init__(emission_type, pos, dnde, name)

    def __repr__(self):
        print(
f"""{type(self).__name__} instance
    {'Name':.<20s}: {self.name}
    {'Emission type':.<20s}: {self.emission_type}
    {'Position':.<20s}: {self.pos}
"""
        )

        return super().__repr__()

    def dndo(self, coord):
        return 1 / (4 * np.pi * u.sr)


class FitsCubeSource(Source):
    def __init__(self, emission_type, file_name, name='fits_source'):
        cube, wcs = self.read_data(file_name)

        pos = wcs.pixel_to_world(0, 0, 0)[0]
        super().__init__(emission_type, pos=pos, dnde=None, name=name)

        self.file_name = file_name
        self.cube = cube
        self.wcs = wcs
        self._cube_interpolator = self._get_cube_interpolator(cube)

    def __repr__(self):
        print(
f"""{type(self).__name__} instance
    {'Name':.<20s}: {self.name}
    {'File name':.<20s}: {self.file_name}
    {'Emission type':.<20s}: {self.emission_type}
    {'Position':.<20s}: {self.pos}
"""
        )

        return super().__repr__()

    @classmethod
    def read_data(cls, file_name):
        with fits.open(file_name) as hdus:
            wcs = WCS(hdus['primary'].header)

            zero = 0
            if 'BZERO' in hdus['primary'].header:
                zero = hdus['primary'].header['BZERO']

            scale = 1
            if 'BSCALE' in hdus['primary'].header:
                scale = hdus['primary'].header['BSCALE']

            if 'BUNIT' in hdus['primary'].header:
                unit = u.Unit(hdus['primary'].header['BUNIT'])
            else:
                raise ValueError("No 'BUNIT' keyword in the primary extension header")

            cube = (hdus['primary'].data.transpose() - zero) * scale * unit

        return cube, wcs

    @classmethod
    def _get_cube_interpolator(self, cube):
        x = np.arange(cube.shape[0])
        y = np.arange(cube.shape[1])
        z = np.arange(cube.shape[2])

        interp = scipy.interpolate.RegularGridInterpolator(
            (x, y, z),
            cube.value,
            bounds_error = False,
            fill_value = 0,
        )

        return interp

    def cube_value(self, x, y, z):
        val = self._cube_interpolator(list(zip(x.flatten(), y.flatten(), z.flatten()))) * self.cube.unit
        return val.reshape(x.shape)

    def dndedo(self, energy, coord):
        x, y, z = self.wcs.world_to_pixel(coord, energy)
        return self.cube_value(x, y, z)
