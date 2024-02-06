import re
import glob
import numpy as np
from dataclasses import dataclass

from astropy.coordinates import SkyCoord
from gammapy.irf.effective_area import EffectiveAreaTable2D
from gammapy.irf.edisp.core import EnergyDispersion2D
from gammapy.irf.background import Background2D
from gammapy.irf.psf.table import PSF3D

from gammapy.irf import load_irf_dict_from_file


@dataclass
class IRFSample:
    altaz: SkyCoord = None
    aeff: EffectiveAreaTable2D = None
    edisp: EnergyDispersion2D = None
    bkg: Background2D = None
    psf: PSF3D = None

    @classmethod
    def from_fits(cls, file_name):
        parsed = re.findall(
            '.*node_corsika_theta_([\d\.]+)_az_([\d\.]+).*',
            file_name
        )
        if not len(parsed):
            raise RuntimeError(
                f'can not parse pointing alt/az from "{file_name}"'
            )
        
        try:
            theta = float(parsed[0][0])
            az = float(parsed[0][1])
            altaz = SkyCoord(az, 90-theta, unit='deg', frame='altaz')
        except Exception as exc:
            raise RuntimeError(
                f'can not convert pointing ({theta}; {az}) to SkyCoord'
            )

        irfs = load_irf_dict_from_file(file_name)

        return cls(altaz, irfs.get('aeff'), irfs.get('edisp'), irfs.get('bkg'), irfs.get('psf'))


class IRFCollection:
    def __init__(self, file_mask=None, samples=None):
        self.file_mask = file_mask

        if samples is None:
            self.samples = self.read_files(file_mask)
        else:
            self.samples = samples

    @classmethod
    def read_files(cls, file_mask):
        file_list = glob.glob(file_mask)

        samples = tuple(
            IRFSample.from_fits(file_name)
            for file_name in file_list
        )

        return samples
    
    def get_closest(self, target_position):
        altaz = SkyCoord([sample.altaz for sample in self.samples])
        separation = altaz.separation(target_position)
        idx = separation.argmin()

        return IRFCollection(samples=(self.samples[idx],))
    
    def get_nearby(self, target_position, search_radius):
        samples = tuple(
            filter(
                lambda sample: sample.altaz.separation(target_position) <= search_radius,
                self.samples
            )
        )

        return IRFCollection(samples=samples)

    def get_in_box(self, target_position, max_lon_offset, max_lat_offset):
        centers = SkyCoord([sample.altaz for sample in self.samples])
        target_position = SkyCoord(target_position.altaz.az, target_position.altaz.alt, frame='altaz')

        lon_offset, lat_offset = centers.altaz.spherical_offsets_to(target_position.altaz)
        inbox = (np.absolute(lon_offset) <= max_lon_offset) & (np.absolute(lat_offset) <= max_lat_offset)

        if sum(inbox):
            samples = tuple(sample for sample, take_it in zip(self.samples, inbox) if take_it)
        else:
            samples = ()

        return IRFCollection(samples=samples)
