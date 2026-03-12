import glob
import logging
import numpy as np
from abc import abstractmethod
from astropy.coordinates import SkyCoord


class CollectionBase:
    def __init__(self, file_mask=None, samples=None, log=None):
        self.file_mask = file_mask

        if log is None:
            self.log = logging.getLogger(__name__)
        else:
            self.log = log.getChild(__name__)

        if samples is None:
            self.samples = self.read_files(file_mask)
        else:
            self.samples = samples

    def __repr__(self):
        print(
f"""{type(self).__name__} instance
    {'File mask':.<20s}: {self.file_mask}
    {'Obs IDs':.<20s}: {tuple(sample.obs_id for sample in self.samples)}
"""
        )

        return super().__repr__()

    @abstractmethod
    def read_file(self, file_name):
        pass

    @classmethod
    def read_files(cls, file_mask):
        file_list = glob.glob(file_mask)

        samples = ()
        for file_name in file_list:
            samples += cls.read_file(file_name)

        return samples

    def get_closest(self, target_position):
        target_position = SkyCoord(target_position.altaz.az, target_position.altaz.alt, frame='altaz')
        tel_pos = SkyCoord([sample.tel_pos for sample in self.samples])
        separation = tel_pos.separation(target_position)
        idx = separation.argmin()

        return self.__class__(samples=(self.samples[idx],))

    def get_nearby(self, target_position, search_radius):
        target_position = SkyCoord(target_position.altaz.az, target_position.altaz.alt, frame='altaz')
        samples = tuple(
            filter(
                lambda sample: sample.tel_pos.separation(target_position) <= search_radius,
                self.samples
            )
        )

        self.log.debug(
            f"found {len(samples)} within {search_radius.to('deg'):.1f} around "
            f"(alt,az) = ({target_position.altaz.alt.to('deg'):.2f} , {target_position.altaz.az.to('deg'):.2f})"
        )

        return self.__class__(samples=samples)

    def get_in_box(self, target_position, max_lon_offset, max_lat_offset):
        target_position = SkyCoord(target_position.altaz.az, target_position.altaz.alt, frame='altaz')
        tel_pos = SkyCoord([sample.tel_pos for sample in self.samples])

        lon_offset, lat_offset = tel_pos.altaz.spherical_offsets_to(target_position.altaz)
        inbox = (np.absolute(lon_offset) <= max_lon_offset) & (np.absolute(lat_offset) <= max_lat_offset)

        if sum(inbox):
            samples = tuple(sample for sample, take_it in zip(self.samples, inbox) if take_it)
        else:
            samples = ()

        self.log.debug(
            f"found {len(samples)} within "
            f"(dlot, dlat) = ({max_lon_offset.to('deg') / 2 :.1f}, {max_lat_offset.to('deg') / 2 :.1f}) around "
            f"(alt,az) = ({target_position.altaz.alt.to('deg'):.2f} , {target_position.altaz.az.to('deg'):.2f})"
        )

        return self.__class__(samples=samples)
