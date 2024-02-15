import yaml
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz

from gammapy.data import Observation, PointingMode, FixedPointingInfo
from gammapy.datasets import MapDataset, MapDatasetEventSampler
from gammapy.makers import MapDatasetMaker
from gammapy.maps import MapAxis, WcsGeom

from .base import DataRun


class SkyDataRun(DataRun):
    mode = PointingMode.POINTING

    def __str__(self):
        frame_start = AltAz(obstime=self.tstart, location=self.obsloc)
        frame_stop = AltAz(obstime=self.tstop, location=self.obsloc)
        desc = f"""{type(self).__name__} instance
    {'ID':.<20s}: {self.id}
    {'Tel. RA/Dec':.<20s}: {self.tel_pos}
    {'Tstart':.<20s}: {self.tstart.isot}
    {'Tstop':.<20s}: {self.tstop.isot}
    {'Tel. azimuth':.<20s}: [{self.tel_pos.transform_to(frame_start).az.to('deg'):.2f} - {self.tel_pos.transform_to(frame_stop).az.to('deg'):.2f}]
    {'Tel. alt':.<20s}: [{self.tel_pos.transform_to(frame_start).alt.to('deg'):.2f} - {self.tel_pos.transform_to(frame_stop).alt.to('deg'):.2f}]
"""
        return desc

    @classmethod
    def from_config(cls, config):
        if isinstance(config, str):
            cfg = yaml.load(open(config, "r"), Loader=yaml.FullLoader)
        else:
            cfg = config

        data_run = cls(
            SkyCoord(
                u.Quantity(cfg['pointing']['ra']),
                u.Quantity(cfg['pointing']['dec']),
                frame='icrs'
            ),
            Time(cfg['time']['start']),
            Time(cfg['time']['stop']),
            EarthLocation(
                lat=u.Quantity(cfg['location']['lat']),
                lon=u.Quantity(cfg['location']['lon']),
                height=u.Quantity(cfg['location']['height']),
            ),
            cfg['id'] if 'id' in cfg else 0
        )

        return data_run

    @property
    def pointing(self):
        return FixedPointingInfo(fixed_icrs=self.tel_pos, mode=PointingMode.POINTING)

    @property
    def tel_pos_center_icrs(self):
        return self.tel_pos.icrs

    def to_dict(self):
        data = {'id': self.id, 'pointing': {}, 'time': {}, 'location': {}}

        data['pointing']['ra'] = str(self.tel_pos.icrs.ra.to('deg').value) + ' deg'
        data['pointing']['dec'] = str(self.tel_pos.icrs.dec.to('deg').value) + ' deg'

        data['time']['start'] = self.tstart.isot
        data['time']['stop'] = self.tstop.isot

        data['location']['lon'] = str(self.obsloc.lon.to('deg').value) + ' deg'
        data['location']['lat'] = str(self.obsloc.lat.to('deg').value) + ' deg'
        data['location']['height'] = str(self.obsloc.height.to('m').to_string())

        return data

    def tel_pos_to_altaz(self, frame):
        return self.tel_pos.transform_to(frame)
