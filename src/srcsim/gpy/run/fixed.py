import yaml
import numpy as np
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz

from gammapy.data import PointingMode
from gammapy.data import Observation, PointingMode, FixedPointingInfo
from gammapy.datasets import MapDataset
# from gammapy.datasets import MapDatasetEventSampler
# from gammapy.makers import MapDatasetMaker
from gammapy.maps import MapAxis, WcsGeom

from .makers import MapDatasetMaker
from .simulate import MapDatasetEventSampler

from .base import DataRun


class FixedPointingDataRun(DataRun):
    mode = PointingMode.DRIFT

    def __str__(self):
        frame_start = AltAz(obstime=self.tstart, location=self.obsloc)
        frame_stop = AltAz(obstime=self.tstop, location=self.obsloc)
        tel_pos_start = SkyCoord(
            self.tel_pos.az,
            self.tel_pos.alt,
            frame=frame_start
        )
        tel_pos_stop = SkyCoord(
            self.tel_pos.az,
            self.tel_pos.alt,
            frame=frame_stop
        )
        desc = f"""{type(self).__name__} instance
    {'ID':.<20s}: {self.id}
    {'Tel. alt/az':.<20s}: {self.tel_pos}
    {'Tstart':.<20s}: {self.tstart.isot}
    {'Tstop':.<20s}: {self.tstop.isot}
    {'Tel. RA':.<20s}: [{tel_pos_start.icrs.ra.to('deg'):.2f} - {tel_pos_stop.icrs.ra.to('deg'):.2f}]
    {'Tel. Dec':.<20s}: [{tel_pos_start.icrs.dec.to('deg'):.2f} - {tel_pos_stop.icrs.dec.to('deg'):.2f}]
"""
        return desc

    @classmethod
    def from_config(cls, config):
        if isinstance(config, str):
            cfg = yaml.load(open(config, "r"), Loader=yaml.FullLoader)
        else:
            cfg = config

        dummy_obstime = Time('2000-01-01T00:00:00')
        location = EarthLocation(
            lat=u.Quantity(cfg['location']['lat']),
            lon=u.Quantity(cfg['location']['lon']),
            height=u.Quantity(cfg['location']['height']),
        )

        data_run = cls(
            tel_pos = SkyCoord(
                u.Quantity(cfg['pointing']['az']),
                u.Quantity(cfg['pointing']['alt']),
                location=location,
                obstime=dummy_obstime
            ),
            tstart = Time(cfg['time']['start']),
            tstop = Time(cfg['time']['stop']),
            obsloc = location,
            id = cfg['id'] if 'id' in cfg else 0,
            mode = PointingMode.DRIFT
        )

        return data_run

    @property
    def pointing(self):
        pointing = FixedPointingInfo(
            fixed_altaz=self.tel_pos, 
            mode=self.mode,
            location=self.obsloc
        )
        return pointing
    
    @property
    def slew_length_ra(self):
        frame_start = AltAz(obstime=self.tstart, location=self.obsloc)
        frame_stop = AltAz(obstime=self.tstop, location=self.obsloc)

        tel_pos_start = SkyCoord(self.tel_pos.az, self.tel_pos.alt, frame=frame_start)
        tel_pos_stop = SkyCoord(self.tel_pos.az, self.tel_pos.alt, frame=frame_stop)

        return tel_pos_stop.icrs.ra - tel_pos_start.icrs.ra
    
    @property
    def tel_pos_center_icrs(self):
        frame_tref = AltAz(
            obstime=Time(
                (self.tstart.mjd + self.tstop.mjd) / 2,
                format='mjd'
            ),
            location=self.obsloc
        )
        return SkyCoord(self.tel_pos.az, self.tel_pos.alt, frame=frame_tref)

    def to_dict(self):
        data = {'id': self.id, 'pointing': {}, 'time': {}, 'location': {}}

        data['pointing']['alt'] = str(self.tel_pos.alt.to('deg').value) + ' deg'
        data['pointing']['az'] = str(self.tel_pos.az.to('deg').value) + ' deg'

        data['time']['start'] = self.tstart.isot
        data['time']['stop'] = self.tstop.isot

        data['location']['lon'] = str(self.obsloc.lon.to('deg').value) + ' deg'
        data['location']['lat'] = str(self.obsloc.lat.to('deg').value) + ' deg'
        data['location']['height'] = str(self.obsloc.height.to('m').to_string())

        return data
    
    def tel_pos_to_altaz(self, frame):
        if frame.obstime.size > 1:
            tel_pos = SkyCoord(
                np.repeat(self.tel_pos.az, frame.obstime.size),
                np.repeat(self.tel_pos.alt, frame.obstime.size),
                frame=frame
            )
        else:
            tel_pos = SkyCoord(
                self.tel_pos.az,
                self.tel_pos.alt,
                frame=frame
            )
        return tel_pos
