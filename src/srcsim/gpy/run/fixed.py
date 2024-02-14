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
    
    def predict(self, irf_collections, model, tel_pos_tolerance=None):
        frame_tref = AltAz(
            obstime=Time(
                (self.tstart.mjd + self.tstop.mjd) / 2,
                format='mjd'
            ),
            location=self.obsloc
        )

        tel_pos = self.tel_pos_to_altaz(frame_tref)
        
        if tel_pos_tolerance is None:
            irfs = irf_collections.get_closest(tel_pos.altaz)
        elif isinstance(tel_pos_tolerance, u.Quantity):
            irfs = irf_collections.get_nearby(tel_pos, tel_pos_tolerance)
        elif isinstance(tel_pos_tolerance, list) or isinstance(tel_pos_tolerance, tuple):
            irfs = irf_collections.get_in_box(
                tel_pos,
                max_lon_offset=tel_pos_tolerance[0],
                max_lat_offset=tel_pos_tolerance[1],
            )
        else:
            raise ValueError(f"Data type '{type(tel_pos_tolerance)}' for argument 'tel_pos_tolerance' is not supported")

        pointing = FixedPointingInfo(
            fixed_altaz=self.tel_pos, 
            mode=self.mode,
            location=self.obsloc
        )

        observation = Observation.create(
            pointing=pointing,
            location=self.obsloc,
            obs_id=self.id,
            tstart=self.tstart,
            tstop=self.tstop,
            irfs=irfs.samples[0].to_dict()
        )
        observation.aeff.meta["TELESCOP"] = 'cta_north'

        energy_axis = MapAxis.from_energy_bounds(
            "0.1 TeV",
            "100 TeV",
            nbin=20,
            per_decade=True
        )
        energy_axis_true = MapAxis.from_energy_bounds(
            "0.03 TeV",
            "300 TeV",
            nbin=20,
            per_decade=True,
            name="energy_true"
        )
        migra_axis = MapAxis.from_bounds(
            0.5,
            2,
            nbin=150,
            node_type="edges",
            name="migra"
        )

        frame_tref = AltAz(
            obstime=Time(
                (self.tstart.mjd + self.tstop.mjd) / 2,
                format='mjd'
            ),
            location=self.obsloc
        )
        frame_start = AltAz(obstime=self.tstart, location=self.obsloc)
        frame_stop = AltAz(obstime=self.tstop, location=self.obsloc)
        
        tel_pos_ref = SkyCoord(self.tel_pos.az, self.tel_pos.alt, frame=frame_tref)
        tel_pos_start = SkyCoord(self.tel_pos.az, self.tel_pos.alt, frame=frame_start)
        tel_pos_stop = SkyCoord(self.tel_pos.az, self.tel_pos.alt, frame=frame_stop)

        ra_width = tel_pos_stop.icrs.ra - tel_pos_start.icrs.ra

        geom = WcsGeom.create(
            skydir=tel_pos_ref.icrs,
            width=(ra_width.to('deg').value + 5, 5),
            binsz=0.1,
            frame="icrs",
            axes=[energy_axis],
        )

        empty = MapDataset.create(
            geom,
            energy_axis_true=energy_axis_true,
            migra_axis=migra_axis,
            name="my-dataset",
        )

        maker = MapDatasetMaker(
            selection=["exposure", "background", "psf", "edisp"]
        )
        dataset = maker.run(empty, observation)

        dataset.models = model

        sampler = MapDatasetEventSampler(random_state=0)
        events = sampler.run(dataset, observation)

        observation._events = events

        return observation
