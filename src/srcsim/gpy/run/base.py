import numpy as np
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, SkyOffsetFrame, AltAz

from gammapy.data import Observation, PointingMode, FixedPointingInfo
from gammapy.datasets import MapDataset
from gammapy.maps import MapAxis, WcsGeom
# from gammapy.datasets import MapDatasetEventSampler
# from gammapy.makers import MapDatasetMaker

from .makers import MapDatasetMaker
from .simulate import MapDatasetEventSampler


class DataRun:
    mode = None

    def __init__(self, tel_pos, tstart, tstop, obsloc, id=0):
        self.id = id
        self.tel_pos = tel_pos
        self.obsloc = obsloc
        self.tstart = tstart
        self.tstop = tstop 

    @classmethod
    def from_config(cls, config):
        pass

    @property
    def pointing(self):
        return None

    @property
    def slew_length_ra(self):
        return 0 * u.deg

    @property
    def slew_length_dec(self):
        return 0 * u.deg

    @property
    def tel_pos_center_icrs(self):
        return None

    def to_dict(self):
        pass

    def tel_pos_to_altaz(self, frame):
        pass

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

        observation = Observation.create(
            pointing=self.pointing,
            location=self.obsloc,
            obs_id=self.id,
            tstart=self.tstart,
            tstop=self.tstop,
            irfs=irfs.samples[0].to_dict()
        )
        observation.aeff.meta["TELESCOP"] = 'cta_north'

        energy_axis = MapAxis.from_energy_bounds(
            "0.03 TeV",
            "100 TeV",
            nbin=20,
            per_decade=True
        )
        energy_axis_true = MapAxis.from_energy_bounds(
            "0.01 TeV",
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

        geom = WcsGeom.create(
            skydir=self.tel_pos_center_icrs,
            width=(
                self.slew_length_ra.to('deg').value + 5,
                self.slew_length_dec.to('deg').value + 5)
            ,
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
