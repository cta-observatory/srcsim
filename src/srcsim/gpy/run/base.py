import numpy as np
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, SkyOffsetFrame, AltAz


class DataRun:
    def __init__(self, tel_pos, tstart, tstop, obsloc, id=0):
        self.id = id
        self.tel_pos = tel_pos
        self.obsloc = obsloc
        self.tstart = tstart
        self.tstop = tstop 

    @classmethod
    def from_config(cls, config):
        pass

    def to_dict(self):
        pass

    def tel_pos_to_altaz(self, frame):
        pass

    def predict(self, irf_collections, model, tel_pos_tolerance=None):
        pass