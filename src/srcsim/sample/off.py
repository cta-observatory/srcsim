import numpy as np
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord

from .sample import SampleBase


class OffSample(SampleBase):
    def __init__(self, file_name=None, obs_id=None, data_table=None):
        self.units = dict(
            energy = u.TeV,
            angle = u.rad,
            distance = u.m,
            viewcone = u.deg,
            solid_angle = u.sr
        )

        # TODO: refine the logic below / implement nicer
        if data_table is not None:
            self.file_name = None
            self.obs_id = obs_id
            self.data_table = data_table
        else:
            self.file_name = file_name
            self.obs_id = obs_id
            self.data_table = self.read_data(file_name).query(f'obs_id == {obs_id}')

        self.n_events = len(self.data_table)
        self.obs_duration = self.calc_obs_duration(self.data_table)

        self.tel_pos = SkyCoord(
            self.data_table['az_tel'].to_numpy(),
            self.data_table['alt_tel'].to_numpy(),
            unit=self.units['angle'],
            frame='altaz'
        )

        alt, az = self.data_table[['reco_alt', 'reco_az']].to_numpy().transpose()
        self.evt_coord = SkyCoord(az, alt, frame='altaz', unit=self.units['angle'])
        self.evt_coord = self.evt_coord.transform_to(self.tel_pos.skyoffset_frame())
        self.evt_energy = self.data_table['reco_energy'].to_numpy() * self.units['energy']
        
    def __repr__(self):
        print(
f"""{type(self).__name__} instance
    {'File name':.<20s}: {self.file_name}
    {'Obs ID':.<20s}: {self.obs_id}
    {'Pointing azimuth':.<20s}: [{self.tel_pos.az.min().to('deg'):.2f} - {self.tel_pos.az.max().to('deg'):.2f}]
    {'Pointing altitude':.<20s}: [{self.tel_pos.alt.min().to('deg'):.2f} - {self.tel_pos.alt.max().to('deg'):.2f}]
    {'N events':.<20s}: {self.n_events}
    {'Obs. duration':.<20s}: {self.obs_duration.to('min')}
"""
        )

        return super().__repr__()
    
    @classmethod
    def calc_obs_duration(self, data_table):
        mjd = Time(data_table['trigger_time'].to_numpy(), format='unix').mjd

        time_diff = np.diff(np.sort(mjd))
        time_diff_max = np.percentile(time_diff, 99.99)
        time_diff = time_diff[time_diff < time_diff_max]

        t_elapsed = u.d * np.sum(time_diff[time_diff < time_diff_max])

        return t_elapsed

    def dndedo(self, energy, coord):
        dummy = 1 + 0 * energy.value
        val = dummy * 1 / (1 / self.obs_duration * u.Unit('1/(s TeV sr)'))
        return val