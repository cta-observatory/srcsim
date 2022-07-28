import glob
import numpy as np
import pandas as pd
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord


class OffSample:
    def __init__(self, file_name=None, obs_id=None, data_table=None):
        self.units = dict(
            energy = u.TeV,
            angle = u.rad,
            distance = u.m,
            viewcone = u.deg,
            solid_angle = u.sr
        )
        
        # TODO: refine this value
        lst_focal_length = 28.01 * u.m
        self.cam2angle = 1 * u.rad / lst_focal_length

        # TODO: refine the logic below / implement nicer
        if data_table is not None:
            self.file_name = None
            self.obs_id = obs_id
            self.data_table = data_table
        else:
            self.file_name = file_name
            self.obs_id = obs_id
            self.data_table = pd.read_hdf(file_name, 'dl2/event/telescope/parameters/LST_LSTCam').query(f'obs_id == {obs_id}')

        self.n_events = len(self.data_table)
        self.obs_duration = self.calc_obs_duration(self.data_table)

        # Getting the telescope pointing
        pointing_data = self.data_table[['az_tel', 'alt_tel']].mean()
        self.tel_pos = SkyCoord(pointing_data['az_tel'], pointing_data['alt_tel'], unit=self.units['angle'], frame='altaz')

        tel_pos = SkyCoord(
            data_table['az_tel'].to_numpy(),
            data_table['alt_tel'].to_numpy(),
            unit=self.units['angle'],
            frame='altaz'
        )
        cam_x, cam_y = self.data_table[['reco_src_x', 'reco_src_y']].to_numpy().transpose() * self.units['distance'] * self.cam2angle
        self.evt_coord = SkyCoord(cam_x, cam_y, frame=tel_pos.skyoffset_frame())
        self.evt_energy = self.data_table['reco_energy'].to_numpy() * self.units['energy']
        
    def __repr__(self):
        print(
f"""{type(self).__name__} instance
    {'File name':.<20s}: {self.file_name}
    {'Obs ID':.<20s}: {self.obs_id}
    {'Pointing':.<20s}: {self.tel_pos}
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


class OffCollection:
    def __init__(self, file_mask=None, samples=None):
        self.file_mask = file_mask

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

    @classmethod
    def read_file(cls, file_name):
        data = pd.read_hdf(file_name, "dl2/event/telescope/parameters/LST_LSTCam")
        obs_ids = np.unique(data['obs_id'].to_numpy())

        samples = tuple(
            OffSample(
                data_table = data.query(f'obs_id == {obs_id}'),
                obs_id = obs_id
            )
            for obs_id in obs_ids
        )

        return samples

    @classmethod
    def read_files(cls, file_mask):
        file_list = glob.glob(file_mask)

        samples = ()
        for file_name in file_list:
            samples += cls.read_file(file_name)

        return samples

    def get_closest(self, target_position):
        tel_pos = SkyCoord([sample.tel_pos for sample in self.samples])
        separation = tel_pos.separation(target_position)
        idx = separation.argmin()

        return OffCollection(samples=(self.samples[idx],))

    def get_nearby(self, target_position, search_radius):
        samples = tuple(
            filter(
                lambda sample: sample.tel_pos.separation(target_position) <= search_radius,
                self.samples
            )
        )

        return OffCollection(samples=samples)

    def get_in_box(self, target_position, max_lon_offset, max_lat_offset):
        tel_pos = SkyCoord([sample.tel_pos for sample in self.samples])
        target_position = SkyCoord(target_position.altaz.az, target_position.altaz.alt, frame='altaz')

        lon_offset, lat_offset = tel_pos.altaz.spherical_offsets_to(target_position.altaz)
        inbox = (np.absolute(lon_offset) <= max_lon_offset) & (np.absolute(lat_offset) <= max_lat_offset)

        if sum(inbox):
            samples = tuple(sample for sample, take_it in zip(self.samples, inbox) if take_it)
        else:
            samples = ()

        return OffCollection(samples=samples)
