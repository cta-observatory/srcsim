import glob
import numpy as np
import pandas as pd
import tables
import astropy.units as u
from astropy.coordinates import SkyCoord


def power_law(e, e0, norm, index):
    return norm * (e/e0).decompose()**index


class MCSample:
    def __init__(self, file_name=None, obs_id=None, data_table=None, config_table=None):
        self.units = dict(
            energy = u.TeV,
            angle = u.rad,
            distance = u.m,
            viewcone = u.deg
        )
        
        # TODO: refine this value
        lst_focal_length = 28.01 * u.m
        self.cam2angle = 1 * u.rad / lst_focal_length

        # TODO: refine the logic below / implement nicer
        if data_table is not None and config_table is not None:
            self.file_name = None
            self.obs_id = config_table['obs_id'].iloc[0]
            self.config_table = config_table
            self.data_table = data_table
        else:
            self.file_name = file_name
            self.obs_id = obs_id
            self.config_table = self.read_config(file_name, obs_id)
            self.data_table = pd.read_hdf(file_name, 'dl2/event/telescope/parameters/LST_LSTCam').query(f'obs_id == {obs_id}')

        # Getting the telescope pointing
        pointing_data = self.data_table[['mc_az_tel', 'mc_alt_tel']].mean()
        self.tel_pos = SkyCoord(pointing_data['mc_az_tel'], pointing_data['mc_alt_tel'], unit=self.units['angle'], frame='altaz')
        
        # Working out the simulation spectrum
        rmin, rmax = self.config_table[['min_scatter_range', 'max_scatter_range']].iloc[0] * self.units['distance']
        ground_area = np.pi * (rmax**2 - rmin**2)
        nevents = self.config_table['num_showers'].iloc[0] * self.config_table['shower_reuse'].iloc[0]
        emin = self.config_table['energy_range_min'].iloc[0] * self.units['energy']
        emax = self.config_table['energy_range_max'].iloc[0] * self.units['energy']
        index = self.config_table['spectral_index'].iloc[0]
        self.spec_data = self.get_spec_data(nevents, emin, emax, index)
        self.spec_data['norm'] /= ground_area

        cam_x, cam_y = self.data_table[['src_x', 'src_y']].to_numpy().transpose() * self.units['distance'] * self.cam2angle
        self.evt_coord = SkyCoord(cam_x, cam_y, frame=self.tel_pos.skyoffset_frame())

        self.evt_energy = self.data_table['mc_energy'].to_numpy() * self.units['energy']

        # Filtering out events with excessive offsets (e.g. due to the simulation numerical accuracy)
        offset_min, offset_max = self.config_table[['min_viewcone_radius', 'max_viewcone_radius']].iloc[0] * self.units['viewcone']
        evt_offset = self.evt_coord.separation(self.tel_pos)

        in_fov = (evt_offset >= offset_min) & (evt_offset <= offset_max)
        self.data_table = self.data_table[in_fov]
        self.evt_coord = self.evt_coord[in_fov]
        self.evt_energy = self.evt_energy[in_fov]
        
    def __repr__(self):
        print(
f"""{type(self).__name__} instance
    {'File name':.<20s}: {self.file_name}
    {'Obs ID':.<20s}: {self.obs_id}
    {'Pointing':.<20s}: {self.tel_pos}
    {'N events':.<20s}: {len(self.data_table)}
    {'Energy range':.<20s}: [{self.config_table['energy_range_min'].iloc[0]:.2e}; {self.config_table['energy_range_max'].iloc[0]:.2e}] {self.units['energy']}
    {'Viewcone':.<20s}: [{self.config_table['min_viewcone_radius'].iloc[0]:.1f}; {self.config_table['max_viewcone_radius'].iloc[0]:.1f}] {self.units['viewcone']}
"""
        )

        return super().__repr__()
    
    @classmethod
    def read_config(cls, file_name, obs_id):
        with tables.open_file(file_name) as table:
            cfg_table = table.root['/simulation/run_config']
            obs_ids = [v['obs_id'] for v in cfg_table.iterrows()]

            columns = (
                'obs_id',
                'num_showers',
                'shower_reuse',
                'min_scatter_range',
                'max_scatter_range',
                'energy_range_min',
                'energy_range_max',
                'spectral_index',
                'min_viewcone_radius',
                'max_viewcone_radius'
            )

            obs_idx = obs_ids.index(obs_id)

            data = {}

            for col_name in columns:
                col_idx = cfg_table.colnames.index(col_name)
                data[col_name] = (cfg_table[obs_idx][col_idx], )

            return pd.DataFrame(data=data)

    def get_spec_data(self, n_events, emin, emax, index=-1):
        e0 = (emin * emax)**0.5

        if index == -1:
            norm = n_events / e0 / (np.log(emax/e0) - np.log(emin/e0))
        else:
            norm = n_events * (index + 1) / e0 / ((emax/e0).decompose()**((index + 1)) - (emin/e0).decompose()**((index + 1)))

        norm = norm.to(1/u.eV)

        sim_spec = {
            'norm': norm,
            'e0': e0,
            'index': index
        }

        return sim_spec
    
    def dnde(self, energy):
        return power_law(energy, **self.spec_data)
    
    def dndo(self, coord):
        offset_min, offset_max = self.config_table[['min_viewcone_radius', 'max_viewcone_radius']].iloc[0] * self.units['viewcone']
        sky_area = 2 * np.pi * (np.cos(offset_min) - np.cos(offset_max)) * u.sr
        norm = 1 / sky_area

        r = self.tel_pos.separation(coord)
        
        return norm * (r >= offset_min) * (r <= offset_max)

    def dndedo(self, energy, coord):
        return self.dnde(energy) * self.dndo(coord)


class MCCollection:
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
    def read_obs_ids(cls, file_name):
        with tables.open_file(file_name, 'r') as data:
            cfg_table = data.root['/simulation/run_config']
            obs_ids = [v['obs_id'] for v in cfg_table.iterrows()]

        return obs_ids

    @classmethod
    def read_config(cls, file_name):
        with tables.open_file(file_name) as table:
            cfg_table = table.root['/simulation/run_config']

            columns = (
                'obs_id',
                'num_showers',
                'shower_reuse',
                'min_scatter_range',
                'max_scatter_range',
                'energy_range_min',
                'energy_range_max',
                'spectral_index',
                'min_viewcone_radius',
                'max_viewcone_radius'
            )

            data = {}

            for col_name in columns:
                data[col_name] = [
                    v[col_name] for v in cfg_table.iterrows()
                ]

        return pd.DataFrame(data=data)

    @classmethod
    def read_file(cls, file_name):
        obs_ids = cls.read_obs_ids(file_name)

        data = pd.read_hdf(file_name, "dl2/event/telescope/parameters/LST_LSTCam")
        config = cls.read_config(file_name)

        samples = tuple(
            MCSample(
                config_table = config.query(f'obs_id == {obs_id}'),
                data_table = data.query(f'obs_id == {obs_id}')
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

        return MCCollection(samples=(self.samples[idx],))

    def get_nearby(self, target_position, search_radius):
        samples = tuple(
            filter(
                lambda sample: sample.tel_pos.separation(target_position) <= search_radius,
                self.samples
            )
        )

        return MCCollection(samples=samples)

    def get_in_box(self, target_position, max_lon_offset, max_lat_offset):
        tel_pos = SkyCoord([sample.tel_pos for sample in self.samples])
        target_position = SkyCoord(target_position.altaz.az, target_position.altaz.alt, frame='altaz')

        lon_offset, lat_offset = tel_pos.altaz.spherical_offsets_to(target_position.altaz)
        inbox = (np.absolute(lon_offset) <= max_lon_offset) & (np.absolute(lat_offset) <= max_lat_offset)

        if sum(inbox):
            samples = tuple(sample for sample, take_it in zip(self.samples, inbox) if take_it)
        else:
            samples = ()

        return MCCollection(samples=samples)
