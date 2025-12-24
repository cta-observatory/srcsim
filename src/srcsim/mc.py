import glob
import numpy as np
import pandas as pd
import tables
import astropy.units as u
from astropy.coordinates import SkyCoord


def power_law(e, e0, norm, index):
    return norm * (e/e0).decompose()**index


class MCBase:
    @classmethod
    def _has_key(cls, file_name, key):
        with tables.open_file(file_name) as table:
            has_key = key in table

        return has_key
    
    @classmethod
    def _choose_first_valid_key(cls, file_name, keys):
        for key in keys:
            if cls._has_key(file_name, key):
                return key

        return None
    
    @classmethod
    def get_config_key(cls, file_name):
        keys = (
            '/simulation/config',
            '/simulation/run_config'
        )
        key = cls._choose_first_valid_key(file_name, keys)
        return key
    
    @classmethod
    def get_events_key(cls, file_name):
        keys = (
            '/events/parameters',
            '/dl2/event/telescope/parameters/LST_LSTCam'
        )
        key = cls._choose_first_valid_key(file_name, keys)
        return key

    @classmethod
    def read_config(cls, file_name):
        with tables.open_file(file_name) as table:
            cfg_table = table.root[cls.get_config_key(file_name)]

            columns = {
                'n_showers': ('num_showers',),
                'shower_reuse': (),
                'min_scatter_range': (),
                'max_scatter_range': (),
                'energy_range_min': (),
                'energy_range_max': (),
                'spectral_index': (),
                'min_viewcone_radius': (),
                'max_viewcone_radius': ()
            }

            data = {}

            for col_name in columns:
                if col_name in cfg_table.colnames:
                    data[col_name] = [
                        v[col_name] for v in cfg_table.iterrows()
                    ]
                else:
                    for alternative in columns[col_name]:
                        if alternative in cfg_table.colnames:
                            data[col_name] = [
                                v[alternative] for v in cfg_table.iterrows()
                            ]
                            break
                if col_name not in data:
                    raise RuntimeError(f"could not load config key {col_name} from '{file_name}'")

            if 'obs_id' in cfg_table.colnames:
                data['obs_id'] = [
                    v['obs_id'] for v in cfg_table.iterrows()
                ]
            else:
                evt_table = table.root[cls.get_events_key(file_name)]
                row = next(evt_table.iterrows())
                data['obs_id'] = [
                    row['obs_id'] for _ in cfg_table.iterrows()
                ]
                print(
                    "WARN: can not find 'obs_id' in the configuration table, "
                    f"assuming {row['obs_id']} from the first event. "
                    "Simulation results may be incorrect."
                )

        return pd.DataFrame(data=data)

    @classmethod
    def read_data(cls, file_name):
        data = pd.read_hdf(file_name, cls.get_events_key(file_name))

        return data


class MCSample(MCBase):
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
            self.data_table = self.read_data(file_name, obs_id)

        self.data_table = self._fix_format(self.data_table)
        self.config_table = self._fix_format(self.config_table)

        # Getting the telescope pointing
        pointing_data = self.data_table[['pointing_az', 'pointing_alt']].mean()
        self.tel_pos = SkyCoord(pointing_data['pointing_az'], pointing_data['pointing_alt'], unit=self.units['angle'], frame='altaz')
        
        # Working out the simulation spectrum
        rmin, rmax = self.config_table[['min_scatter_range', 'max_scatter_range']].iloc[0].values * self.units['distance']
        ground_area = np.pi * (rmax**2 - rmin**2)
        nevents = self.config_table['n_showers'].iloc[0] * self.config_table['shower_reuse'].iloc[0]
        emin = self.config_table['energy_range_min'].iloc[0] * self.units['energy']
        emax = self.config_table['energy_range_max'].iloc[0] * self.units['energy']
        index = self.config_table['spectral_index'].iloc[0]
        self.spec_data = self.get_spec_data(nevents, emin, emax, index)
        self.spec_data['norm'] /= ground_area

        cam_x, cam_y = self.data_table[['src_x', 'src_y']].to_numpy().transpose() * self.units['distance'] * self.cam2angle
        self.evt_coord = SkyCoord(cam_x, cam_y, frame=self.tel_pos.skyoffset_frame())

        self.evt_energy = self.data_table['mc_energy'].to_numpy() * self.units['energy']
        
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
        config = super().read_config(file_name)
        return config.query(f'obs_id == {obs_id}')
        
    @classmethod
    def read_data(cls, file_name, obs_id):
        data = super().read_data(file_name)
        return data.query(f'obs_id == {obs_id}')

    def _fix_format(self, table):
        table = table.copy()
        old2new = dict(
            mc_energy = 'true_energy',
            mc_az = 'true_az',
            mc_alt = 'true_alt',
            mc_az_tel = 'pointing_az',
            mc_alt_tel = 'pointing_alt',
            num_showers = 'n_showers'
        )
        for key in old2new:
            if key in table.columns:
                table[old2new[key]] = table[key]

        return table

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
        offset_min, offset_max = self.config_table[['min_viewcone_radius', 'max_viewcone_radius']].iloc[0].values * self.units['viewcone']
        sky_area = 2 * np.pi * (np.cos(offset_min) - np.cos(offset_max)) * u.sr
        norm = 1 / sky_area

        r = self.tel_pos.separation(coord)
        
        return norm * (r >= offset_min) * (r <= offset_max)

    def dndedo(self, energy, coord):
        return self.dnde(energy) * self.dndo(coord)


class MCCollection(MCBase):
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
        obs_ids = cls.read_config(file_name)['obs_id'].values

        return obs_ids

    @classmethod
    def read_file(cls, file_name):
        obs_ids = cls.read_obs_ids(file_name)

        data = cls.read_data(file_name)
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
