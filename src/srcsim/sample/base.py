import tables
import pandas as pd
from abc import abstractmethod

from .hdfkeys import get_events_key, get_config_key

class SampleBase:
    @classmethod
    def read_data(cls, file_name):
        data = pd.read_hdf(file_name, get_events_key(file_name))

        return data
    
    @abstractmethod
    def dndedo(self, energy, coord):
        pass


class MCSampleBase(SampleBase):
    @classmethod
    def read_config(cls, file_name):
        with tables.open_file(file_name) as table:
            cfg_table = table.root[get_config_key(file_name)]

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
