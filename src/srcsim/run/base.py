import numpy as np

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

    def predict(self, mccollections, source, tel_pos_tolerance, time_step):
        pass

    @classmethod
    def time_sort(cls, events):
        timestamp = np.array(events["trigger_time"])
        sorting = np.argsort(timestamp)

        return events.iloc[sorting]

    @classmethod
    def update_time_delta(cls, events):
        event_time = events['trigger_time'].to_numpy()
        delta_time = np.zeros_like(event_time)

        if len(event_time) > 0:
            argsorted = event_time.argsort()
            dt = np.diff(event_time[argsorted])
            delta_time[argsorted[:-1]] = dt
            delta_time[argsorted[-1]] = dt[-1]

        events = events.drop(
            columns=['delta_t'],
            errors='ignore'
        )
        events = events.assign(
            delta_t = delta_time,
        )

        return events
    