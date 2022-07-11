from imports import *

def adjust_time(ds):
    _tt = []
    t2 = datetime.datetime.strptime(str(ds.base_time.values)[:-3], '%Y-%m-%dT%H:%M:%S.%f')
    for i in range(len(ds.time_offset)):
        t1 = datetime.datetime.strptime(str(ds.time_offset[i].values)[:-3], '%Y-%m-%dT%H:%M:%S.%f')
        _tt.append(t1 + datetime.timedelta(hours=t2.hour, minutes=t2.minute, seconds=t2.second))

    return(_tt)