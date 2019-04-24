import json
from os import listdir,path
from params import LOGGING_LEVEL

def load_trace(_folder):
    all_cooked_time, all_cooked_bw = list(), list()

    for _file in listdir(_folder):
        with open(path.join(_folder, _file), 'r') as f:
            cooked_time, cooked_bw = list(), list()
            for line in f:
                parse = line.split()
                cooked_time.append(float(parse[0]))
                cooked_bw.append(float(parse[1]))
                pass
            all_cooked_time.append(cooked_time)
            all_cooked_bw.append(cooked_bw)
        pass
    
    return all_cooked_time, all_cooked_bw

def printh(_str):
    if LOGGING_LEVEL:
        _str = str(_str)
        _len = 50
        _tmp = '=' * int((_len-len(_str))/2)
        print('%s %s %s'%(_tmp,_str,_tmp))
    pass

class logger:
    def __init__(self, _file):
        self.counter = 0
        self.file    = _file
        self.format  = '[%6d] %f \t %s\n' #(idx, timestamp, message)
        pass
    
    def write(self, log, time=-1.0):
        with open(self.file, 'wb') as f:
            f.write(self.format%(self.counter, time, log))
            pass
        pass

    def loadAll(self):
        ret = list()
        with open(self.file, 'rb') as f:
            lines = f.readlines()
            for line in lines:
                tmp = line.split('[')
                _id = int(tmp[0].split(']')[0])
                tmp = tmp[1].split()
                _timestamp = float(tmp[0])
                _message = str(tmp[1])
                ret.append((_id, _timestamp, _message))
            pass
        return ret

    pass