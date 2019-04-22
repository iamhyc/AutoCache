import json
from os import listdir

def load_trace(_folder):
    timestamps, bandwidths = [], []

    for _file in listdir(_folder):
        with open(_folder+_file, 'r') as f:
            for line in f:
                parse = line.split()
                timestamps.append(float(parse[0]))
                bandwidths.append(float(parse[1]))
                pass
        pass

    return timestamps, bandwidths
