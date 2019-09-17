from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from alignment import sw_alignment

import os
import numpy as np
import math
dirpath = os.path.dirname(__file__)
score_note = [47, 44, 44, 44, 47, 44, 44, 44, 47, 47, 49, 47, 45, 45, 42, 42, 42, 45, 42, 42, 42, 47, 47, 45, 42,
              40, 47, 47, 49, 49, 47, 47, 44, 42, 40, 42, 44, 47, 47, 47, 49, 49, 47, 47, 44, 42, 45, 44, 42, 40]
onset_frame = [11, 34, 74, 93, 128, 159, 192, 226, 261, 303, 343, 366, 439, 640, 642, 763, 797, 844, 873, 906, 943,
               977, 1016, 1206, 1354, 1356, 1400, 1474, 1510, 1542, 1577, 1617, 1768, 1801, 1836, 1877, 1911, 1947,
               2052, 2062, 2101, 2138, 2177, 2218, 2239]

def load(f0_file):
    f0_array = []
    with open(f0_file, 'r+') as f:
        f0_list = f.readlines()
        for f0 in f0_list:
            try:
                f0 = float(f0.strip())
            except BaseException as e:
                print(e)
            f0 = (69 + 12 * math.log(f0 / 440) / math.log(2)) if f0 > 0 else 0
            f0_array.append(f0)
    f.close()
    pitches = np.array(f0_array)
    return pitches

if __name__ == "__main__":
    f0_file = os.path.join(dirpath, "1011_f0.txt")
    pitches = load(f0_file)
    match_loc_info = sw_alignment(pitches,onset_frame,score_note)
    print(match_loc_info)