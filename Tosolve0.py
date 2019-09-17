from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import numpy as np
import math
dirpath = os.path.dirname(__file__)
onset_frame = [11, 34, 74, 93, 128, 159, 192, 226, 261, 303, 343, 366, 439, 640, 642, 763,\
 797, 844, 873, 906, 943, 977, 1016, 1206, 1354, 1356, 1400, 1474, 1510, 1542, \
 1577, 1617, 1768, 1801, 1836, 1877, 1911, 1947, 2052, 2062, 2101, 2138, 2177, 2218, 2239]
match_loc_info = {}

def trans_onset_and_offset(match_loc_info,onset_frame,pitches):
    '''
        after sw alignment to modify onset
    return:
        dict include {onset_frame,offset,pitches,add zero_loc}
    '''
    modify_onset,modify_index = [],[]
    pading_zero_loc = match_loc_info['zero_loc']
    locate_info = match_loc_info['loc_info']
    for i,info in enumerate(locate_info):
        if i not in pading_zero_loc:
            modify_onset.append(onset_frame[info[0]])
            modify_index.append(i)
    modify_onset = sorted(modify_onset)
    modify_index = np.array(modify_index)
    add_onset = []
    for i in pading_zero_loc:
        if i==0:
            modify_onset.append(1)
            modify_index = np.append(modify_index,0)
        else:
            insert_index1 = np.where(modify_index>i)[0]
            insert_index2 = np.where(modify_index<i)[0]
            if len(insert_index1)>0 and len(insert_index2)>0:
                modify_onset.append((modify_onset[insert_index1[0]]+modify_onset[insert_index2[-1]])//2)
                modify_index = np.append(modify_index,i)
            elif len(insert_index1)==0:
                modify_onset.append(modify_onset[-1]+20)
        modify_onset =  sorted(modify_onset)
        modify_index = np.sort(modify_index)

    offset_frame = modify_onset[1:]
    offset_frame = np.append(offset_frame,len(pitches)-1)
    onset_frame = modify_onset

    onset_offset_pitches = {
        'onset_frame':onset_frame,
        'offset_frame':offset_frame,
        'pitches':pitches,
        'add_zero_loc':pading_zero_loc
    }
    return onset_offset_pitches

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

if __name__ == '__main__':
    f0_file = os.path.join(dirpath,"1011_f0.txt")
    pitches = load(f0_file)
    onset_frame = [11, 34, 74, 93, 128, 159, 192, 226, 261, 303, 343, 366, 439, 640, 642, 763, 797, 844, 873, 906, 943,
                   977, 1016, 1206, 1354, 1356, 1400, 1474, 1510, 1542, 1577, 1617, 1768, 1801, 1836, 1877, 1911, 1947,
                   2052, 2062, 2101, 2138, 2177, 2218, 2239]

    match_loc_info['zero_loc'] = [15, 26, 33, 41, 44, 48]

    match_loc_info['loc_info'] = [(0, 0), (1, 1), (3, 2), (4, 3), (5, 4), (6, 5), (7, 6), (8, 7), (9, 8), (10, 9), (11, 10),
                                  (12, 11), (13, 12), (14, 13), (15, 14), (9985, 15), (16, 16), (17, 17), (18, 18), (19, 19),
                                  (20, 20), (21, 21), (22, 22), (23, 23), (24, 24), (25, 25), (9974, 26), (26, 27), (27, 28),
                                  (28, 29), (29, 30), (30, 31), (31, 32), (9967, 33), (32, 34), (33, 35), (34, 36), (35, 37),
                                  (36, 38), (37, 39), (38, 40), (9959, 41), (39, 42), (40, 43), (9956, 44), (41, 45), (42, 46),
                                  (43, 47), (9952, 48), (44, 49)]

    onset_offset_pitches = trans_onset_and_offset(match_loc_info,
                                                  onset_frame,
                                                  pitches)
    print("onset_offset_pitches:",onset_offset_pitches)
