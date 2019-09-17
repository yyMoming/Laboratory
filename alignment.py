# coding=utf-8
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from collections import Counter

import numpy as np

MATCH_COST = 1
INSERT_COST = 2
DELETE_COST = 3

def filter_pitch(cur_pitches, bool_zero_loc=False):
    '''
            smooth pitch and add pause
    param:
            cur_pitches  音调值
            bool_zero_loc 是否需要添加休止符
    return:
            pitches
    '''
    max_note, min_note = 70, 25
    cur_pitches = np.array(cur_pitches)
    cur_pitches[np.where(cur_pitches > max_note)[0]] = 0.0
    cur_pitches[np.where(cur_pitches < min_note)[0]] = 0.0
    dpitches = np.copy(cur_pitches)
    for i in range(len(dpitches) - 2):
        indices = np.argsort(cur_pitches[i:i + 3])
        diff1, diff2 = abs(cur_pitches[i + indices[0]] - cur_pitches[i + indices[1]]),\
            abs(cur_pitches[i + indices[1]] - cur_pitches[i + indices[2]])
        if diff1 > 2 and diff2 <= 2:
            dpitches[i + indices[0]] = dpitches[i + indices[2]]
        elif diff1 <= 2 and diff2 > 2:
            dpitches[i + indices[2]] = dpitches[i + indices[0]]
    zero_indices = np.where(dpitches == 0)[0]
    if len(zero_indices) <= 15 and len(zero_indices) > 0:
        dpitches[zero_indices] = dpitches[0]
    elif len(zero_indices) > 15:
        dpitches[zero_indices[0]:] = 0.0
    if bool_zero_loc and len(zero_indices) <= 15:
        dpitches = np.append(dpitches, np.zeros(15))
    return dpitches.tolist()

def offset_loc(pitches):
    '''
            find offset loc
    '''
    pitches_ = np.array(pitches).copy()
    pitches_ = pitches_.astype(int)
    number, start_loc = 1, 0        ####修改  Number=1
    for i, _det in enumerate(pitches_, start=1):
        if i == len(pitches_):
            break
        pitch_range = sorted(pitches_[i:i + 4])
        smooth_pitch = True
        if len(pitch_range) == 4:
            max_pitch = pitch_range[-2:]
            if abs(max_pitch[0] - max_pitch[1]) <= 2 and max_pitch[0] > 25:
                smooth_pitch = False
        diff = abs(pitches_[i] - pitches_[i - 1])
        if (_det == 0 or _det < 20 or (diff > 2 and diff != 12 and diff != 11 and diff != 13)) and smooth_pitch:
            if number >= 8:
                break
            number = 1
            start_loc = 0
        elif diff <= 2 or (diff >= 11 and diff <= 13) or (not smooth_pitch):
            if number == 0:
                start_loc = i - 1
            number += 1
    flag = number + start_loc
    return flag

def smooth_pitches(cur_pitches):
    '''
            also smooth pitches
    '''
    pitches_ = cur_pitches.astype(int)
    indices = np.where(pitches_ > 25)[0]
    std_pitches = pitches_[indices]
    counts = np.bincount(std_pitches)
    if len(counts) > 0:
        mode_pitch = np.argmax(counts) #众数最多的音高，视为标准
        for i, pitch in enumerate(pitches_):
            cur_pitches[i] = mode_pitch if abs(
                pitch - mode_pitch) > 8 and pitch > 20 else cur_pitches[i]
    flag = offset_loc(cur_pitches)
    pitch = cur_pitches[0:flag]
    pitch = pitch.astype(int)
    unique_pitch = np.unique(pitch)
    if len(pitch) > 0:
        maxnum_pitch = Counter(pitch).most_common(1)[0][0]  #众数最多音高
        max_indices = np.where(pitch == maxnum_pitch)[0]
        pitches = cur_pitches.copy()
        for _p in unique_pitch:
            if abs(_p - maxnum_pitch) > 1:
                indices = np.where(pitch == _p)[0]
                for idx in indices:
                    rand_id = np.random.permutation(len(max_indices))[0]
                    cur_pitches[idx] = cur_pitches[max_indices[rand_id]]
    return cur_pitches

def process_pitch(pitches, onset_frame, score_note):
    result_info = []
    offset_frame = onset_frame[1:]
    offset_frame = np.append(offset_frame, len(pitches) - 1)
    for idx, cur_onset_frame in enumerate(onset_frame):
        pitch_info = {}
        cur_offset_frame = offset_frame[idx]
        pitch = pitches[cur_onset_frame:cur_offset_frame]
        pitch = smooth_pitches(pitch)
        voiced_length = offset_loc(pitch)
        pitch_info['onset'] = cur_onset_frame
        pitch_info['flag'] = voiced_length
        pitch_info['pitches'] = filter_pitch(pitch)
        result_info.append(pitch_info)
    return result_info

def pitch_Note(pitches, onset_frame, score_note):
    '''
            将连续的pitch转化为note
    '''
    result_info = process_pitch(pitches, onset_frame, score_note)
    det_pitches = []
    for _info in result_info:
        loc_flag = _info['flag']
        pitches = np.round(np.array(_info['pitches'][:loc_flag])).astype(int)
        pitches = pitches[np.where(pitches > 20)[0]]
        unique_pitch = np.unique(pitches)
        number_dict = {}
        for _det in unique_pitch:
            count = pitches.tolist().count(_det)
            number_dict[_det] = count
        number_values = np.array(number_dict.values())
        if len(number_values) > 0:
            max_index = np.argmax(number_values)
            det_pitches.append(number_dict.keys()[max_index])
    return det_pitches

def sw_alignment(pitches, onset_frame, score_note):
    '''
            sw alignment algorithm
    '''

    det_note = pitch_Note(pitches, onset_frame, score_note)
    score_note = np.array(score_note)
    if (len(score_note) - len(det_note)) > 0.15 * len(score_note):
        return {
            'loc_info': [],
            'zero_loc': []
        }
    det_note = np.array(det_note)
    score_diff = score_note[1:] - score_note[0:-1]
    _lt = np.where(score_diff < 0)[0]
    _eq = np.where(score_diff == 0)[0]
    _gt = np.where(score_diff > 0)[0]
    score_diff[_lt] = 1
    score_diff[_eq] = 2
    score_diff[_gt] = 3

    det_diff = det_note[1:] - det_note[0:-1]
    _lt = np.where(det_diff < 0)[0]
    _eq = np.where(det_diff == 0)[0]
    _gt = np.where(det_diff > 0)[0]
    det_diff[_lt] = 1
    det_diff[_eq] = 2
    det_diff[_gt] = 3

    query_str, ref_str = '', ''
    for x in det_diff:
        query_str += str(x)
    for x in score_diff:
        ref_str += str(x)
    query_str += str(2)
    ref_str += str(2)
    sw_ref_str, sw_query_str = WaterMan(ref_str, query_str)
    match_loc_info = locate(ref_str, query_str, sw_ref_str, sw_query_str)
    return match_loc_info


def WaterMan(s1, s2):
    x = len(s1)
    y = len(s2)
    opt = np.zeros((x + 1, y + 1))
    for i in range(x):
        opt[i][y] = DELETE_COST * (x - i)
    for j in range(y):
        opt[x][j] = DELETE_COST * (y - j)
    opt[x][y] = 0
    minxy = min(x, y)
    for k in range(1, minxy + 1):
        for i in range(x - 1, -1, -1):
            opt[i][y - k] = getMin(opt, i, y - k, s1, s2)
        for j in range(y - 1, -1, -1):
            opt[x - k][j] = getMin(opt, x - k, j, s1, s2)
    for k in range(x - minxy, -1, -1):
        opt[k][0] = getMin(opt, k, 0, s1, s2)
    for k in range(y - minxy, -1, -1):
        opt[0][k] = getMin(opt, 0, k, s1, s2)
    i, j, a1, a2 = 0, 0, "", ""
    while (i < x and j < y):
        t = MATCH_COST + \
            opt[i + 1][j + 1] if s1[i] == s2[j] else INSERT_COST + opt[i + 1][j + 1]
        if opt[i][j] == t:
            a1 += s1[i]
            a2 += s2[j]
            i += 1
            j += 1
        elif opt[i][j] == (opt[i + 1][j] + DELETE_COST):
            a1 += s1[i]
            a2 += '-'
            i += 1
        elif opt[i][j] == (opt[i][j + 1] + DELETE_COST):
            a1 += '-'
            a2 += s2[j]
            j += 1
    lenDiff = len(a1) - len(a2)
    for k in range(-lenDiff):
        a1 += '-'
    for k in range(lenDiff):
        a2 += '-'
    return a1, a2


def getMin(opt, x, y, s1, s2):
    x1 = opt[x][y + 1] + 2
    x2 = opt[x + 1][y + 1] + \
         MATCH_COST if s1[x] == s2[y] else INSERT_COST + opt[x + 1][y + 1]
    x3 = opt[x + 1][y] + DELETE_COST
    return min(x1, min(x2, x3))


def locate(ref_str, query_str, sw_ref_str, sw_query_str):
    locate_info = {}
    pading_zero_loc = []
    delte_loc = []
    ref_str = [str(x) for x in ref_str]
    query_str = [str(x) for x in query_str]
    for i in range(len(sw_ref_str)):
        if sw_ref_str[i] != '-' and sw_query_str[i] != '-':
            loc_ref = ref_str.index(sw_ref_str[i])
            loc_query = query_str.index(sw_query_str[i])
            ref_str[loc_ref] = -1
            query_str[loc_query] = -1
            locate_info[loc_query] = loc_ref
        elif sw_ref_str[i] != '-' and sw_query_str[i] == '-':
            loc_ref = ref_str.index(sw_ref_str[i])
            ref_str[loc_ref] = -1
            pading_zero_loc.append(loc_ref)
        elif sw_ref_str[i] == '-' and sw_query_str[i] != '-':
            loc_query = query_str.index(sw_query_str[i])
            query_str[loc_query] = -1
            delte_loc.append(loc_query)

    values = locate_info.values()
    for i in range(len(ref_str)):
        if (i not in values) and (i not in pading_zero_loc):
            pading_zero_loc.append(i)
    for zero_loc in pading_zero_loc:
        locate_info[10000 - zero_loc] = zero_loc

    locate_info = sorted(locate_info.items(), lambda x, y: cmp(x[1], y[1]))
    match_loc_info = {
        'loc_info': locate_info,
        'zero_loc': pading_zero_loc
    }

    return match_loc_info
