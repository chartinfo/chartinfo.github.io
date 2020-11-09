
import os
import sys
import json
import math
import itertools
import editdistance
import numpy as np
import scipy.optimize
import scipy.spatial.distance

def pprint(obj):
    print(json.dumps(obj, indent=4, sort_keys=True))


def preprocess(outputs, gt_type):
    # normalize box json format
    if 'box' in gt_type:
        if len(outputs) >= 1 and isinstance(outputs[0]['data'], list):
            if debug:
                print("Converting Box format")
            outputs = convert_box_dataseries(outputs)
            #pprint(outputs)

    # remove tuples whose y-values are not numbers
    new_outputs = []
    for ds in outputs:
        if isinstance(ds['data'], list):
            new_ds = {'data': [], 'name': ds['name']}
            for datapoint in ds['data']:
                if 'y2' in datapoint:
                    datapoint['y'] = datapoint['y2']
                    del datapoint['y2']
                if 'y' in datapoint:
                    if not is_number(datapoint['y']):
                        if debug:
                            print("Omitting %r from data series %s" % (datapoint, new_ds['name']))
                        continue
                new_ds['data'].append(datapoint)
            new_outputs.append(new_ds)
        else:
            new_outputs.append(ds)
    return new_outputs

def convert_box_dataseries(outputs):
    new_outputs = []
    for idx in range(len(outputs)):
        global_name = outputs[idx]['name']
        for ds in outputs[idx]['data']:
            if 'x' in ds:
                name = ds['x']
                del ds['x']
            elif 'y' in ds:
                name = ds['y']
                del ds['y']
            else:
                name = global_name
            new_ds = {'name': name, 'data': ds}
            new_outputs.append(new_ds)
    return new_outputs


def is_number(s):
    try:
        float(s)
    except ValueError:
        return False
    return True


def check_discrete(xy_list):
    for point in xy_list:
        x = point['x']
        if not is_number(x):
            return True
    return False


def euclid(p1, p2):
    x1 = float(p1['x'])
    y1 = float(p1['y'])
    x2 = float(p2['x'])
    y2 = float(p2['y'])
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def box_to_discrete(ds):
    out = []
    #print(ds)
    for x in ['first_quartile', 'max', 'min', 'median', 'third_quartile']: 
        out.append( {'x': x, 'y': ds[x]} )
    return out


def compare_box(pred_ds, gt_ds, alpha, gamma):
    pred_ds = box_to_discrete(pred_ds)
    gt_ds = box_to_discrete(gt_ds)
    return compare_discrete(pred_ds, gt_ds, alpha, gamma)


def compare_discrete(pred_ds, gt_ds, alpha, gamma):
    #print("compare_discrete")
    pred_names = list(map(lambda ds: str(ds['x']), pred_ds))
    gt_names = list(map(lambda ds: ds['x'], gt_ds))
    name_compare = lambda s1, s2: 1 - norm_edit_dist(s1, s2) ** alpha
    name_match_scores = create_dist_mat(pred_names, gt_names, name_compare)

    pred_vals = arr_to_np_1d(pred_ds)
    gt_vals = arr_to_np_1d(gt_ds)
    gt_mean = gt_vals.mean(axis=0)
    VI = min(400 / gt_mean ** 2, 1 / np.cov(gt_vals.T))
    value_match_scores = 1 - np.fmin(1, scipy.spatial.distance.cdist(pred_vals, gt_vals, metric='mahalanobis', VI=VI) / gamma)
    
    cost_mat = 1 - (name_match_scores * value_match_scores)
    #if np.isnan(cost_mat).any():
    #    print(pred_ds)
    #    print(gt_ds)
    return get_score(cost_mat)


def arr_to_np(ds):
    n = np.zeros( (len(ds), 2))
    for i,p in enumerate(ds):
        n[i,0] = float(p['x'])
        n[i,1] = float(p['y'])
    return n


def arr_to_np_1d(ds):
    n = np.zeros( (len(ds), 1))
    for i,p in enumerate(ds):
        n[i,0] = float(p['y'])
    return n


def compare_scatter(pred_ds, gt_ds, gamma):
    #print("compare_scatter")
    pred_ds = arr_to_np(pred_ds)
    gt_ds = arr_to_np(gt_ds)
    gt_means = gt_ds.mean(axis=0)

    V = np.cov(gt_ds.T)
    #print(V)
    try:
        VI = np.linalg.inv(V).T
        #print(VI)
        for i in range(VI.shape[0]):
            VI[i,i] = min(VI[i,i], 400 / gt_means[i] ** 2)
    except:
        VI = np.asarray([ [400 / gt_means[0] ** 0, 0], [0, 400 / gt_means[1]] ])

    cost_mat = np.fmin(1, scipy.spatial.distance.cdist(pred_ds, gt_ds, metric='mahalanobis', VI=VI) / gamma)
    return get_score(cost_mat)


def get_score(cost_mat):
    cost_mat = pad_mat(cost_mat)
    k = cost_mat.shape[0]

    row_ind, col_ind = scipy.optimize.linear_sum_assignment(cost_mat)
    cost = cost_mat[row_ind, col_ind].sum()
    score = 1 - (cost / k)
    return score


def get_cont_recall(p_xs, p_ys, g_xs, g_ys, epsilon):
    total_score = 0
    total_interval = 0
    for i in range(g_xs.shape[0]):
        x = g_xs[i]
        if g_xs.shape[0] == 1:
            interval = 1
        elif i == 0:
            interval = (g_xs[i+1] - x) / 2
        elif i == (g_xs.shape[0] - 1):
            interval = (x - g_xs[i-1]) / 2
        else:
            interval = (g_xs[i+1] - g_xs[i-1]) / 2

        y = g_ys[i]
        y_interp = np.interp(x, p_xs, p_ys)
        error = min(1, abs( (y - y_interp) / (abs(y) + epsilon)))
        total_score += (1 - error) * interval
        total_interval += interval
    if g_xs.shape[0] != 1:
        assert np.isclose(total_interval, g_xs[-1] - g_xs[0])
    return total_score / total_interval if total_interval else 0


def compare_continuous(pred_ds, gt_ds):
    #print("compare_cont")
    # filter out any string `x`s.  string y's were already filtered
    # TODO: is this the right thing to do?
    pred_ds = [p for p in pred_ds if is_number(p['x'])]
    gt_ds = [p for p in gt_ds if is_number(p['x'])]

    pred_ds = list(sorted(pred_ds, key=lambda p: float(p['x'])))
    gt_ds = list(sorted(gt_ds, key=lambda p: float(p['x'])))
    if not pred_ds and not gt_ds:
        # empty matches empty
        return 1.
    elif not pred_ds and gt_ds:
        # empty does not match non-empty
        return 0.
    elif pred_ds and not gt_ds:
        # empty does not match non-empty
        return 0.

    p_xs = np.array([float(ds['x']) for ds in pred_ds])
    p_ys = np.array([float(ds['y']) for ds in pred_ds])
    g_xs = np.array([float(ds['x']) for ds in gt_ds])
    g_ys = np.array([float(ds['y']) for ds in gt_ds])


    epsilon = (g_ys.max() - g_ys.min()) / 100.
    recall = get_cont_recall(p_xs, p_ys, g_xs, g_ys, epsilon)
    precision = get_cont_recall(g_xs, g_ys, p_xs, p_ys, epsilon)

    return (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.


def norm_edit_dist(s1, s2):
    if s2.startswith('[unnamed category #'):
        if s1 in [None, ''] or s1.startswith('[unnamed category #'):
            return 0

    try:
        return editdistance.eval(s1, s2) / float(max(len(s1), len(s2), 1))
    except:
        print(repr(s1), repr(s2))
        raise


def create_dist_mat(seq1, seq2, compare):
    l1 = len(seq1)
    l2 = len(seq2)
    mat = np.full( (l1, l2), -1.)
    for i in range(l1):
        for j in range(l2):
            mat[i,j] = compare(seq1[i], seq2[j])
    return mat


def pad_mat(mat):
    h,w = mat.shape
    if h == w:
        return mat
    elif h > w:
        new_mat = np.ones( (h, h) )
        new_mat[:,:w] = mat
        return new_mat
    else:
        new_mat = np.ones( (w, w) )
        new_mat[:h,:] = mat
        return new_mat


def metric_6b(pred_data_series, gt_data_series, gt_type, alpha=1, beta=2, gamma=1, debug=False):
    if 'box' in gt_type.lower():
        compare = lambda ds1, ds2: compare_box(ds1, ds2, alpha, gamma)
    else:
        is_discrete = any(map(lambda ds: check_discrete(ds['data']), gt_data_series))
        if is_discrete:
            compare = lambda ds1, ds2: compare_discrete(ds1, ds2, alpha, gamma)
        elif gt_type.lower() == 'scatter':
            compare = lambda ds1, ds2: compare_scatter(ds1, ds2, gamma)
        elif gt_type.lower() == 'line':
            compare = compare_continuous
        elif 'bar' in gt_type.lower():
            compare = lambda ds1, ds2: compare_discrete(ds1, ds2, alpha, gamma)
        else:
            raise Exception("Odd Case: " + gt_type)

    pred_no_names = list(map(lambda ds: ds['data'], pred_data_series))
    gt_no_names = list(map(lambda ds: ds['data'], gt_data_series))
    ds_match_scores = create_dist_mat(pred_no_names, gt_no_names, compare)
    if debug:
        print("Data Series Match Scores:")
        print(ds_match_scores)

    pred_names = list(map(lambda ds: str(ds['name']), pred_data_series))
    gt_names = list(map(lambda ds: str(ds['name']), gt_data_series))
    name_compare = lambda s1, s2: 1 - norm_edit_dist(s1, s2) ** alpha
    #print("HERE")
    #print(pred_names)
    #print(gt_names)
    name_match_costs = create_dist_mat(pred_names, gt_names, name_compare)
    if debug:
        print("\nName Match Scores:")
        print(name_match_costs)

    mat1 = 1 - (ds_match_scores / beta)
    mat2 = 1 - (name_match_costs * ds_match_scores)
    cost_mat = np.fmin(mat1, mat2)
    if debug:
        print("\nCost Matrix:")
        print(cost_mat)
    return get_score(cost_mat)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("USAGE: python metric6b.py pred_file|pred_dir gt_file|gt_dir [alpha] [beta] [gamma] [debug]")
        exit()
    pred_infile = sys.argv[1]
    gt_infile = sys.argv[2]

    try:
        alpha = float(sys.argv[3])
    except:
        alpha = 1
    try:
        beta = float(sys.argv[4])
    except:
        beta = 2
    try:
        gamma = float(sys.argv[5])
    except:
        gamma = 1
    try:
        debug = sys.argv[6]
    except:
        debug = False

    if os.path.isfile(pred_infile) and os.path.isfile(gt_infile):
        pred_json = json.load(open(pred_infile))
        gt_json = json.load(open(gt_infile))

        pred_outputs = pred_json['task6']['output']['data series']
        gt_outputs = gt_json['task6']['output']['data series']
        gt_type = gt_json['task1']['output']['chart_type']

        if debug:
            print("GT Chart Type:", gt_type)

        pred_outputs = preprocess(pred_outputs, gt_type)
        gt_outputs = preprocess(gt_outputs, gt_type)

        score = metric_6b(pred_outputs, gt_outputs, gt_type, alpha, beta, gamma, debug)
        print(score)
    elif os.path.isdir(pred_infile) and os.path.isdir(gt_infile):
        scores = []
        idx = 0
        for x in os.listdir(pred_infile):
            pred_file = os.path.join(pred_infile, x)
            gt_file = os.path.join(gt_infile, x)
            if debug:
                print()
                print("Idx:", idx)
                print("Pred File:", pred_file)
                print("GT File:", gt_file)

            pred_json = json.load(open(pred_file))
            gt_json = json.load(open(gt_file))

            pred_outputs = pred_json['task6']['output']['data series']
            gt_outputs = gt_json['task6']['output']['data series']
            gt_type = gt_json['task1']['output']['chart_type']
            if debug:
                print("GT Chart Type:", gt_type)

            #pprint(gt_outputs)
            pred_outputs = preprocess(pred_outputs, gt_type)
            gt_outputs = preprocess(gt_outputs, gt_type)
            #pprint(gt_outputs)

            score = metric_6b(pred_outputs, gt_outputs, gt_type, alpha, beta, gamma, debug)
            scores.append(score)
            print("%s: %f" % (x, score))
            idx += 1
        avg_score = sum(scores) / len(scores)
        print("Average Score: %f" % avg_score)
    else:
        print("Error: pred_file and gt_file must both be files or both be directories")
        exit()


