import os
import sys
import cv2
import json
import numpy as np

LOW_THRESHOLD = 0.005
HIGH_THRESHOLD = 0.02

IMG_FORMAT = 'png'

def extract_tick_point_pairs(js):
    def get_coords(tpp):
        ID = tpp['id']
        x, y = tpp['tick_pt']['x'], tpp['tick_pt']['y']
        return (ID, (x, y))
    axes = js['task4']['output']['axes']
    tpp_x = [get_coords(tpp) for tpp in axes['x-axis']]
    tpp_x = {ID: coords for ID, coords in tpp_x}
    tpp_y = [get_coords(tpp) for tpp in axes['y-axis']]
    tpp_y = {ID: coords for ID, coords in tpp_y}
    return tpp_x, tpp_y


def get_distance(pt1, pt2):
    x1, y1 = pt1
    x2, y2 = pt2
    return np.linalg.norm([x1 - x2, y1 - y2])


def get_distance_score(distance, low, high):
    if distance <= low:
        return 1.
    if distance >= high:
        return 0.
    return 1. - ((high - distance) / (high - low))

def get_axis_score(gt, res, lt, ht):
    score = 0.
    for ID, gt_coords in gt.items():
        if ID not in res:
            continue
        distance = get_distance(gt_coords, res[ID])
        score += get_distance_score(distance, lt, ht)
    return score

def eval_task4(gt_folder, result_folder, img_folder):
    total_recall = 0.
    total_precision = 0.
    gt_files = os.listdir(gt_folder)
    for gt_file in gt_files:
        gt_id = ''.join(gt_file.split('.')[:-1])
        if not os.path.isfile(os.path.join(result_folder, gt_id + '.json')):
            continue
        with open(os.path.join(gt_folder, gt_file), 'r') as f:
            gt = json.load(f)
        gt_x, gt_y = extract_tick_point_pairs(gt)
        with open(os.path.join(result_folder, gt_id + '.json'), 'r') as f:
            res = json.load(f)
        res_x, res_y = extract_tick_point_pairs(res)
        h, w, _ = cv2.imread('{}/{}.{}'.format(img_folder, gt_id, IMG_FORMAT)).shape
        lt, ht = LOW_THRESHOLD * min(w, h), HIGH_THRESHOLD * min(w, h)
        score_x = get_axis_score(gt_x, res_x, lt, ht)
        score_y = get_axis_score(gt_y, res_y, lt, ht)
        total_recall += score_x / len(gt_x)
        total_recall += score_y / len(gt_y)
        total_precision += score_x / len(res_x)
        total_precision += score_y / len(res_y)
    total_recall /= len(gt_files)
    total_precision /= len(gt_files)
    if total_recall == 0 and total_precision == 0:
        f_measure = 0
    else:
        f_measure = 2 * total_recall * total_precision / (total_recall + total_precision)
    print('Average Recall:', total_recall)
    print('Average Precision:', total_precision)
    print('Average F-Measure:', f_measure)

if __name__ == '__main__':
    try:
        eval_task4(sys.argv[1], sys.argv[2], sys.argv[3])
    except Exception as e:
        print(e)
        print('Usage Guide: python eval_task4.py <ground_truth_folder> <result_folder> <img_folder>')










