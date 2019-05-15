import os
import json
import sys


def eval_task1(result_folder, gt_folder):
    gt_label_map = {}
    result_label_map = {}
    metrics = {}
    result_files = os.listdir(result_folder)
    gt_files = os.listdir(gt_folder)
    for result_file in result_files:
        result_id = ''.join(result_file.split('.')[:-1])
        with open(os.path.join(result_folder, result_file), 'r') as f:
            result = json.load(f)
        try:
            pred = result['task1']['output']['chart_type'].lower().strip()
            if 'stacked' in pred or 'grouped' in pred:
                pred = ' '.join(pred.split(' ')[1:])
        except Exception as e:
            print(e)
            print('invalid result json format in {} please check against provided samples'.format(result_file))
            continue
        result_label_map[pred] = result_label_map[pred] + [result_id] if pred in result_label_map else [result_id]
    for gt_file in gt_files:
        gt_id = ''.join(gt_file.split('.')[:-1])
        with open(os.path.join(gt_folder, gt_file), 'r') as f:
            gt = json.load(f)
            truth = gt['task1']['output']['chart_type'].lower().strip()
        gt_label_map[truth] = gt_label_map[truth] + [gt_id] if truth in gt_label_map else [gt_id]
    total_recall = 0.
    total_precision = 0.
    total_fmeasure = 0.
    for label, gt_imgs in gt_label_map.items():
        res_imgs = set(result_label_map[label])
        gt_imgs = set(gt_imgs)
        intersection = gt_imgs.intersection(res_imgs)
        recall = len(intersection) / float(len(gt_imgs))
        precision = len(intersection) / float(len(res_imgs))
        f_measure = 2 * recall * precision / (recall + precision)
        total_recall += recall
        total_precision += precision
        total_fmeasure += f_measure
        metrics[label] = (recall, precision, f_measure)
        if 'bar' in label:
            print('Grouped/Stacked will be ignored in PMC eval, only Horizontal/Vertical is considered')
        print('Recall for class {}: {}'.format(label, recall))
        print('Recall for class {}: {}'.format(label, precision))
        print('Recall for class {}: {}'.format(label, f_measure))
    total_recall /= len(gt_label_map)
    total_precision /= len(gt_label_map)
    total_fmeasure /= len(gt_label_map)
    print('Average Recall across {} classes: {}'.format(len(gt_label_map), total_recall))
    print('Average Precision across {} classes: {}'.format(len(gt_label_map), total_precision))
    print('Average F-Measure across {} classes: {}'.format(len(gt_label_map), total_fmeasure))


if __name__ == '__main__':
    try:
        eval_task1(sys.argv[2], sys.argv[3])
    except Exception as e:
        print(e)
        print('Usage Guide: python eval_task1.py <result_folder> <ground_truth_folder>')

