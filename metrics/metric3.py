import os
import json
import sys


def eval_task3(result_folder, gt_folder):
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
            text_roles = result['task3']['output']['text_roles']
            for text_role in text_roles:
                text_id = text_role['id']
                role = text_role['role'].lower().strip()
                result_label_map[role] = result_label_map[role] + ['{}__sep__{}'.format(result_id, text_id)]\
                    if role in result_label_map else ['{}__sep__{}'.format(result_id, text_id)]
        except Exception as e:
            print(e)
            print('invalid result json format in {} please check against provided samples'.format(result_file))
            continue
    for gt_file in gt_files:
        gt_id = ''.join(gt_file.split('.')[:-1])
        with open(os.path.join(gt_folder, gt_file), 'r') as f:
            gt = json.load(f)
        text_roles = gt['task3']['output']['text_roles']
        for text_role in text_roles:
            text_id = text_role['id']
            role = text_role['role'].lower().strip()
            # VALUE LABEL IN PMC IS NOT PRESENT IN SYNTHETIC, TO BE CONSIDERED AS OTHER FOR EVAL
            if role == 'value_label':
                role = 'other'
            gt_label_map[role] = gt_label_map[role] + ['{}__sep__{}'.format(gt_id, text_id)] \
                if role in gt_label_map else ['{}__sep__{}'.format(gt_id, text_id)]
    total_recall = 0.
    total_precision = 0.
    total_fmeasure = 0.
    for label, gt_instances in gt_label_map.items():
        res_instances = set(result_label_map[label])
        gt_instances = set(gt_instances)
        intersection = gt_instances.intersection(res_instances)
        recall = len(intersection) / float(len(gt_instances))
        precision = len(intersection) / float(len(res_instances))
        f_measure = 2 * recall * precision / (recall + precision + 1e-9)
        total_recall += recall
        total_precision += precision
        total_fmeasure += f_measure
        metrics[label] = (recall, precision, f_measure)
        print('Recall for class {}: {}'.format(label, recall))
        print('Precision for class {}: {}'.format(label, precision))
        print('F-measure for class {}: {}'.format(label, f_measure))
    total_recall /= len(gt_label_map)
    total_precision /= len(gt_label_map)
    total_fmeasure /= len(gt_label_map)
    print('Average Recall across {} classes: {}'.format(len(gt_label_map), total_recall))
    print('Average Precision across {} classes: {}'.format(len(gt_label_map), total_precision))
    print('Average F-Measure across {} classes: {}'.format(len(gt_label_map), total_fmeasure))


if __name__ == '__main__':
    try:
        eval_task3(sys.argv[1], sys.argv[2])
    except Exception as e:
        print(e)
        print('Usage Guide: python metric3.py <result_folder> <ground_truth_folder>')
