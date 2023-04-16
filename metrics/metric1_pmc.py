import os
import json
import sys
import numpy as np
import matplotlib.pyplot as plt

def get_confusion_matrix(confusion, unique_labels):
    label_idx_map = {label : i for i, label in enumerate(unique_labels)}
    idx_label_map = {i : label for label, i in label_idx_map.items()}
    cmat = np.zeros((len(label_idx_map), len(label_idx_map)))
    for ID, pair in confusion.items():
        truth, pred = pair
        if pred is None or pred not in label_idx_map:
            continue
        t = label_idx_map[truth]
        p = label_idx_map[pred]
        cmat[t, p] += 1
    norm = cmat.sum(axis=1).reshape(-1, 1)
    cmat /= norm
    return cmat, idx_label_map

def plot_confusion_matrix(cm, classes, output_img_path):
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title='Confusion Matrix',
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    # fig.tight_layout()
    fig.savefig(output_img_path, bbox_inches='tight')
    plt.show()


def eval_task1(gt_folder, result_folder, output_img_path, class_auto_mapping):
    gt_label_map = {}
    result_label_map = {}
    metrics = {}
    confusion = {}
    result_files = os.listdir(result_folder)
    gt_files = os.listdir(gt_folder)
    for gt_file in gt_files:
        gt_id = '.'.join(gt_file.split('.')[:-1])
        with open(os.path.join(gt_folder, gt_file), 'r') as f:
            gt = json.load(f)
            truth = gt['task1']['output']['chart_type'].lower().strip()
        gt_label_map[truth] = gt_label_map[truth] + [gt_id] if truth in gt_label_map else [gt_id]
        confusion[gt_id] = [truth, None]
    for result_file in result_files:
        result_id = '.'.join(result_file.split('.')[:-1])
        with open(os.path.join(result_folder, result_file), 'r') as f:
            result = json.load(f)
        try:
            if 'task1.1' in result:
                result['task1'] = result['task1.1']                                          
            
            if not 'task1' in result:
                print("results file {0:s} does not follow a supported format".format(result_file))
                continue

            if 'output' in result['task1']:
                if 'chart_type' in result['task1']['output']:
                    # IDEAL/expected format ....
                    pred = result['task1']['output']['chart_type']
                else:
                    # try raw value in output
                    pred = result['task1']['output']
            else:
                # less ideal format, try directly value at task1
                pred = result['task1']

            if isinstance(pred, list):
                pred = pred[0]

            pred = pred.lower()

            if 'stacked' in pred or 'grouped' in pred:
                pred = ' '.join(pred.split(' ')[1:])
            pred = pred.strip()

            # check for synonym classes ...
            if pred in class_auto_mapping:
                pred = class_auto_mapping[pred]
        except Exception as e:
            print(e)
            print('invalid result json format in {} please check against provided samples'.format(result_file))
            continue
        result_label_map[pred] = result_label_map[pred] + [result_id] if pred in result_label_map else [result_id]
        confusion[result_id][1] = pred
        # print(confusion[result_id])

    # compute the overall statistics
    total_recall = 0.0
    total_precision = 0.0
    total_fmeasure = 0.0

    all_recall = []
    all_precision = []
    all_fmeasure = []

    total_images = 0
    total_correct = 0

    print("GT labels:")
    print(gt_label_map.keys())
    print("Pred Labels:")
    print(result_label_map.keys())

    for label, gt_imgs in gt_label_map.items():
        res_imgs = set(result_label_map[label])        
        gt_imgs = set(gt_imgs)
        intersection = gt_imgs.intersection(res_imgs)
        recall = len(intersection) / float(len(gt_imgs))
        precision = len(intersection) / float(len(res_imgs))
        if recall == 0 and precision == 0:
            f_measure = 0.0
        else:
            f_measure = 2.0 * recall * precision / (recall + precision)

        all_recall.append(recall)
        all_precision.append(precision)
        all_fmeasure.append(f_measure)

        total_recall += recall
        total_precision += precision
        total_fmeasure += f_measure

        metrics[label] = (recall, precision, f_measure)

        if 'bar' in label:
            print('Grouped/Stacked will be ignored in PMC eval, only Horizontal/Vertical is considered')

        print('Recall for class {}: {}'.format(label, recall))
        print('Precision for class {}: {}'.format(label, precision))
        print('F-measure for class {}: {}'.format(label, f_measure))

        total_images += len(gt_imgs)
        total_correct += len(intersection)

    total_recall /= len(gt_label_map)
    total_precision /= len(gt_label_map)
    total_fmeasure /= len(gt_label_map)

    all_recall = np.array(all_recall)
    all_precision = np.array(all_precision)
    all_fmeasure = np.array(all_fmeasure)

    print('Average Recall across {} classes: {}'.format(len(gt_label_map), total_recall))
    print('Average Precision across {} classes: {}'.format(len(gt_label_map), total_precision))
    print('Average F-Measure across {} classes: {}'.format(len(gt_label_map), total_fmeasure))

    print('Average Recall across {} classes: {} ({})'.format(len(gt_label_map), all_recall.mean(), all_recall.std()))
    print('Average Precision across {} classes: {} ({})'.format(len(gt_label_map), all_precision.mean(), all_precision.std()))
    print('Average F-Measure across {} classes: {} ({})'.format(len(gt_label_map), all_fmeasure.mean(), all_fmeasure.std()))

    print("Overall Accuracy: {} / {} = {}".format(total_correct, total_images, total_correct / total_images))

    print('Computing Confusion Matrix')
    classes = sorted(list(gt_label_map.keys()))
    cm, idx_label_map = get_confusion_matrix(confusion, classes)
    plot_confusion_matrix(cm, classes, output_img_path)

def main():
    if len(sys.argv) < 4:
        print("Usage:")
        print("\tpython metric1_pmc.py <ground_truth_folder> <result_folder> <confusion_matrix_path> [mapping]")
        return

    gt_folder = sys.argv[1]
    result_folder = sys.argv[2]
    confusion_matrix_path = sys.argv[3]

    if len(sys.argv) >= 5:
        auto_mappings_filename = sys.argv[4]
        with open(auto_mappings_filename, "r") as mappings_file:
            class_auto_mapping = json.load(mappings_file)
    else:
        class_auto_mapping = {}

    eval_task1(gt_folder, result_folder, confusion_matrix_path, class_auto_mapping)
    """
    try:
        eval_task1(gt_folder, result_folder, confusion_matrix_path)
    except Exception as e:
        print("Error evaluating results: ")
        print(e)
    """

if __name__ == '__main__':
    main()
