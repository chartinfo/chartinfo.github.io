import os
import sys
import json
import numpy as np
import editdistance

from shapely.geometry import Polygon

from unicodeit import replace

IOU_THRESHOLD = 0.5

# Gets the pair-wise IOUs for GT bounding boxes and predicted Bounding Boxes
def bbox_iou(bboxes1, bboxes2, return_intersections=False):
    x11, y11, x12, y12 = np.split(bboxes1[:, :4], 4, axis=1)
    x21, y21, x22, y22 = np.split(bboxes2[:, :4], 4, axis=1)
    xA = np.maximum(x11, np.transpose(x21))
    yA = np.maximum(y11, np.transpose(y21))
    xB = np.minimum(x12, np.transpose(x22))
    yB = np.minimum(y12, np.transpose(y22))
    interArea = np.maximum((xB - xA + 1), 0) * np.maximum((yB - yA + 1), 0)
    boxAArea = (x12 - x11 + 1) * (y12 - y11 + 1)
    boxBArea = (x22 - x21 + 1) * (y22 - y21 + 1)
    unionArea = (boxAArea + np.transpose(boxBArea) - interArea)
    iou = interArea / unionArea
    if return_intersections:
        sigmas = interArea / boxAArea
        taus = interArea / np.transpose(boxBArea)
        return iou, sigmas, taus
    else:
        return iou

# Gets the pair-wise IOUs for GT quadrilaterals and predicted quadrilaterals
def quadrilateral_IOU(quads_1, quads_2):
    interArea = np.zeros((quads_1.shape[0], quads_2.shape[0]))
    unionArea = np.zeros((quads_1.shape[0], quads_2.shape[0]))

    all_polygons_1 = [Polygon(quad) for quad in quads_1]
    all_polygons_2 = [Polygon(quad) for quad in quads_2]

    for idx_1 in range(len(all_polygons_1)):
        poly_1 = all_polygons_1[idx_1]
        for idx_2 in range(len(all_polygons_2)):
            poly_2 = all_polygons_2[idx_2]

            interArea[idx_1, idx_2] = poly_1.intersection(poly_2).area
            unionArea[idx_1, idx_2] = poly_1.union(poly_2).area

    iou = interArea / unionArea

    return iou


def sanitize_text(text):
    text = text.replace('\n', ' ')
    text = text.replace('\r', ' ')
    text = text.lower().strip()

    # treat aas latex ...
    if '\\' in text or '_{' in text or '^{' in text:
        text = replace([text])[0]
    return text


def extract_bboxes(js):
    if not 'task2' in js:
        raise Exception("No Task 2 output found")

    if 'output' in js['task2']:
        # check
        if 'text_blocks' in js['task2']['output']:
            # expected format
            text_blocks = js['task2']['output']['text_blocks']
        else:
            # try recovery
            text_blocks = js['task2']['output']
    else:
        # try recovery
        text_blocks = js['task2']

    bboxes = []
    ids = []
    texts = []
    for text_block in text_blocks:
        if 'bb' in text_block:
            text_region = text_block['bb']
        elif 'polygon' in text_block:
            text_region = text_block['polygon']
        else:
            raise Exception("No valid text region block found in JSON")

        x1, y1 = text_region['x0'], text_region['y0']

        if 'width' in text_region:
            # old format ... extract bounding box
            w, h = text_region['width'], text_region['height']
            x2, y2 = x1 + w, y1
            x3, y3 = x1 + w, y1 + h
            x4, y4 = x1, y1 + h
        else:
            # assume new format
            x2, y2 = text_region['x1'], text_region['y1']
            x3, y3 = text_region['x2'], text_region['y2']
            x4, y4 = text_region['x3'], text_region['y3']

        if 'id' in text_block:
            ids += [text_block['id']]

        raw_text = text_block['text']
        text = sanitize_text(raw_text)
        if '__' in text:
            continue
        texts += [text]
        bboxes += [((x1, y1), (x2, y2), (x3, y3), (x4, y4))]
    bboxes = np.asarray(bboxes)
    return bboxes, ids, texts


def eval_task2(gt_folder, result_folder):
    total_iou_score = 0.0
    total_text_score = 0.0

    for gt_file in os.listdir(gt_folder):
        gt_id = ''.join(gt_file.split('.')[:-1])

        # print(" " * 120, end="\r")
        print("Processing: " + gt_id)

        # if this image has not been processed at all by a submission, it counts as zero for IOU and OCR scores
        if not os.path.isfile(os.path.join(result_folder, gt_id + '.json')):
            continue

        # read Ground Truth file ...
        with open(os.path.join(gt_folder, gt_file), 'r') as f:
            gt = json.load(f)
        gt_quads, gt_ids, gt_texts = extract_bboxes(gt)

        # read the corresponding result
        try:
            with open(os.path.join(result_folder, gt_id + '.json'), 'r') as f:
                res = json.load(f)
            res_quads, res_ids, res_texts = extract_bboxes(res)
        except Exception as e:
            print("Error processing: " + gt_id)
            print(e)
            continue

        # iou = bbox_iou(gt_bboxes, res_bboxes)
        iou = quadrilateral_IOU(gt_quads, res_quads)

        iou_flag = iou >= IOU_THRESHOLD

        # do the matching ...
        # fp_count = len(res_bboxes)
        # fn_count = len(gt_bboxes)
        iou_score = 0.0
        text_score = 0.0
        for g in range(gt_quads.shape[0]):
            # exact match or one-many match
            if iou_flag[g, :].sum() >= 1:
                # take the best match in case of multiple predictions mapping to one gt
                # rest are considered FP
                r = np.argmax(iou_flag[g, :])
                iou_score += iou[g, r]
                ncer = editdistance.eval(gt_texts[g], res_texts[r]) / float(len(gt_texts[g]))
                text_score += max(1. - ncer, 0.)
                # fp_count -= 1.
                # fn_count -= 1.

        # add score ....
        iou_score /= max(len(gt_quads), len(res_quads))
        text_score /= len(gt_quads)
        total_iou_score += iou_score
        total_text_score += text_score

    total_iou_score /= len(os.listdir(gt_folder))
    total_text_score /= len(os.listdir(gt_folder))
    hmean_score = 2 * total_iou_score * total_text_score / (total_iou_score + total_text_score)

    print('Total IOU Score over all ground truth images: {}'.format(total_iou_score))
    print('Total OCR Score over all ground truth images: {}'.format(total_text_score))
    print('Harmonic Mean of overall IOU and OCR scores: {}'.format(hmean_score))


def main():
    if len(sys.argv) < 3:
        print("Usage:")
        print('\tpython eval_task2.py <ground_truth_folder> <result_folder>')
        return

    gt_folder = sys.argv[1]
    result_folder = sys.argv[2]

    eval_task2(gt_folder, result_folder)
    """
    try:
        eval_task2(gt_folder, result_folder)
    except Exception as e:
        print(e)
    """

if __name__ == '__main__':
    main()
