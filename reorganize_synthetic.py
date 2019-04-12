"""
Usage: python3 sort_synthetic.py /path/to/dataset_root

This script assumes a folder hierarchy like this:

dataset_root (passed as argument to this script)		
	|----> png # folder with images (decompressed images.tar.gz)
	|----> json_gt # folder with json files (decompressed json_gt.tar.gz)
"""

import json
import os
import sys

IMAGES_PER_FOLDER = 1000

def read_gt_file(gt_root, index):
	with open('{}/{}.json'.format(gt_root, index), 'r') as f:
		gt = json.load(f)
	return gt

def get_chart_type(gt):
	return gt['task1']['output']['chart_type']

def reorganize_dataset(png_root, gt_root):
	label_to_index = {}
	indices = [int(f.split('.')[0]) for f in os.listdir(gt_root) if '.json' in f]
	for index in indices:
		gt = read_gt_file(gt_root, index)
		label = get_chart_type(gt)
		label_to_index[label] = label_to_index[label] + [index] if label in label_to_index else [index]
	for label, members in label_to_index.items():
		c = 0
		d = 0
		print(label, ':', len(members))
		label_path = '{}/{}'.format(png_root, '_'.join(label.split(' ')))
		label_path_gt = '{}/{}'.format(gt_root, '_'.join(label.split(' ')))
		os.makedirs(label_path, exist_ok=True)
		os.makedirs(label_path_gt, exist_ok=True)
		for index in members:
			label_subdir = '{}/{}'.format(label_path, d)
			os.makedirs(label_subdir, exist_ok=True)
			label_subdir_gt = '{}/{}'.format(label_path_gt, d)
			os.makedirs(label_subdir_gt, exist_ok=True)
			orig_path = '{}/{}.png'.format(png_root, index)
			new_path = '{}/{}.png'.format(label_subdir, index)
			orig_path_gt = '{}/{}.json'.format(gt_root, index)
			new_path_gt = '{}/{}.json'.format(label_subdir_gt, index)
			os.rename(orig_path, new_path)
			os.rename(orig_path_gt, new_path_gt)
			c += 1
			if c % IMAGES_PER_FOLDER == 0:
				c = 0
				d += 1

if __name__ == '__main__':	
	dataset_root = sys.argv[1]
	png_root = os.path.join(dataset_root, 'png')
	gt_root = os.path.join(dataset_root, 'json_gt')	
	reorganize_dataset(png_root, gt_root)

	











