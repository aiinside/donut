import os
import json
import numpy as np
import cv2
from nxclient import *

IMDIR = '/data/murayama/k8s/ocr_dxs1/donut/dataset/kie/invoice_nttd/images'
COCO_JSON_PATH='/data/murayama/k8s/ocr_dxs1/donut/dataset/kie/invoice_nttd/annotations/instances_test.json'
OUT_NAME_TR = 'data.json'
# 人工データ用
# TARGET_CAT_ID = [8,9,57,58,59]
# nttd追加データ用
# TARGET_CAT_ID = [8,9,53,54,55]
TARGET_CAT_ID = [7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,53,54,55]
print(len(TARGET_CAT_ID))
client = NxClient()

def box_text(box_data, img):
    target_boxes = [box[0] for box in box_data]
    target_boxes = np.array(target_boxes).astype(np.int32)

    inputs = []
    for bb in target_boxes:
        part = img[bb[1]:bb[3],bb[0]:bb[2]]
        inp = NxInput(api_key='', model_name='katsuji', image=part)
        inputs.append(inp)

    results = client.post_to_nx(inputs)
    texts = [rr.text for rr in results]

    return texts

def key_func(ind):
    box, _ = ind
    x0,y0,_,_ = box
    return int(y0), int(x0)

with open(COCO_JSON_PATH) as fp:
    coco_data = json.load(fp)

cat_dict = {}
for  cat  in coco_data['categories']:
    id = cat['id']
    name = cat['name']
    cat_dict[id] = name

im_dict = {}
for im in coco_data['images']:
    id = im['id']
    fn = im['file_name']
    hh = im['height']
    ww = im['width']
    im_dict[id] = (fn, hh, ww)

box_dict = {kk:[] for kk in im_dict}
for ann in coco_data['annotations']:
    cat = ann['category_id']
    if not (cat in TARGET_CAT_ID):
        continue
    bbox = np.array(ann['bbox'])
    bbox[2] += bbox[0]
    bbox[3] += bbox[1]
    imid = ann['image_id']
    box_dict[imid].append((bbox, cat))

gt_list = []
for kk in im_dict:
    fn, hh, ww = im_dict[kk]
    fn = os.path.basename(fn)
    print(fn)
    img = cv2.imread(os.path.join(IMDIR, fn))
    box_texts = box_text(box_dict[kk], img)

    gt_dict = {'gt_parse':{'invoice':[]}, 'bboxes':[]}
    for box, text in zip(box_dict[kk], box_texts):
        bbox = box[0].tolist()
        cat_id = box[1]
        cat_name = cat_dict[cat_id]
        gt_dict['gt_parse']['invoice'].append({cat_name:text})
        gt_dict['bboxes'].append({cat_name:bbox})

    gt_dict['image'] = os.path.join(IMDIR, fn)

    gt_list.append(gt_dict)

with open(OUT_NAME_TR, 'w') as fp:
    json.dump(gt_list, fp, ensure_ascii=False, indent=4)
