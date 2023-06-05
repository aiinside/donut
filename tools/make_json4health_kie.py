import os
import json
import numpy as np

IMDIR = '/data/murayama/k8s/ocr_dxs1/detr/datadir/coco211006/val2017'
COCO_JSON_PATH='/data/murayama/k8s/ocr_dxs1/detr/datadir/coco211006/annotations/instances_val2017.json'
FOCR_DIR = '/data/murayama/k8s/ocr_dxs1/dxsx-fullocr-engine/src/results/health_valid'
OUT_NAME = 'valid_data.json'

# IMDIR = '/data/murayama/k8s/ocr_dxs1/detr/datadir/coco211006/train2017'
# COCO_JSON_PATH='/data/murayama/k8s/ocr_dxs1/detr/datadir/coco211006/annotations/instances_train2017.json'
# FOCR_DIR = '/data/murayama/k8s/ocr_dxs1/dxsx-fullocr-engine/src/results/health_train_back'
# OUT_NAME = 'train_data.json'


def box_text(box_data, ocr_data, hh, ww):
    chars = [res['characters'] for res in ocr_data['results']]
    chars = sum(chars, [])

    char_boxes = []
    for char in chars:
        x0 = char['bbox']['left'] * ww
        y0 = char['bbox']['top'] * hh
        x1 = char['bbox']['right'] * ww
        y1 = char['bbox']['bottom'] * hh
        char_boxes.append(np.array([x0,y0,x1,y1]))
    char_boxes = np.array(char_boxes)
    char_centers = char_boxes.reshape(-1,2,2).mean(axis=1, keepdims=True)
    target_boxes = [box[0] for box in box_data]
    target_boxes = np.array(target_boxes).reshape(1,-1,4)

    x0 = target_boxes[:,:,0] < char_centers[:,:,0]
    x1 = char_centers[:,:,0] < target_boxes[:,:,2]
    y0 = target_boxes[:,:,1] < char_centers[:,:,1]
    y1 = char_centers[:,:,1] < target_boxes[:,:,3]

    keep = np.stack([x0,x1,y0,y1], axis=-1).all(axis=-1)
    keep = np.argwhere(keep)

    box_texts = ['' for _ in range(target_boxes.shape[1])]

    for ii in range(len(keep)):
        ic, ib = keep[ii]
        box_texts[ib] += chars[ic]['char']

    return box_texts

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
    bbox = np.array(ann['bbox'])
    bbox[2] += bbox[0]
    bbox[3] += bbox[1]
    imid = ann['image_id']
    box_dict[imid].append((bbox, cat))

gt_list = []
for kk in im_dict:
    fn, hh, ww = im_dict[kk]
    ocr_path = os.path.splitext(fn)[0] + '.json'
    ocr_path = os.path.join(FOCR_DIR, ocr_path)
    with open(ocr_path) as fp:
        ocr_data = json.load(fp)

    box_data = sorted(box_dict[kk], key=key_func)
    box_texts = box_text(box_data, ocr_data, hh, ww)

    gt_dict = {'gt_parse':{'health':[]}, 'bboxes':[]}
    for box, text in zip(box_data, box_texts):
        bbox = box[0].tolist()
        cat_id = box[1]
        cat_name = cat_dict[cat_id]
        gt_dict['gt_parse']['health'].append({cat_name:text})
        gt_dict['bboxes'].append({cat_name:bbox})

    gt_dict['image'] = os.path.join(IMDIR, fn)

    gt_list.append(gt_dict)

with open(OUT_NAME, 'w') as fp:
    json.dump(gt_list, fp, ensure_ascii=False, indent=4)