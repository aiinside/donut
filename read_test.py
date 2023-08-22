import json
import cv2
from atypical_postproc import modify_ocr_text, replace_tougou
from nxclient import *

JSON_DIR = '/data/murayama/k8s/ocr_dxs1/donut/TEST_RESULTS/read_500epoch'
OUTDIR = '/data/murayama/k8s/ocr_dxs1/donut/OCR_RESULTS/read_500epoch'
IMDIR = '/data/murayama/k8s/ocr_dxs1/donut/dataset/VHR_TESTIMG_HANDCROP'
RECORD_THRESH = 0.8

def proc_one(jp, imp, outdir):
    img = cv2.imread(imp)
    viz = img.copy()
    hh,ww = img.shape[:2]
    with open(jp) as fp:
        data = json.load(fp)

    if isinstance(data, dict):
        data = [data,]

    _data = []
    for dd in data:
        if len(dd) == 0:
            continue
        if not isinstance(dd, dict):
            continue
        vv = list(dd.values())[0]
        if not isinstance(vv, dict):
            continue
        if vv['score'] < RECORD_THRESH:
            continue
        _data.append(dd)

    data = _data

    parts = []
    data = [dd for dd in data if len(dd) != 0]
    data = [dd for dd in data if isinstance(list(dd.values())[0], dict)]
    for ind in data:
        try:
            cname, content = list(ind.items())[0]
        except Exception as ex:
            print(ind)
            raise ex
        box = content['box']
        x0 = int(box[0] - box[2]//2)
        x0 = min(max(0, x0),ww)

        y0 = int(box[1] - box[3]//2)
        y0 = min(max(0, y0),hh)
        
        x1 = int(box[0] + box[2]//2)
        x1 = min(max(0, x1),ww)
        
        y1 = int(box[1] + box[3]//2)
        y1 = min(max(0, y1),hh)

        parts.append(img[y0:y1,x0:x1])

        cv2.rectangle(viz, (int(x0),int(y0)), (int(x1), int(y1)), (0,255,0), 2)

    inputs = []
    for part in parts:
        inp = NxInput(api_key='', model_name='katsuji', image=part)
        inputs.append(inp)

    client = NxClient()
    results = client.post_to_nx(inputs)

    for ii, rr in enumerate(results):
        ind = data[ii]
        for kk in ind:
            ocr = rr.text
            mk = replace_tougou(kk, ocr)
            mod, _ = modify_ocr_text(mk, ocr)
            ind[kk]['tougou'] = mk
            ind[kk]['ocr'] = mod

    fn = os.path.basename(jp)
    out = os.path.join(outdir, fn)
    with open(out, 'w') as fp:
        json.dump(data,fp, ensure_ascii=False, indent=4)

    fn = os.path.splitext(out)[0] + '.jpg'
    cv2.imwrite(fn, viz)

jsons = os.listdir(JSON_DIR)
jsons = [jj for jj in jsons if 'json' in jj]
os.makedirs(OUTDIR, exist_ok=True)

for jj in jsons:
    print(jj)
    fn = os.path.splitext(jj)[0]
    jp = os.path.join(JSON_DIR, jj)
    imp = os.path.join(IMDIR, fn) + '.jpg'
    proc_one(jp, imp, OUTDIR)