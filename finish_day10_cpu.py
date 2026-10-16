"""CPU-only finalizer: verify evidence; measure wrong coord; draw six GT/pred comparisons.
Run from repo root AFTER extracting day10_submission_evidence.zip.
Do not modify teacher scripts/tests.
"""
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from scripts.labcommon import read_jsonl
from scripts.score import parse_output, iou_xyxy, evaluate

ROOT = Path.cwd()
EVID = ROOT/'report'/'evidence'
FIG = ROOT/'report'/'figures'
FIG.mkdir(parents=True,exist_ok=True)
KEYS=['e0','e1','e2','e3','e4_rank_4','e4_rank_64']
meta=json.loads((ROOT/'data'/'splits'/'meta.json').read_text())
assert meta['dataset_version']=='d10-21e4070301'
test=read_jsonl(ROOT/'data'/'splits'/'test.jsonl')
assert len(test)==311
for key in KEYS:
    d=json.loads((EVID/f'{key}.json').read_text())
    assert d['score_version']==1 and d['split']=='test'
    assert d['n_images']==311 and d['missing_predictions']==0
    assert d['dataset_version']==meta['dataset_version']
    x=read_jsonl(ROOT/'work'/f'{key}.jsonl')
    assert len(x)==311 and len(set(p['image_id'] for p in x))==311
    print(f'PASS: {key}: F1={d["micro"]["f1"]:.4f}')

# Wrong coord check uses original E3 JSON predictions while lying to scorer
preds={r['image_id']:r['raw'] for r in read_jsonl(ROOT/'work'/'e3.jsonl')}
wrong=evaluate(test,preds,coord='pixel',n_boot=1000,seed=0)
correct=json.loads((EVID/'e3.json').read_text())
print('E3 normal norm1000 F1:',correct['micro']['f1'])
print('E3 WRONG pixel interpretation F1:',wrong['micro']['f1'])
assert abs(wrong['micro']['f1']-correct['micro']['f1'])>0.01
report_file=ROOT/'REPORT.md'
report=report_file.read_text(encoding='utf-8')
if 'PENDING_WRONG_COORD_F1' in report:
    report=report.replace('PENDING_WRONG_COORD_F1',
        f'chấm E3 trên cùng 311 ảnh với `coord=norm1000` cho F1 **{correct["micro"]["f1"]:.4f}**, '
        f'nhưng dùng nhầm `coord=pixel` cho F1 **{wrong["micro"]["f1"]:.4f}** '
        f'(chênh lệch **{wrong["micro"]["f1"]-correct["micro"]["f1"]:+.4f}**). '
        'Đây là thử nghiệm đổi cách *diễn giải output* khi chấm, không phải train lại model.')
    report_file.write_text(report,encoding='utf-8')

# Official-like matching per image (same-label greedy, IoU>=0.5)
def gt_boxes(sample):
    out=[]
    for obj in sample['objects']:
        xc,yc,w,h=obj['yolo'];out.append({'label':obj['label'],'bbox':((xc-w/2)*1000,(yc-h/2)*1000,(xc+w/2)*1000,(yc+h/2)*1000)})
    return out

def match_case(sample,predicted):
    gt=gt_boxes(sample)
    classes=sorted({x['label'] for x in gt+predicted})
    tp=fp=fn=0
    for lab in classes:
        g=[x for x in gt if x['label']==lab]
        p=[x for x in predicted if x['label']==lab]
        edges=sorted([(iou_xyxy(a['bbox'],b['bbox']),i,j) for i,a in enumerate(g) for j,b in enumerate(p)],reverse=True)
        used_g=set();used_p=set()
        for iou,i,j in edges:
            if iou<0.5:break
            if i not in used_g and j not in used_p:
                used_g.add(i);used_p.add(j)
        tp+=len(used_g);fp+=len(p)-len(used_p);fn+=len(g)-len(used_g)
    return {'tp':tp,'fp':fp,'fn':fn,'gt':gt,'pred':predicted}

byid={p['image_id']:p['raw'] for p in read_jsonl(ROOT/'work'/'e4_rank_64.jsonl')}
items=[]
for s in test:
    pred=parse_output(byid[s['image_id']])
    assert pred is not None
    case=match_case(s,pred)
    items.append((s,case))

# 2 strong hits, 2 hallucinated/FP, 2 missed/FN (6 unique image IDs)
used=set();selected=[]
choices=[
 ('Good localization',sorted(items,key=lambda x:(-x[1]['tp'],x[1]['fp']+x[1]['fn'],x[0]['image_id']))),
 ('Good localization',sorted(items,key=lambda x:(-x[1]['tp'],x[1]['fp']+x[1]['fn'],x[0]['image_id']))),
 ('False positives',sorted(items,key=lambda x:(-x[1]['fp'],x[1]['tp'],x[0]['image_id']))),
 ('False positives',sorted(items,key=lambda x:(-x[1]['fp'],x[1]['fn'],x[0]['image_id']))),
 ('Missed objects',sorted(items,key=lambda x:(-x[1]['fn'],x[1]['tp'],x[0]['image_id']))),
 ('Missed objects',sorted(items,key=lambda x:(-x[1]['fn'],x[1]['fp'],x[0]['image_id']))),
]
# Correct typo-protect: python executes only valid local helper
for category,candidates in choices:
    selected_one=None
    for s,c in candidates:
        if s['image_id'] in used:continue
        if category=='Good localization' and c['tp']==0:continue
        if category=='False positives' and c['fp']==0:continue
        if category=='Missed objects' and c['fn']==0:continue
        selected_one=(s,c);break
    assert selected_one is not None,category
    s,c=selected_one;used.add(s['image_id']);selected.append((category,s,c))

COLORS={'pothole':(255,190,40),'crack':(255,90,190),'manhole':(40,210,255)}
CAP=[]
for i,(category,s,c) in enumerate(selected,1):
    img=Image.open(ROOT/'data'/s['image_path']).convert('RGB')
    W,H=img.size
    panels=[]
    for title,boxes in [('Ground truth',c['gt']),('E4 rank64 prediction',c['pred'])]:
        canvas=img.copy();d=ImageDraw.Draw(canvas)
        for obj in boxes:
            x1,y1,x2,y2=obj['bbox'];px=[int(round(x1*W/1000)),int(round(y1*H/1000)),int(round(x2*W/1000)),int(round(y2*H/1000))]
            px=[min(max(v,0),W if j%2==0 else H) for j,v in enumerate(px)]
            if px[2]<=px[0] or px[3]<=px[1]:continue
            color=COLORS.get(obj['label'],(255,255,255))
            d.rectangle(px,outline=color,width=3)
            d.text((px[0],max(0,px[1]-13)),obj['label'],fill=color)
        panels.append(canvas)
    out=Image.new('RGB',(2*W,H+42),(20,20,24))
    out.paste(panels[0],(0,42));out.paste(panels[1],(W,42))
    d=ImageDraw.Draw(out)
    d.text((8,8),f'GT | {s["image_id"][:45]}',fill=(255,255,255))
    d.text((W+8,8),f'Pred | TP={c["tp"]} FP={c["fp"]} FN={c["fn"]}',fill=(255,255,255))
    path=FIG/f'compare_{i:02d}.png';out.save(path,optimize=True)
    CAP.append(f'- **{path.name}** — {category}; `image_id={s["image_id"]}`; TP={c["tp"]}, FP={c["fp"]}, FN={c["fn"]}; GT left vs E4 rank64 right; colors: pothole=amber, crack=pink, manhole=cyan.')
    print('Saved:',path)
(FIG/'README.md').write_text('# Day 10 visual evidence\n\nMatching: same-class greedy IoU≥0.5. GT on left, E4 rank64 predictions right.\n\n'+'\n'.join(CAP)+'\n',encoding='utf-8')
print('PASS: 6 figures, 6 evidence scores, REPORT coord-scale experiment')
