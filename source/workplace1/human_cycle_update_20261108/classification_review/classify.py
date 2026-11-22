import json,hashlib,csv
from pathlib import Path
B=Path(__file__).resolve().parent;R=Path('/mnt/ojc/workplace1/asset_library/human')
# Explicit per-asset temporal visual review, not a speed-threshold classifier.
notes={
'HN0043':('standing_or_small_gesture','屈膝抬脚姿态持续保持，低头抱持/操作物品；未见左右脚交替步态。', 'medium'),
'HN0090':('standing_or_small_gesture','脚基本固定，转头和轻微手部调整。','high'),
'HN0137':('standing','原地站立，主要转头与轻微重心变化。','high'),
'HN0175':('gesturing','原地抬手、摆手和低头，无持续步行。','high'),
'HN0356':('gesturing','原地低头、弯腰和手臂下伸，未见连续步态。','high'),
'HN0380':('standing_or_small_gesture','低头操作手中物体，身体轻晃，足部未呈交替行走。','medium'),
'HN0387':('gesturing','原地触碰颈肩、摆手和转头。','high'),
'HN0411':('gesturing','原地摆臂、看向侧方，脚部基本保持站姿。','high'),
'HN0428':('gesturing','原地手臂抬起/摆动并改变上身朝向。','high'),
'HN0431':('gesturing','原地抬手整理头发，腿部仅站姿调整。','high'),
'HN0434':('gesturing','原地将手臂向身后调整，上身轻微前倾；重建边缘有噪声。','medium'),
'HN0453':('standing_or_small_gesture','双手持物、低头，轻微屈膝与站姿调整，无持续交替行走。','medium'),
'HN0571':('standing_or_small_gesture','原地低头持物，手部小动作与重心轻移。','high'),
'HN0577':('gesturing','原地抬手靠近脸部并前倾，脚部保持站位。','high'),
'HN0646':('standing_or_small_gesture','双手靠近腰部，原地调整站姿与头部方向。','high'),
'HN0668':('standing_or_small_gesture','一手靠近脸部，屈膝站立、小幅晃动，无连续步态。','medium'),
'HN0675':('unclassified_motion','大面积非人体高斯残留遮住躯干，无法可靠判断运行资产的动作；不能靠改标签认证。','unobservable'),
'H0002':('gesturing','原地双臂抬起、放下，脚部保持站位。','high'),
'H0029':('standing_or_small_gesture','低头、手臂小幅调整、原地重心移动。','high'),
'H0041':('standing_or_small_gesture','一手靠近脸部，原地转头和小幅晃动。','high'),
'H0112':('gesturing','低头操作物品后下放手臂；末段局部站姿调整，不是持续步行。','medium'),
'H0131':('gesturing','抬脚站姿逐步放下，伴随手臂动作；不是重复交替行走。','medium'),
'H0167':('standing_or_small_gesture','原地侧身与小幅屈膝换重心，持续持物姿态。','medium'),
'H0202':('standing_or_small_gesture','原地转头抬头、手部小动作，未见交替迈步。','high'),
'H0371':('standing_or_small_gesture','原地低头调整双手与衣物，轻微前倾。','high'),
'H0390':('standing_or_small_gesture','交叉腿站姿后调整站位，低头/转头，无持续步行。','medium'),
'H0409':('standing_or_small_gesture','屈膝单腿姿态基本固定，手部轻动；末尾有重建残留。','medium'),
'H0501':('standing_or_small_gesture','原地双手持物，腿部左右调整站姿，未形成连续交替步态。','medium'),
'H0566':('gesturing','原地向外伸手和收回，另一手靠近脸部。','high'),
'H0593':('gesturing','原地低头前倾，双手靠近腰部调整。','high'),
'H0597':('standing','站姿基本不变，少量头部和重心变化。','high'),
'H0654':('standing_or_small_gesture','轻微低头转头与原地换重心。','high'),
'H0680':('gesturing','原地抬手、挥手并转头，脚部无持续交替。','high'),
'H0699':('gesturing','双手从两侧移到腰部，原地轻转身。','high'),
'H0736':('gesturing','原地将手抬到颈肩附近，腿部基本固定。','high'),
'H0937':('standing_or_small_gesture','原地头部和手部小幅动作；衣物侧面有重建污染。','medium'),
'H1032':('standing_or_small_gesture','双手持物、转头，足部仅站姿调整。','high'),
'H1038':('standing_or_small_gesture','原地低头、手臂微动及重心调整。','medium'),
'H1106':('standing_or_small_gesture','原地左右看，手部在身前/口袋附近小动作。','high'),
'H1112':('standing','双臂自然下垂，站位保持，身体略微晃动。','high'),
'H1192':('standing_or_small_gesture','持物站立，轻微屈膝和上身调整，无连续步态。','medium'),
'H1200':('gesturing','原地抬手、较大幅度转身并前倾；这是非周期手势/转身片段，循环需单独验收。','medium'),
'H1306':('standing_or_small_gesture','原地左右转头并前倾，足部站位调整。','high'),
'H1623':('standing_or_small_gesture','原地转头、轻微抬头和手臂小动作。','high'),
'H1738':('standing_or_small_gesture','原地侧倾后直起、转头，未见连续步态。','high'),
'H1825':('gesturing','双手由身侧移至身前交叠，脚部站位基本固定。','high'),
'H1849':('standing_or_small_gesture','原地转头、低头和轻微上身晃动。','high'),
'H2144':('standing_or_small_gesture','可辨主体保持站位并轻微手部动作，但腿部和身侧有严重重影，动作标签不代表质量可用。','low'),
'H2182':('standing_or_small_gesture','原地看手中物体，手臂小幅调整、轻微低头。','high'),
'H2249':('standing_or_small_gesture','原地左右转头、微小侧转，双脚基本固定。','high')}
rows=[];index={r['folder']:r for r in json.loads((B.parent/'v2/cycle_updates.json').read_text())['assets']}
for folder,(label,note,confidence) in notes.items():
 m=json.loads((R/folder/'asset.json').read_text());assert m['motion_type']=='unclassified_motion'
 rows.append({'id':m['id'],'folder':folder,'runtime_file_sha256':index[folder]['runtime_file_sha256'],'source_sha256':index[folder]['source_sha256'],'old_motion_type':m['motion_type'],'motion_type':label,'classification':{'method':'assistant_review_of_six_temporal_samples_in_two_reconstruction_views','review_date':'2026-11-08','confidence':confidence,'observation':note,'continuous_locomotion_observed':False,'recommended_trajectory':'stationary' if label!='unclassified_motion' else 'do_not_enable','quality_certified':False,'cycle_requires_separate_validation':True,'cycle_review_blocked':folder in ['HN0675','H2144'],'evidence_sheet':f'sheet_{list(notes).index(folder)//5:02d}.jpg'}})
assert len(rows)==50
(B/'classification_updates.json').write_text(json.dumps({'format':'human_motion_classification_review_v1','assets':rows},ensure_ascii=False,indent=2))
from collections import Counter
print(Counter(r['motion_type'] for r in rows))
with (B/'classification.csv').open('w',newline='') as f:
 w=csv.writer(f);w.writerow(['folder','id','motion_type','confidence','observation','cycle_review_blocked']);w.writerows([r['folder'],r['id'],r['motion_type'],r['classification']['confidence'],r['classification']['observation'],r['classification']['cycle_review_blocked']] for r in rows)
