import copy,csv,hashlib,html,json,shutil
from pathlib import Path

LIB=Path('/mnt/ojc/workplace1/asset_library')
OUT=LIB/'deliveries/subset_16_categories_20261103'
rigid=json.loads(Path('/tmp/subset_rigid_selected.json').read_text())
human=json.loads(Path('/tmp/subset_human_root_entries.json').read_text())
release=json.loads((LIB/'releases/rigid/drivelab_asset.json').read_text())
assets=[];groups=[];cards=[]
def write(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def picture(src,dst,label):
    if not src or not Path(src).is_file():return ''
    destination=OUT/dst;destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,destination)
    return '<figure><a href="'+html.escape(dst)+'"><img loading="lazy" src="'+html.escape(dst)+'"></a><figcaption>'+html.escape(label)+'</figcaption></figure>'
for a in rigid:
    category=a['presentation_category'];entry=copy.deepcopy(a['release_binding'])
    if entry is None:
        original=a['manifest_entry'];canonical=original['canonical'];size=canonical['size_xyz'];axis=max(range(3),key=lambda i:size[i]);factor=canonical['metric_preview']['isotropic_factor']
        entry=dict(global_uid=a['uid'],dataset=original['dataset'],**{'class':'traffic_sign'},category='traffic_sign',asset_path='assets/rigid/traffic_sign/emergency_warning_triangle.ply',size_xyz=size,scale_m=size[axis]*factor,scale_axis='xyz'[axis],canonical=dict(rotation=canonical['rotation'],translation=canonical['translation'],heading='+X'))
    destination=OUT/entry['asset_path'];destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(a['model_path'],destination)
    assert hashlib.sha256(destination.read_bytes()).hexdigest()==a['sha256']
    assets.append(entry)
    meta=OUT/'metadata'/category;meta.mkdir(parents=True,exist_ok=True)
    write(meta/'original_asset_record.json',a['manifest_entry'])
    note=dict(display_category=a['display_name'],global_uid=a['uid'],source_status=a['status'],selection_notes=a['selection_notes'],payload_sha256=a['sha256'],visual_review=a['visual_review'],original_input_available=a['source_image_path'] is not None)
    groups.append(note)
    images=picture(a['evidence_path'],f'previews/{category}/orbit_16.jpg','已有资产的 16 视角审核图')
    images+=picture(a['source_image_path'],f'previews/{category}/source.jpg','原始参考图')
    cards.append('<article><h2>'+html.escape(a['display_name'])+'</h2><p>'+html.escape(a['selection_notes'])+'；原状态：'+a['status']+'</p><div class="pictures">'+images+'</div></article>')
    if category=='emergency_sign':
        source=Path(a['model_path']).parent
        for name in ['placement.json','quality.json','sign_semantics.json','source_camera.json']:
            shutil.copy2(source/name,meta/name)
for entry in human:
    assets.append(entry);identity=Path(entry['package_path']).name;action=entry['person']['action'];label='行人（走路）' if action=='walking' else '行人（站立）'
    groups.append(dict(display_category=label,global_uid=entry['global_uid'],source_status='visual_review_pending',selection_notes='121帧，25fps，约4.8秒；保留原 quality_approved=false',payload_sha256=hashlib.sha256((OUT/entry['asset_path']).read_bytes()).hexdigest(),visual_review='首中末及五时刻双视角检查动作；不等于完整质量准入'))
    image=f'human/{identity}/subset_motion_review.jpg';video=f'human/{identity}/demo.mp4'
    cards.append('<article><h2>'+label+'</h2><p>121帧 / 25fps；人体仍为待全面质量批准样本。</p><a href="'+video+'">查看动作视频</a><div class="pictures"><figure><a href="'+image+'"><img loading="lazy" src="'+image+'"></a></figure></div><video controls preload="metadata" src="'+video+'"></video></article>')
assert len(assets)==16 and len({a['global_uid'] for a in assets})==16
order=['小汽车','交通锥桶','行人（走路）','行人（站立）','倒伏树木','杆状设施','道路隔离护栏','道路散落物','防撞柱','交通信号灯','交通标志牌','卡车','厢式车','公共汽车','骑行车辆（电动车）','道路应急标识']
groups.sort(key=lambda x:order.index(x['display_category']));byid={a['global_uid']:a for a in assets};assets=[byid[g['global_uid']] for g in groups]
write(OUT/'drivelab_asset.json',dict(schema_version=1,manifest_type='drivelab_asset',path_policy='repository_relative_single_asset_path',total=16,scale_policy=release['scale_policy'],heading_policy=release['heading_policy'],subset_groups=groups,quality_note='13 rigid keep, 1 emergency-sign review, 2 human visual_review_pending. Inclusion in this subset does not promote quality status.',human_compatibility_index='human/drivelab_asset.json',assets=assets))
with (OUT/'subset.csv').open('w',newline='') as f:
    writer=csv.writer(f);writer.writerow(['类别','UID','模型路径','原质量状态','备注'])
    for g in groups:writer.writerow([g['display_category'],g['global_uid'],byid[g['global_uid']]['asset_path'],g['source_status'],g['selection_notes']])
page='''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>16个代表资产 subset</title><style>body{font:16px/1.6 system-ui;background:#15202b;color:#eef;max-width:1300px;margin:auto;padding:24px}a{color:#9bd3ff}article{border-top:1px solid #567;padding:18px 0}.pictures{display:flex;flex-wrap:wrap;gap:12px}figure{width:min(100%,560px);margin:0}img{width:100%;max-height:650px;object-fit:contain}video{width:min(100%,960px);margin-top:16px}</style><h1>每类一个 · 16个代表资产</h1><p>14个刚体 + 走路人体 + 站立人体。骑行车辆选择蓝色 NIU 电动车与一位固定坐姿骑手。</p><p><a href="drivelab_asset.json">统一总 JSON</a> · <a href="subset.csv">分类清单</a> · <a href="README.txt">使用说明</a></p><p>保留原审核状态：13个刚体 keep，1个应急标识 review，2个人体尚未全面质量批准。电动车原始输入照片已缺失，保留现有模型与审核图。</p>'''
(OUT/'index.html').write_text(page+''.join(cards)+'</html>')
(OUT/'README.txt').write_text('''16个代表资产 subset

入口：drivelab_asset.json（16条总清单）；index.html（离线图片、人体动作视频）；subset.csv（类别名单）。
内容：旧13类各一个，其中人体细分走路和站立各一个；新增骑行车辆选电动车一个、道路应急标识一个。共14刚体+2人体。
骑行车辆为蓝色NIU MQi GT电动车，包含一名固定坐姿骑手；不含自行车。

解压后保留目录结构。总JSON中的asset_path均相对于解压根目录，模型无需原服务器路径。
human/drivelab_asset.json是Hasim读取人体所需的内部兼容索引；用户总入口仍是根drivelab_asset.json。
刚体PLY已烘焙规范变换，不要再次应用canonical.rotation/translation；米制缩放只用一次 scale_m / size_xyz[scale_axis]。
人体保留121帧、25fps、SH3、放置及运动数组。使用支持SH的人体loader和原有放置矩阵；不要重复减根运动。单次时长4.8秒，不保证无缝循环。
人体NPZ集中放在assets/nonrigid/human/，每份只保存一次；human/<ID>/manifest.json已改为包内相对路径。
human/<ID>/runtime与refs也保留。现有烘焙帧回放不需要重训；连续形变运行仍需相应Grid4D软件环境。

审核边界：13个刚体原状态keep；应急三角警示牌仍为review（背面与厚度未完整核实）；两个人体原quality_approved=false。此合集未将它们升级为全面合格。
电动车原始输入照片已缺失，已包含模型和现有16视角审核图；不把审核图冒充原图。
metadata中的原记录保留历史路径用于溯源，不是此包的运行依赖。人体源目录README和校验记录为历史生产记录，搬迁后的实际文件校验以根FILES.sha256为准。

验证：python tools/verify_asset_subset.py . --hasim-source /path/to/Hasim_trajectory_assets
需要numpy、scipy、plyfile和现有Hasim源码。验证脚本只读，不启动训练或GPU任务。
''')
(OUT/'tools').mkdir(exist_ok=True);shutil.copy2('/tmp/verify_asset_subset.py',OUT/'tools/verify_asset_subset.py')
print('Built total manifest and previews:',OUT)
