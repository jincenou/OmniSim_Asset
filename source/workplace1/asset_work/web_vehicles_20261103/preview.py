"""Small, local review page for actual web-source pilot outputs."""
import html
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = Path('/mnt/ojc/workplace1/asset_library/previews/web_vehicles_20261103')
OUT.mkdir(parents=True, exist_ok=True)
def load(path):
    return json.loads(path.read_text())
def picture(source, name, caption):
    if not source.is_file():
        return ''
    destination = OUT / name
    shutil.copy2(source, destination)
    return '<figure><a href="'+html.escape(name)+'"><img loading="lazy" src="'+html.escape(name)+'"></a><figcaption>'+html.escape(caption)+'</figcaption></figure>'

page = '''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>网络车辆试跑</title><style>body{font:16px/1.65 system-ui;background:#15202b;color:#eee;max-width:1300px;margin:24px auto;padding:16px}a{color:#9bd3ff}.images{display:flex;flex-wrap:wrap;gap:12px}figure{margin:0;width:min(100%,400px)}img{width:100%;max-height:450px;object-fit:contain;background:#fff}article{border-top:1px solid #567;padding:18px 0}small{color:#bdcbd5}</style><h1>网络车辆 · 首批真实输出</h1><p>来源照片 → H 路线生成 → 完整性与比例检查。明显比例异常时移除 real 图，必要时重做 FLUX，保留前后两次记录。点击图片查看完整 16 视角。</p><p>这些是网络照片条件生成的资产，不代表对原车型的精确几何复原；世界尺寸未实测。</p>'''
records = []
for batch in sorted(ROOT.glob('chunks/batch_00000/batch.json')):
    chunk = batch.parent
    final_path = chunk / 'curation/curation.json'
    final = load(final_path)['assets'] if final_path.exists() else {}
    for i, row in enumerate(load(batch)['rows']):
        uid = row['obj_id']
        source = ROOT / 'sources' / uid
        provenance = load(source / 'acquisition.json')
        item = final.get(uid, {})
        status = item.get('status', '生成／审核中，尚未准入')
        page += '<article><h2>'+html.escape(row['category']+' · '+uid)+'</h2><p>最终状态：'+html.escape(status)+'</p>'
        page += '<p><a href="'+html.escape(provenance['page_url'],quote=True)+'">原始来源与授权信息</a></p><div class="images">'
        page += picture(source / 'source.jpg', uid+'_source.jpg', '最初的网络输入照片')
        page += picture(chunk / 'curation' / f'asset_{i:05d}' / 'axis_render/evidence.jpg', uid+'_initial.jpg', '初次 H 重建，16 视角；不等于已准入')
        retry = chunk / 'shape_retry' / uid
        page += picture(retry / 'curation/asset_00000/axis_render/evidence.jpg', uid+'_retry.jpg', '去除 real 图后的重试，16 视角')
        page += '</div>'
        plan = load(retry / 'plan.json') if (retry / 'plan.json').exists() else None
        if plan:
            reason = plan['trigger'].get('proportion_reason', '')
            page += '<p>重试原因：'+html.escape(reason)+'；移除 real 图 '+str(plan['removed_real_count'])+' 张；重新生成 FLUX：'+str(plan['regenerate_flux'])+'。</p>'
        page += '</article>'
        records.append(dict(obj_id=uid,category=row['category'],status=status,source=provenance['page_url'],retry_plan=plan))
(OUT / 'index.html').write_text(page+'</html>')
(OUT / 'review.json').write_text(json.dumps(records,ensure_ascii=False,indent=2))
print(OUT / 'index.html')
