import json
from pathlib import Path
W=Path(__file__).resolve().parents[1]
notes={
'V01':('selected','150 / 5.005 s；单个展开三角牌和配重底座。视频为产品照片动画，无真实互补视角。'),
'V02':('download_blocked','原始页和 MP4 返回 403 / challenge；未绕过。'),
'V03':('download_blocked','另一演示页与 MP4 同样返回 403；不能与 V02 假定为同一实物。'),
'V04':('reference_only','实际解码 174.174 s 的完整部署状态有人站在后方，右下标题遮挡支脚；69–86 s 是产品图拼接。未选作核心资产。'),
'V05':('selected','frame 1176 / 49.049 s，ROAD WORK AHEAD，完整短立杆与黑色底座；尺寸型号未确认。'),
'V06':('reference_only','ROAD CLOSED AHEAD / 500 FT 和 RIGHT LANE CLOSED AHEAD 分配置；主要演示均裁掉底座。88–94 s 产品拼图不能当作同一实物多视图。'),
'V07':('download_blocked','YouTube metadata 要求登录/机器人验证；未绕过。'),
'V08':('download_blocked','YouTube metadata 要求登录/机器人验证；未绕过。'),
'V09':('selected_reconstruction_rejected','270.0 s 主视图 Unfall，264.0 s 同一连续镜头补充 Feuerwehr 与第三面；灯状态固定。两次 FLUX、real-only B0/H1 均有字样或结构失真，不能发布为合格资产。'),
'V10':('reference_only','真实解码 10.473 s、1920×1080、30 fps；两件 ROAD CLOSED 装置相互重叠，杆、长路栏和沙袋归属难分，未强行合并。'),
'V11':('not_attempted','条件候选；须核查是否仅牌面样品。'),
'V12':('user_reference_only','用户报告 14.7 s、720²、30 fps；躺放和局部近摄。本工作区未重新下载，不能记作本轮解码。'),
'V13':('deferred','便携警示灯为第二阶段，不计入本轮标识资产。'),
'V14':('not_attempted','Triflash 带灯产品补充；不能与三面折叠牌混为同结构。'),
'V15':('not_attempted','条件候选；车辆压过和变形段不能用于刚体重建。'),
'P01':('selected','厂家/商家原始 1000² 照片 full2：完整四脚 ROAD CLOSED。其它两张裁脚，排除互补参考。尺寸页与照片比例有冲突，参数化支架单独标记。'),
'P02':('reference_only','7 张 1400×1000 产品图片外观为渲染图；左右箭头/空架/背面不具同一实物证据，未作为真实多视图。'),
'P03':('unavailable','Proteclight 旧产品链接失效；类别页仅可作为后续入口。'),
'P04':('download_blocked','Commons 原图下载 429，未记作获得素材。'),
'P05':('reference_only','Geograph 825234 / Phillip Perry / CC-BY-SA 2.0；640×480，立杆底部裁切，不是完整独立部署单元。'),
'P06':('reference_only','Geograph 3312126 / Walter Baxter；640×478，封道与绕行标识支杆埋入雪中，未见独立底座。'),
'P07':('selected','Tennants 官方 PDF 中 269×403 的完整右转 Diversion 实拍；黄色杆和黑底座完整，低分辨率；同页施工图是另一独立资产。'),
'P08':('selected','Tennants 官方页下载 1200×800 施工图（工人图标＋400 yds＋完整黄杆底座），优于 PDF 低清图。后背/连接件图片只证明产品家族结构，不冒充同实物参考。'),
'D01':('dimension_reference','Deflecto 官方 PDF 与视频尺寸图；单件约 16.55 inch 高 / 17.25 inch 宽；折叠尺寸描述不用于展开深度。')}
rows=[]
for sid,(status,note) in notes.items():
 d=W/'sources'/sid;files=list(d.glob('*acquisition.json'));records=[json.loads(p.read_text()) for p in files]
 row=dict(source_id=sid,screening_status=status,decision=note,acquisition_records=[str(p.relative_to(W)) for p in files])
 if records:
  a=records[-1];row.update(page_url=a.get('page_url'),sha256=a.get('sha256'),download_status=a.get('status'),license_status=a.get('license_status',a.get('license','unknown')))
  for rec in records:
   if 'probe' in rec:
    st=next((s for s in rec['probe']['streams'] if s.get('codec_type')=='video'),{});row['actual_media']={k:st.get(k) for k in ['width','height','avg_frame_rate','nb_frames','duration','codec_name']}
 rows.append(row)
(W/'reports/source_screening.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2));(W/'sources/sources_manifest.json').write_text(json.dumps(dict(sources=rows,policy='Downloaded and decoded evidence only; user-reported and unattempted sources explicitly separate'),ensure_ascii=False,indent=2))
