"""Portable meeting materials from existing verified results; no GPU work."""
import asyncio
import csv
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import zipfile
from zoneinfo import ZoneInfo

R = Path(__file__).resolve().parents[1]
os.environ['MPLCONFIGDIR'] = str(R / 'cache/matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import av
from PIL import Image

B = ('ftgs_base5k', 'ftgs_opt15k', '4dgs_grid25', '4dgs_grid61')
NAMES = ('FTGS baseline 5k', 'FTGS optimized 15k', '4DGaussians grid25', '4DGaussians grid61')
CN = ('FTGS原5k', 'FTGS优化15k', '4DGaussians网格25', '4DGaussians网格61')
K = ('full_psnr', 'foreground_psnr', 'ssim', 'crop_lpips', 'silhouette_iou', 'outside_alpha_mean')
HEAD = ('方法', '全图PSNR_dB_越高越好', '前景PSNR_dB_越高越好', '全图SSIM_越高越好',
        '人体裁剪LPIPS_越低越好', '轮廓IoU_越高越好', 'mask外平均alpha_越低越好')
ROOT = R / 'deliverables/meeting_20261015'
ARCHIVE = ROOT.with_suffix('.zip')


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def figure(summary):
    fig = plt.figure(figsize=(20, 11.25), facecolor='white')
    fig.text(.045, .955, '4DAnyone + Dynamic Gaussian Reconstruction', fontsize=25, weight='bold', color='#17334d')
    fig.text(.045, .916, 'Mean metrics | same generated targets, held-out cameras and evaluation times', fontsize=15, color='#526576')
    headers = ['Backend recipe', 'Full-image\nPSNR (dB) ↑', 'Foreground\nPSNR (dB) ↑',
               'Full-image\nSSIM ↑', 'Cropped\nLPIPS ↓', 'Silhouette\nIoU ↑', 'Outside-mask\nalpha ↓']
    for cohort, title, y in [('main', 'MAIN COHORT: 7 reconstructed tracks / 8 preregistered', .54),
                             ('reserve', 'RESERVE: N11 only / 3 attempted reserves — reported separately', .19)]:
        fig.text(.045, y + .285, title, fontsize=16, weight='bold', color='#17334d')
        rows = []
        for name, branch in zip(NAMES, B):
            values = summary['cohorts'][cohort]['matched_complete_four']['branches'][branch]['mean']
            rows.append([name, *[f'{values[key]:.3f}' if key.endswith('psnr') else f'{values[key]:.5f}' for key in K]])
        axis = fig.add_axes([.04, y, .92, .25]);axis.axis('off')
        table = axis.table(cellText=rows, colLabels=headers, cellLoc='center', loc='center',
                           colWidths=[.25, .125, .125, .12, .125, .12, .135], bbox=[0,0,1,1])
        table.auto_set_font_size(False);table.set_fontsize(13)
        for (row, col), cell in table.get_celld().items():
            cell.set_edgecolor('#d9e1e8');cell.set_linewidth(.6)
            if row == 0:
                cell.set_facecolor('#17334d');cell.get_text().set_color('white');cell.get_text().set_weight('bold')
            else:
                cell.set_facecolor('#edf6f4' if row == 4 else '#f3f6f9' if row % 2 == 0 else 'white')
                if col == 0:cell.get_text().set_ha('left');cell.PAD=.04
                if row == 4:cell.get_text().set_weight('bold');cell.get_text().set_color('#176653')
    fig.text(.045, .125, 'Protocol: generated-heldout reconstruction, not real Waymo novel-view ground truth.', fontsize=13, color='#17334d', weight='bold')
    fig.text(.045, .095, '352×640 | gray background 0.5 | views {0, 8, 16} × 13 times | per-track means, then equal-weight track average.', fontsize=12, color='#526576')
    fig.text(.045, .067, 'Foreground: target alpha > 0.5. Full-image PSNR includes background. HUST: 3k coarse + 30k fine (32,999 updates).', fontsize=12, color='#526576')
    fig.text(.045, .039, 'N05 / N09 / N10 frontend failures retained. N11 still has strong arm/texture artifacts; numerical gains do not imply visual approval.', fontsize=12, color='#9a3737')
    fig.savefig(ROOT / 'mean_metrics.png', dpi=160, facecolor='white')
    plt.close(fig)


async def main():
    assert not ROOT.exists() and not ARCHIVE.exists(), 'Preserve existing package; select a new explicit destination for another delivery.'
    status = read(R / 'FINAL_STATUS.json');summary = read(R / 'metrics_summary.json')
    assert status['completed_final_versions'] == 32
    selection = {c['sample_id']:c for c in read(R / 'selection.json')['main'] + read(R / 'selection.json')['reserve']}
    for cohort in ('main', 'reserve'):
        group = summary['cohorts'][cohort]['matched_complete_four']
        assert group['tracks'] == (7 if cohort == 'main' else 1)
        for branch in B:
            actual = [read(R/'samples'/label/'evaluations'/branch/'results.json')['mean'] for label in group['sample_ids']]
            for key in K:
                values = [row[key] for row in actual]
                assert np.isfinite(values).all()
                assert abs(float(np.mean(values)) - group['branches'][branch]['mean'][key]) < 1e-9
    ROOT.mkdir(parents=True)
    mappings = [];actors = []

    def copy(source, target, expected):
        assert source.is_file() and sha(source) == expected, str(source)
        destination = ROOT / target;destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        assert sha(destination) == expected
        mappings.append({'file':target.as_posix(), 'source':str(source), 'sha256':expected, 'bytes':source.stat().st_size})

    review = read(R / 'research/source_action_review.json')
    labels = status['completed_tracks']
    for label in labels:
        base = R/'samples'/label;candidate = selection[label];mid = (candidate['output_slots'] - 1)//2
        validation = read(base/'comparison/validation.json')
        assert set(validation['successful_branches']) == set(B) and not validation.get('missing_branches')
        by_name = {Path(row['file']).name:row for row in [*validation['videos'], *validation['preview_images']]}
        for view in (0,8,16):
            for source_name, target_name in [(f'heldout_view{view:02d}_five.mp4', f'view{view:02d}.mp4'),
                                              (f'heldout_view{view:02d}_frame{mid:03d}_five.jpg', f'view{view:02d}.jpg')]:
                copy(base/'comparison'/source_name, Path('cases')/label/target_name, by_name[source_name]['sha256'])
        item = review['cases'][label]
        copy(R/item['source_contact'], Path('cases')/label/'source.jpg', item['source_contact_sha256'])
        source_media = read(base/'source/visualization/manifest.json')
        copy(base/'source/visualization/source_chain.mp4', Path('cases')/label/'pipeline.mp4', source_media['videos']['source_chain.mp4']['sha256'])
        parts = read(base/'comparison/details/manifest.json')
        row = next(row for row in parts['images'] if Path(row['file']).name == f'parts_view00_frame{mid:03d}.jpg')
        copy(Path(row['file']), Path('cases')/label/'pose_details.jpg', row['sha256'])
        actors.append({'label':label, 'cohort':candidate['cohort'], 'source_scene':candidate['scene'],
                       'source_camera':candidate['camera_index'], 'track_id':candidate['track_id'],
                       'native_observations':candidate['native_frame_count'], 'video_slots':candidate['output_slots'],
                       'descriptive_action':item['category'], 'selected_still_frame':mid})

    with (ROOT/'mean_metrics.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['统计组','轨迹数',*HEAD])
        for cohort, title in [('main','主样本成功重建7人；原登记8人'),('reserve','备用N11；备用共尝试3人')]:
            group = summary['cohorts'][cohort]['matched_complete_four']
            for name, branch in zip(CN, B):writer.writerow([title,group['tracks'],name,*[group['branches'][branch]['mean'][key] for key in K]])
    with (ROOT/'per_person_metrics.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['人物','统计组',*HEAD])
        for label in labels:
            for name, branch in zip(CN, B):
                row=read(R/'samples'/label/'evaluations'/branch/'results.json')['mean']
                writer.writerow([label,selection[label]['cohort'],name,*[row[key] for key in K]])
    figure(summary)
    descriptions = {'N01':'站立或小步', 'N02':'行走，途中有立柱遮挡', 'N03':'行走，暗部观测',
                    'N04':'行走，浅色夹克', 'N06':'行走，输入像素较小', 'N07':'小步或停顿、手持物',
                    'N08':'行走、背包', 'N11':'大幅举臂与跨步，明显重建失败案例'}
    actor_lines = '\n'.join(f"{label}：{'备用' if selection[label]['cohort']=='reserve' else '主样本'}；{descriptions[label]}；{selection[label]['native_frame_count']}张实拍→{selection[label]['output_slots']}槽。" for label in labels)
    (ROOT/'README.txt').write_text(f'''4DAnyone / Waymo 四组动态重建组会素材
整理日期：{datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y-%m-%d %H:%M')}（北京时间）

本文件夹可整体下载，所有媒体都是实际文件，不依赖服务器路径或软链接。

建议汇报顺序
1. 打开 mean_metrics.png，说明全图PSNR与人体前景PSNR的区别。
2. 看 cases/N04/pipeline.mp4 与 view00.mp4，介绍行走人物的完整流程和四组对照。
3. 看 N08 的背包/行走与 N01 的站立或小步，说明不同观测的表现。
4. 看 N11/view00.mp4 和 pose_details.jpg，展示大幅动作仍有明显重影和纹理丢失。
八个人物都已包含，以上只是汇报建议，未根据好坏删除其他人物。

目录
mean_metrics.png：一张平均指标图，上半主样本七人，下半备用N11，分开统计。
mean_metrics.csv：相同表格的可编辑原始数值，Excel可直接打开。
per_person_metrics.csv：八个人物×四组，共32行逐人指标，方便答疑。
cases/<人物>/source.jpg：真实Waymo观测的五时刻拼图。
cases/<人物>/pipeline.mp4：实拍观测→实际输入→root-local骨架→4DAnyone生成预览。
cases/<人物>/view00.mp4、view08.mp4、view16.mp4：三个留出生成相机的完整五栏对照。
cases/<人物>/view00.jpg、view08.jpg、view16.jpg：对应相机的中间时刻截图，直接用于PPT。
cases/<人物>/pose_details.jpg：同一中间时刻、view00的手脚条件骨架投影代理区域对照。
manifest.json：来源、哈希、样本身份及协议说明。

五栏从左到右
参考生成视频｜FTGS原5k｜FTGS优化15k｜4DGaussians网格25｜4DGaussians网格61。
源实拍与生成相机没有配准。pipeline中的实拍列按观测来源帧保持显示，不能当作逐像素GT。
view00/08/16属于生成相机编号，不是Waymo真实相机编号。
截图对每人统一选视频中间时刻，不挑选最好看的帧；视频保留完整61/121槽、25fps。

人物
{actor_lines}
以上动作是实拍抽样后的描述，不是官方行为标注或世界速度真值。

平均指标口径
主样本原登记8人，N05共同前端硬失败；主表是另外7人四组都完整的配对均值。
备用依次尝试N09/N10/N11，前两人共同前端失败，N11成功进入四后端；备用表n=1。
三条前端失败保留在成功率分母中，没有给失败人物伪造PSNR，也没有把备用混入主样本均值。
每条轨迹：352×640、灰背景0.5、留出生成视角H={{0,8,16}}×13个固定时刻；先逐人平均，再对人等权平均。
全图PSNR和SSIM包含背景；前景PSNR仅在参考alpha>0.5处计算；LPIPS使用参考mask确定的统一人体裁剪。
全图背景占比较大，主七人评测图的人体平均约15.6%，所以全图PSNR明显高于前景PSNR。
轮廓IoU越高越好，mask外平均alpha越低越好；前景PSNR不能单独反映身体外的漂浮点。
所有指标基于原始渲染计算，未用压缩视频或截图重新算分。
这些是generated-heldout reconstruction，衡量高斯对4DAnyone生成视频的重建，不是真实Waymo新视角精度。

四组训练
FTGS原5k：独立冷启动5k完整LR，零初速度，初始sigma15/(F-1)。
FTGS优化15k：独立冷启动15k完整LR，零初速度，初始sigma8/(F-1)；速度后续可学习。
两组FTGS是完整配方对照，不能把提升单独归因于训练步数。
HUST两网格：相同初始化、数据、seed42和采样顺序，3k coarse＋30k fine；原生末轮不step，实际32999次更新。
grid25/61改变时间网格，也改变网络容量及离散正则的影响，不能称为固定容量的纯消融。
前端使用官方4DAnyone-Turbo公开权重，每个人只生成一次24路视频，再供四组后端共享。
后端是独立FreeTimeGsVanilla与HUST 4DGaussians，不是4DAnyone论文作者尚未公开的内部FreeTimeGS。

结果边界
技术运行、动态包重放和完整评测通过，不代表所有人物外观合格。
N11的大幅摆臂仍有明显失败；其他人物的脸手、鞋、白边、拉丝与附件也有缺陷。
pose_details只是统一投影代理区，不是可靠语义手脚真值；图片放大没有增加真实细节。
源动作准确性、真实世界尺度、地面接触、任意换动作及场景融合均未验证；production_approved=false。
本包只整理已完成素材，未重新训练或修改原有模型、指标、原图和视频。
''',encoding='utf-8')

    all_files = [{'file':str(path.relative_to(ROOT)), 'sha256':sha(path), 'bytes':path.stat().st_size}
                 for path in sorted(ROOT.rglob('*')) if path.is_file()]
    manifest = {'created_at':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),'source_run':str(R),
                'source_summary_sha256':sha(R/'metrics_summary.json'), 'actors':actors,
                'columns':['generated reference',*B], 'cohorts_separate':True, 'new_training':False,
                'new_media_encoding':False,'metric_scope':'generated-heldout reconstruction',
                'production_approved':False,'source_media_copies':mappings,'files_except_this_manifest':all_files}
    save(ROOT/'manifest.json',manifest)
    await finalize()


async def finalize():
    assert not ARCHIVE.exists(), 'Do not replace an existing archive.'
    manifest=read(ROOT/'manifest.json');mappings=manifest['source_media_copies'];all_files=manifest['files_except_this_manifest']
    assert len(all_files)==76 and len(mappings)==72
    for row in all_files:
        path=ROOT/row['file'];assert path.stat().st_size==row['bytes'] and sha(path)==row['sha256']
    for row in mappings:assert sha(Path(row['source']))==row['sha256']
    frames={row['label']:row['video_slots'] for row in manifest['actors']}
    semaphore=asyncio.Semaphore(4)
    probe='''import av,json,sys
av.logging.set_level(av.logging.PANIC)
with av.open(sys.argv[1]) as container:
 stream=container.streams.video[0];stream.thread_count=1
 assert float(stream.average_rate)==25
 count=sum(1 for _ in container.decode(video=0))
 assert count==int(sys.argv[2]),(count,sys.argv[2])
print(json.dumps({'frames':count}))
'''

    async def validate_video(path):
        async with semaphore:
            child=await asyncio.create_subprocess_exec(sys.executable,'-c',probe,str(path),str(frames[path.parent.name]),
                 stdout=asyncio.subprocess.PIPE,stderr=asyncio.subprocess.PIPE)
            stdout,stderr=await child.communicate()
            if child.returncode:raise RuntimeError(f'Video decode failed:{path};exit{child.returncode};{stderr.decode()}')
            result=json.loads(stdout);result['file']=str(path.relative_to(ROOT));return result

    videos = await asyncio.gather(*map(validate_video,sorted(ROOT.rglob('*.mp4'))))
    for path in ROOT.rglob('*.jpg'):
        with Image.open(path) as image:image.verify()
    with Image.open(ROOT/'mean_metrics.png') as image:image.verify()
    assert len(videos)==32 and len(list(ROOT.rglob('*.jpg')))==40
    assert not any(path.is_symlink() for path in ROOT.rglob('*'))
    with zipfile.ZipFile(ARCHIVE,'x',zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for path in sorted(ROOT.rglob('*')):
            if path.is_file():archive.write(path,Path(ROOT.name)/path.relative_to(ROOT))
    with zipfile.ZipFile(ARCHIVE) as archive:
        assert archive.testzip() is None
        assert len(archive.infolist())==len(all_files)+1
    verification={'state':'portable_meeting_package_verified','folder':str(ROOT),'archive':str(ARCHIVE),
                  'zip_bytes':ARCHIVE.stat().st_size,'zip_sha256':sha(ARCHIVE),'zip_crc_passed':True,
                  'source_copies_byte_identical':len(mappings),'videos_fully_decoded':videos,
                  'still_images_verified':41,'media_reencoded':False,'symlinks':0,
                  'source_models_or_scores_changed':False}
    save(R/'research/meeting_package_verification.json',verification)
    print(json.dumps({key:value for key,value in verification.items() if key!='videos_fully_decoded'},ensure_ascii=False))


if __name__=='__main__':asyncio.run(finalize() if '--verify-existing' in sys.argv else main())
