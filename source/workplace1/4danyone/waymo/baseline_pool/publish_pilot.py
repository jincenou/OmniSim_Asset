"""Publish complete input/skeleton videos with explicit, limited review scope."""
import html
import json
from pathlib import Path
import sys

import av
from PIL import Image

from scan_native import ROOT, write

sys.path.insert(0, str(ROOT.parent / "scripts"))
from prepare import encode


def main():
    rows = json.loads((ROOT / "pilot_manifest.json").read_text())["candidates"]
    notes = {
        "B001": "夜间低亮度；分割保留了部分背景状纹理，末帧手臂估计需复核。不判质量通过。",
        "B002": "五个固定时刻站立骨架与轮廓大致一致；脸手细节有限，完整动态仍待复核。",
        "B003": "细杆遮挡时出现白色缺口，末段手腿也有分割损伤；数值通过不能代表遮挡可用。",
        "B004": "人物随自车运动显著放大；固定裁剪保留尺度变化，五个抽样时刻骨架大致对齐。",
    }
    cards, validation = [], []
    for row in rows:
        source = Path(row["sample_root"])
        native = json.loads((source / "native.json").read_text())
        observed = source / "observed_native.mp4"
        if not observed.exists():
            encode(observed, [Image.open(f["path"]).convert("RGB") for f in native["frames"]], 10)
        videos = [(observed, row["native_frame_count"]),
                  (source / "inputs/main.mp4", row["output_slots"]),
                  (source / "motion_pair/projection_comparison.mp4", row["output_slots"])]
        for path, expected in videos:
            with av.open(str(path)) as con:
                decoded = sum(1 for _ in con.decode(video=0))
            assert decoded == expected, (path, decoded, expected)
            validation.append({"path": str(path), "decoded_frames": decoded, "expected_frames": expected})
        key = row["review_id"]
        players = "".join(
            f'<div><p>{title}</p><video controls preload="metadata" src="{html.escape(str(path.relative_to(ROOT)))}"></video></div>'
            for (path, _), title in zip(videos, ["原生观测裁剪（10fps）", "实际模型输入（25fps）", "实际输入＋GVHMR骨架（25fps）"]))
        cards.append(f'<section><h2>{key} · {row["split"]}/{row["scene"]} · camera {row["camera_index"]}</h2>'
                     f'<p>{notes[key]}</p><div class="videos">{players}</div></section>')
    document = '''<!doctype html><meta charset="utf-8"><title>Waymo 基线输入与骨架试运行</title>
<style>body{font:16px sans-serif;background:#17191d;color:#eee;margin:24px}section{padding:18px;background:#252830;margin:20px 0}.videos{display:flex;gap:12px}.videos>div{flex:1;min-width:0}video{width:100%;max-height:480px}p{line-height:1.6}a{color:#9bd}</style>
<h1>基线输入与骨架：四个工程试运行样本</h1>
<p>这不是全量筛选结果，也不是4DAnyone生成视频。每条视频完整解码；视觉检查仅覆盖五个固定时刻，未宣称完整动作质量通过。绿色为GVHMR投影骨架，红色为二维检测点。无CondMDI、RoHM或O2。24视角骨架条件单独准备，不运行多视角RGB生成。</p>
<p>两种视频帧率不同，但覆盖同一源片段；RIFE插值帧不是新的真实观测。原生观测以10fps播放，精确曝光时间保存在各输入包。</p>'''
    (ROOT / "index.html").write_text(document + "".join(cards))
    write(ROOT / "pilot/input_skeleton_validation.json",
          {"videos": validation, "review_notes": notes, "visual_scope": "five fixed frames per clip",
           "quality_accepted": False, "generation_started": False})


if __name__ == "__main__":
    main()
