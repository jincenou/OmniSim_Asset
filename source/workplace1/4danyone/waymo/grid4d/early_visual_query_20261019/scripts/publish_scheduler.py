"""Only completed comparisons; CPU subprocesses, no training permission changes."""
import asyncio,json,os,shutil,time
from pathlib import Path
NEW=Path(__file__).resolve().parents[1]
async def command(script,args,log):
    with (NEW/'logs'/log).open('x') as f:
        p=await asyncio.create_subprocess_exec('bash',str(NEW/'scripts/run.sh'),script,*args,stdout=f,stderr=asyncio.subprocess.STDOUT)
        return await p.wait()
async def main():
    plan=json.loads((NEW/'PLAN.json').read_text());running={};failed={}
    while True:
        for label,task in list(running.items()):
            if task.done():
                rc=task.result()
                if rc:failed[label]=rc
                del running[label]
        for label in plan['labels']:
            if label in failed or label in running or (NEW/'comparison'/label/'complete.json').exists():continue
            if len(running)>=2 or shutil.disk_usage(NEW).free<5.5*2**30:continue
            if all((NEW/'results'/f'{label}_{a}_32999/parts.json').exists() for a in plan['arms']):running[label]=asyncio.create_task(command('media',['--label',label],f'media_{label}.log'))
        status=dict(state='publishing_complete_cases_only',pid=os.getpid(),running=list(running),failed=failed,complete=[l for l in plan['labels'] if (NEW/'comparison'/l/'complete.json').exists()]);tmp=NEW/'PUBLISH_STATUS.writing';tmp.write_text(json.dumps(status,indent=2)+'\n');tmp.replace(NEW/'PUBLISH_STATUS.json')
        if (NEW/'post_queue.complete.json').exists() and len(status['complete'])==3:
            rc=await command('analyze',[],'analyze.log');assert rc==0
            html=['<!doctype html><meta charset="utf-8"><title>EVQ完整消融</title><style>body{max-width:1500px;margin:auto;font-family:sans-serif;background:#eee}video,img{width:100%}a{margin:8px}</style><h1>三人 × 四组 EVQ 完整终态</h1><p>生成开发参考；R0/R1/R2/R3 同协议32999更新。历史 A/F/HUST 算力不同。数值检查与人工视觉结论见报告。</p><a href="../METRICS.md">指标</a><a href="../ASSETS.json">资产</a><a href="../gate_results.json">保护线</a>']
            for label in plan['labels']:
                html.append('<h2>'+label+'</h2>');meta=json.loads((NEW/'comparison'/label/'complete.json').read_text())
                for f in meta['files']:
                    p=Path(f['path']).relative_to('comparison');html.append(f'<p>{p.name}：'+', '.join(f['columns'])+f'</p><video controls preload="none" src="{p}"></video>')
                if label in ['N06','N11']:html.append(f'<p>固定历史坏窗口，参考定义左侧裁剪</p><img src="{label}/fixed_bad_window_left.jpg">')
            (NEW/'comparison/index.html').write_text('\n'.join(html));(NEW/'publication.complete.json').write_text(json.dumps(dict(passed=True,manual_visual_review='pending',goal_complete=False))+'\n');return
        await asyncio.sleep(25)
if __name__=='__main__':asyncio.run(main())
