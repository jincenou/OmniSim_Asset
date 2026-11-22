"""Detached generation and technical validation; never auto-approve visual quality."""
from pathlib import Path
import argparse, json, os, shutil, subprocess, sys, time, traceback

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('identity', choices=['QWH0002', 'QWH0003', 'QWH0004'])
    identity = parser.parse_args().identity
    status = ROOT / 'jobs' / f'{identity}_background.json'
    status.parent.mkdir(exist_ok=True)
    state = dict(identity=identity, pid=os.getpid(), started=time.time(),
                 final_figure_ready=False, target_total_assets=5)

    def save(stage, **extra):
        state.update(stage=stage, updated=time.time(), **extra)
        temp = status.with_suffix('.tmp')
        temp.write_text(json.dumps(state, indent=2) + '\n')
        temp.replace(status)

    try:
        for stage in ['generate', 'validate']:
            save(stage)
            subprocess.run([sys.executable, str(ROOT/'scripts/maintained_stage.py'),
                            'frontend', stage, identity], check=True)
        # Keep the expensive generated videos on persistent storage when they fit.
        # The root reserve is checked immediately before each copied file.
        source = ROOT/'outputs'/identity/'generation'
        destination = ROOT/'persisted_generation'/identity
        files = [p for p in source.rglob('*') if p.is_file()]
        total = sum(p.stat().st_size for p in files)
        backed_up = False
        if total < 2 * 2**30 and shutil.disk_usage(ROOT).free > total + 10 * 2**30:
            save('persisting_generated_videos', generation_bytes=total)
            for file in files:
                if shutil.disk_usage(ROOT).free < file.stat().st_size + 10 * 2**30:
                    raise RuntimeError('Persistent storage reserve reached during backup')
                target = destination/file.relative_to(source)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(file, target)
            backed_up = True
        save('waiting_frontend_visual_review', frontend_complete=True,
             technical_validation=str(ROOT/'reports'/f'{identity}_frontend_validation.json'),
             persistent_generation=str(destination) if backed_up else None,
             next_action='Review generated body, identity and motion before Grid4D reconstruction. No automatic visual approval.',
             completed=time.time())
    except BaseException as error:
        save('failed', error=str(error), traceback=traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
