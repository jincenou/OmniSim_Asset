from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[1]
path=ROOT/'inputs/random10_additional_matrix.json';rows=json.loads(path.read_text())
for r in rows:
 run=ROOT/'results'/r['case_id']/f"{r['variant']}_seed{r['seed']}"
 if (run/'evaluation/dino_metrics.json').exists() and (run/'evaluation/clip_metrics.json').exists():r['status']='evaluated'
 elif (run/'asset/heading_review.json').exists():r['status']='heading_verified_evaluation_pending'
 elif (run/'asset/turntable.mp4').exists():r['status']='rendered_heading_review_pending'
 elif (run/'raw.ply').exists():r['status']='generated_export_pending'
 elif (run/'failure.txt').exists():r['status']='failed'
 elif run.exists():r['status']='running'
path.write_text(json.dumps(rows,indent=2))
print([(r['variant'],r['seed'],r['status']) for r in rows])
