"""Short four-rank CUDA and NCCL check; this does not train a model."""
import json
import os
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist

rank = int(os.environ['LOCAL_RANK'])
assert os.environ['CUDA_VISIBLE_DEVICES'] == '0,1,2,3'
assert torch.cuda.device_count() == 4
torch.cuda.set_device(rank)
dist.init_process_group('nccl', timeout=timedelta(seconds=60))
try:
    value = torch.tensor([rank + 1.0], device=f'cuda:{rank}')
    dist.all_reduce(value)
    assert value.item() == 10.0
    matrix = torch.randn(128, 128, device=f'cuda:{rank}', dtype=torch.bfloat16)
    product = matrix @ matrix.T
    assert torch.isfinite(product).all().item()
    torch.cuda.synchronize()
    record = {'rank': rank, 'device': f'cuda:{rank}', 'gpu_name': torch.cuda.get_device_name(rank), 'torch': torch.__version__, 'cuda_build': torch.version.cuda, 'nccl': torch.cuda.nccl.version(), 'all_reduce_sum': value.item(), 'bf16_matmul_finite': True}
    path = Path('/mnt/ojc/workplace1/4danyone/research') / f'cuda-check-rank{rank}.json'
    path.write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record), flush=True)
finally:
    dist.destroy_process_group()
