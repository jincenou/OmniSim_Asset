"""Numerically stable evaluation of the unchanged published vertex-regressor weights."""
import inspect,hashlib

def build(original):
 source=inspect.getsource(original)
 for space in ['global','incam']:
  before=f'(vertices_{space}[:, support] * weights[None, :, :, None]).sum(dim=2)'
  after=f'(vertices_{space}[:, support].double() * weights.double()[None, :, :, None]).sum(dim=2).to(vertices_{space}.dtype)'
  assert before in source
  source=source.replace(before,after)
 scope=dict(original.__globals__);exec(compile(source,'<random8 stable vertex regression>','exec'),scope);function=scope[original.__name__];function.random8_precision_source_sha256=hashlib.sha256(source.encode()).hexdigest();return function
