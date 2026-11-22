"""Explicit live integration check; never imported by production workers."""
import asyncio
import base64
import json
from pathlib import Path
import time
from openai import AsyncOpenAI

ROOT=Path(__file__).resolve().parents[1]

async def main():
    async with AsyncOpenAI(base_url='http://127.0.0.1:8086/v1',api_key='EMPTY',max_retries=0,timeout=60) as client:
        models=await client.models.list()
        assert any(x.id=='Qwen/Qwen3-VL-8B-Instruct' for x in models.data)
        image=ROOT/'validation/end_to_end_smoke/generation/asset_00000/flux_0.png'
        uri='data:image/png;base64,'+base64.b64encode(image.read_bytes()).decode()
        async def one(index):
            messages=[{'role':'user','content':[{'type':'text','text':'Return a JSON object with keys color and object describing the main object in this image.'},
                       {'type':'image_url','image_url':{'url':uri}}]}] if index==0 else [{'role':'user','content':'Return JSON only: {"ok": true}'}]
            started=time.monotonic()
            raw=await client.chat.completions.with_raw_response.create(model='Qwen/Qwen3-VL-8B-Instruct',messages=messages,
                temperature=0,max_tokens=100,response_format={'type':'json_object'})
            result=raw.parse();value=json.loads(result.choices[0].message.content)
            assert result.choices[0].finish_reason=='stop'
            return dict(index=index,backend=raw.headers.get('x-qwen-backend'),seconds=time.monotonic()-started,response=value)
        records=await asyncio.gather(*(one(i) for i in range(6)))
        assert {x['backend'] for x in records}=={'8083','8084'},records
        stream=await client.chat.completions.create(model='Qwen/Qwen3-VL-8B-Instruct',messages=[{'role':'user','content':'Return JSON only: {"ok":true}'}],
            response_format={'type':'json_object'},max_tokens=30,temperature=0,stream=True)
        chunks=[]
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:chunks.append(chunk.choices[0].delta.content)
        streamed=json.loads(''.join(chunks));assert streamed['ok'] is True
        report=dict(checked=time.time(),records=records,streaming=streamed,status='passed',
            scope='Actual image input, forced JSON, concurrent distribution across both new replicas, and SSE compatibility')
        (ROOT/'vlm_cluster/integration_test.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2))

if __name__=='__main__':asyncio.run(main())
