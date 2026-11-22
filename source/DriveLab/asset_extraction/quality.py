"""Insertion-oriented audit and one provenance-preserving, FLUX-only shape retry."""
from copy import deepcopy
from collections import Counter
import asyncio
import json
import math
from pathlib import Path
from jsonschema import ValidationError

from .export import AUDIT_SCHEMA, _schema, _vlm_messages, _verdict, _needs_heading, FACING_CATEGORIES
from .generation import digest, sha256, generate
from .io_utils import read_json, write_json
from .vlm import AsyncOpenAIClient

POLICY = 'insertion_quality_shape_retry_v2'
SEVERE = ['major_missing_geometry', 'severe_deformation', 'wrong_category',
          'large_floaters', 'unrenderable', 'functional_semantics_wrong']
SCHEMA = deepcopy(AUDIT_SCHEMA)
SCHEMA['properties'].update(
    severe_defects=dict(type='array', uniqueItems=True, items={'enum': SEVERE}),
    proportion_status={'enum': ['acceptable', 'uncertain', 'severely_mismatched']},
    proportion_reason=dict(type='string', maxLength=240))
SCHEMA['required'] = list(SCHEMA['properties'])
FLUX_SCHEMA = _schema(dict(
    proportion_status={'enum': ['acceptable', 'uncertain', 'severely_mismatched']},
    reason=dict(type='string', maxLength=240)))


def proportion_guard(row, canonical, audit):
    """Broad, documented shape limits; never use a partial crop as a length measurement."""
    size = canonical['size_xyz']
    if len(size) != 3 or any(not math.isfinite(v) or v <= 0 for v in size):
        raise ValueError('Invalid canonical extents')
    length_width = max(size[:2]) / min(size[:2])
    description = ' '.join(row.get('shape_expectation_prompts', row['prompts'])).lower()
    category = row['category'].lower()
    minimum, reason = None, None
    # Deliberately broad engineering limits for gross truncation, not metric size estimates.
    if category == 'bus':
        if any(w in description for w in ['minibus', 'mini bus', 'short bus']):
            minimum, reason = 1.8, 'input describes a short/minibus'
        elif 'articulated' in description:
            minimum, reason = 4.0, 'input describes an articulated bus'
        elif any(w in description for w in ['public transit', 'full-size', 'full size', 'long, rectangular']):
            minimum, reason = 3.0, 'input describes a long/full-size transit bus'
        else:
            minimum, reason = 2.2, 'bus subtype unspecified; only grossly short bodies rejected'
    elif category == 'truck' and ('semi-truck' in description or 'semi truck' in description) and 'trailer' in description:
        minimum, reason = 3.2, 'input describes a semi-truck with trailer, not a short box truck'
    diagnostic = dict(length_width=length_width, minimum=minimum, basis=reason,
                      protocol='Broad engineering bounds v1; no sensor dimensions or source crop ratio used',
                      triggered=bool(audit['category_match'] and minimum and length_width < minimum))
    updated = deepcopy(audit)
    if diagnostic['triggered']:
        updated.update(keep=False, proportion_status='severely_mismatched',
            proportion_reason=f'Generated L/W={length_width:.3f} < broad subtype minimum {minimum}: {reason}')
    return updated, diagnostic


def audit_prompt(row, canonical):
    size = canonical['size_xyz']
    ratio = [round(v / max(size[1], 1e-8), 3) for v in size]
    return (
        f'Inspect one generated rigid asset, expected category {row["category"]}. '
        f'Input-only description used before reconstruction: {json.dumps(row.get("shape_expectation_prompts", row["prompts"]))}. '
        'This description is evidence of intended subtype, not proof that the source was segmented correctly. '
        'The LEFT panel is the source observation and the RIGHT panel is 16 orbit renders of ONE 3D asset. '
        'Judge practical usability for insertion into a driving scene, not exact instance replication. '
        'Accept complete, plausible assets despite minor blur, unreadable/invented plate or brand text, '
        'small decal changes, minor texture noise, slight local detail variation, or small isolated artifacts. '
        'Source blur, missing source views, cleaner renders, or uncertainty about unseen details are NOT defects '
        'of generated geometry. Different sides may have different markings, dirt or moss; this is not temporal '
        'inconsistency. Do not hallucinate holes or missing parts. Do not require rotational symmetry. '
        'Record appearance deviations in defects/identity_consistent without lowering keep solely for them. '
        'Reject major missing body sections, severe deformation, wrong category, large detached floaters, '
        'unrenderable output, or wrong functional semantics such as an incorrect traffic-sign instruction or '
        'missing signal lamps. A source that is actually a building/pole fragment rather than the stated object '
        'is wrong_category even if the generated object looks nice. But ordinary subtype differences within '
        'a category are not automatically wrong_category. '
        'Separately check gross proportions: a long bus reconstructed as half a bus, an extremely short truck '
        'cargo body, or grossly implausible length/width/height is severely_mismatched and must not pass. '
        'Use subtype-aware reasoning: minibus, full-size bus and articulated bus differ. Do NOT infer the full '
        'object length from a truncated source crop. Do NOT invent a precise ground-truth length. '
        'Do not relabel a compressed full-size transit bus as a minibus just to make it pass. '
        'A semi-truck with a trailer reduced to a tiny box directly behind a large sleeper cab has a gross '
        'structural/proportion mismatch even if all rendered surfaces look clean. A complete plausible small '
        'box truck is different from a semi-truck whose trailer was truncated. Compare the intended assembly. '
        'Use uncertain when subtype/proportions cannot be established; do not equate uncertainty with failure. '
        f'Canonical normalized extent ratio x:y:z is {ratio}; this is generated geometry, NOT metric truth. '
        'Set keep=true for usable assets with only minor defects and quality_score>=4; false for severe issues. '
        'severe_defects must contain only clearly observed severe issues; otherwise []. '
        'proportion_reason must state visible evidence for a severe mismatch, otherwise concise uncertainty or OK. '
        'Estimate positive physical dimensions in meters from subtype; low confidence when uncertain. '
        'Keep geometry usability separate from metric-scale/heading confidence. Return concise JSON only; '
        'at most 5 defects, each <=160 characters; sub_type <=80 characters; object_desc <=160 characters; '
        'usable_as at most 5 short items. Do not copy long input descriptions. No speculation about microscopic details.')


async def call_audit(client, messages, schema, evidence, index=0):
    """One explicit format repair, preserving the invalid response instead of inventing values."""
    try:
        return await client.call_json(messages, client_index=index, response_schema=schema)
    except ValidationError as exc:
        write_json(evidence, dict(error=str(exc), invalid_response=exc.instance))
        correction = dict(role='user', content='Your previous response failed JSON schema validation: '
            + str(exc)[:800] + '. Return a fresh concise JSON satisfying all fields and length bounds. '
            'Use short descriptions; preserve your evidence-based quality judgment.')
        return await client.call_json([*messages, correction], client_index=index, response_schema=schema)


async def reassess(state, generation, config, destination):
    """Reuse immutable renders for old rejects; never regenerate merely to change the audit."""
    destination = Path(destination)
    signature = digest(dict(state=state, generation=generation['signature'],
                            code=sha256(__file__), model=config.vlm.model))
    if destination.exists():
        cached = read_json(destination)
        if cached['policy_signature'] != signature:
            raise ValueError('Quality review inputs changed; use a new version directory')
        return cached
    result = deepcopy(state)
    client = AsyncOpenAIClient(config, None)
    semaphore = asyncio.Semaphore(config.vlm.concurrency)
    jobs = {r['obj_id']:r for r in generation['assets']}

    async def one(uid, record):
        if 'entry' not in record or record['status'] in {'keep', 'review'}:
            return  # Published packages are immutable; historical acceptance stays traceable.
        entry = record['entry']; quality = entry['quality']
        async with semaphore:
            audit = await call_audit(client, _vlm_messages(audit_prompt(jobs[uid], entry['canonical']),
                quality['audit_image'], SCHEMA), SCHEMA, destination.parent/(uid+'_format_error.json'))
        model_audit = deepcopy(audit)
        audit, guard = proportion_guard(jobs[uid], entry['canonical'], audit)
        record['previous_quality_verdict'] = deepcopy(record)
        quality.update(previous_audit=quality['audit'], audit=audit, model_audit=model_audit,
                       geometry_guard=guard, admission_policy=POLICY,
                       policy_code_sha256=sha256(__file__))
        status = _verdict(audit, quality['rendering'], quality['heading_decision'],
            _needs_heading(entry['category']), semantic_face=entry['category'] in FACING_CATEGORIES)
        record.update(status=status, visual_curation_status=status)
        entry.update(status=status, quality_score=audit['quality_score'], defects=audit['defects'],
                     category_match=audit['category_match'], symmetry_360_ok=audit['symmetry_360_ok'])
        # Existing scale/heading and transforms are not changed by appearance re-auditing.
        quality['reassessment_scale_not_applied'] = audit['scale']
        if entry['scale']['scale_confidence'] == 'low' and status == 'keep':
            record['status'] = entry['status'] = record['visual_curation_status'] = 'review'

    try:
        await client.check_endpoints()
        await asyncio.gather(*(one(uid, record) for uid,record in result['assets'].items()))
    finally:
        await client.close()
    result.update(counts=dict(queries=len(result['assets']), **Counter(r['status'] for r in result['assets'].values())),
                  policy_signature=signature, admission_policy=POLICY)
    write_json(destination, result)
    return result


def retry_bundle(original, row, regenerate_flux, subtype):
    job = {k:deepcopy(v) for k,v in row.items() if k not in
           {'stages','status','raw_path','raw_sha256','raw_up_axis','gaussian_count','stage_attempts'}}
    job.update(item_index=0, real_images=[], input_mode='flux_only',
        shape_expectation_prompts=deepcopy(row.get('shape_expectation_prompts',row['prompts'])),
        real_view_selection=dict(policy=POLICY, reason='Gross proportion mismatch; all real reconstruction inputs removed',
                                 original_selection=row['real_view_selection'], original_real_count=len(row['real_images'])))
    if regenerate_flux:
        suffix = (f' Show ONE COMPLETE {row["category"]}, subtype {subtype}, with plausible full-body proportions. '
                  'The reference may show only a partial object; do not use the crop boundary as the object boundary. '
                  'Restore the whole body and all main sections, leave margin around it, avoid foreshortened side views. '
                  'Retain visible color cues but do not preserve truncation. Do not invent extra attached objects.')
        job['prompts'] = [p + suffix for p in row['prompts']]
        job['flux_seed_index'] = row.get('flux_seed_index', row['item_index']) + 100000
    config = dict(original['config'], seed=original['config']['seed'] + 1)
    signature = digest(dict(parent=original['signature'], job=job, config=config, policy_sha=sha256(__file__)))
    return dict(schema_version=1, signature=signature, config=config, assets=[job])


def needs_shape_retry(record):
    audit = record.get('entry', {}).get('quality', {}).get('audit', {})
    return (audit.get('proportion_status') == 'severely_mismatched'
            and audit.get('category_match') is True
            and 'wrong_category' not in audit.get('severe_defects', []))


async def repair_shapes(generation_path, state, config, work, *, renderer_python, gpus):
    """At most one no-real attempt per query; original and retry outputs stay separate."""
    from .export import curate
    work = Path(work); work.mkdir(parents=True, exist_ok=True)
    generation = read_json(generation_path)
    signature = digest(dict(generation_sha=sha256(generation_path), state=state, policy_sha=sha256(__file__)))
    receipt = work/'completed.json'
    if receipt.exists():
        saved = read_json(receipt)
        if saved['signature'] != signature:
            raise ValueError('Shape retry inputs changed; use a new attempt directory')
        return read_json(work/'curation.json'), read_json(work/'selected_generation.json')
    final, selected = deepcopy(state), deepcopy(generation)
    client = AsyncOpenAIClient(config, None)
    network = asyncio.Semaphore(config.vlm.concurrency)
    reconstruction = asyncio.Semaphore(max(1, min(2, len(gpus))))

    async def one(row):
        uid = row['obj_id']; record = state['assets'][uid]
        if not needs_shape_retry(record):
            return
        target = work/uid; target.mkdir(exist_ok=True)
        plan_path = target/'plan.json'
        if plan_path.exists():
            plan = read_json(plan_path)
        else:
            # Check generated views themselves, not the partial real crop.
            async def check(image):
                async with network:
                    return await call_audit(client, _vlm_messages(
                        f'Inspect this standardized generated {row["category"]} view. Is the whole object present '
                        'with plausible subtype-specific proportions? Only obvious cutoff, missing large body '
                        'sections or grossly compressed length is severely_mismatched. An end-on view alone '
                        'cannot establish length: use uncertain. Ignore logos, textures and minor defects. '
                        'Never require all buses to have identical length. Return JSON.', image['path'], FLUX_SCHEMA),
                        FLUX_SCHEMA, target/(Path(image['path']).stem+'_format_error.json'))
            checks = await asyncio.gather(*(check(x) for x in row['stages']['flux']['images']))
            # Uncertain views are regenerated too: do not feed an unverified defective pair again.
            guard_trigger = record['entry']['quality'].get('geometry_guard',{}).get('triggered',False)
            regen = guard_trigger or any(x['proportion_status']=='severely_mismatched' for x in checks) or not any(
                x['proportion_status']=='acceptable' for x in checks)
            plan = dict(uid=uid, checks=checks, regenerate_flux=regen, removed_real_count=len(row['real_images']),
                        trigger=record['entry']['quality']['audit'], geometry_guard=guard_trigger,
                        max_reconstruction_retries=1)
            write_json(plan_path, plan)
        bundle = retry_bundle(generation, row, plan['regenerate_flux'], 'as described in the original input')
        gen_dir = target/'generation'; gen_dir.mkdir(exist_ok=True)
        gen_path = gen_dir/'generation.json'
        if not plan['regenerate_flux'] and not gen_path.exists():
            cached = deepcopy(row['stages']['flux']); cached['item_index']=0
            write_json(gen_path, dict(schema_version=1, signature=bundle['signature'], config=bundle['config'],
                assets=[dict(bundle['assets'][0], stages={'flux':cached}, status='pending')]))
        async with reconstruction:
            regenerated = await generate(bundle, gen_dir, gpus)
            audit_path = target/'curation/curation.json'
            retry = read_json(audit_path) if audit_path.exists() else {}
            if 'counts' not in retry:
                retry = await curate(gen_path, target/'library', config, renderer_python=renderer_python,
                                     gpus=gpus, work_dir=target/'curation')
        chosen = deepcopy(retry['assets'][uid])
        chosen['shape_retry_history'] = dict(initial_generation=str(generation_path), initial_record=record,
            plan=str(plan_path), retry_generation=str(gen_path), attempt=1, query_identity_unchanged=True)
        if 'entry' in chosen:
            chosen['entry']['provenance']['shape_retry'] = dict(plan=str(plan_path), attempt=1,
                original_raw_sha256=row.get('raw_sha256'), original_real_count=len(row['real_images']),
                real_count=0, regenerated_flux=plan['regenerate_flux'])
        # A failed retry remains failed/rejected; never publish the malformed original instead.
        final['assets'][uid] = chosen
        index = next(i for i,x in enumerate(selected['assets']) if x['obj_id']==uid)
        selected['assets'][index] = dict(regenerated['assets'][0], original_item_index=row['item_index'],
                                          selected_attempt=1)

    try:
        results = await asyncio.gather(*(one(row) for row in generation['assets']), return_exceptions=True)
        for result in results:
            if isinstance(result, BaseException):
                raise result
    finally:
        await client.close()
    final['counts'] = dict(queries=len(final['assets']), **Counter(r['status'] for r in final['assets'].values()))
    selected['counts'] = dict(queries=len(selected['assets']), generated=sum(r['status']=='generated' for r in selected['assets']))
    selected['selection_policy'] = POLICY
    write_json(work/'curation.json', final)
    write_json(work/'selected_generation.json', selected)
    write_json(receipt, dict(signature=signature, queries=len(final['assets']), counts=final['counts'],
                            retries=sum(bool(r.get('shape_retry_history')) for r in final['assets'].values())))
    return final, selected
