"""Opt-in fixed rider/vehicle contract; ordinary motorcycles keep their old route."""
from copy import deepcopy

POLICY_VERSION = 'rider_vehicle_rigid_v1'
KINDS = {'motorcycle', 'bicycle'}
PROPULSION = {'electric_confirmed', 'electric_probable', 'unknown', 'non_electric'}


def assembly(record):
    value = record.get('assembly')
    if value is None:
        return None
    if not isinstance(value, dict) or value.get('asset_kind') != 'rider_vehicle_rigid':
        raise ValueError('Unknown assembly contract')
    required = {'fixed_pose', 'rider_count', 'vehicle_subtype', 'propulsion_label',
        'propulsion_evidence', 'pose_usage', 'pose_description', 'rider_appearance',
        'vehicle_appearance', 'hand_handlebar_contact', 'seat_contact', 'foot_support',
        'source_id', 'shot_id', 'identity', 'pose_cluster_id', 'internal_articulation',
        'wheel_animation', 'anchor_review'}
    missing = required - value.keys()
    if missing:
        raise ValueError('Incomplete assembly metadata: ' + ', '.join(sorted(missing)))
    if record.get('category') not in KINDS:
        raise ValueError('Rider assemblies require motorcycle or bicycle category')
    if (value['fixed_pose'] is not True or type(value['rider_count']) is not int
            or value['rider_count'] != 1 or value['internal_articulation'] is not False
            or value['wheel_animation'] is not False):
        raise ValueError('Require exactly one fixed rider with no internal articulation')
    if value['propulsion_label'] not in PROPULSION or value['pose_usage'] not in {'riding', 'stopped', 'uncertain'}:
        raise ValueError('Invalid propulsion or pose usage')
    if value['propulsion_label'] == 'electric_confirmed' and not value['propulsion_evidence']:
        raise ValueError('Confirmed electric requires evidence')
    for name in ['source_id', 'shot_id', 'identity', 'pose_cluster_id']:
        if not isinstance(value[name],str) or not value[name].strip():
            raise ValueError('Missing assembly identity: ' + name)
    review=value['anchor_review']
    if not isinstance(review,dict) or not all(review.get(k) is True for k in ['accepted','complete_assembly','exactly_one_rider','seated']):
        raise ValueError('Assembly requires a reviewed complete seated rider anchor')
    if not review.get('source'):
        raise ValueError('Anchor review needs a reviewer/source')
    return deepcopy(value)


def is_assembly(record):
    return assembly(record) is not None


def uses_h(record):
    from .real_views import VEHICLES
    return is_assembly(record) or record['category'].lower() in VEHICLES


def prompt(record, view='same observed view'):
    a = assembly(record)
    if a is None:
        raise ValueError('No rider assembly')
    return (f'Create a clean {view} of the same seated rider and the same {a["vehicle_subtype"]} '
        'together as one complete asset. Preserve exactly one rider, the same vehicle and the fixed seated pose. '
        f'Preserve the observed rider appearance: {a["rider_appearance"]}. '
        f'Preserve the observed vehicle appearance: {a["vehicle_appearance"]}. '
        f'Preserve the posture: {a["pose_description"]}. '
        f'Preserve hand-to-handlebar contact: {a["hand_handlebar_contact"]}; '
        f'hip-to-seat contact: {a["seat_contact"]}; foot support: {a["foot_support"]}. '
        'Keep head, torso, hands, feet, handlebars, frame and wheels inside the frame. '
        'Do not add passengers, headwear, logos or unobserved accessories. '
        'Do not remove the person or vehicle, change the pose, detach limbs, or duplicate wheels. '
        'Use a plain white background with comfortable margin. Preserve plausible self-occlusion.')


def union_masks(rider, vehicle):
    import numpy as np
    a,b=np.asarray(rider,dtype=bool),np.asarray(vehicle,dtype=bool)
    if a.ndim!=2 or a.shape!=b.shape or not a.any() or not b.any():
        raise ValueError('Both nonempty rider and vehicle masks of equal 2D shape are required')
    return a | b


def validate_reference(record, reference):
    a=assembly(record)
    if a is None:
        return
    review=reference.get('assembly_review',{})
    if (reference.get('identity')!=a['identity'] or reference.get('pose_cluster_id')!=a['pose_cluster_id']
            or not all(review.get(k) is True for k in ['accepted','same_identity','same_pose','complete_assembly','complementary_view'])
            or not review.get('source')):
        raise ValueError('Real reference lacks same-identity, fixed-pose and complementary-view review')


def validate_flux_review(row, flux_result):
    if not is_assembly(row):
        return
    if row.get('assembly', {}).get('admission_policy') == 'user_direct_flux_v1':
        return  # Explicit user opt-in: no source identity/pose admission gate.
    reviews={x['sha256']:x for x in row.get('flux_assembly_reviews',[])}
    for image in flux_result['images']:
        r=reviews.get(image['sha256'],{})
        if not all(r.get(k) is True for k in ['accepted','complete_assembly','same_pose','same_identity','contacts_preserved']) or not r.get('source'):
            raise ValueError('Generated assembly must pass explicit identity/pose/contact review before reconstruction')
