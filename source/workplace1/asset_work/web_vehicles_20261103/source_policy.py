"""Auditable source corrections and conservative whole-vehicle input checks."""
import json
from PIL import Image

def effective_review(review, folder):
    result = dict(review)
    override = folder / 'review_override.json'
    if override.exists():
        correction = json.loads(override.read_text())
        if correction['source_sha256'] != result['source_sha256']:
            raise ValueError('Source override hash mismatch')
        result.update(correction)
    box = result['bbox_xyxy_1000']
    if result['accepted'] and (min(box[:2]) <= 5 or max(box[2:]) >= 995):
        result.update(accepted=False, reason='Whole-vehicle box touches source frame; select an easier complete photograph.', source_edge_guard=True)
    return result

def mask_frame_contact(path):
    mask = Image.open(path).convert('L')
    w, h = mask.size
    border = [mask.getpixel((x, y)) for y in (0, h-1) for x in range(w)]
    border += [mask.getpixel((x, y)) for x in (0, w-1) for y in range(1, h-1)]
    touched = sum(v > 0 for v in border)
    return dict(contact_fraction=touched / len(border), touched_pixels=touched,
                perimeter_pixels=len(border), measurement='SAM mask on original source image perimeter')
