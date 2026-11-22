"""Prioritize non-car vehicles without changing identities or frozen chunks."""
from collections import deque, Counter
from copy import deepcopy

POLICY = 'noncar_three_to_one_within_scene_v1'


def prioritize(selection, prefix):
    rows = selection['rows']
    if not 0 <= prefix <= len(rows):
        raise ValueError('Protected prefix outside selection')
    if selection.get('vehicle_priority', {}).get('policy') == POLICY:
        return selection
    if len({r['obj_id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate query identities in selection')
    categories = ('bus', 'truck', 'van')
    queues = {c: deque(r for r in rows[prefix:] if r['category'] == c) for c in categories}
    cars = deque(r for r in rows[prefix:] if r['category'] == 'car')
    other = [r for r in rows[prefix:] if r['category'] not in (*categories, 'car')]
    # Round-robin the three preferred classes, borrowing empty classes' slots.
    preferred = deque()
    while any(queues.values()):
        for c in categories:
            if queues[c]:
                preferred.append(queues[c].popleft())
    tail = []
    while preferred or cars:
        for _ in range(3):
            if preferred:
                tail.append(preferred.popleft())
        if cars:
            tail.append(cars.popleft())
    tail.extend(other)
    result = deepcopy(selection)
    result['rows'] = deepcopy(rows[:prefix] + tail)
    route_map = {r['obj_id']: r for r in selection['routes']}
    result['routes'] = [deepcopy(route_map[r['obj_id']]) for r in result['rows']]
    assert Counter(r['obj_id'] for r in result['rows']) == Counter(r['obj_id'] for r in rows)
    assert result['rows'][:prefix] == rows[:prefix]
    result['vehicle_priority'] = dict(
        policy=POLICY, protected_prefix=prefix,
        preferred_categories=list(categories), preferred_per_car=3,
        remaining_counts=dict(Counter(r['category'] for r in tail)),
        limitation='Ordering only; exhausted classes borrow capacity. No duplication, quota or guaranteed final class ratio.')
    return result
