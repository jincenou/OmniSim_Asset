"""Apply live category scope while validating immutable resumed chunk identities."""
def scoped_rows(rows, allowed, frozen=None, finish_existing=False):
    selected=[r for r in rows if not allowed or r['category'] in allowed]
    deferred=[r for r in rows if allowed and r['category'] not in allowed]
    if frozen is not None:
        expected=rows if finish_existing else selected
        if frozen != expected:
            raise ValueError('Frozen chunk differs from its ordered production slice and category scope')
        return frozen, [] if finish_existing else deferred
    return selected,deferred
