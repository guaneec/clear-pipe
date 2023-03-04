from contextlib import contextmanager


@contextmanager
def patched(victim, prop, new_prop):
    orig = getattr(victim, prop)
    setattr(victim, prop, new_prop)
    yield
    setattr(victim, prop, orig)
