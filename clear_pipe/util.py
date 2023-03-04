from contextlib import contextmanager


@contextmanager
def patched(victim, prop, new_prop):
    orig = getattr(victim, prop)
    setattr(victim, prop, new_prop)
    try:
        yield
    finally:
        setattr(victim, prop, orig)
