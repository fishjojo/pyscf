from pyscf import __config__
JAXNUMPY = getattr(__config__, "jaxnumpy", False)
if JAXNUMPY:
    import jax

class _Indexable(object):
  """
  see https://github.com/google/jax/blob/97d00584f8b87dfe5c95e67892b54db993f34486/jax/_src/ops/scatter.py#L87
  """
  __slots__ = ()

  def __getitem__(self, index):
    return index

index = _Indexable()

def index_update(a, idx, value):
    if JAXNUMPY:
        a = jax.ops.index_update(a, idx, value)
    else:
        a[idx] = value
    return a

def index_add(a, idx, value):
    if JAXNUMPY:
        a = jax.ops.index_add(a, idx, value)
    else:
        a[idx] += value
    return a

def index_mul(a, idx, value):
    if JAXNUMPY:
        a = jax.ops.index_mul(a, idx, value)
    else:
        a[idx] *= value
    return a
