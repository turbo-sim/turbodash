"""Array backend seam; retain current NumPy execution.

A future JAX migration also needs traceable validators and context managers; changing
this import alone does not promise JIT or automatic differentiation.
"""

import numpy as jnp

# Future backend: import jax.numpy as jnp
