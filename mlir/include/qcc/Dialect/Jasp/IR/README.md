# Jasp IR

The jasp dialect is not owned by us. We have copied it from
[this folder](https://github.com/eclipse-qrisp/Qrisp/tree/main/src/qrisp/jasp/mlir/dialect_definition)
from within the qrisp repository.

In order to upgrade copy the files over here and possibly add include guards (or pragma once).

TODO: Agree on a single source of truth for the dialect definition.
See also issue https://github.com/FullStaQD/compiler/issues/16.

Note that qrisp defines the dialects twice: On the one hand in the tablegen files already mentioned.
Then on the other hand in xdsl (see e.g.
[xdsl_dialect.py](https://github.com/eclipse-qrisp/Qrisp/blob/main/src/qrisp/jasp/mlir/xdsl_dialect.py).
The former is needed for JAX, the latter they use to implement passes in python without the full MLIR dependency.
