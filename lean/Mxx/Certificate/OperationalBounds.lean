import Mxx.Certificate.OperationalBounds.Core
import Mxx.Certificate.OperationalBounds.DirectCarrier
import Mxx.Certificate.OperationalBounds.IndexedEngine
import Mxx.Certificate.OperationalBounds.Progress
import Mxx.Certificate.OperationalBounds.Evaluation
import Mxx.Certificate.OperationalBounds.Fixtures

/-! # Linear operational hard-bound estimator

This facade preserves the historical module path. The implementation is split by responsibility:
core syntax and polynomials, direct indexed storage, indexed transport, evaluation, and fixtures.
-/
