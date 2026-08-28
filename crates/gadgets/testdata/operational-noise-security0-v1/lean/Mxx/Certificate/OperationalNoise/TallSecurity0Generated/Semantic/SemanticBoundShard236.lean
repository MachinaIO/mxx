import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard235

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound35954
def owner : Owner := ⟨.program ⟨214⟩, ⟨18876⟩⟩
def transferEvent : Nat := 35954
def frameStart : Nat := 0
def rule : BoundRule := .sum [.transfer 35952, .transfer 35953]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 35952)
      LeftBound35952.bound (LeftBound35952.actual selector witness) := by
  exact .transfer (LeftBound35952.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 35953)
      LeftBound35953.bound (LeftBound35953.actual selector witness) := by
  exact .transfer (LeftBound35953.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound35952.bound, LeftBound35953.bound]
def bound : CoeffClass := .finite ⟨6507539580055607486593200, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound35952.bound, LeftBound35953.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound35952.actual selector witness, LeftBound35953.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound35954

namespace LeftBound35955
def owner : Owner := ⟨.program ⟨214⟩, ⟨18876⟩⟩
def transferEvent : Nat := 35955
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨214⟩, ⟨6387⟩⟩, ⟨.program ⟨214⟩, ⟨17173⟩⟩], []⟩ [⟨.result 643 .coefficient, true, some 1⟩, ⟨.result 2163 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 643 .coefficient)
      LeftAuthority642.bound (LeftAuthority642.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨6387⟩⟩) (rawTerms := some (Proof.Events002.exact643RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority642.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority642.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 2163 .coefficient)
      LeftAuthority2162.bound (LeftAuthority2162.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨17173⟩⟩) (rawTerms := some (Proof.Events008.exact2163RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority2162.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority2162.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority642.bound [LeftAuthority2162.bound]
def bound : CoeffClass := .finite ⟨222230617312560576599880, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority642.bound, LeftAuthority2162.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority642.actual selector witness) * ([LeftAuthority2162.actual selector witness].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.cons (.intro (input1 selector witness)) (.nil))
end LeftBound35955

namespace LeftBound35956
def owner : Owner := ⟨.program ⟨214⟩, ⟨18876⟩⟩
def transferEvent : Nat := 35956
def frameStart : Nat := 0
def rule : BoundRule := .sum [.transfer 35954, .transfer 35955]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 35954)
      LeftBound35954.bound (LeftBound35954.actual selector witness) := by
  exact .transfer (LeftBound35954.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 35955)
      LeftBound35955.bound (LeftBound35955.actual selector witness) := by
  exact .transfer (LeftBound35955.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound35954.bound, LeftBound35955.bound]
def bound : CoeffClass := .finite ⟨6729770197368168063193080, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound35954.bound, LeftBound35955.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound35954.actual selector witness, LeftBound35955.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound35956

namespace LeftBound35957
def owner : Owner := ⟨.program ⟨214⟩, ⟨18876⟩⟩
def transferEvent : Nat := 35957
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨214⟩, ⟨6391⟩⟩, ⟨.program ⟨214⟩, ⟨17229⟩⟩], []⟩ [⟨.result 653 .coefficient, true, some 1⟩, ⟨.result 2171 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 653 .coefficient)
      LeftAuthority652.bound (LeftAuthority652.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨6391⟩⟩) (rawTerms := some (Proof.Events002.exact653RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority652.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority652.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 2171 .coefficient)
      LeftAuthority2170.bound (LeftAuthority2170.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨17229⟩⟩) (rawTerms := some (Proof.Events008.exact2171RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority2170.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority2170.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority652.bound [LeftAuthority2170.bound]
def bound : CoeffClass := .finite ⟨220778129617707239497920, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority652.bound, LeftAuthority2170.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority652.actual selector witness) * ([LeftAuthority2170.actual selector witness].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.cons (.intro (input1 selector witness)) (.nil))
end LeftBound35957

namespace LeftBound35958
def owner : Owner := ⟨.program ⟨214⟩, ⟨18876⟩⟩
def transferEvent : Nat := 35958
def frameStart : Nat := 0
def rule : BoundRule := .sum [.transfer 35956, .transfer 35957]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 35956)
      LeftBound35956.bound (LeftBound35956.actual selector witness) := by
  exact .transfer (LeftBound35956.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 35957)
      LeftBound35957.bound (LeftBound35957.actual selector witness) := by
  exact .transfer (LeftBound35957.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound35956.bound, LeftBound35957.bound]
def bound : CoeffClass := .finite ⟨6950548326985875302691000, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound35956.bound, LeftBound35957.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound35956.actual selector witness, LeftBound35957.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound35958

namespace LeftBound35959
def owner : Owner := ⟨.program ⟨214⟩, ⟨18876⟩⟩
def transferEvent : Nat := 35959
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨214⟩, ⟨6398⟩⟩, ⟨.program ⟨214⟩, ⟨17446⟩⟩], []⟩ [⟨.result 663 .coefficient, true, some 1⟩, ⟨.result 2179 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 663 .coefficient)
      LeftAuthority662.bound (LeftAuthority662.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨6398⟩⟩) (rawTerms := some (Proof.Events002.exact663RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority662.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority662.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 2179 .coefficient)
      LeftAuthority2178.bound (LeftAuthority2178.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨17446⟩⟩) (rawTerms := some (Proof.Events008.exact2179RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority2178.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority2178.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority662.bound [LeftAuthority2178.bound]
def bound : CoeffClass := .finite ⟨216532396355828254122960, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority662.bound, LeftAuthority2178.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority662.actual selector witness) * ([LeftAuthority2178.actual selector witness].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.cons (.intro (input1 selector witness)) (.nil))
end LeftBound35959

namespace LeftBound35960
def owner : Owner := ⟨.program ⟨214⟩, ⟨18876⟩⟩
def transferEvent : Nat := 35960
def frameStart : Nat := 0
def rule : BoundRule := .sum [.transfer 35958, .transfer 35959]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 35958)
      LeftBound35958.bound (LeftBound35958.actual selector witness) := by
  exact .transfer (LeftBound35958.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 35959)
      LeftBound35959.bound (LeftBound35959.actual selector witness) := by
  exact .transfer (LeftBound35959.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound35958.bound, LeftBound35959.bound]
def bound : CoeffClass := .finite ⟨7167080723341703556813960, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound35958.bound, LeftBound35959.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound35958.actual selector witness, LeftBound35959.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound35960

namespace LeftBound35961
def owner : Owner := ⟨.program ⟨214⟩, ⟨18876⟩⟩
def transferEvent : Nat := 35961
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨214⟩, ⟨6407⟩⟩, ⟨.program ⟨214⟩, ⟨17830⟩⟩], []⟩ [⟨.result 673 .coefficient, true, some 1⟩, ⟨.result 2187 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 673 .coefficient)
      LeftAuthority672.bound (LeftAuthority672.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨6407⟩⟩) (rawTerms := some (Proof.Events002.exact673RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority672.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority672.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 2187 .coefficient)
      LeftAuthority2186.bound (LeftAuthority2186.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨17830⟩⟩) (rawTerms := some (Proof.Events008.exact2187RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority2186.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority2186.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority672.bound [LeftAuthority2186.bound]
def bound : CoeffClass := .finite ⟨213251602471649038151400, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority672.bound, LeftAuthority2186.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority672.actual selector witness) * ([LeftAuthority2186.actual selector witness].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.cons (.intro (input1 selector witness)) (.nil))
end LeftBound35961

namespace LeftBound35962
def owner : Owner := ⟨.program ⟨214⟩, ⟨18876⟩⟩
def transferEvent : Nat := 35962
def frameStart : Nat := 0
def rule : BoundRule := .sum [.transfer 35960, .transfer 35961]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 35960)
      LeftBound35960.bound (LeftBound35960.actual selector witness) := by
  exact .transfer (LeftBound35960.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 35961)
      LeftBound35961.bound (LeftBound35961.actual selector witness) := by
  exact .transfer (LeftBound35961.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound35960.bound, LeftBound35961.bound]
def bound : CoeffClass := .finite ⟨7380332325813352594965360, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound35960.bound, LeftBound35961.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound35960.actual selector witness, LeftBound35961.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound35962

namespace LeftBound35963
def owner : Owner := ⟨.program ⟨214⟩, ⟨18876⟩⟩
def transferEvent : Nat := 35963
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨214⟩, ⟨6427⟩⟩, ⟨.program ⟨214⟩, ⟨15526⟩⟩], []⟩ [⟨.result 683 .coefficient, true, some 1⟩, ⟨.result 2195 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 683 .coefficient)
      LeftAuthority682.bound (LeftAuthority682.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨6427⟩⟩) (rawTerms := some (Proof.Events002.exact683RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority682.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority682.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 2195 .coefficient)
      LeftAuthority2194.bound (LeftAuthority2194.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨15526⟩⟩) (rawTerms := some (Proof.Events008.exact2195RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority2194.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority2194.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority682.bound [LeftAuthority2194.bound]
def bound : CoeffClass := .finite ⟨201065796616126235971320, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority682.bound, LeftAuthority2194.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority682.actual selector witness) * ([LeftAuthority2194.actual selector witness].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.cons (.intro (input1 selector witness)) (.nil))
end LeftBound35963

namespace LeftBound35964
def owner : Owner := ⟨.program ⟨214⟩, ⟨18876⟩⟩
def transferEvent : Nat := 35964
def frameStart : Nat := 0
def rule : BoundRule := .sum [.transfer 35962, .transfer 35963]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 35962)
      LeftBound35962.bound (LeftBound35962.actual selector witness) := by
  exact .transfer (LeftBound35962.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 35963)
      LeftBound35963.bound (LeftBound35963.actual selector witness) := by
  exact .transfer (LeftBound35963.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound35962.bound, LeftBound35963.bound]
def bound : CoeffClass := .finite ⟨7581398122429478830936680, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound35962.bound, LeftBound35963.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound35962.actual selector witness, LeftBound35963.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound35964

namespace LeftBound35965
def owner : Owner := ⟨.program ⟨214⟩, ⟨18876⟩⟩
def transferEvent : Nat := 35965
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨214⟩, ⟨6452⟩⟩, ⟨.program ⟨214⟩, ⟨15218⟩⟩], []⟩ [⟨.result 693 .coefficient, true, some 1⟩, ⟨.result 2203 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 693 .coefficient)
      LeftAuthority692.bound (LeftAuthority692.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨6452⟩⟩) (rawTerms := some (Proof.Events002.exact693RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority692.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority692.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 2203 .coefficient)
      LeftAuthority2202.bound (LeftAuthority2202.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨15218⟩⟩) (rawTerms := some (Proof.Events008.exact2203RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority2202.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority2202.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority692.bound [LeftAuthority2202.bound]
def bound : CoeffClass := .finite ⟨187661410175051153573232, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority692.bound, LeftAuthority2202.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority692.actual selector witness) * ([LeftAuthority2202.actual selector witness].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.cons (.intro (input1 selector witness)) (.nil))
end LeftBound35965

namespace LeftBound35966
def owner : Owner := ⟨.program ⟨214⟩, ⟨18876⟩⟩
def transferEvent : Nat := 35966
def frameStart : Nat := 0
def rule : BoundRule := .sum [.transfer 35964, .transfer 35965]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 35964)
      LeftBound35964.bound (LeftBound35964.actual selector witness) := by
  exact .transfer (LeftBound35964.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 35965)
      LeftBound35965.bound (LeftBound35965.actual selector witness) := by
  exact .transfer (LeftBound35965.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound35964.bound, LeftBound35965.bound]
def bound : CoeffClass := .finite ⟨7769059532604529984509912, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound35964.bound, LeftBound35965.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound35964.actual selector witness, LeftBound35965.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound35966

namespace LeftBound35967
def owner : Owner := ⟨.program ⟨214⟩, ⟨18876⟩⟩
def transferEvent : Nat := 35967
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨214⟩, ⟨6475⟩⟩, ⟨.program ⟨214⟩, ⟨15057⟩⟩], []⟩ [⟨.result 703 .coefficient, true, some 1⟩, ⟨.result 2211 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 703 .coefficient)
      LeftAuthority702.bound (LeftAuthority702.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨6475⟩⟩) (rawTerms := some (Proof.Events002.exact703RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority702.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority702.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 2211 .coefficient)
      LeftAuthority2210.bound (LeftAuthority2210.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨15057⟩⟩) (rawTerms := some (Proof.Events008.exact2211RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority2210.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority2210.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority702.bound [LeftAuthority2210.bound]
def bound : CoeffClass := .finite ⟨175932572039110456474905, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority702.bound, LeftAuthority2210.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority702.actual selector witness) * ([LeftAuthority2210.actual selector witness].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.cons (.intro (input1 selector witness)) (.nil))
end LeftBound35967

namespace LeftBound35968
def owner : Owner := ⟨.program ⟨214⟩, ⟨18876⟩⟩
def transferEvent : Nat := 35968
def frameStart : Nat := 0
def rule : BoundRule := .sum [.transfer 35966, .transfer 35967]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 35966)
      LeftBound35966.bound (LeftBound35966.actual selector witness) := by
  exact .transfer (LeftBound35966.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 35967)
      LeftBound35967.bound (LeftBound35967.actual selector witness) := by
  exact .transfer (LeftBound35967.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound35966.bound, LeftBound35967.bound]
def bound : CoeffClass := .finite ⟨7944992104643640440984817, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound35966.bound, LeftBound35967.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound35966.actual selector witness, LeftBound35967.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound35968

namespace LeftBound35969
def owner : Owner := ⟨.program ⟨214⟩, ⟨18876⟩⟩
def transferEvent : Nat := 35969
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨214⟩, ⟨6495⟩⟩, ⟨.program ⟨214⟩, ⟨14896⟩⟩], []⟩ [⟨.result 713 .coefficient, true, some 1⟩, ⟨.result 2219 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 713 .coefficient)
      LeftAuthority712.bound (LeftAuthority712.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨6495⟩⟩) (rawTerms := some (Proof.Events002.exact713RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority712.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority712.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 2219 .coefficient)
      LeftAuthority2218.bound (LeftAuthority2218.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨14896⟩⟩) (rawTerms := some (Proof.Events008.exact2219RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority2218.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority2218.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority712.bound [LeftAuthority2218.bound]
def bound : CoeffClass := .finite ⟨156384508479209294644360, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority712.bound, LeftAuthority2218.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority712.actual selector witness) * ([LeftAuthority2218.actual selector witness].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.cons (.intro (input1 selector witness)) (.nil))
end LeftBound35969

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
