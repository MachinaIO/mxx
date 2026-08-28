import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard019
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard030
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard111

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound18172
def owner : Owner := ⟨.program ⟨214⟩, ⟨22066⟩⟩
def transferEvent : Nat := 18172
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 18170 .coefficient) (.value (.predecessor 1 18171 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 18170 .coefficient)
      LeftAuthority18168.bound (LeftAuthority18168.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events070.exact18169RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority18168.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority18168.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 18171 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority18168.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority18168.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority18168.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound18172

namespace LeftBound18176
def owner : Owner := ⟨.program ⟨214⟩, ⟨22067⟩⟩
def transferEvent : Nat := 18176
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 18174 .coefficient) (.predecessor 1 18175 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 18174 .coefficient)
      LeftBound6558.bound (LeftBound6558.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events025.exact6561RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6558.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6558.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 18175 .coefficient)
      LeftBound18172.bound (LeftBound18172.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events070.exact18173RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound18172.bound, RecordedBoundRefines] <;> decide)
      (LeftBound18172.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound6558.bound LeftBound18172.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound6558.bound, LeftBound18172.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound6558.actual selector witness) * (LeftBound18172.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound18176

namespace LeftBound18177
def owner : Owner := ⟨.program ⟨214⟩, ⟨22067⟩⟩
def transferEvent : Nat := 18177
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨22064⟩⟩]⟩ [⟨.result 18169 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 18169 .coefficient)
      LeftAuthority18168.bound (LeftAuthority18168.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨22064⟩⟩) (rawTerms := some (Proof.Events070.exact18169RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority18168.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority18168.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority18168.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority18168.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority18168.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound18177

namespace LeftBound18178
def owner : Owner := ⟨.program ⟨214⟩, ⟨22067⟩⟩
def transferEvent : Nat := 18178
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 6561 .summary) (.transfer 18177) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 6561 .summary)
      LeftBound6559.bound (LeftBound6559.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5565⟩⟩) (rawTerms := some (Proof.Events025.exact6561RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound6559.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 18177)
      LeftBound18177.bound (LeftBound18177.actual selector witness) := by
  exact .transfer (LeftBound18177.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound6559.bound LeftBound18177.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound6559.bound, LeftBound18177.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound6559.actual selector witness) * (LeftBound18177.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound18178

namespace LeftBound18273
def owner : Owner := ⟨.program ⟨214⟩, ⟨16482⟩⟩
def transferEvent : Nat := 18273
def frameStart : Nat := 18234
def rule : BoundRule := .identity (.predecessor 0 18272 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 18272 .coefficient)
      LeftAuthority18270.bound (LeftAuthority18270.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events071.exact18271RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority18270.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority18270.derived selector witness)

def rawBound : CoeffClass := LeftAuthority18270.bound
def bound : CoeffClass := .finite ⟨40, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority18270.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority18270.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound18273

namespace LeftBound18290
def owner : Owner := ⟨.program ⟨214⟩, ⟨16521⟩⟩
def transferEvent : Nat := 18290
def frameStart : Nat := 18234
def rule : BoundRule := .sum [.predecessor 0 18288 .coefficient, .predecessor 1 18289 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 18288 .coefficient)
      LeftBound18273.bound (LeftBound18273.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound18273.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 18289 .coefficient)
      LeftAuthority18286.bound (LeftAuthority18286.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority18286.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound18273.bound, LeftAuthority18286.bound]
def bound : CoeffClass := .finite ⟨40, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound18273.bound, LeftAuthority18286.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound18273.actual selector witness, LeftAuthority18286.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound18290

namespace LeftBound18293
def owner : Owner := ⟨.program ⟨214⟩, ⟨16522⟩⟩
def transferEvent : Nat := 18293
def frameStart : Nat := 18234
def rule : BoundRule := .identity (.predecessor 0 18292 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 18292 .coefficient)
      LeftBound18290.bound (LeftBound18290.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound18290.derived selector witness)

def rawBound : CoeffClass := LeftBound18290.bound
def bound : CoeffClass := .finite ⟨40, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound18290.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound18290.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound18293

namespace LeftBound18299
def owner : Owner := ⟨.program ⟨214⟩, ⟨16523⟩⟩
def transferEvent : Nat := 18299
def frameStart : Nat := 18234
def rule : BoundRule := .product (.predecessor 0 18297 .coefficient) (.predecessor 1 18298 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 18297 .coefficient)
      LeftAuthority18295.bound (LeftAuthority18295.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events071.exact18296RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority18295.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority18295.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 18298 .coefficient)
      LeftBound18293.bound (LeftBound18293.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events071.exact18294RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound18293.bound, RecordedBoundRefines] <;> decide)
      (LeftBound18293.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority18295.bound LeftBound18293.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority18295.bound, LeftBound18293.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority18295.actual selector witness) * (LeftBound18293.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound18299

namespace LeftBound18307
def owner : Owner := ⟨.program ⟨214⟩, ⟨16524⟩⟩
def transferEvent : Nat := 18307
def frameStart : Nat := 18234
def rule : BoundRule := .sum [.predecessor 0 18305 .coefficient, .predecessor 1 18306 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 18305 .coefficient)
      LeftAuthority18303.bound (LeftAuthority18303.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events071.exact18304RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority18303.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority18303.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 18306 .coefficient)
      LeftBound18299.bound (LeftBound18299.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events071.exact18301RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound18299.bound, RecordedBoundRefines] <;> decide)
      (LeftBound18299.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority18303.bound, LeftBound18299.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority18303.bound, LeftBound18299.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority18303.actual selector witness, LeftBound18299.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound18307

namespace LeftBound18311
def owner : Owner := ⟨.program ⟨214⟩, ⟨28997⟩⟩
def transferEvent : Nat := 18311
def frameStart : Nat := 18234
def rule : BoundRule := .product (.predecessor 0 18309 .coefficient) (.predecessor 1 18310 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 18309 .coefficient)
      LeftBound18307.bound (LeftBound18307.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events071.exact18308RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound18307.bound, RecordedBoundRefines] <;> decide)
      (LeftBound18307.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 18310 .coefficient)
      LeftAuthority18284.bound (LeftAuthority18284.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events071.exact18285RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority18284.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority18284.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound18307.bound LeftAuthority18284.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound18307.bound, LeftAuthority18284.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound18307.actual selector witness) * (LeftAuthority18284.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound18311

namespace LeftBound18322
def owner : Owner := ⟨.program ⟨214⟩, ⟨17568⟩⟩
def transferEvent : Nat := 18322
def frameStart : Nat := 18234
def rule : BoundRule := .product (.predecessor 0 18320 .coefficient) (.predecessor 1 18321 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 18320 .coefficient)
      LeftAuthority18295.bound (LeftAuthority18295.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events071.exact18296RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority18295.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority18295.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 18321 .coefficient)
      LeftAuthority18318.bound (LeftAuthority18318.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events071.exact18319RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority18318.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority18318.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority18295.bound LeftAuthority18318.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority18295.bound, LeftAuthority18318.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority18295.actual selector witness) * (LeftAuthority18318.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound18322

namespace LeftBound18330
def owner : Owner := ⟨.program ⟨214⟩, ⟨17569⟩⟩
def transferEvent : Nat := 18330
def frameStart : Nat := 18234
def rule : BoundRule := .sum [.predecessor 0 18328 .coefficient, .predecessor 1 18329 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 18328 .coefficient)
      LeftAuthority18326.bound (LeftAuthority18326.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events071.exact18327RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority18326.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority18326.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 18329 .coefficient)
      LeftBound18322.bound (LeftBound18322.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events071.exact18324RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound18322.bound, RecordedBoundRefines] <;> decide)
      (LeftBound18322.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority18326.bound, LeftBound18322.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority18326.bound, LeftBound18322.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority18326.actual selector witness, LeftBound18322.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound18330

namespace LeftBound18334
def owner : Owner := ⟨.program ⟨214⟩, ⟨29002⟩⟩
def transferEvent : Nat := 18334
def frameStart : Nat := 18234
def rule : BoundRule := .sum [.predecessor 0 18332 .coefficient, .predecessor 1 18333 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 18332 .coefficient)
      LeftBound18330.bound (LeftBound18330.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events071.exact18331RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound18330.bound, RecordedBoundRefines] <;> decide)
      (LeftBound18330.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 18333 .coefficient)
      LeftBound18311.bound (LeftBound18311.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events071.exact18316RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound18311.bound, RecordedBoundRefines] <;> decide)
      (LeftBound18311.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound18330.bound, LeftBound18311.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound18330.bound, LeftBound18311.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound18330.actual selector witness, LeftBound18311.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound18334

namespace LeftBound18347
def owner : Owner := ⟨.program ⟨214⟩, ⟨28999⟩⟩
def transferEvent : Nat := 18347
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 18345 .coefficient, .predecessor 1 18346 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 18345 .coefficient)
      LeftBound18176.bound (LeftBound18176.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events071.exact18344RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound18176.bound, RecordedBoundRefines] <;> decide)
      (LeftBound18176.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 18346 .coefficient)
      LeftBound18159.bound (LeftBound18159.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events070.exact18166RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound18159.bound, RecordedBoundRefines] <;> decide)
      (LeftBound18159.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound18176.bound, LeftBound18159.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound18176.bound, LeftBound18159.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound18176.actual selector witness, LeftBound18159.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound18347

namespace LeftBound18350
def owner : Owner := ⟨.program ⟨214⟩, ⟨28999⟩⟩
def transferEvent : Nat := 18350
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 18344 .summary, .result 18166 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 18344 .summary)
      LeftBound18178.bound (LeftBound18178.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨22067⟩⟩) (rawTerms := some (Proof.Events071.exact18344RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound18178.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 18166 .summary)
      LeftBound18161.bound (LeftBound18161.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28998⟩⟩) (rawTerms := some (Proof.Events070.exact18166RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound18161.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound18178.bound, LeftBound18161.bound]
def bound : CoeffClass := .finite ⟨1292315010834812776448, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound18178.bound, LeftBound18161.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound18178.actual selector witness, LeftBound18161.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound18350

namespace LeftBound18354
def owner : Owner := ⟨.program ⟨214⟩, ⟨29000⟩⟩
def transferEvent : Nat := 18354
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 18352 .coefficient) (.predecessor 1 18353 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 18352 .coefficient)
      LeftBound18347.bound (LeftBound18347.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events071.exact18351RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound18347.bound, RecordedBoundRefines] <;> decide)
      (LeftBound18347.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 18353 .coefficient)
      LeftBound5618.bound (LeftBound5618.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events021.exact5619RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5618.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5618.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound18347.bound LeftBound5618.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound18347.bound, LeftBound5618.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound18347.actual selector witness) * (LeftBound5618.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound18354

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
