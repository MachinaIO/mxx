import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard340
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard357

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound53154
def owner : Owner := ⟨.program ⟨214⟩, ⟨12385⟩⟩
def transferEvent : Nat := 53154
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 53149 .summary, .result 53119 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 53149 .summary)
      LeftBound53144.bound (LeftBound53144.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨9829⟩⟩) (rawTerms := some (Proof.Events207.exact53149RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound53144.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 53119 .summary)
      LeftBound53116.bound (LeftBound53116.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨12384⟩⟩) (rawTerms := some (Proof.Events207.exact53119RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound53116.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound53144.bound, LeftBound53116.bound]
def bound : CoeffClass := .finite ⟨95453696, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound53144.bound, LeftBound53116.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound53144.actual selector witness, LeftBound53116.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound53154

namespace LeftBound53158
def owner : Owner := ⟨.program ⟨214⟩, ⟨25379⟩⟩
def transferEvent : Nat := 53158
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 53156 .coefficient) (.predecessor 1 53157 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 53156 .coefficient)
      LeftBound53152.bound (LeftBound53152.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events207.exact53155RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound53152.bound, RecordedBoundRefines] <;> decide)
      (LeftBound53152.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 53157 .coefficient)
      LeftAuthority53090.bound (LeftAuthority53090.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events207.exact53091RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority53090.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority53090.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound53152.bound LeftAuthority53090.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound53152.bound, LeftAuthority53090.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound53152.actual selector witness) * (LeftAuthority53090.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound53158

namespace LeftBound53159
def owner : Owner := ⟨.program ⟨214⟩, ⟨25379⟩⟩
def transferEvent : Nat := 53159
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨25378⟩⟩]⟩ [⟨.result 53091 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 53091 .coefficient)
      LeftAuthority53090.bound (LeftAuthority53090.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨25378⟩⟩) (rawTerms := some (Proof.Events207.exact53091RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority53090.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority53090.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority53090.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority53090.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority53090.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound53159

namespace LeftBound53160
def owner : Owner := ⟨.program ⟨214⟩, ⟨25379⟩⟩
def transferEvent : Nat := 53160
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 53155 .summary) (.transfer 53159) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 53155 .summary)
      LeftBound53154.bound (LeftBound53154.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨12385⟩⟩) (rawTerms := some (Proof.Events207.exact53155RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound53154.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 53159)
      LeftBound53159.bound (LeftBound53159.actual selector witness) := by
  exact .transfer (LeftBound53159.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound53154.bound LeftBound53159.bound
def bound : CoeffClass := .finite ⟨350316591579136, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound53154.bound, LeftBound53159.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound53154.actual selector witness) * (LeftBound53159.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound53160

namespace LeftBound53171
def owner : Owner := ⟨.program ⟨214⟩, ⟨19894⟩⟩
def transferEvent : Nat := 53171
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 53169 .coefficient) (.value (.predecessor 1 53170 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 53169 .coefficient)
      LeftAuthority53167.bound (LeftAuthority53167.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events207.exact53168RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority53167.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority53167.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 53170 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority53167.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority53167.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority53167.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound53171

namespace LeftBound53175
def owner : Owner := ⟨.program ⟨214⟩, ⟨19895⟩⟩
def transferEvent : Nat := 53175
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 53173 .coefficient) (.predecessor 1 53174 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 53173 .coefficient)
      LeftBound50759.bound (LeftBound50759.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events198.exact50762RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50759.bound, RecordedBoundRefines] <;> decide)
      (LeftBound50759.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 53174 .coefficient)
      LeftBound53171.bound (LeftBound53171.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events207.exact53172RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound53171.bound, RecordedBoundRefines] <;> decide)
      (LeftBound53171.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound50759.bound LeftBound53171.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50759.bound, LeftBound53171.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound50759.actual selector witness) * (LeftBound53171.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound53175

namespace LeftBound53176
def owner : Owner := ⟨.program ⟨214⟩, ⟨19895⟩⟩
def transferEvent : Nat := 53176
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨19892⟩⟩]⟩ [⟨.result 53168 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 53168 .coefficient)
      LeftAuthority53167.bound (LeftAuthority53167.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨19892⟩⟩) (rawTerms := some (Proof.Events207.exact53168RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority53167.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority53167.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority53167.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority53167.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority53167.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound53176

namespace LeftBound53177
def owner : Owner := ⟨.program ⟨214⟩, ⟨19895⟩⟩
def transferEvent : Nat := 53177
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 50762 .summary) (.transfer 53176) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 50762 .summary)
      LeftBound50760.bound (LeftBound50760.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5547⟩⟩) (rawTerms := some (Proof.Events198.exact50762RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound50760.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 53176)
      LeftBound53176.bound (LeftBound53176.actual selector witness) := by
  exact .transfer (LeftBound53176.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound50760.bound LeftBound53176.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50760.bound, LeftBound53176.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound50760.actual selector witness) * (LeftBound53176.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound53177

namespace LeftBound53256
def owner : Owner := ⟨.program ⟨214⟩, ⟨12379⟩⟩
def transferEvent : Nat := 53256
def frameStart : Nat := 53227
def rule : BoundRule := .product (.predecessor 0 53254 .coefficient) (.predecessor 1 53255 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 53254 .coefficient)
      LeftAuthority53252.bound (LeftAuthority53252.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events208.exact53253RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority53252.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority53252.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 53255 .coefficient)
      LeftAuthority53249.bound (LeftAuthority53249.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events208.exact53250RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority53249.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority53249.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority53252.bound LeftAuthority53249.bound
def bound : CoeffClass := .finite ⟨1600, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority53252.bound, LeftAuthority53249.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority53252.actual selector witness) * (LeftAuthority53249.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound53256

namespace LeftBound53260
def owner : Owner := ⟨.program ⟨214⟩, ⟨12380⟩⟩
def transferEvent : Nat := 53260
def frameStart : Nat := 53227
def rule : BoundRule := .identity (.predecessor 0 53259 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 53259 .coefficient)
      LeftBound53256.bound (LeftBound53256.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events208.exact53258RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound53256.bound, RecordedBoundRefines] <;> decide)
      (LeftBound53256.derived selector witness)

def rawBound : CoeffClass := LeftBound53256.bound
def bound : CoeffClass := .finite ⟨1600, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound53256.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound53256.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound53260

namespace LeftBound53277
def owner : Owner := ⟨.program ⟨214⟩, ⟨12470⟩⟩
def transferEvent : Nat := 53277
def frameStart : Nat := 53227
def rule : BoundRule := .sum [.predecessor 0 53275 .coefficient, .predecessor 1 53276 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 53275 .coefficient)
      LeftBound53260.bound (LeftBound53260.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound53260.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 53276 .coefficient)
      LeftAuthority53273.bound (LeftAuthority53273.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority53273.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound53260.bound, LeftAuthority53273.bound]
def bound : CoeffClass := .finite ⟨1600, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound53260.bound, LeftAuthority53273.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound53260.actual selector witness, LeftAuthority53273.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound53277

namespace LeftBound53280
def owner : Owner := ⟨.program ⟨214⟩, ⟨12471⟩⟩
def transferEvent : Nat := 53280
def frameStart : Nat := 53227
def rule : BoundRule := .identity (.predecessor 0 53279 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 53279 .coefficient)
      LeftBound53277.bound (LeftBound53277.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound53277.derived selector witness)

def rawBound : CoeffClass := LeftBound53277.bound
def bound : CoeffClass := .finite ⟨1600, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound53277.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound53277.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound53280

namespace LeftBound53286
def owner : Owner := ⟨.program ⟨214⟩, ⟨12472⟩⟩
def transferEvent : Nat := 53286
def frameStart : Nat := 53227
def rule : BoundRule := .product (.predecessor 0 53284 .coefficient) (.predecessor 1 53285 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 53284 .coefficient)
      LeftAuthority53282.bound (LeftAuthority53282.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events208.exact53283RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority53282.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority53282.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 53285 .coefficient)
      LeftBound53280.bound (LeftBound53280.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events208.exact53281RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound53280.bound, RecordedBoundRefines] <;> decide)
      (LeftBound53280.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority53282.bound LeftBound53280.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority53282.bound, LeftBound53280.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority53282.actual selector witness) * (LeftBound53280.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound53286

namespace LeftBound53302
def owner : Owner := ⟨.program ⟨214⟩, ⟨7868⟩⟩
def transferEvent : Nat := 53302
def frameStart : Nat := 53227
def rule : BoundRule := .scale (.predecessor 0 53300 .coefficient) (.value (.predecessor 1 53301 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 53300 .coefficient)
      LeftAuthority53298.bound (LeftAuthority53298.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events208.exact53299RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority53298.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority53298.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 53301 .coefficient)
      LeftAuthority53289.bound (LeftAuthority53289.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority53289.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority53298.bound LeftAuthority53289.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority53298.bound, LeftAuthority53289.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority53298.actual selector witness) * (LeftAuthority53289.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound53302

namespace LeftBound53305
def owner : Owner := ⟨.program ⟨214⟩, ⟨6765⟩⟩
def transferEvent : Nat := 53305
def frameStart : Nat := 53227
def rule : BoundRule := .identity (.predecessor 0 53304 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 53304 .coefficient)
      LeftAuthority53292.bound (LeftAuthority53292.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events208.exact53293RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority53292.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority53292.derived selector witness)

def rawBound : CoeffClass := LeftAuthority53292.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority53292.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority53292.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound53305

namespace LeftBound53309
def owner : Owner := ⟨.program ⟨214⟩, ⟨7869⟩⟩
def transferEvent : Nat := 53309
def frameStart : Nat := 53227
def rule : BoundRule := .product (.predecessor 0 53307 .coefficient) (.predecessor 1 53308 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 53307 .coefficient)
      LeftBound53305.bound (LeftBound53305.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events208.exact53306RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound53305.bound, RecordedBoundRefines] <;> decide)
      (LeftBound53305.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 53308 .coefficient)
      LeftBound53302.bound (LeftBound53302.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events208.exact53303RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound53302.bound, RecordedBoundRefines] <;> decide)
      (LeftBound53302.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound53305.bound LeftBound53302.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound53305.bound, LeftBound53302.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound53305.actual selector witness) * (LeftBound53302.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound53309

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
