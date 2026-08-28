import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard238
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard253

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound38211
def owner : Owner := ⟨.program ⟨214⟩, ⟨25463⟩⟩
def transferEvent : Nat := 38211
def frameStart : Nat := 38120
def rule : BoundRule := .product (.predecessor 0 38209 .coefficient) (.predecessor 1 38210 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 38209 .coefficient)
      LeftBound38207.bound (LeftBound38207.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events149.exact38208RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound38207.bound, RecordedBoundRefines] <;> decide)
      (LeftBound38207.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 38210 .coefficient)
      LeftAuthority38164.bound (LeftAuthority38164.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events149.exact38165RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority38164.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority38164.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound38207.bound LeftAuthority38164.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound38207.bound, LeftAuthority38164.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound38207.actual selector witness) * (LeftAuthority38164.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound38211

namespace LeftBound38222
def owner : Owner := ⟨.program ⟨214⟩, ⟨16559⟩⟩
def transferEvent : Nat := 38222
def frameStart : Nat := 38120
def rule : BoundRule := .product (.predecessor 0 38220 .coefficient) (.predecessor 1 38221 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 38220 .coefficient)
      LeftAuthority38175.bound (LeftAuthority38175.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events149.exact38176RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority38175.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority38175.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 38221 .coefficient)
      LeftAuthority38218.bound (LeftAuthority38218.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events149.exact38219RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority38218.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority38218.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority38175.bound LeftAuthority38218.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority38175.bound, LeftAuthority38218.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority38175.actual selector witness) * (LeftAuthority38218.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound38222

namespace LeftBound38230
def owner : Owner := ⟨.program ⟨214⟩, ⟨16560⟩⟩
def transferEvent : Nat := 38230
def frameStart : Nat := 38120
def rule : BoundRule := .sum [.predecessor 0 38228 .coefficient, .predecessor 1 38229 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 38228 .coefficient)
      LeftAuthority38226.bound (LeftAuthority38226.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events149.exact38227RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority38226.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority38226.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 38229 .coefficient)
      LeftBound38222.bound (LeftBound38222.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events149.exact38224RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound38222.bound, RecordedBoundRefines] <;> decide)
      (LeftBound38222.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority38226.bound, LeftBound38222.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority38226.bound, LeftBound38222.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority38226.actual selector witness, LeftBound38222.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound38230

namespace LeftBound38234
def owner : Owner := ⟨.program ⟨214⟩, ⟨25464⟩⟩
def transferEvent : Nat := 38234
def frameStart : Nat := 38120
def rule : BoundRule := .sum [.predecessor 0 38232 .coefficient, .predecessor 1 38233 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 38232 .coefficient)
      LeftBound38230.bound (LeftBound38230.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events149.exact38231RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound38230.bound, RecordedBoundRefines] <;> decide)
      (LeftBound38230.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 38233 .coefficient)
      LeftBound38211.bound (LeftBound38211.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events149.exact38216RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound38211.bound, RecordedBoundRefines] <;> decide)
      (LeftBound38211.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound38230.bound, LeftBound38211.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound38230.bound, LeftBound38211.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound38230.actual selector witness, LeftBound38211.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound38234

namespace LeftBound38247
def owner : Owner := ⟨.program ⟨214⟩, ⟨25462⟩⟩
def transferEvent : Nat := 38247
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 38245 .coefficient, .predecessor 1 38246 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 38245 .coefficient)
      LeftBound38068.bound (LeftBound38068.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events149.exact38244RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound38068.bound, RecordedBoundRefines] <;> decide)
      (LeftBound38068.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 38246 .coefficient)
      LeftBound38051.bound (LeftBound38051.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events148.exact38058RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound38051.bound, RecordedBoundRefines] <;> decide)
      (LeftBound38051.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound38068.bound, LeftBound38051.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound38068.bound, LeftBound38051.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound38068.actual selector witness, LeftBound38051.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound38247

namespace LeftBound38250
def owner : Owner := ⟨.program ⟨214⟩, ⟨25462⟩⟩
def transferEvent : Nat := 38250
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 38244 .summary, .result 38058 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 38244 .summary)
      LeftBound38070.bound (LeftBound38070.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨19971⟩⟩) (rawTerms := some (Proof.Events149.exact38244RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound38070.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 38058 .summary)
      LeftBound38053.bound (LeftBound38053.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25461⟩⟩) (rawTerms := some (Proof.Events148.exact38058RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound38053.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound38070.bound, LeftBound38053.bound]
def bound : CoeffClass := .finite ⟨352134001995776, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound38070.bound, LeftBound38053.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound38070.actual selector witness, LeftBound38053.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound38250

namespace LeftBound38254
def owner : Owner := ⟨.program ⟨214⟩, ⟨29196⟩⟩
def transferEvent : Nat := 38254
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 38252 .coefficient) (.predecessor 1 38253 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 38252 .coefficient)
      LeftBound38247.bound (LeftBound38247.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events149.exact38251RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound38247.bound, RecordedBoundRefines] <;> decide)
      (LeftBound38247.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 38253 .coefficient)
      LeftAuthority37973.bound (LeftAuthority37973.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events148.exact37974RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority37973.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority37973.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound38247.bound LeftAuthority37973.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound38247.bound, LeftAuthority37973.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound38247.actual selector witness) * (LeftAuthority37973.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound38254

namespace LeftBound38255
def owner : Owner := ⟨.program ⟨214⟩, ⟨29196⟩⟩
def transferEvent : Nat := 38255
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨29194⟩⟩]⟩ [⟨.result 37974 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 37974 .coefficient)
      LeftAuthority37973.bound (LeftAuthority37973.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨29194⟩⟩) (rawTerms := some (Proof.Events148.exact37974RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority37973.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority37973.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority37973.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority37973.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority37973.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound38255

namespace LeftBound38256
def owner : Owner := ⟨.program ⟨214⟩, ⟨29196⟩⟩
def transferEvent : Nat := 38256
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 38251 .summary) (.transfer 38255) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 38251 .summary)
      LeftBound38250.bound (LeftBound38250.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25462⟩⟩) (rawTerms := some (Proof.Events149.exact38251RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound38250.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 38255)
      LeftBound38255.bound (LeftBound38255.actual selector witness) := by
  exact .transfer (LeftBound38255.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound38250.bound LeftBound38255.bound
def bound : CoeffClass := .finite ⟨1292337421468529852416, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound38250.bound, LeftBound38255.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound38250.actual selector witness) * (LeftBound38255.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound38256

namespace LeftBound38267
def owner : Owner := ⟨.program ⟨214⟩, ⟨22274⟩⟩
def transferEvent : Nat := 38267
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 38265 .coefficient) (.value (.predecessor 1 38266 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 38265 .coefficient)
      LeftAuthority38263.bound (LeftAuthority38263.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events149.exact38264RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority38263.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority38263.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 38266 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority38263.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority38263.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority38263.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound38267

namespace LeftBound38271
def owner : Owner := ⟨.program ⟨214⟩, ⟨22275⟩⟩
def transferEvent : Nat := 38271
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 38269 .coefficient) (.predecessor 1 38270 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 38269 .coefficient)
      LeftBound36134.bound (LeftBound36134.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events141.exact36137RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound36134.bound, RecordedBoundRefines] <;> decide)
      (LeftBound36134.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 38270 .coefficient)
      LeftBound38267.bound (LeftBound38267.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events149.exact38268RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound38267.bound, RecordedBoundRefines] <;> decide)
      (LeftBound38267.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound36134.bound LeftBound38267.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound36134.bound, LeftBound38267.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound36134.actual selector witness) * (LeftBound38267.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound38271

namespace LeftBound38272
def owner : Owner := ⟨.program ⟨214⟩, ⟨22275⟩⟩
def transferEvent : Nat := 38272
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨22272⟩⟩]⟩ [⟨.result 38264 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 38264 .coefficient)
      LeftAuthority38263.bound (LeftAuthority38263.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨22272⟩⟩) (rawTerms := some (Proof.Events149.exact38264RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority38263.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority38263.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority38263.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority38263.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority38263.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound38272

namespace LeftBound38273
def owner : Owner := ⟨.program ⟨214⟩, ⟨22275⟩⟩
def transferEvent : Nat := 38273
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 36137 .summary) (.transfer 38272) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 36137 .summary)
      LeftBound36135.bound (LeftBound36135.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5553⟩⟩) (rawTerms := some (Proof.Events141.exact36137RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound36135.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 38272)
      LeftBound38272.bound (LeftBound38272.actual selector witness) := by
  exact .transfer (LeftBound38272.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound36135.bound LeftBound38272.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound36135.bound, LeftBound38272.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound36135.actual selector witness) * (LeftBound38272.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound38273

namespace LeftBound38368
def owner : Owner := ⟨.program ⟨214⟩, ⟨16558⟩⟩
def transferEvent : Nat := 38368
def frameStart : Nat := 38329
def rule : BoundRule := .identity (.predecessor 0 38367 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 38367 .coefficient)
      LeftAuthority38365.bound (LeftAuthority38365.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events149.exact38366RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority38365.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority38365.derived selector witness)

def rawBound : CoeffClass := LeftAuthority38365.bound
def bound : CoeffClass := .finite ⟨42, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority38365.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority38365.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound38368

namespace LeftBound38385
def owner : Owner := ⟨.program ⟨214⟩, ⟨16597⟩⟩
def transferEvent : Nat := 38385
def frameStart : Nat := 38329
def rule : BoundRule := .sum [.predecessor 0 38383 .coefficient, .predecessor 1 38384 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 38383 .coefficient)
      LeftBound38368.bound (LeftBound38368.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound38368.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 38384 .coefficient)
      LeftAuthority38381.bound (LeftAuthority38381.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority38381.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound38368.bound, LeftAuthority38381.bound]
def bound : CoeffClass := .finite ⟨42, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound38368.bound, LeftAuthority38381.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound38368.actual selector witness, LeftAuthority38381.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound38385

namespace LeftBound38388
def owner : Owner := ⟨.program ⟨214⟩, ⟨16598⟩⟩
def transferEvent : Nat := 38388
def frameStart : Nat := 38329
def rule : BoundRule := .identity (.predecessor 0 38387 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 38387 .coefficient)
      LeftBound38385.bound (LeftBound38385.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound38385.derived selector witness)

def rawBound : CoeffClass := LeftBound38385.bound
def bound : CoeffClass := .finite ⟨42, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound38385.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound38385.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound38388

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
