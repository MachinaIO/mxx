import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard340
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard379
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard380

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound56233
def owner : Owner := ⟨.program ⟨214⟩, ⟨25998⟩⟩
def transferEvent : Nat := 56233
def frameStart : Nat := 56119
def rule : BoundRule := .sum [.predecessor 0 56231 .coefficient, .predecessor 1 56232 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 56231 .coefficient)
      LeftBound56229.bound (LeftBound56229.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events219.exact56230RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound56229.bound, RecordedBoundRefines] <;> decide)
      (LeftBound56229.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 56232 .coefficient)
      LeftBound56210.bound (LeftBound56210.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events219.exact56215RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound56210.bound, RecordedBoundRefines] <;> decide)
      (LeftBound56210.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound56229.bound, LeftBound56210.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound56229.bound, LeftBound56210.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound56229.actual selector witness, LeftBound56210.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound56233

namespace LeftBound56246
def owner : Owner := ⟨.program ⟨214⟩, ⟨25996⟩⟩
def transferEvent : Nat := 56246
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 56244 .coefficient, .predecessor 1 56245 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 56244 .coefficient)
      LeftBound56067.bound (LeftBound56067.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events219.exact56243RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound56067.bound, RecordedBoundRefines] <;> decide)
      (LeftBound56067.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 56245 .coefficient)
      LeftBound56050.bound (LeftBound56050.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events218.exact56057RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound56050.bound, RecordedBoundRefines] <;> decide)
      (LeftBound56050.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound56067.bound, LeftBound56050.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound56067.bound, LeftBound56050.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound56067.actual selector witness, LeftBound56050.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound56246

namespace LeftBound56249
def owner : Owner := ⟨.program ⟨214⟩, ⟨25996⟩⟩
def transferEvent : Nat := 56249
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 56243 .summary, .result 56057 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 56243 .summary)
      LeftBound56069.bound (LeftBound56069.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨19463⟩⟩) (rawTerms := some (Proof.Events219.exact56243RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound56069.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 56057 .summary)
      LeftBound56052.bound (LeftBound56052.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25995⟩⟩) (rawTerms := some (Proof.Events218.exact56057RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound56052.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound56069.bound, LeftBound56052.bound]
def bound : CoeffClass := .finite ⟨352054612209664, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound56069.bound, LeftBound56052.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound56069.actual selector witness, LeftBound56052.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound56249

namespace LeftBound56253
def owner : Owner := ⟨.program ⟨214⟩, ⟨27664⟩⟩
def transferEvent : Nat := 56253
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 56251 .coefficient) (.predecessor 1 56252 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 56251 .coefficient)
      LeftBound56246.bound (LeftBound56246.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events219.exact56250RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound56246.bound, RecordedBoundRefines] <;> decide)
      (LeftBound56246.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 56252 .coefficient)
      LeftAuthority55972.bound (LeftAuthority55972.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events218.exact55973RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority55972.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority55972.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound56246.bound LeftAuthority55972.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound56246.bound, LeftAuthority55972.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound56246.actual selector witness) * (LeftAuthority55972.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound56253

namespace LeftBound56254
def owner : Owner := ⟨.program ⟨214⟩, ⟨27664⟩⟩
def transferEvent : Nat := 56254
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨27662⟩⟩]⟩ [⟨.result 55973 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 55973 .coefficient)
      LeftAuthority55972.bound (LeftAuthority55972.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨27662⟩⟩) (rawTerms := some (Proof.Events218.exact55973RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority55972.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority55972.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority55972.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority55972.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority55972.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound56254

namespace LeftBound56255
def owner : Owner := ⟨.program ⟨214⟩, ⟨27664⟩⟩
def transferEvent : Nat := 56255
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 56250 .summary) (.transfer 56254) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 56250 .summary)
      LeftBound56249.bound (LeftBound56249.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25996⟩⟩) (rawTerms := some (Proof.Events219.exact56250RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound56249.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 56254)
      LeftBound56254.bound (LeftBound56254.actual selector witness) := by
  exact .transfer (LeftBound56254.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound56249.bound LeftBound56254.bound
def bound : CoeffClass := .finite ⟨1292046059683262234624, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound56249.bound, LeftBound56254.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound56249.actual selector witness) * (LeftBound56254.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound56255

namespace LeftBound56266
def owner : Owner := ⟨.program ⟨214⟩, ⟨21262⟩⟩
def transferEvent : Nat := 56266
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 56264 .coefficient) (.value (.predecessor 1 56265 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 56264 .coefficient)
      LeftAuthority56262.bound (LeftAuthority56262.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events219.exact56263RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority56262.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority56262.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 56265 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority56262.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority56262.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority56262.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound56266

namespace LeftBound56270
def owner : Owner := ⟨.program ⟨214⟩, ⟨21263⟩⟩
def transferEvent : Nat := 56270
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 56268 .coefficient) (.predecessor 1 56269 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 56268 .coefficient)
      LeftBound50759.bound (LeftBound50759.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events198.exact50762RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50759.bound, RecordedBoundRefines] <;> decide)
      (LeftBound50759.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 56269 .coefficient)
      LeftBound56266.bound (LeftBound56266.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events219.exact56267RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound56266.bound, RecordedBoundRefines] <;> decide)
      (LeftBound56266.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound50759.bound LeftBound56266.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50759.bound, LeftBound56266.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound50759.actual selector witness) * (LeftBound56266.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound56270

namespace LeftBound56271
def owner : Owner := ⟨.program ⟨214⟩, ⟨21263⟩⟩
def transferEvent : Nat := 56271
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨21260⟩⟩]⟩ [⟨.result 56263 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 56263 .coefficient)
      LeftAuthority56262.bound (LeftAuthority56262.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨21260⟩⟩) (rawTerms := some (Proof.Events219.exact56263RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority56262.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority56262.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority56262.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority56262.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority56262.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound56271

namespace LeftBound56272
def owner : Owner := ⟨.program ⟨214⟩, ⟨21263⟩⟩
def transferEvent : Nat := 56272
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 50762 .summary) (.transfer 56271) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 50762 .summary)
      LeftBound50760.bound (LeftBound50760.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5547⟩⟩) (rawTerms := some (Proof.Events198.exact50762RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound50760.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 56271)
      LeftBound56271.bound (LeftBound56271.actual selector witness) := by
  exact .transfer (LeftBound56271.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound50760.bound LeftBound56271.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50760.bound, LeftBound56271.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound50760.actual selector witness) * (LeftBound56271.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound56272

namespace LeftBound56367
def owner : Owner := ⟨.program ⟨214⟩, ⟨15826⟩⟩
def transferEvent : Nat := 56367
def frameStart : Nat := 56328
def rule : BoundRule := .identity (.predecessor 0 56366 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 56366 .coefficient)
      LeftAuthority56364.bound (LeftAuthority56364.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events220.exact56365RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority56364.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority56364.derived selector witness)

def rawBound : CoeffClass := LeftAuthority56364.bound
def bound : CoeffClass := .finite ⟨16, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority56364.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority56364.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound56367

namespace LeftBound56384
def owner : Owner := ⟨.program ⟨214⟩, ⟨15900⟩⟩
def transferEvent : Nat := 56384
def frameStart : Nat := 56328
def rule : BoundRule := .sum [.predecessor 0 56382 .coefficient, .predecessor 1 56383 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 56382 .coefficient)
      LeftBound56367.bound (LeftBound56367.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound56367.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 56383 .coefficient)
      LeftAuthority56380.bound (LeftAuthority56380.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority56380.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound56367.bound, LeftAuthority56380.bound]
def bound : CoeffClass := .finite ⟨16, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound56367.bound, LeftAuthority56380.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound56367.actual selector witness, LeftAuthority56380.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound56384

namespace LeftBound56387
def owner : Owner := ⟨.program ⟨214⟩, ⟨15901⟩⟩
def transferEvent : Nat := 56387
def frameStart : Nat := 56328
def rule : BoundRule := .identity (.predecessor 0 56386 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 56386 .coefficient)
      LeftBound56384.bound (LeftBound56384.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound56384.derived selector witness)

def rawBound : CoeffClass := LeftBound56384.bound
def bound : CoeffClass := .finite ⟨16, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound56384.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound56384.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound56387

namespace LeftBound56393
def owner : Owner := ⟨.program ⟨214⟩, ⟨15902⟩⟩
def transferEvent : Nat := 56393
def frameStart : Nat := 56328
def rule : BoundRule := .product (.predecessor 0 56391 .coefficient) (.predecessor 1 56392 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 56391 .coefficient)
      LeftAuthority56389.bound (LeftAuthority56389.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events220.exact56390RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority56389.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority56389.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 56392 .coefficient)
      LeftBound56387.bound (LeftBound56387.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events220.exact56388RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound56387.bound, RecordedBoundRefines] <;> decide)
      (LeftBound56387.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority56389.bound LeftBound56387.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority56389.bound, LeftBound56387.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority56389.actual selector witness) * (LeftBound56387.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound56393

namespace LeftBound56401
def owner : Owner := ⟨.program ⟨214⟩, ⟨15903⟩⟩
def transferEvent : Nat := 56401
def frameStart : Nat := 56328
def rule : BoundRule := .sum [.predecessor 0 56399 .coefficient, .predecessor 1 56400 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 56399 .coefficient)
      LeftAuthority56397.bound (LeftAuthority56397.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events220.exact56398RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority56397.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority56397.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 56400 .coefficient)
      LeftBound56393.bound (LeftBound56393.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events220.exact56395RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound56393.bound, RecordedBoundRefines] <;> decide)
      (LeftBound56393.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority56397.bound, LeftBound56393.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority56397.bound, LeftBound56393.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority56397.actual selector witness, LeftBound56393.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound56401

namespace LeftBound56405
def owner : Owner := ⟨.program ⟨214⟩, ⟨27663⟩⟩
def transferEvent : Nat := 56405
def frameStart : Nat := 56328
def rule : BoundRule := .product (.predecessor 0 56403 .coefficient) (.predecessor 1 56404 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 56403 .coefficient)
      LeftBound56401.bound (LeftBound56401.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events220.exact56402RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound56401.bound, RecordedBoundRefines] <;> decide)
      (LeftBound56401.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 56404 .coefficient)
      LeftAuthority56378.bound (LeftAuthority56378.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events220.exact56379RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority56378.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority56378.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound56401.bound LeftAuthority56378.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound56401.bound, LeftAuthority56378.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound56401.actual selector witness) * (LeftAuthority56378.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound56405

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
