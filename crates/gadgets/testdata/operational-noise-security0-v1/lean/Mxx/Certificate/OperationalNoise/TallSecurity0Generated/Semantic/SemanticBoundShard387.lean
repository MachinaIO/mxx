import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard340
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard386

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound57010
def owner : Owner := ⟨.program ⟨214⟩, ⟨13573⟩⟩
def transferEvent : Nat := 57010
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 57005 .summary, .result 56975 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 57005 .summary)
      LeftBound57000.bound (LeftBound57000.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨13572⟩⟩) (rawTerms := some (Proof.Events222.exact57005RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound57000.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 56975 .summary)
      LeftBound56972.bound (LeftBound56972.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨13568⟩⟩) (rawTerms := some (Proof.Events222.exact56975RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound56972.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound57000.bound, LeftBound56972.bound]
def bound : CoeffClass := .finite ⟨95428736, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound57000.bound, LeftBound56972.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound57000.actual selector witness, LeftBound56972.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound57010

namespace LeftBound57014
def owner : Owner := ⟨.program ⟨214⟩, ⟨25841⟩⟩
def transferEvent : Nat := 57014
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 57012 .coefficient) (.predecessor 1 57013 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 57012 .coefficient)
      LeftBound57008.bound (LeftBound57008.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events222.exact57011RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound57008.bound, RecordedBoundRefines] <;> decide)
      (LeftBound57008.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 57013 .coefficient)
      LeftAuthority56946.bound (LeftAuthority56946.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events222.exact56947RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority56946.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority56946.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound57008.bound LeftAuthority56946.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound57008.bound, LeftAuthority56946.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound57008.actual selector witness) * (LeftAuthority56946.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound57014

namespace LeftBound57015
def owner : Owner := ⟨.program ⟨214⟩, ⟨25841⟩⟩
def transferEvent : Nat := 57015
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨25840⟩⟩]⟩ [⟨.result 56947 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 56947 .coefficient)
      LeftAuthority56946.bound (LeftAuthority56946.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨25840⟩⟩) (rawTerms := some (Proof.Events222.exact56947RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority56946.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority56946.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority56946.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority56946.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority56946.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound57015

namespace LeftBound57016
def owner : Owner := ⟨.program ⟨214⟩, ⟨25841⟩⟩
def transferEvent : Nat := 57016
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 57011 .summary) (.transfer 57015) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 57011 .summary)
      LeftBound57010.bound (LeftBound57010.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨13573⟩⟩) (rawTerms := some (Proof.Events222.exact57011RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound57010.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 57015)
      LeftBound57015.bound (LeftBound57015.actual selector witness) := by
  exact .transfer (LeftBound57015.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound57010.bound LeftBound57015.bound
def bound : CoeffClass := .finite ⟨350224987979776, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound57010.bound, LeftBound57015.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound57010.actual selector witness) * (LeftBound57015.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound57016

namespace LeftBound57027
def owner : Owner := ⟨.program ⟨214⟩, ⟨19318⟩⟩
def transferEvent : Nat := 57027
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 57025 .coefficient) (.value (.predecessor 1 57026 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 57025 .coefficient)
      LeftAuthority57023.bound (LeftAuthority57023.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events222.exact57024RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority57023.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority57023.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 57026 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority57023.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority57023.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority57023.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound57027

namespace LeftBound57031
def owner : Owner := ⟨.program ⟨214⟩, ⟨19319⟩⟩
def transferEvent : Nat := 57031
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 57029 .coefficient) (.predecessor 1 57030 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 57029 .coefficient)
      LeftBound50759.bound (LeftBound50759.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events198.exact50762RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50759.bound, RecordedBoundRefines] <;> decide)
      (LeftBound50759.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 57030 .coefficient)
      LeftBound57027.bound (LeftBound57027.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events222.exact57028RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound57027.bound, RecordedBoundRefines] <;> decide)
      (LeftBound57027.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound50759.bound LeftBound57027.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50759.bound, LeftBound57027.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound50759.actual selector witness) * (LeftBound57027.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound57031

namespace LeftBound57032
def owner : Owner := ⟨.program ⟨214⟩, ⟨19319⟩⟩
def transferEvent : Nat := 57032
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨19316⟩⟩]⟩ [⟨.result 57024 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 57024 .coefficient)
      LeftAuthority57023.bound (LeftAuthority57023.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨19316⟩⟩) (rawTerms := some (Proof.Events222.exact57024RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority57023.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority57023.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority57023.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority57023.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority57023.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound57032

namespace LeftBound57033
def owner : Owner := ⟨.program ⟨214⟩, ⟨19319⟩⟩
def transferEvent : Nat := 57033
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 50762 .summary) (.transfer 57032) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 50762 .summary)
      LeftBound50760.bound (LeftBound50760.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5547⟩⟩) (rawTerms := some (Proof.Events198.exact50762RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound50760.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 57032)
      LeftBound57032.bound (LeftBound57032.actual selector witness) := by
  exact .transfer (LeftBound57032.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound50760.bound LeftBound57032.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50760.bound, LeftBound57032.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound50760.actual selector witness) * (LeftBound57032.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound57033

namespace LeftBound57112
def owner : Owner := ⟨.program ⟨214⟩, ⟨13566⟩⟩
def transferEvent : Nat := 57112
def frameStart : Nat := 57083
def rule : BoundRule := .product (.predecessor 0 57110 .coefficient) (.predecessor 1 57111 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 57110 .coefficient)
      LeftAuthority57108.bound (LeftAuthority57108.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events223.exact57109RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority57108.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority57108.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 57111 .coefficient)
      LeftAuthority57105.bound (LeftAuthority57105.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events223.exact57106RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority57105.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority57105.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority57108.bound LeftAuthority57105.bound
def bound : CoeffClass := .finite ⟨100, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority57108.bound, LeftAuthority57105.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority57108.actual selector witness) * (LeftAuthority57105.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound57112

namespace LeftBound57116
def owner : Owner := ⟨.program ⟨214⟩, ⟨13567⟩⟩
def transferEvent : Nat := 57116
def frameStart : Nat := 57083
def rule : BoundRule := .identity (.predecessor 0 57115 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 57115 .coefficient)
      LeftBound57112.bound (LeftBound57112.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events223.exact57114RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound57112.bound, RecordedBoundRefines] <;> decide)
      (LeftBound57112.derived selector witness)

def rawBound : CoeffClass := LeftBound57112.bound
def bound : CoeffClass := .finite ⟨100, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound57112.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound57112.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound57116

namespace LeftBound57133
def owner : Owner := ⟨.program ⟨214⟩, ⟨13667⟩⟩
def transferEvent : Nat := 57133
def frameStart : Nat := 57083
def rule : BoundRule := .sum [.predecessor 0 57131 .coefficient, .predecessor 1 57132 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 57131 .coefficient)
      LeftBound57116.bound (LeftBound57116.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound57116.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 57132 .coefficient)
      LeftAuthority57129.bound (LeftAuthority57129.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority57129.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound57116.bound, LeftAuthority57129.bound]
def bound : CoeffClass := .finite ⟨100, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound57116.bound, LeftAuthority57129.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound57116.actual selector witness, LeftAuthority57129.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound57133

namespace LeftBound57136
def owner : Owner := ⟨.program ⟨214⟩, ⟨13668⟩⟩
def transferEvent : Nat := 57136
def frameStart : Nat := 57083
def rule : BoundRule := .identity (.predecessor 0 57135 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 57135 .coefficient)
      LeftBound57133.bound (LeftBound57133.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound57133.derived selector witness)

def rawBound : CoeffClass := LeftBound57133.bound
def bound : CoeffClass := .finite ⟨100, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound57133.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound57133.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound57136

namespace LeftBound57142
def owner : Owner := ⟨.program ⟨214⟩, ⟨13669⟩⟩
def transferEvent : Nat := 57142
def frameStart : Nat := 57083
def rule : BoundRule := .product (.predecessor 0 57140 .coefficient) (.predecessor 1 57141 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 57140 .coefficient)
      LeftAuthority57138.bound (LeftAuthority57138.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events223.exact57139RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority57138.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority57138.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 57141 .coefficient)
      LeftBound57136.bound (LeftBound57136.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events223.exact57137RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound57136.bound, RecordedBoundRefines] <;> decide)
      (LeftBound57136.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority57138.bound LeftBound57136.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority57138.bound, LeftBound57136.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority57138.actual selector witness) * (LeftBound57136.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound57142

namespace LeftBound57158
def owner : Owner := ⟨.program ⟨214⟩, ⟨7844⟩⟩
def transferEvent : Nat := 57158
def frameStart : Nat := 57083
def rule : BoundRule := .scale (.predecessor 0 57156 .coefficient) (.value (.predecessor 1 57157 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 57156 .coefficient)
      LeftAuthority57154.bound (LeftAuthority57154.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events223.exact57155RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority57154.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority57154.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 57157 .coefficient)
      LeftAuthority57145.bound (LeftAuthority57145.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority57145.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority57154.bound LeftAuthority57145.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority57154.bound, LeftAuthority57145.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority57154.actual selector witness) * (LeftAuthority57145.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound57158

namespace LeftBound57161
def owner : Owner := ⟨.program ⟨214⟩, ⟨6793⟩⟩
def transferEvent : Nat := 57161
def frameStart : Nat := 57083
def rule : BoundRule := .identity (.predecessor 0 57160 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 57160 .coefficient)
      LeftAuthority57148.bound (LeftAuthority57148.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events223.exact57149RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority57148.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority57148.derived selector witness)

def rawBound : CoeffClass := LeftAuthority57148.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority57148.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority57148.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound57161

namespace LeftBound57165
def owner : Owner := ⟨.program ⟨214⟩, ⟨7845⟩⟩
def transferEvent : Nat := 57165
def frameStart : Nat := 57083
def rule : BoundRule := .product (.predecessor 0 57163 .coefficient) (.predecessor 1 57164 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 57163 .coefficient)
      LeftBound57161.bound (LeftBound57161.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events223.exact57162RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound57161.bound, RecordedBoundRefines] <;> decide)
      (LeftBound57161.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 57164 .coefficient)
      LeftBound57158.bound (LeftBound57158.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events223.exact57159RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound57158.bound, RecordedBoundRefines] <;> decide)
      (LeftBound57158.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound57161.bound LeftBound57158.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound57161.bound, LeftBound57158.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound57161.actual selector witness) * (LeftBound57158.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound57165

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
