import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard137
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard174
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard219

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound33849
def owner : Owner := ⟨.program ⟨214⟩, ⟨27900⟩⟩
def transferEvent : Nat := 33849
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 26518 .summary) (.transfer 33848) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 26518 .summary)
      LeftBound26517.bound (LeftBound26517.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26083⟩⟩) (rawTerms := some (Proof.Events103.exact26518RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound26517.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 33848)
      LeftBound33848.bound (LeftBound33848.actual selector witness) := by
  exact .transfer (LeftBound33848.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound26517.bound LeftBound33848.bound
def bound : CoeffClass := .finite ⟨1292068472128282820608, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound26517.bound, LeftBound33848.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound26517.actual selector witness) * (LeftBound33848.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound33849

namespace LeftBound33860
def owner : Owner := ⟨.program ⟨214⟩, ⟨21342⟩⟩
def transferEvent : Nat := 33860
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 33858 .coefficient) (.value (.predecessor 1 33859 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 33858 .coefficient)
      LeftAuthority33856.bound (LeftAuthority33856.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events132.exact33857RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority33856.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority33856.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 33859 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority33856.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority33856.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority33856.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound33860

namespace LeftBound33864
def owner : Owner := ⟨.program ⟨214⟩, ⟨21343⟩⟩
def transferEvent : Nat := 33864
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 33862 .coefficient) (.predecessor 1 33863 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 33862 .coefficient)
      LeftBound21509.bound (LeftBound21509.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events084.exact21512RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21509.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21509.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 33863 .coefficient)
      LeftBound33860.bound (LeftBound33860.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events132.exact33861RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound33860.bound, RecordedBoundRefines] <;> decide)
      (LeftBound33860.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound21509.bound LeftBound33860.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21509.bound, LeftBound33860.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound21509.actual selector witness) * (LeftBound33860.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound33864

namespace LeftBound33865
def owner : Owner := ⟨.program ⟨214⟩, ⟨21343⟩⟩
def transferEvent : Nat := 33865
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨21340⟩⟩]⟩ [⟨.result 33857 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 33857 .coefficient)
      LeftAuthority33856.bound (LeftAuthority33856.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨21340⟩⟩) (rawTerms := some (Proof.Events132.exact33857RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority33856.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority33856.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority33856.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority33856.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority33856.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound33865

namespace LeftBound33866
def owner : Owner := ⟨.program ⟨214⟩, ⟨21343⟩⟩
def transferEvent : Nat := 33866
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 21512 .summary) (.transfer 33865) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 21512 .summary)
      LeftBound21510.bound (LeftBound21510.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5559⟩⟩) (rawTerms := some (Proof.Events084.exact21512RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound21510.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 33865)
      LeftBound33865.bound (LeftBound33865.actual selector witness) := by
  exact .transfer (LeftBound33865.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound21510.bound LeftBound33865.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21510.bound, LeftBound33865.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound21510.actual selector witness) * (LeftBound33865.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound33866

namespace LeftBound33961
def owner : Owner := ⟨.program ⟨214⟩, ⟨15953⟩⟩
def transferEvent : Nat := 33961
def frameStart : Nat := 33922
def rule : BoundRule := .identity (.predecessor 0 33960 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 33960 .coefficient)
      LeftAuthority33958.bound (LeftAuthority33958.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events132.exact33959RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority33958.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority33958.derived selector witness)

def rawBound : CoeffClass := LeftAuthority33958.bound
def bound : CoeffClass := .finite ⟨18, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority33958.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority33958.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound33961

namespace LeftBound33978
def owner : Owner := ⟨.program ⟨214⟩, ⟨16027⟩⟩
def transferEvent : Nat := 33978
def frameStart : Nat := 33922
def rule : BoundRule := .sum [.predecessor 0 33976 .coefficient, .predecessor 1 33977 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 33976 .coefficient)
      LeftBound33961.bound (LeftBound33961.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound33961.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 33977 .coefficient)
      LeftAuthority33974.bound (LeftAuthority33974.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority33974.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound33961.bound, LeftAuthority33974.bound]
def bound : CoeffClass := .finite ⟨18, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound33961.bound, LeftAuthority33974.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound33961.actual selector witness, LeftAuthority33974.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound33978

namespace LeftBound33981
def owner : Owner := ⟨.program ⟨214⟩, ⟨16028⟩⟩
def transferEvent : Nat := 33981
def frameStart : Nat := 33922
def rule : BoundRule := .identity (.predecessor 0 33980 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 33980 .coefficient)
      LeftBound33978.bound (LeftBound33978.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound33978.derived selector witness)

def rawBound : CoeffClass := LeftBound33978.bound
def bound : CoeffClass := .finite ⟨18, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound33978.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound33978.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound33981

namespace LeftBound33987
def owner : Owner := ⟨.program ⟨214⟩, ⟨16029⟩⟩
def transferEvent : Nat := 33987
def frameStart : Nat := 33922
def rule : BoundRule := .product (.predecessor 0 33985 .coefficient) (.predecessor 1 33986 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 33985 .coefficient)
      LeftAuthority33983.bound (LeftAuthority33983.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events132.exact33984RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority33983.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority33983.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 33986 .coefficient)
      LeftBound33981.bound (LeftBound33981.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events132.exact33982RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound33981.bound, RecordedBoundRefines] <;> decide)
      (LeftBound33981.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority33983.bound LeftBound33981.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority33983.bound, LeftBound33981.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority33983.actual selector witness) * (LeftBound33981.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound33987

namespace LeftBound33995
def owner : Owner := ⟨.program ⟨214⟩, ⟨16030⟩⟩
def transferEvent : Nat := 33995
def frameStart : Nat := 33922
def rule : BoundRule := .sum [.predecessor 0 33993 .coefficient, .predecessor 1 33994 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 33993 .coefficient)
      LeftAuthority33991.bound (LeftAuthority33991.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events132.exact33992RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority33991.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority33991.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 33994 .coefficient)
      LeftBound33987.bound (LeftBound33987.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events132.exact33989RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound33987.bound, RecordedBoundRefines] <;> decide)
      (LeftBound33987.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority33991.bound, LeftBound33987.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority33991.bound, LeftBound33987.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority33991.actual selector witness, LeftBound33987.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound33995

namespace LeftBound33999
def owner : Owner := ⟨.program ⟨214⟩, ⟨27899⟩⟩
def transferEvent : Nat := 33999
def frameStart : Nat := 33922
def rule : BoundRule := .product (.predecessor 0 33997 .coefficient) (.predecessor 1 33998 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 33997 .coefficient)
      LeftBound33995.bound (LeftBound33995.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events132.exact33996RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound33995.bound, RecordedBoundRefines] <;> decide)
      (LeftBound33995.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 33998 .coefficient)
      LeftAuthority33972.bound (LeftAuthority33972.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events132.exact33973RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority33972.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority33972.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound33995.bound LeftAuthority33972.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound33995.bound, LeftAuthority33972.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound33995.actual selector witness) * (LeftAuthority33972.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound33999

namespace LeftBound34010
def owner : Owner := ⟨.program ⟨214⟩, ⟨17179⟩⟩
def transferEvent : Nat := 34010
def frameStart : Nat := 33922
def rule : BoundRule := .product (.predecessor 0 34008 .coefficient) (.predecessor 1 34009 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 34008 .coefficient)
      LeftAuthority33983.bound (LeftAuthority33983.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events132.exact33984RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority33983.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority33983.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 34009 .coefficient)
      LeftAuthority34006.bound (LeftAuthority34006.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events132.exact34007RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority34006.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority34006.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority33983.bound LeftAuthority34006.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority33983.bound, LeftAuthority34006.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority33983.actual selector witness) * (LeftAuthority34006.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound34010

namespace LeftBound34018
def owner : Owner := ⟨.program ⟨214⟩, ⟨17180⟩⟩
def transferEvent : Nat := 34018
def frameStart : Nat := 33922
def rule : BoundRule := .sum [.predecessor 0 34016 .coefficient, .predecessor 1 34017 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 34016 .coefficient)
      LeftAuthority34014.bound (LeftAuthority34014.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events132.exact34015RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority34014.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority34014.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 34017 .coefficient)
      LeftBound34010.bound (LeftBound34010.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events132.exact34012RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound34010.bound, RecordedBoundRefines] <;> decide)
      (LeftBound34010.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority34014.bound, LeftBound34010.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority34014.bound, LeftBound34010.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority34014.actual selector witness, LeftBound34010.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound34018

namespace LeftBound34022
def owner : Owner := ⟨.program ⟨214⟩, ⟨27904⟩⟩
def transferEvent : Nat := 34022
def frameStart : Nat := 33922
def rule : BoundRule := .sum [.predecessor 0 34020 .coefficient, .predecessor 1 34021 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 34020 .coefficient)
      LeftBound34018.bound (LeftBound34018.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events132.exact34019RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound34018.bound, RecordedBoundRefines] <;> decide)
      (LeftBound34018.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 34021 .coefficient)
      LeftBound33999.bound (LeftBound33999.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events132.exact34004RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound33999.bound, RecordedBoundRefines] <;> decide)
      (LeftBound33999.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound34018.bound, LeftBound33999.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound34018.bound, LeftBound33999.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound34018.actual selector witness, LeftBound33999.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound34022

namespace LeftBound34035
def owner : Owner := ⟨.program ⟨214⟩, ⟨27901⟩⟩
def transferEvent : Nat := 34035
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 34033 .coefficient, .predecessor 1 34034 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 34033 .coefficient)
      LeftBound33864.bound (LeftBound33864.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events132.exact34032RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound33864.bound, RecordedBoundRefines] <;> decide)
      (LeftBound33864.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 34034 .coefficient)
      LeftBound33847.bound (LeftBound33847.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events132.exact33854RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound33847.bound, RecordedBoundRefines] <;> decide)
      (LeftBound33847.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound33864.bound, LeftBound33847.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound33864.bound, LeftBound33847.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound33864.actual selector witness, LeftBound33847.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound34035

namespace LeftBound34038
def owner : Owner := ⟨.program ⟨214⟩, ⟨27901⟩⟩
def transferEvent : Nat := 34038
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 34032 .summary, .result 33854 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 34032 .summary)
      LeftBound33866.bound (LeftBound33866.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨21343⟩⟩) (rawTerms := some (Proof.Events132.exact34032RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound33866.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 33854 .summary)
      LeftBound33849.bound (LeftBound33849.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27900⟩⟩) (rawTerms := some (Proof.Events132.exact33854RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound33849.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound33866.bound, LeftBound33849.bound]
def bound : CoeffClass := .finite ⟨1292068473939586330624, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound33866.bound, LeftBound33849.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound33866.actual selector witness, LeftBound33849.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound34038

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
