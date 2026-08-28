import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard238
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard250
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard311

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound46972
def owner : Owner := ⟨.program ⟨214⟩, ⟨29625⟩⟩
def transferEvent : Nat := 46972
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨6661⟩⟩]⟩ [⟨.result 5555 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 5555 .coefficient)
      LeftAuthority5554.bound (LeftAuthority5554.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨6661⟩⟩) (rawTerms := some (Proof.Events021.exact5555RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority5554.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority5554.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority5554.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority5554.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority5554.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound46972

namespace LeftBound46973
def owner : Owner := ⟨.program ⟨214⟩, ⟨29625⟩⟩
def transferEvent : Nat := 46973
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 46968 .summary) (.transfer 46972) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 46968 .summary)
      LeftBound46967.bound (LeftBound46967.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29624⟩⟩) (rawTerms := some (Proof.Events183.exact46968RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound46967.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 46972)
      LeftBound46972.bound (LeftBound46972.actual selector witness) := by
  exact .transfer (LeftBound46972.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound46967.bound LeftBound46972.bound
def bound : CoeffClass := .finite ⟨4743310290994884271912517632, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound46967.bound, LeftBound46972.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound46967.actual selector witness) * (LeftBound46972.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound46973

namespace LeftBound46988
def owner : Owner := ⟨.program ⟨214⟩, ⟨29406⟩⟩
def transferEvent : Nat := 46988
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 46986 .coefficient) (.predecessor 1 46987 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 46986 .coefficient)
      LeftBound37765.bound (LeftBound37765.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events147.exact37769RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound37765.bound, RecordedBoundRefines] <;> decide)
      (LeftBound37765.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 46987 .coefficient)
      LeftAuthority46984.bound (LeftAuthority46984.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events183.exact46985RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority46984.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority46984.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound37765.bound LeftAuthority46984.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound37765.bound, LeftAuthority46984.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound37765.actual selector witness) * (LeftAuthority46984.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound46988

namespace LeftBound46989
def owner : Owner := ⟨.program ⟨214⟩, ⟨29406⟩⟩
def transferEvent : Nat := 46989
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨29404⟩⟩]⟩ [⟨.result 46985 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 46985 .coefficient)
      LeftAuthority46984.bound (LeftAuthority46984.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨29404⟩⟩) (rawTerms := some (Proof.Events183.exact46985RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority46984.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority46984.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority46984.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority46984.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority46984.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound46989

namespace LeftBound46990
def owner : Owner := ⟨.program ⟨214⟩, ⟨29406⟩⟩
def transferEvent : Nat := 46990
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 37769 .summary) (.transfer 46989) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 37769 .summary)
      LeftBound37768.bound (LeftBound37768.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25539⟩⟩) (rawTerms := some (Proof.Events147.exact37769RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound37768.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 46989)
      LeftBound46989.bound (LeftBound46989.actual selector witness) := by
  exact .transfer (LeftBound46989.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound37768.bound LeftBound46989.bound
def bound : CoeffClass := .finite ⟨1292382246358571024384, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound37768.bound, LeftBound46989.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound37768.actual selector witness) * (LeftBound46989.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound46990

namespace LeftBound47001
def owner : Owner := ⟨.program ⟨214⟩, ⟨22346⟩⟩
def transferEvent : Nat := 47001
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 46999 .coefficient) (.value (.predecessor 1 47000 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 46999 .coefficient)
      LeftAuthority46997.bound (LeftAuthority46997.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events183.exact46998RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority46997.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority46997.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 47000 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority46997.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority46997.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority46997.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound47001

namespace LeftBound47005
def owner : Owner := ⟨.program ⟨214⟩, ⟨22347⟩⟩
def transferEvent : Nat := 47005
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 47003 .coefficient) (.predecessor 1 47004 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 47003 .coefficient)
      LeftBound36134.bound (LeftBound36134.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events141.exact36137RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound36134.bound, RecordedBoundRefines] <;> decide)
      (LeftBound36134.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 47004 .coefficient)
      LeftBound47001.bound (LeftBound47001.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events183.exact47002RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound47001.bound, RecordedBoundRefines] <;> decide)
      (LeftBound47001.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound36134.bound LeftBound47001.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound36134.bound, LeftBound47001.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound36134.actual selector witness) * (LeftBound47001.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound47005

namespace LeftBound47006
def owner : Owner := ⟨.program ⟨214⟩, ⟨22347⟩⟩
def transferEvent : Nat := 47006
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨22344⟩⟩]⟩ [⟨.result 46998 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 46998 .coefficient)
      LeftAuthority46997.bound (LeftAuthority46997.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨22344⟩⟩) (rawTerms := some (Proof.Events183.exact46998RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority46997.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority46997.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority46997.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority46997.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority46997.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound47006

namespace LeftBound47007
def owner : Owner := ⟨.program ⟨214⟩, ⟨22347⟩⟩
def transferEvent : Nat := 47007
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 36137 .summary) (.transfer 47006) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 36137 .summary)
      LeftBound36135.bound (LeftBound36135.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5553⟩⟩) (rawTerms := some (Proof.Events141.exact36137RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound36135.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 47006)
      LeftBound47006.bound (LeftBound47006.actual selector witness) := by
  exact .transfer (LeftBound47006.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound36135.bound LeftBound47006.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound36135.bound, LeftBound47006.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound36135.actual selector witness) * (LeftBound47006.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound47007

namespace LeftBound47102
def owner : Owner := ⟨.program ⟨214⟩, ⟨16642⟩⟩
def transferEvent : Nat := 47102
def frameStart : Nat := 47063
def rule : BoundRule := .identity (.predecessor 0 47101 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 47101 .coefficient)
      LeftAuthority47099.bound (LeftAuthority47099.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events183.exact47100RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority47099.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority47099.derived selector witness)

def rawBound : CoeffClass := LeftAuthority47099.bound
def bound : CoeffClass := .finite ⟨46, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority47099.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority47099.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound47102

namespace LeftBound47119
def owner : Owner := ⟨.program ⟨214⟩, ⟨16716⟩⟩
def transferEvent : Nat := 47119
def frameStart : Nat := 47063
def rule : BoundRule := .sum [.predecessor 0 47117 .coefficient, .predecessor 1 47118 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 47117 .coefficient)
      LeftBound47102.bound (LeftBound47102.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound47102.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 47118 .coefficient)
      LeftAuthority47115.bound (LeftAuthority47115.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority47115.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound47102.bound, LeftAuthority47115.bound]
def bound : CoeffClass := .finite ⟨46, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound47102.bound, LeftAuthority47115.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound47102.actual selector witness, LeftAuthority47115.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound47119

namespace LeftBound47122
def owner : Owner := ⟨.program ⟨214⟩, ⟨16717⟩⟩
def transferEvent : Nat := 47122
def frameStart : Nat := 47063
def rule : BoundRule := .identity (.predecessor 0 47121 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 47121 .coefficient)
      LeftBound47119.bound (LeftBound47119.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound47119.derived selector witness)

def rawBound : CoeffClass := LeftBound47119.bound
def bound : CoeffClass := .finite ⟨46, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound47119.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound47119.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound47122

namespace LeftBound47128
def owner : Owner := ⟨.program ⟨214⟩, ⟨16718⟩⟩
def transferEvent : Nat := 47128
def frameStart : Nat := 47063
def rule : BoundRule := .product (.predecessor 0 47126 .coefficient) (.predecessor 1 47127 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 47126 .coefficient)
      LeftAuthority47124.bound (LeftAuthority47124.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events184.exact47125RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority47124.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority47124.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 47127 .coefficient)
      LeftBound47122.bound (LeftBound47122.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events184.exact47123RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound47122.bound, RecordedBoundRefines] <;> decide)
      (LeftBound47122.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority47124.bound LeftBound47122.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority47124.bound, LeftBound47122.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority47124.actual selector witness) * (LeftBound47122.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound47128

namespace LeftBound47136
def owner : Owner := ⟨.program ⟨214⟩, ⟨16719⟩⟩
def transferEvent : Nat := 47136
def frameStart : Nat := 47063
def rule : BoundRule := .sum [.predecessor 0 47134 .coefficient, .predecessor 1 47135 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 47134 .coefficient)
      LeftAuthority47132.bound (LeftAuthority47132.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events184.exact47133RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority47132.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority47132.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 47135 .coefficient)
      LeftBound47128.bound (LeftBound47128.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events184.exact47130RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound47128.bound, RecordedBoundRefines] <;> decide)
      (LeftBound47128.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority47132.bound, LeftBound47128.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority47132.bound, LeftBound47128.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority47132.actual selector witness, LeftBound47128.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound47136

namespace LeftBound47140
def owner : Owner := ⟨.program ⟨214⟩, ⟨29405⟩⟩
def transferEvent : Nat := 47140
def frameStart : Nat := 47063
def rule : BoundRule := .product (.predecessor 0 47138 .coefficient) (.predecessor 1 47139 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 47138 .coefficient)
      LeftBound47136.bound (LeftBound47136.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events184.exact47137RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound47136.bound, RecordedBoundRefines] <;> decide)
      (LeftBound47136.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 47139 .coefficient)
      LeftAuthority47113.bound (LeftAuthority47113.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events184.exact47114RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority47113.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority47113.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound47136.bound LeftAuthority47113.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound47136.bound, LeftAuthority47113.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound47136.actual selector witness) * (LeftAuthority47113.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound47140

namespace LeftBound47151
def owner : Owner := ⟨.program ⟨214⟩, ⟨17728⟩⟩
def transferEvent : Nat := 47151
def frameStart : Nat := 47063
def rule : BoundRule := .product (.predecessor 0 47149 .coefficient) (.predecessor 1 47150 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 47149 .coefficient)
      LeftAuthority47124.bound (LeftAuthority47124.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events184.exact47125RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority47124.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority47124.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 47150 .coefficient)
      LeftAuthority47147.bound (LeftAuthority47147.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events184.exact47148RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority47147.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority47147.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority47124.bound LeftAuthority47147.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority47124.bound, LeftAuthority47147.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority47124.actual selector witness) * (LeftAuthority47147.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound47151

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
