import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard340
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard346

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound51698
def owner : Owner := ⟨.program ⟨214⟩, ⟨10144⟩⟩
def transferEvent : Nat := 51698
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 51693 .summary) (.transfer 51697) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 51693 .summary)
      LeftBound51691.bound (LeftBound51691.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨10143⟩⟩) (rawTerms := some (Proof.Events201.exact51693RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound51691.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 51697)
      LeftBound51697.bound (LeftBound51697.actual selector witness) := by
  exact .transfer (LeftBound51697.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound51691.bound LeftBound51697.bound
def bound : CoeffClass := .finite ⟨95420416, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound51691.bound, LeftBound51697.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound51691.actual selector witness) * (LeftBound51697.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound51698

namespace LeftBound51706
def owner : Owner := ⟨.program ⟨214⟩, ⟨12973⟩⟩
def transferEvent : Nat := 51706
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 51704 .coefficient, .predecessor 1 51705 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 51704 .coefficient)
      LeftBound51696.bound (LeftBound51696.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events201.exact51703RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound51696.bound, RecordedBoundRefines] <;> decide)
      (LeftBound51696.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 51705 .coefficient)
      LeftBound51668.bound (LeftBound51668.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events201.exact51673RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound51668.bound, RecordedBoundRefines] <;> decide)
      (LeftBound51668.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound51696.bound, LeftBound51668.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound51696.bound, LeftBound51668.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound51696.actual selector witness, LeftBound51668.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound51706

namespace LeftBound51708
def owner : Owner := ⟨.program ⟨214⟩, ⟨12973⟩⟩
def transferEvent : Nat := 51708
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 51703 .summary, .result 51673 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 51703 .summary)
      LeftBound51698.bound (LeftBound51698.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨10144⟩⟩) (rawTerms := some (Proof.Events201.exact51703RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound51698.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 51673 .summary)
      LeftBound51670.bound (LeftBound51670.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨12972⟩⟩) (rawTerms := some (Proof.Events201.exact51673RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound51670.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound51698.bound, LeftBound51670.bound]
def bound : CoeffClass := .finite ⟨95463680, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound51698.bound, LeftBound51670.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound51698.actual selector witness, LeftBound51670.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound51708

namespace LeftBound51712
def owner : Owner := ⟨.program ⟨214⟩, ⟨25610⟩⟩
def transferEvent : Nat := 51712
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 51710 .coefficient) (.predecessor 1 51711 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 51710 .coefficient)
      LeftBound51706.bound (LeftBound51706.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events201.exact51709RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound51706.bound, RecordedBoundRefines] <;> decide)
      (LeftBound51706.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 51711 .coefficient)
      LeftAuthority51644.bound (LeftAuthority51644.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events201.exact51645RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority51644.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority51644.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound51706.bound LeftAuthority51644.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound51706.bound, LeftAuthority51644.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound51706.actual selector witness) * (LeftAuthority51644.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound51712

namespace LeftBound51713
def owner : Owner := ⟨.program ⟨214⟩, ⟨25610⟩⟩
def transferEvent : Nat := 51713
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨25609⟩⟩]⟩ [⟨.result 51645 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 51645 .coefficient)
      LeftAuthority51644.bound (LeftAuthority51644.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨25609⟩⟩) (rawTerms := some (Proof.Events201.exact51645RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority51644.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority51644.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority51644.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority51644.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority51644.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound51713

namespace LeftBound51714
def owner : Owner := ⟨.program ⟨214⟩, ⟨25610⟩⟩
def transferEvent : Nat := 51714
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 51709 .summary) (.transfer 51713) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 51709 .summary)
      LeftBound51708.bound (LeftBound51708.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨12973⟩⟩) (rawTerms := some (Proof.Events201.exact51709RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound51708.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 51713)
      LeftBound51713.bound (LeftBound51713.actual selector witness) := by
  exact .transfer (LeftBound51713.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound51708.bound LeftBound51713.bound
def bound : CoeffClass := .finite ⟨350353233018880, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound51708.bound, LeftBound51713.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound51708.actual selector witness) * (LeftBound51713.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound51714

namespace LeftBound51725
def owner : Owner := ⟨.program ⟨214⟩, ⟨20110⟩⟩
def transferEvent : Nat := 51725
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 51723 .coefficient) (.value (.predecessor 1 51724 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 51723 .coefficient)
      LeftAuthority51721.bound (LeftAuthority51721.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events202.exact51722RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority51721.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority51721.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 51724 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority51721.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority51721.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority51721.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound51725

namespace LeftBound51729
def owner : Owner := ⟨.program ⟨214⟩, ⟨20111⟩⟩
def transferEvent : Nat := 51729
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 51727 .coefficient) (.predecessor 1 51728 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 51727 .coefficient)
      LeftBound50759.bound (LeftBound50759.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events198.exact50762RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50759.bound, RecordedBoundRefines] <;> decide)
      (LeftBound50759.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 51728 .coefficient)
      LeftBound51725.bound (LeftBound51725.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events202.exact51726RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound51725.bound, RecordedBoundRefines] <;> decide)
      (LeftBound51725.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound50759.bound LeftBound51725.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50759.bound, LeftBound51725.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound50759.actual selector witness) * (LeftBound51725.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound51729

namespace LeftBound51730
def owner : Owner := ⟨.program ⟨214⟩, ⟨20111⟩⟩
def transferEvent : Nat := 51730
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨20108⟩⟩]⟩ [⟨.result 51722 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 51722 .coefficient)
      LeftAuthority51721.bound (LeftAuthority51721.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨20108⟩⟩) (rawTerms := some (Proof.Events202.exact51722RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority51721.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority51721.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority51721.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority51721.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority51721.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound51730

namespace LeftBound51731
def owner : Owner := ⟨.program ⟨214⟩, ⟨20111⟩⟩
def transferEvent : Nat := 51731
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 50762 .summary) (.transfer 51730) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 50762 .summary)
      LeftBound50760.bound (LeftBound50760.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5547⟩⟩) (rawTerms := some (Proof.Events198.exact50762RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound50760.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 51730)
      LeftBound51730.bound (LeftBound51730.actual selector witness) := by
  exact .transfer (LeftBound51730.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound50760.bound LeftBound51730.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50760.bound, LeftBound51730.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound50760.actual selector witness) * (LeftBound51730.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound51731

namespace LeftBound51810
def owner : Owner := ⟨.program ⟨214⟩, ⟨12967⟩⟩
def transferEvent : Nat := 51810
def frameStart : Nat := 51781
def rule : BoundRule := .product (.predecessor 0 51808 .coefficient) (.predecessor 1 51809 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 51808 .coefficient)
      LeftAuthority51806.bound (LeftAuthority51806.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events202.exact51807RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority51806.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority51806.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 51809 .coefficient)
      LeftAuthority51803.bound (LeftAuthority51803.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events202.exact51804RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority51803.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority51803.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority51806.bound LeftAuthority51803.bound
def bound : CoeffClass := .finite ⟨2704, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority51806.bound, LeftAuthority51803.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority51806.actual selector witness) * (LeftAuthority51803.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound51810

namespace LeftBound51814
def owner : Owner := ⟨.program ⟨214⟩, ⟨12968⟩⟩
def transferEvent : Nat := 51814
def frameStart : Nat := 51781
def rule : BoundRule := .identity (.predecessor 0 51813 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 51813 .coefficient)
      LeftBound51810.bound (LeftBound51810.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events202.exact51812RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound51810.bound, RecordedBoundRefines] <;> decide)
      (LeftBound51810.derived selector witness)

def rawBound : CoeffClass := LeftBound51810.bound
def bound : CoeffClass := .finite ⟨2704, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound51810.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound51810.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound51814

namespace LeftBound51831
def owner : Owner := ⟨.program ⟨214⟩, ⟨13058⟩⟩
def transferEvent : Nat := 51831
def frameStart : Nat := 51781
def rule : BoundRule := .sum [.predecessor 0 51829 .coefficient, .predecessor 1 51830 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 51829 .coefficient)
      LeftBound51814.bound (LeftBound51814.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound51814.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 51830 .coefficient)
      LeftAuthority51827.bound (LeftAuthority51827.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority51827.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound51814.bound, LeftAuthority51827.bound]
def bound : CoeffClass := .finite ⟨2704, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound51814.bound, LeftAuthority51827.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound51814.actual selector witness, LeftAuthority51827.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound51831

namespace LeftBound51834
def owner : Owner := ⟨.program ⟨214⟩, ⟨13059⟩⟩
def transferEvent : Nat := 51834
def frameStart : Nat := 51781
def rule : BoundRule := .identity (.predecessor 0 51833 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 51833 .coefficient)
      LeftBound51831.bound (LeftBound51831.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound51831.derived selector witness)

def rawBound : CoeffClass := LeftBound51831.bound
def bound : CoeffClass := .finite ⟨2704, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound51831.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound51831.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound51834

namespace LeftBound51840
def owner : Owner := ⟨.program ⟨214⟩, ⟨13060⟩⟩
def transferEvent : Nat := 51840
def frameStart : Nat := 51781
def rule : BoundRule := .product (.predecessor 0 51838 .coefficient) (.predecessor 1 51839 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 51838 .coefficient)
      LeftAuthority51836.bound (LeftAuthority51836.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events202.exact51837RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority51836.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority51836.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 51839 .coefficient)
      LeftBound51834.bound (LeftBound51834.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events202.exact51835RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound51834.bound, RecordedBoundRefines] <;> decide)
      (LeftBound51834.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority51836.bound LeftBound51834.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority51836.bound, LeftBound51834.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority51836.actual selector witness) * (LeftBound51834.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound51840

namespace LeftBound51856
def owner : Owner := ⟨.program ⟨214⟩, ⟨7877⟩⟩
def transferEvent : Nat := 51856
def frameStart : Nat := 51781
def rule : BoundRule := .scale (.predecessor 0 51854 .coefficient) (.value (.predecessor 1 51855 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 51854 .coefficient)
      LeftAuthority51852.bound (LeftAuthority51852.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events202.exact51853RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority51852.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority51852.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 51855 .coefficient)
      LeftAuthority51843.bound (LeftAuthority51843.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority51843.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority51852.bound LeftAuthority51843.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority51852.bound, LeftAuthority51843.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority51852.actual selector witness) * (LeftAuthority51843.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound51856

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
