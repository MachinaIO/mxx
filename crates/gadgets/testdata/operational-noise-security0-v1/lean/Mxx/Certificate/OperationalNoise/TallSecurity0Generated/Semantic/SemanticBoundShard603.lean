import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard095
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard096
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard538
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard541
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard602

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound88042
def owner : Owner := ⟨.program ⟨214⟩, ⟨15312⟩⟩
def transferEvent : Nat := 88042
def frameStart : Nat := 87954
def rule : BoundRule := .product (.predecessor 0 88040 .coefficient) (.predecessor 1 88041 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 88040 .coefficient)
      LeftAuthority88015.bound (LeftAuthority88015.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events343.exact88016RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority88015.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority88015.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 88041 .coefficient)
      LeftAuthority88038.bound (LeftAuthority88038.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events343.exact88039RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority88038.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority88038.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority88015.bound LeftAuthority88038.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority88015.bound, LeftAuthority88038.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority88015.actual selector witness) * (LeftAuthority88038.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound88042

namespace LeftBound88050
def owner : Owner := ⟨.program ⟨214⟩, ⟨15313⟩⟩
def transferEvent : Nat := 88050
def frameStart : Nat := 87954
def rule : BoundRule := .sum [.predecessor 0 88048 .coefficient, .predecessor 1 88049 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 88048 .coefficient)
      LeftAuthority88046.bound (LeftAuthority88046.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events343.exact88047RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority88046.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority88046.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 88049 .coefficient)
      LeftBound88042.bound (LeftBound88042.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events343.exact88044RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound88042.bound, RecordedBoundRefines] <;> decide)
      (LeftBound88042.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority88046.bound, LeftBound88042.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority88046.bound, LeftBound88042.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority88046.actual selector witness, LeftBound88042.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound88050

namespace LeftBound88054
def owner : Owner := ⟨.program ⟨214⟩, ⟨26569⟩⟩
def transferEvent : Nat := 88054
def frameStart : Nat := 87954
def rule : BoundRule := .sum [.predecessor 0 88052 .coefficient, .predecessor 1 88053 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 88052 .coefficient)
      LeftBound88050.bound (LeftBound88050.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events343.exact88051RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound88050.bound, RecordedBoundRefines] <;> decide)
      (LeftBound88050.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 88053 .coefficient)
      LeftBound88031.bound (LeftBound88031.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events343.exact88036RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound88031.bound, RecordedBoundRefines] <;> decide)
      (LeftBound88031.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound88050.bound, LeftBound88031.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound88050.bound, LeftBound88031.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound88050.actual selector witness, LeftBound88031.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound88054

namespace LeftBound88067
def owner : Owner := ⟨.program ⟨214⟩, ⟨26567⟩⟩
def transferEvent : Nat := 88067
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 88065 .coefficient, .predecessor 1 88066 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 88065 .coefficient)
      LeftBound87896.bound (LeftBound87896.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events344.exact88064RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound87896.bound, RecordedBoundRefines] <;> decide)
      (LeftBound87896.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 88066 .coefficient)
      LeftBound87879.bound (LeftBound87879.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events343.exact87886RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound87879.bound, RecordedBoundRefines] <;> decide)
      (LeftBound87879.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound87896.bound, LeftBound87879.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound87896.bound, LeftBound87879.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound87896.actual selector witness, LeftBound87879.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound88067

namespace LeftBound88070
def owner : Owner := ⟨.program ⟨214⟩, ⟨26567⟩⟩
def transferEvent : Nat := 88070
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 88064 .summary, .result 87886 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 88064 .summary)
      LeftBound87898.bound (LeftBound87898.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨20539⟩⟩) (rawTerms := some (Proof.Events344.exact88064RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound87898.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 87886 .summary)
      LeftBound87881.bound (LeftBound87881.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26566⟩⟩) (rawTerms := some (Proof.Events343.exact87886RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound87881.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound87898.bound, LeftBound87881.bound]
def bound : CoeffClass := .finite ⟨1291900380601931935744, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound87898.bound, LeftBound87881.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound87898.actual selector witness, LeftBound87881.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound88070

namespace LeftBound88094
def owner : Owner := ⟨.program ⟨214⟩, ⟨10483⟩⟩
def transferEvent : Nat := 88094
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 88092 .coefficient) (.predecessor 1 88093 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 88092 .coefficient)
      LeftAuthority4218.bound (LeftAuthority4218.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events016.exact4219RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority4218.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority4218.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 88093 .coefficient)
      LeftBound79918.bound (LeftBound79918.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events312.exact79920RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound79918.bound, RecordedBoundRefines] <;> decide)
      (LeftBound79918.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32 ⟨true, false, none, none, none⟩ LeftAuthority4218.bound LeftBound79918.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority4218.bound, LeftBound79918.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := tensorFactor 32 ⟨true, false, none, none, none⟩ * (LeftAuthority4218.actual selector witness) * (LeftBound79918.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound88094

namespace LeftBound88099
def owner : Owner := ⟨.program ⟨214⟩, ⟨7228⟩⟩
def transferEvent : Nat := 88099
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 88097 .coefficient) (.predecessor 1 88098 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 88097 .coefficient)
      LeftBound79789.bound (LeftBound79789.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events311.exact79790RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound79789.bound, RecordedBoundRefines] <;> decide)
      (LeftBound79789.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 88098 .coefficient)
      LeftBound14988.bound (LeftBound14988.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events058.exact14989RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14988.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14988.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound79789.bound LeftBound14988.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound79789.bound, LeftBound14988.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound79789.actual selector witness) * (LeftBound14988.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound88099

namespace LeftBound88104
def owner : Owner := ⟨.program ⟨214⟩, ⟨10484⟩⟩
def transferEvent : Nat := 88104
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 88102 .coefficient, .predecessor 1 88103 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 88102 .coefficient)
      LeftBound88099.bound (LeftBound88099.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events344.exact88101RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound88099.bound, RecordedBoundRefines] <;> decide)
      (LeftBound88099.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 88103 .coefficient)
      LeftBound88094.bound (LeftBound88094.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events344.exact88096RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound88094.bound, RecordedBoundRefines] <;> decide)
      (LeftBound88094.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound88099.bound, LeftBound88094.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound88099.bound, LeftBound88094.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound88099.actual selector witness, LeftBound88094.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound88104

namespace LeftBound88108
def owner : Owner := ⟨.program ⟨214⟩, ⟨10485⟩⟩
def transferEvent : Nat := 88108
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 88106 .coefficient, .predecessor 1 88107 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 88106 .coefficient)
      LeftBound88104.bound (LeftBound88104.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events344.exact88105RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound88104.bound, RecordedBoundRefines] <;> decide)
      (LeftBound88104.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 88107 .coefficient)
      LeftBound14980.bound (LeftBound14980.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events058.exact14981RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14980.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14980.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound88104.bound, LeftBound14980.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound88104.bound, LeftBound14980.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound88104.actual selector witness, LeftBound14980.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound88108

namespace LeftBound88109
def owner : Owner := ⟨.program ⟨214⟩, ⟨10485⟩⟩
def transferEvent : Nat := 88109
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨86⟩⟩]⟩ [⟨.result 14981 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 14981 .coefficient)
      LeftBound14980.bound (LeftBound14980.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨86⟩⟩) (rawTerms := some (Proof.Events058.exact14981RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14980.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14980.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound14980.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound14980.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftBound14980.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound88109

namespace LeftBound88114
def owner : Owner := ⟨.program ⟨214⟩, ⟨10486⟩⟩
def transferEvent : Nat := 88114
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 88112 .coefficient) (.predecessor 1 88113 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 88112 .coefficient)
      LeftBound88108.bound (LeftBound88108.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events344.exact88111RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound88108.bound, RecordedBoundRefines] <;> decide)
      (LeftBound88108.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 88113 .coefficient)
      LeftAuthority4221.bound (LeftAuthority4221.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events016.exact4222RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority4221.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority4221.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftBound88108.bound LeftAuthority4221.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound88108.bound, LeftAuthority4221.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftBound88108.actual selector witness) * (LeftAuthority4221.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound88114

namespace LeftBound88115
def owner : Owner := ⟨.program ⟨214⟩, ⟨10486⟩⟩
def transferEvent : Nat := 88115
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨214⟩, ⟨9400⟩⟩], []⟩ [⟨.result 4222 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 4222 .coefficient)
      LeftAuthority4221.bound (LeftAuthority4221.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨9400⟩⟩) (rawTerms := some (Proof.Events016.exact4222RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority4221.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority4221.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority4221.bound []
def bound : CoeffClass := .finite ⟨2, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority4221.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority4221.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound88115

namespace LeftBound88116
def owner : Owner := ⟨.program ⟨214⟩, ⟨10486⟩⟩
def transferEvent : Nat := 88116
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 88111 .summary) (.transfer 88115) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 88111 .summary)
      LeftBound88109.bound (LeftBound88109.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨10485⟩⟩) (rawTerms := some (Proof.Events344.exact88111RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound88109.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 88115)
      LeftBound88115.bound (LeftBound88115.actual selector witness) := by
  exact .transfer (LeftBound88115.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound88109.bound LeftBound88115.bound
def bound : CoeffClass := .finite ⟨1664, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound88109.bound, LeftBound88115.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound88109.actual selector witness) * (LeftBound88115.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound88116

namespace LeftBound88122
def owner : Owner := ⟨.program ⟨214⟩, ⟨9401⟩⟩
def transferEvent : Nat := 88122
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 88120 .coefficient) (.predecessor 1 88121 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 88120 .coefficient)
      LeftAuthority4221.bound (LeftAuthority4221.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events016.exact4222RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority4221.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority4221.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 88121 .coefficient)
      LeftBound79918.bound (LeftBound79918.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events312.exact79920RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound79918.bound, RecordedBoundRefines] <;> decide)
      (LeftBound79918.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32 ⟨true, false, none, none, none⟩ LeftAuthority4221.bound LeftBound79918.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority4221.bound, LeftBound79918.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := tensorFactor 32 ⟨true, false, none, none, none⟩ * (LeftAuthority4221.actual selector witness) * (LeftBound79918.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound88122

namespace LeftBound88127
def owner : Owner := ⟨.program ⟨214⟩, ⟨7227⟩⟩
def transferEvent : Nat := 88127
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 88125 .coefficient) (.predecessor 1 88126 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 88125 .coefficient)
      LeftBound79789.bound (LeftBound79789.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events311.exact79790RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound79789.bound, RecordedBoundRefines] <;> decide)
      (LeftBound79789.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 88126 .coefficient)
      LeftBound15029.bound (LeftBound15029.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events058.exact15030RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound15029.bound, RecordedBoundRefines] <;> decide)
      (LeftBound15029.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound79789.bound LeftBound15029.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound79789.bound, LeftBound15029.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound79789.actual selector witness) * (LeftBound15029.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound88127

namespace LeftBound88132
def owner : Owner := ⟨.program ⟨214⟩, ⟨9402⟩⟩
def transferEvent : Nat := 88132
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 88130 .coefficient, .predecessor 1 88131 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 88130 .coefficient)
      LeftBound88127.bound (LeftBound88127.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events344.exact88129RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound88127.bound, RecordedBoundRefines] <;> decide)
      (LeftBound88127.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 88131 .coefficient)
      LeftBound88122.bound (LeftBound88122.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events344.exact88124RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound88122.bound, RecordedBoundRefines] <;> decide)
      (LeftBound88122.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound88127.bound, LeftBound88122.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound88127.bound, LeftBound88122.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound88127.actual selector witness, LeftBound88122.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound88132

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
