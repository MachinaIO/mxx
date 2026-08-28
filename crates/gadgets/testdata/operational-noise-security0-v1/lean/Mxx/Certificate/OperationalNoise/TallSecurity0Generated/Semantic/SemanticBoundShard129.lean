import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard021
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard025
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard123
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard124
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard126
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard127
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard128

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound20916
def owner : Owner := ⟨.program ⟨214⟩, ⟨7368⟩⟩
def transferEvent : Nat := 20916
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 20914 .coefficient) (.predecessor 1 20915 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 20914 .coefficient)
      LeftBound6313.bound (LeftBound6313.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events024.exact6314RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6313.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6313.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 20915 .coefficient)
      LeftBound5872.bound (LeftBound5872.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events022.exact5873RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5872.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5872.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound6313.bound LeftBound5872.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound6313.bound, LeftBound5872.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound6313.actual selector witness) * (LeftBound5872.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound20916

namespace LeftBound20921
def owner : Owner := ⟨.program ⟨214⟩, ⟨7769⟩⟩
def transferEvent : Nat := 20921
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 20919 .coefficient, .predecessor 1 20920 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 20919 .coefficient)
      LeftBound20916.bound (LeftBound20916.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events081.exact20918RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound20916.bound, RecordedBoundRefines] <;> decide)
      (LeftBound20916.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 20920 .coefficient)
      LeftBound20911.bound (LeftBound20911.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events081.exact20913RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound20911.bound, RecordedBoundRefines] <;> decide)
      (LeftBound20911.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound20916.bound, LeftBound20911.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound20916.bound, LeftBound20911.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound20916.actual selector witness, LeftBound20911.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound20921

namespace LeftBound20925
def owner : Owner := ⟨.program ⟨214⟩, ⟨7770⟩⟩
def transferEvent : Nat := 20925
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 20923 .coefficient, .predecessor 1 20924 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 20923 .coefficient)
      LeftBound20921.bound (LeftBound20921.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events081.exact20922RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound20921.bound, RecordedBoundRefines] <;> decide)
      (LeftBound20921.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 20924 .coefficient)
      LeftBound20907.bound (LeftBound20907.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events081.exact20908RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound20907.bound, RecordedBoundRefines] <;> decide)
      (LeftBound20907.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound20921.bound, LeftBound20907.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound20921.bound, LeftBound20907.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound20921.actual selector witness, LeftBound20907.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound20925

namespace LeftBound20926
def owner : Owner := ⟨.program ⟨214⟩, ⟨7770⟩⟩
def transferEvent : Nat := 20926
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨74⟩⟩]⟩ [⟨.result 20908 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 20908 .coefficient)
      LeftBound20907.bound (LeftBound20907.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨74⟩⟩) (rawTerms := some (Proof.Events081.exact20908RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound20907.bound, RecordedBoundRefines] <;> decide)
      (LeftBound20907.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound20907.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound20907.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftBound20907.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound20926

namespace LeftBound20931
def owner : Owner := ⟨.program ⟨214⟩, ⟨7812⟩⟩
def transferEvent : Nat := 20931
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 20929 .coefficient, .predecessor 1 20930 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 20929 .coefficient)
      LeftBound20925.bound (LeftBound20925.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events081.exact20928RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound20925.bound, RecordedBoundRefines] <;> decide)
      (LeftBound20925.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 20930 .coefficient)
      LeftBound20925.bound (LeftBound20925.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events081.exact20928RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound20925.bound, RecordedBoundRefines] <;> decide)
      (LeftBound20925.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound20925.bound, LeftBound20925.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound20925.bound, LeftBound20925.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound20925.actual selector witness, LeftBound20925.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound20931

namespace LeftBound20934
def owner : Owner := ⟨.program ⟨214⟩, ⟨7812⟩⟩
def transferEvent : Nat := 20934
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 20928 .summary, .result 20928 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 20928 .summary)
      LeftBound20926.bound (LeftBound20926.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨7770⟩⟩) (rawTerms := some (Proof.Events081.exact20928RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound20926.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 20928 .summary)
      LeftBound20926.bound (LeftBound20926.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨7770⟩⟩) (rawTerms := some (Proof.Events081.exact20928RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound20926.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound20926.bound, LeftBound20926.bound]
def bound : CoeffClass := .finite ⟨52, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound20926.bound, LeftBound20926.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound20926.actual selector witness, LeftBound20926.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound20934

namespace LeftBound20938
def owner : Owner := ⟨.program ⟨214⟩, ⟨26404⟩⟩
def transferEvent : Nat := 20938
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 20936 .coefficient, .predecessor 1 20937 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 20936 .coefficient)
      LeftBound20931.bound (LeftBound20931.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events081.exact20935RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound20931.bound, RecordedBoundRefines] <;> decide)
      (LeftBound20931.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 20937 .coefficient)
      LeftBound20898.bound (LeftBound20898.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events081.exact20905RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound20898.bound, RecordedBoundRefines] <;> decide)
      (LeftBound20898.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound20931.bound, LeftBound20898.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound20931.bound, LeftBound20898.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound20931.actual selector witness, LeftBound20898.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound20938

namespace LeftBound20939
def owner : Owner := ⟨.program ⟨214⟩, ⟨26404⟩⟩
def transferEvent : Nat := 20939
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 20935 .summary, .result 20905 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 20935 .summary)
      LeftBound20934.bound (LeftBound20934.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨7812⟩⟩) (rawTerms := some (Proof.Events081.exact20935RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound20934.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 20905 .summary)
      LeftBound20900.bound (LeftBound20900.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26403⟩⟩) (rawTerms := some (Proof.Events081.exact20905RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound20900.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound20934.bound, LeftBound20900.bound]
def bound : CoeffClass := .finite ⟨4741253940199267499646124084, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound20934.bound, LeftBound20900.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound20934.actual selector witness, LeftBound20900.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound20939

namespace LeftBound20943
def owner : Owner := ⟨.program ⟨214⟩, ⟨26614⟩⟩
def transferEvent : Nat := 20943
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 20941 .coefficient, .predecessor 1 20942 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 20941 .coefficient)
      LeftBound20938.bound (LeftBound20938.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events081.exact20940RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound20938.bound, RecordedBoundRefines] <;> decide)
      (LeftBound20938.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 20942 .coefficient)
      LeftBound20686.bound (LeftBound20686.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events080.exact20693RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound20686.bound, RecordedBoundRefines] <;> decide)
      (LeftBound20686.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound20938.bound, LeftBound20686.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound20938.bound, LeftBound20686.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound20938.actual selector witness, LeftBound20686.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound20943

namespace LeftBound20944
def owner : Owner := ⟨.program ⟨214⟩, ⟨26614⟩⟩
def transferEvent : Nat := 20944
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 20940 .summary, .result 20693 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 20940 .summary)
      LeftBound20939.bound (LeftBound20939.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26404⟩⟩) (rawTerms := some (Proof.Events081.exact20940RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound20939.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 20693 .summary)
      LeftBound20688.bound (LeftBound20688.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26613⟩⟩) (rawTerms := some (Proof.Events080.exact20693RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound20688.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound20939.bound, LeftBound20688.bound]
def bound : CoeffClass := .finite ⟨9482549007414447334737575988, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound20939.bound, LeftBound20688.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound20939.actual selector witness, LeftBound20688.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound20944

namespace LeftBound20948
def owner : Owner := ⟨.program ⟨214⟩, ⟨26831⟩⟩
def transferEvent : Nat := 20948
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 20946 .coefficient, .predecessor 1 20947 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 20946 .coefficient)
      LeftBound20943.bound (LeftBound20943.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events081.exact20945RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound20943.bound, RecordedBoundRefines] <;> decide)
      (LeftBound20943.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 20947 .coefficient)
      LeftBound20474.bound (LeftBound20474.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events080.exact20481RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound20474.bound, RecordedBoundRefines] <;> decide)
      (LeftBound20474.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound20943.bound, LeftBound20474.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound20943.bound, LeftBound20474.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound20943.actual selector witness, LeftBound20474.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound20948

namespace LeftBound20949
def owner : Owner := ⟨.program ⟨214⟩, ⟨26831⟩⟩
def transferEvent : Nat := 20949
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 20945 .summary, .result 20481 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 20945 .summary)
      LeftBound20944.bound (LeftBound20944.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26614⟩⟩) (rawTerms := some (Proof.Events081.exact20945RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound20944.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 20481 .summary)
      LeftBound20476.bound (LeftBound20476.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26830⟩⟩) (rawTerms := some (Proof.Events080.exact20481RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound20476.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound20944.bound, LeftBound20476.bound]
def bound : CoeffClass := .finite ⟨14223885201645539505274355764, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound20944.bound, LeftBound20476.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound20944.actual selector witness, LeftBound20476.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound20949

namespace LeftBound20953
def owner : Owner := ⟨.program ⟨214⟩, ⟨27048⟩⟩
def transferEvent : Nat := 20953
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 20951 .coefficient, .predecessor 1 20952 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 20951 .coefficient)
      LeftBound20948.bound (LeftBound20948.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events081.exact20950RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound20948.bound, RecordedBoundRefines] <;> decide)
      (LeftBound20948.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 20952 .coefficient)
      LeftBound20262.bound (LeftBound20262.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events079.exact20269RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound20262.bound, RecordedBoundRefines] <;> decide)
      (LeftBound20262.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound20948.bound, LeftBound20262.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound20948.bound, LeftBound20262.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound20948.actual selector witness, LeftBound20262.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound20953

namespace LeftBound20954
def owner : Owner := ⟨.program ⟨214⟩, ⟨27048⟩⟩
def transferEvent : Nat := 20954
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 20950 .summary, .result 20269 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 20950 .summary)
      LeftBound20949.bound (LeftBound20949.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26831⟩⟩) (rawTerms := some (Proof.Events081.exact20950RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound20949.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 20269 .summary)
      LeftBound20264.bound (LeftBound20264.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27047⟩⟩) (rawTerms := some (Proof.Events079.exact20269RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound20264.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound20949.bound, LeftBound20264.bound]
def bound : CoeffClass := .finite ⟨18965303649908456346701791284, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound20949.bound, LeftBound20264.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound20949.actual selector witness, LeftBound20264.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound20954

namespace LeftBound20958
def owner : Owner := ⟨.program ⟨214⟩, ⟨27265⟩⟩
def transferEvent : Nat := 20958
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 20956 .coefficient, .predecessor 1 20957 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 20956 .coefficient)
      LeftBound20953.bound (LeftBound20953.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events081.exact20955RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound20953.bound, RecordedBoundRefines] <;> decide)
      (LeftBound20953.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 20957 .coefficient)
      LeftBound20050.bound (LeftBound20050.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events078.exact20057RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound20050.bound, RecordedBoundRefines] <;> decide)
      (LeftBound20050.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound20953.bound, LeftBound20050.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound20953.bound, LeftBound20050.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound20953.actual selector witness, LeftBound20050.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound20958

namespace LeftBound20959
def owner : Owner := ⟨.program ⟨214⟩, ⟨27265⟩⟩
def transferEvent : Nat := 20959
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 20955 .summary, .result 20057 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 20955 .summary)
      LeftBound20954.bound (LeftBound20954.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27048⟩⟩) (rawTerms := some (Proof.Events081.exact20955RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound20954.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 20057 .summary)
      LeftBound20052.bound (LeftBound20052.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27264⟩⟩) (rawTerms := some (Proof.Events078.exact20057RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound20052.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound20954.bound, LeftBound20052.bound]
def bound : CoeffClass := .finite ⟨23706886606235022529910538292, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound20954.bound, LeftBound20052.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound20954.actual selector witness, LeftBound20052.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound20959

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
