import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard025
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard030
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard072

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound12027
def owner : Owner := ⟨.program ⟨214⟩, ⟨7366⟩⟩
def transferEvent : Nat := 12027
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 12025 .coefficient) (.predecessor 1 12026 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 12025 .coefficient)
      LeftBound6313.bound (LeftBound6313.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events024.exact6314RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6313.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6313.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 12026 .coefficient)
      LeftBound12023.bound (LeftBound12023.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events046.exact12024RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound12023.bound, RecordedBoundRefines] <;> decide)
      (LeftBound12023.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound6313.bound LeftBound12023.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound6313.bound, LeftBound12023.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound6313.actual selector witness) * (LeftBound12023.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound12027

namespace LeftBound12032
def owner : Owner := ⟨.program ⟨214⟩, ⟨14031⟩⟩
def transferEvent : Nat := 12032
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 12030 .coefficient, .predecessor 1 12031 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 12030 .coefficient)
      LeftBound12027.bound (LeftBound12027.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events046.exact12029RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound12027.bound, RecordedBoundRefines] <;> decide)
      (LeftBound12027.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 12031 .coefficient)
      LeftBound12019.bound (LeftBound12019.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events046.exact12021RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound12019.bound, RecordedBoundRefines] <;> decide)
      (LeftBound12019.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound12027.bound, LeftBound12019.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound12027.bound, LeftBound12019.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound12027.actual selector witness, LeftBound12019.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound12032

namespace LeftBound12036
def owner : Owner := ⟨.program ⟨214⟩, ⟨14032⟩⟩
def transferEvent : Nat := 12036
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 12034 .coefficient, .predecessor 1 12035 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 12034 .coefficient)
      LeftBound12032.bound (LeftBound12032.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events047.exact12033RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound12032.bound, RecordedBoundRefines] <;> decide)
      (LeftBound12032.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 12035 .coefficient)
      LeftBound12015.bound (LeftBound12015.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events046.exact12016RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound12015.bound, RecordedBoundRefines] <;> decide)
      (LeftBound12015.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound12032.bound, LeftBound12015.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound12032.bound, LeftBound12015.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound12032.actual selector witness, LeftBound12015.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound12036

namespace LeftBound12037
def owner : Owner := ⟨.program ⟨214⟩, ⟨14032⟩⟩
def transferEvent : Nat := 12037
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨72⟩⟩]⟩ [⟨.result 12016 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 12016 .coefficient)
      LeftBound12015.bound (LeftBound12015.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨72⟩⟩) (rawTerms := some (Proof.Events046.exact12016RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound12015.bound, RecordedBoundRefines] <;> decide)
      (LeftBound12015.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound12015.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound12015.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftBound12015.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound12037

namespace LeftBound12042
def owner : Owner := ⟨.program ⟨214⟩, ⟨14033⟩⟩
def transferEvent : Nat := 12042
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 12040 .coefficient) (.predecessor 1 12041 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 12040 .coefficient)
      LeftBound12036.bound (LeftBound12036.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events047.exact12039RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound12036.bound, RecordedBoundRefines] <;> decide)
      (LeftBound12036.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 12041 .coefficient)
      LeftBound12012.bound (LeftBound12012.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events046.exact12013RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound12012.bound, RecordedBoundRefines] <;> decide)
      (LeftBound12012.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound12036.bound LeftBound12012.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound12036.bound, LeftBound12012.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound12036.actual selector witness) * (LeftBound12012.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound12042

namespace LeftBound12043
def owner : Owner := ⟨.program ⟨214⟩, ⟨14033⟩⟩
def transferEvent : Nat := 12043
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨7849⟩⟩]⟩ [⟨.result 12009 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 12009 .coefficient)
      LeftAuthority12008.bound (LeftAuthority12008.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨7849⟩⟩) (rawTerms := some (Proof.Events046.exact12009RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority12008.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority12008.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority12008.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority12008.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority12008.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound12043

namespace LeftBound12044
def owner : Owner := ⟨.program ⟨214⟩, ⟨14033⟩⟩
def transferEvent : Nat := 12044
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 12039 .summary) (.transfer 12043) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 12039 .summary)
      LeftBound12037.bound (LeftBound12037.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨14032⟩⟩) (rawTerms := some (Proof.Events047.exact12039RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound12037.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 12043)
      LeftBound12043.bound (LeftBound12043.actual selector witness) := by
  exact .transfer (LeftBound12043.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound12037.bound LeftBound12043.bound
def bound : CoeffClass := .finite ⟨95420416, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound12037.bound, LeftBound12043.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound12037.actual selector witness) * (LeftBound12043.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound12044

namespace LeftBound12052
def owner : Owner := ⟨.program ⟨214⟩, ⟨14034⟩⟩
def transferEvent : Nat := 12052
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 12050 .coefficient, .predecessor 1 12051 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 12050 .coefficient)
      LeftBound12042.bound (LeftBound12042.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events047.exact12049RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound12042.bound, RecordedBoundRefines] <;> decide)
      (LeftBound12042.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 12051 .coefficient)
      LeftBound12001.bound (LeftBound12001.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events046.exact12006RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound12001.bound, RecordedBoundRefines] <;> decide)
      (LeftBound12001.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound12042.bound, LeftBound12001.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound12042.bound, LeftBound12001.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound12042.actual selector witness, LeftBound12001.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound12052

namespace LeftBound12054
def owner : Owner := ⟨.program ⟨214⟩, ⟨14034⟩⟩
def transferEvent : Nat := 12054
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 12049 .summary, .result 12006 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 12049 .summary)
      LeftBound12044.bound (LeftBound12044.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨14033⟩⟩) (rawTerms := some (Proof.Events047.exact12049RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound12044.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 12006 .summary)
      LeftBound12003.bound (LeftBound12003.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨14029⟩⟩) (rawTerms := some (Proof.Events046.exact12006RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound12003.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound12044.bound, LeftBound12003.bound]
def bound : CoeffClass := .finite ⟨95433728, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound12044.bound, LeftBound12003.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound12044.actual selector witness, LeftBound12003.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound12054

namespace LeftBound12058
def owner : Owner := ⟨.program ⟨214⟩, ⟨26010⟩⟩
def transferEvent : Nat := 12058
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 12056 .coefficient) (.predecessor 1 12057 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 12056 .coefficient)
      LeftBound12052.bound (LeftBound12052.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events047.exact12055RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound12052.bound, RecordedBoundRefines] <;> decide)
      (LeftBound12052.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 12057 .coefficient)
      LeftAuthority11971.bound (LeftAuthority11971.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events046.exact11972RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority11971.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority11971.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound12052.bound LeftAuthority11971.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound12052.bound, LeftAuthority11971.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound12052.actual selector witness) * (LeftAuthority11971.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound12058

namespace LeftBound12059
def owner : Owner := ⟨.program ⟨214⟩, ⟨26010⟩⟩
def transferEvent : Nat := 12059
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨26009⟩⟩]⟩ [⟨.result 11972 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 11972 .coefficient)
      LeftAuthority11971.bound (LeftAuthority11971.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨26009⟩⟩) (rawTerms := some (Proof.Events046.exact11972RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority11971.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority11971.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority11971.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority11971.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority11971.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound12059

namespace LeftBound12060
def owner : Owner := ⟨.program ⟨214⟩, ⟨26010⟩⟩
def transferEvent : Nat := 12060
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 12055 .summary) (.transfer 12059) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 12055 .summary)
      LeftBound12054.bound (LeftBound12054.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨14034⟩⟩) (rawTerms := some (Proof.Events047.exact12055RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound12054.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 12059)
      LeftBound12059.bound (LeftBound12059.actual selector witness) := by
  exact .transfer (LeftBound12059.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound12054.bound LeftBound12059.bound
def bound : CoeffClass := .finite ⟨350243308699648, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound12054.bound, LeftBound12059.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound12054.actual selector witness) * (LeftBound12059.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound12060

namespace LeftBound12071
def owner : Owner := ⟨.program ⟨214⟩, ⟨19474⟩⟩
def transferEvent : Nat := 12071
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 12069 .coefficient) (.value (.predecessor 1 12070 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 12069 .coefficient)
      LeftAuthority12067.bound (LeftAuthority12067.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events047.exact12068RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority12067.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority12067.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 12070 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority12067.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority12067.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority12067.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound12071

namespace LeftBound12075
def owner : Owner := ⟨.program ⟨214⟩, ⟨19475⟩⟩
def transferEvent : Nat := 12075
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 12073 .coefficient) (.predecessor 1 12074 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 12073 .coefficient)
      LeftBound6558.bound (LeftBound6558.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events025.exact6561RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6558.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6558.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 12074 .coefficient)
      LeftBound12071.bound (LeftBound12071.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events047.exact12072RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound12071.bound, RecordedBoundRefines] <;> decide)
      (LeftBound12071.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound6558.bound LeftBound12071.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound6558.bound, LeftBound12071.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound6558.actual selector witness) * (LeftBound12071.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound12075

namespace LeftBound12076
def owner : Owner := ⟨.program ⟨214⟩, ⟨19475⟩⟩
def transferEvent : Nat := 12076
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨19472⟩⟩]⟩ [⟨.result 12068 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 12068 .coefficient)
      LeftAuthority12067.bound (LeftAuthority12067.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨19472⟩⟩) (rawTerms := some (Proof.Events047.exact12068RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority12067.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority12067.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority12067.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority12067.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority12067.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound12076

namespace LeftBound12077
def owner : Owner := ⟨.program ⟨214⟩, ⟨19475⟩⟩
def transferEvent : Nat := 12077
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 6561 .summary) (.transfer 12076) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 6561 .summary)
      LeftBound6559.bound (LeftBound6559.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5565⟩⟩) (rawTerms := some (Proof.Events025.exact6561RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound6559.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 12076)
      LeftBound12076.bound (LeftBound12076.actual selector witness) := by
  exact .transfer (LeftBound12076.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound6559.bound LeftBound12076.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound6559.bound, LeftBound12076.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound6559.actual selector witness) * (LeftBound12076.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound12077

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
