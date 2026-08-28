import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard000
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard088
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard641
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard694

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound100916
def owner : Owner := ⟨.program ⟨214⟩, ⟨10959⟩⟩
def transferEvent : Nat := 100916
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 100911 .summary) (.transfer 100915) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 100911 .summary)
      LeftBound100909.bound (LeftBound100909.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨10958⟩⟩) (rawTerms := some (Proof.Events394.exact100911RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound100909.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 100915)
      LeftBound100915.bound (LeftBound100915.actual selector witness) := by
  exact .transfer (LeftBound100915.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound100909.bound LeftBound100915.bound
def bound : CoeffClass := .finite ⟨3328, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound100909.bound, LeftBound100915.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound100909.actual selector witness) * (LeftBound100915.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound100916

namespace LeftBound100922
def owner : Owner := ⟨.program ⟨214⟩, ⟨10828⟩⟩
def transferEvent : Nat := 100922
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 100920 .coefficient) (.predecessor 1 100921 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 100920 .coefficient)
      LeftAuthority4913.bound (LeftAuthority4913.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events019.exact4914RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority4913.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority4913.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 100921 .coefficient)
      LeftBound30.bound (LeftBound30.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events000.exact32RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound30.bound, RecordedBoundRefines] <;> decide)
      (LeftBound30.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32 ⟨true, false, none, none, none⟩ LeftAuthority4913.bound LeftBound30.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority4913.bound, LeftBound30.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := tensorFactor 32 ⟨true, false, none, none, none⟩ * (LeftAuthority4913.actual selector witness) * (LeftBound30.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound100922

namespace LeftBound100927
def owner : Owner := ⟨.program ⟨214⟩, ⟨7128⟩⟩
def transferEvent : Nat := 100927
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 100925 .coefficient) (.predecessor 1 100926 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 100925 .coefficient)
      LeftBound26.bound (LeftBound26.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events000.exact27RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound26.bound, RecordedBoundRefines] <;> decide)
      (LeftBound26.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 100926 .coefficient)
      LeftBound14027.bound (LeftBound14027.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events054.exact14028RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14027.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14027.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound26.bound LeftBound14027.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound26.bound, LeftBound14027.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound26.actual selector witness) * (LeftBound14027.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound100927

namespace LeftBound100932
def owner : Owner := ⟨.program ⟨214⟩, ⟨10829⟩⟩
def transferEvent : Nat := 100932
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 100930 .coefficient, .predecessor 1 100931 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 100930 .coefficient)
      LeftBound100927.bound (LeftBound100927.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events394.exact100929RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound100927.bound, RecordedBoundRefines] <;> decide)
      (LeftBound100927.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 100931 .coefficient)
      LeftBound100922.bound (LeftBound100922.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events394.exact100924RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound100922.bound, RecordedBoundRefines] <;> decide)
      (LeftBound100922.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound100927.bound, LeftBound100922.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound100927.bound, LeftBound100922.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound100927.actual selector witness, LeftBound100922.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound100932

namespace LeftBound100936
def owner : Owner := ⟨.program ⟨214⟩, ⟨10830⟩⟩
def transferEvent : Nat := 100936
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 100934 .coefficient, .predecessor 1 100935 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 100934 .coefficient)
      LeftBound100932.bound (LeftBound100932.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events394.exact100933RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound100932.bound, RecordedBoundRefines] <;> decide)
      (LeftBound100932.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 100935 .coefficient)
      LeftBound14019.bound (LeftBound14019.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events054.exact14020RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14019.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14019.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound100932.bound, LeftBound14019.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound100932.bound, LeftBound14019.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound100932.actual selector witness, LeftBound14019.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound100936

namespace LeftBound100937
def owner : Owner := ⟨.program ⟨214⟩, ⟨10830⟩⟩
def transferEvent : Nat := 100937
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨105⟩⟩]⟩ [⟨.result 14020 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 14020 .coefficient)
      LeftBound14019.bound (LeftBound14019.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨105⟩⟩) (rawTerms := some (Proof.Events054.exact14020RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14019.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14019.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound14019.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound14019.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftBound14019.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound100937

namespace LeftBound100942
def owner : Owner := ⟨.program ⟨214⟩, ⟨10831⟩⟩
def transferEvent : Nat := 100942
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 100940 .coefficient) (.predecessor 1 100941 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 100940 .coefficient)
      LeftBound100936.bound (LeftBound100936.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events394.exact100939RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound100936.bound, RecordedBoundRefines] <;> decide)
      (LeftBound100936.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 100941 .coefficient)
      LeftBound14016.bound (LeftBound14016.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events054.exact14017RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14016.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14016.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound100936.bound LeftBound14016.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound100936.bound, LeftBound14016.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound100936.actual selector witness) * (LeftBound14016.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound100942

namespace LeftBound100943
def owner : Owner := ⟨.program ⟨214⟩, ⟨10831⟩⟩
def transferEvent : Nat := 100943
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨7837⟩⟩]⟩ [⟨.result 14013 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 14013 .coefficient)
      LeftAuthority14012.bound (LeftAuthority14012.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨7837⟩⟩) (rawTerms := some (Proof.Events054.exact14013RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority14012.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority14012.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority14012.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority14012.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority14012.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound100943

namespace LeftBound100944
def owner : Owner := ⟨.program ⟨214⟩, ⟨10831⟩⟩
def transferEvent : Nat := 100944
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 100939 .summary) (.transfer 100943) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 100939 .summary)
      LeftBound100937.bound (LeftBound100937.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨10830⟩⟩) (rawTerms := some (Proof.Events394.exact100939RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound100937.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 100943)
      LeftBound100943.bound (LeftBound100943.actual selector witness) := by
  exact .transfer (LeftBound100943.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound100937.bound LeftBound100943.bound
def bound : CoeffClass := .finite ⟨95420416, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound100937.bound, LeftBound100943.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound100937.actual selector witness) * (LeftBound100943.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound100944

namespace LeftBound100952
def owner : Owner := ⟨.program ⟨214⟩, ⟨10960⟩⟩
def transferEvent : Nat := 100952
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 100950 .coefficient, .predecessor 1 100951 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 100950 .coefficient)
      LeftBound100942.bound (LeftBound100942.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events394.exact100949RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound100942.bound, RecordedBoundRefines] <;> decide)
      (LeftBound100942.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 100951 .coefficient)
      LeftBound100914.bound (LeftBound100914.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events394.exact100919RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound100914.bound, RecordedBoundRefines] <;> decide)
      (LeftBound100914.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound100942.bound, LeftBound100914.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound100942.bound, LeftBound100914.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound100942.actual selector witness, LeftBound100914.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound100952

namespace LeftBound100954
def owner : Owner := ⟨.program ⟨214⟩, ⟨10960⟩⟩
def transferEvent : Nat := 100954
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 100949 .summary, .result 100919 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 100949 .summary)
      LeftBound100944.bound (LeftBound100944.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨10831⟩⟩) (rawTerms := some (Proof.Events394.exact100949RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound100944.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 100919 .summary)
      LeftBound100916.bound (LeftBound100916.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨10959⟩⟩) (rawTerms := some (Proof.Events394.exact100919RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound100916.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound100944.bound, LeftBound100916.bound]
def bound : CoeffClass := .finite ⟨95423744, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound100944.bound, LeftBound100916.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound100944.actual selector witness, LeftBound100916.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound100954

namespace LeftBound100958
def owner : Owner := ⟨.program ⟨214⟩, ⟨25053⟩⟩
def transferEvent : Nat := 100958
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 100956 .coefficient) (.predecessor 1 100957 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 100956 .coefficient)
      LeftBound100952.bound (LeftBound100952.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events394.exact100955RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound100952.bound, RecordedBoundRefines] <;> decide)
      (LeftBound100952.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 100957 .coefficient)
      LeftAuthority100890.bound (LeftAuthority100890.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events394.exact100891RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority100890.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority100890.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound100952.bound LeftAuthority100890.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound100952.bound, LeftAuthority100890.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound100952.actual selector witness) * (LeftAuthority100890.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound100958

namespace LeftBound100959
def owner : Owner := ⟨.program ⟨214⟩, ⟨25053⟩⟩
def transferEvent : Nat := 100959
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨25052⟩⟩]⟩ [⟨.result 100891 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 100891 .coefficient)
      LeftAuthority100890.bound (LeftAuthority100890.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨25052⟩⟩) (rawTerms := some (Proof.Events394.exact100891RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority100890.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority100890.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority100890.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority100890.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority100890.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound100959

namespace LeftBound100960
def owner : Owner := ⟨.program ⟨214⟩, ⟨25053⟩⟩
def transferEvent : Nat := 100960
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 100955 .summary) (.transfer 100959) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 100955 .summary)
      LeftBound100954.bound (LeftBound100954.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨10960⟩⟩) (rawTerms := some (Proof.Events394.exact100955RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound100954.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 100959)
      LeftBound100959.bound (LeftBound100959.actual selector witness) := by
  exact .transfer (LeftBound100959.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound100954.bound LeftBound100959.bound
def bound : CoeffClass := .finite ⟨350206667259904, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound100954.bound, LeftBound100959.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound100954.actual selector witness) * (LeftBound100959.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound100960

namespace LeftBound100971
def owner : Owner := ⟨.program ⟨214⟩, ⟨19159⟩⟩
def transferEvent : Nat := 100971
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 100969 .coefficient) (.value (.predecessor 1 100970 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 100969 .coefficient)
      LeftAuthority100967.bound (LeftAuthority100967.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events394.exact100968RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority100967.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority100967.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 100970 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority100967.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority100967.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority100967.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound100971

namespace LeftBound100975
def owner : Owner := ⟨.program ⟨214⟩, ⟨19160⟩⟩
def transferEvent : Nat := 100975
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 100973 .coefficient) (.predecessor 1 100974 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 100973 .coefficient)
      LeftBound94459.bound (LeftBound94459.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events368.exact94462RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound94459.bound, RecordedBoundRefines] <;> decide)
      (LeftBound94459.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 100974 .coefficient)
      LeftBound100971.bound (LeftBound100971.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events394.exact100972RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound100971.bound, RecordedBoundRefines] <;> decide)
      (LeftBound100971.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound94459.bound LeftBound100971.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94459.bound, LeftBound100971.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound94459.actual selector witness) * (LeftBound100971.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound100975

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
