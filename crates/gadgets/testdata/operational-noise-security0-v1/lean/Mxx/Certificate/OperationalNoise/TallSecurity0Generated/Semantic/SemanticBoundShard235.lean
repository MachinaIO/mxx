import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard234

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound35938
def owner : Owner := ⟨.program ⟨214⟩, ⟨18876⟩⟩
def transferEvent : Nat := 35938
def frameStart : Nat := 0
def rule : BoundRule := .sum [.transfer 35936, .transfer 35937]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 35936)
      LeftBound35936.bound (LeftBound35936.actual selector witness) := by
  exact .transfer (LeftBound35936.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 35937)
      LeftBound35937.bound (LeftBound35937.actual selector witness) := by
  exact .transfer (LeftBound35937.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound35936.bound, LeftBound35937.bound]
def bound : CoeffClass := .finite ⟨4683713856341753228595600, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound35936.bound, LeftBound35937.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound35936.actual selector witness, LeftBound35937.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound35938

namespace LeftBound35939
def owner : Owner := ⟨.program ⟨214⟩, ⟨18876⟩⟩
def transferEvent : Nat := 35939
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨214⟩, ⟨6449⟩⟩, ⟨.program ⟨214⟩, ⟨17502⟩⟩], []⟩ [⟨.result 563 .coefficient, true, some 1⟩, ⟨.result 2099 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 563 .coefficient)
      LeftAuthority562.bound (LeftAuthority562.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨6449⟩⟩) (rawTerms := some (Proof.Events002.exact563RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority562.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority562.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 2099 .coefficient)
      LeftAuthority2098.bound (LeftAuthority2098.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨17502⟩⟩) (rawTerms := some (Proof.Events008.exact2099RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority2098.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority2098.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority562.bound [LeftAuthority2098.bound]
def bound : CoeffClass := .finite ⟨230150786063741980797360, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority562.bound, LeftAuthority2098.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority562.actual selector witness) * ([LeftAuthority2098.actual selector witness].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.cons (.intro (input1 selector witness)) (.nil))
end LeftBound35939

namespace LeftBound35940
def owner : Owner := ⟨.program ⟨214⟩, ⟨18876⟩⟩
def transferEvent : Nat := 35940
def frameStart : Nat := 0
def rule : BoundRule := .sum [.transfer 35938, .transfer 35939]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 35938)
      LeftBound35938.bound (LeftBound35938.actual selector witness) := by
  exact .transfer (LeftBound35938.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 35939)
      LeftBound35939.bound (LeftBound35939.actual selector witness) := by
  exact .transfer (LeftBound35939.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound35938.bound, LeftBound35939.bound]
def bound : CoeffClass := .finite ⟨4913864642405495209392960, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound35938.bound, LeftBound35939.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound35938.actual selector witness, LeftBound35939.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound35940

namespace LeftBound35941
def owner : Owner := ⟨.program ⟨214⟩, ⟨18876⟩⟩
def transferEvent : Nat := 35941
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨214⟩, ⟨6459⟩⟩, ⟨.program ⟨214⟩, ⟨17726⟩⟩], []⟩ [⟨.result 573 .coefficient, true, some 1⟩, ⟨.result 2107 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 573 .coefficient)
      LeftAuthority572.bound (LeftAuthority572.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨6459⟩⟩) (rawTerms := some (Proof.Events002.exact573RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority572.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority572.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 2107 .coefficient)
      LeftAuthority2106.bound (LeftAuthority2106.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨17726⟩⟩) (rawTerms := some (Proof.Events008.exact2107RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority2106.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority2106.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority572.bound [LeftAuthority2106.bound]
def bound : CoeffClass := .finite ⟨229585767767349815541720, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority572.bound, LeftAuthority2106.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority572.actual selector witness) * ([LeftAuthority2106.actual selector witness].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.cons (.intro (input1 selector witness)) (.nil))
end LeftBound35941

namespace LeftBound35942
def owner : Owner := ⟨.program ⟨214⟩, ⟨18876⟩⟩
def transferEvent : Nat := 35942
def frameStart : Nat := 0
def rule : BoundRule := .sum [.transfer 35940, .transfer 35941]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 35940)
      LeftBound35940.bound (LeftBound35940.actual selector witness) := by
  exact .transfer (LeftBound35940.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 35941)
      LeftBound35941.bound (LeftBound35941.actual selector witness) := by
  exact .transfer (LeftBound35941.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound35940.bound, LeftBound35941.bound]
def bound : CoeffClass := .finite ⟨5143450410172845024934680, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound35940.bound, LeftBound35941.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound35940.actual selector witness, LeftBound35941.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound35942

namespace LeftBound35943
def owner : Owner := ⟨.program ⟨214⟩, ⟨18876⟩⟩
def transferEvent : Nat := 35943
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨214⟩, ⟨6467⟩⟩, ⟨.program ⟨214⟩, ⟨17957⟩⟩], []⟩ [⟨.result 583 .coefficient, true, some 1⟩, ⟨.result 2115 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 583 .coefficient)
      LeftAuthority582.bound (LeftAuthority582.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨6467⟩⟩) (rawTerms := some (Proof.Events002.exact583RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority582.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority582.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 2115 .coefficient)
      LeftAuthority2114.bound (LeftAuthority2114.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨17957⟩⟩) (rawTerms := some (Proof.Events008.exact2115RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority2114.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority2114.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority582.bound [LeftAuthority2114.bound]
def bound : CoeffClass := .finite ⟨229121489167213617734760, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority582.bound, LeftAuthority2114.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority582.actual selector witness) * ([LeftAuthority2114.actual selector witness].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.cons (.intro (input1 selector witness)) (.nil))
end LeftBound35943

namespace LeftBound35944
def owner : Owner := ⟨.program ⟨214⟩, ⟨18876⟩⟩
def transferEvent : Nat := 35944
def frameStart : Nat := 0
def rule : BoundRule := .sum [.transfer 35942, .transfer 35943]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 35942)
      LeftBound35942.bound (LeftBound35942.actual selector witness) := by
  exact .transfer (LeftBound35942.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 35943)
      LeftBound35943.bound (LeftBound35943.actual selector witness) := by
  exact .transfer (LeftBound35943.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound35942.bound, LeftBound35943.bound]
def bound : CoeffClass := .finite ⟨5372571899340058642669440, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound35942.bound, LeftBound35943.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound35942.actual selector witness, LeftBound35943.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound35944

namespace LeftBound35945
def owner : Owner := ⟨.program ⟨214⟩, ⟨18876⟩⟩
def transferEvent : Nat := 35945
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨214⟩, ⟨6473⟩⟩, ⟨.program ⟨214⟩, ⟨17558⟩⟩], []⟩ [⟨.result 593 .coefficient, true, some 1⟩, ⟨.result 2123 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 593 .coefficient)
      LeftAuthority592.bound (LeftAuthority592.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨6473⟩⟩) (rawTerms := some (Proof.Events002.exact593RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority592.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority592.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 2123 .coefficient)
      LeftAuthority2122.bound (LeftAuthority2122.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨17558⟩⟩) (rawTerms := some (Proof.Events008.exact2123RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority2122.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority2122.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority592.bound [LeftAuthority2122.bound]
def bound : CoeffClass := .finite ⟨228855378262257504357600, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority592.bound, LeftAuthority2122.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority592.actual selector witness) * ([LeftAuthority2122.actual selector witness].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.cons (.intro (input1 selector witness)) (.nil))
end LeftBound35945

namespace LeftBound35946
def owner : Owner := ⟨.program ⟨214⟩, ⟨18876⟩⟩
def transferEvent : Nat := 35946
def frameStart : Nat := 0
def rule : BoundRule := .sum [.transfer 35944, .transfer 35945]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 35944)
      LeftBound35944.bound (LeftBound35944.actual selector witness) := by
  exact .transfer (LeftBound35944.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 35945)
      LeftBound35945.bound (LeftBound35945.actual selector witness) := by
  exact .transfer (LeftBound35945.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound35944.bound, LeftBound35945.bound]
def bound : CoeffClass := .finite ⟨5601427277602316147027040, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound35944.bound, LeftBound35945.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound35944.actual selector witness, LeftBound35945.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound35946

namespace LeftBound35947
def owner : Owner := ⟨.program ⟨214⟩, ⟨18876⟩⟩
def transferEvent : Nat := 35947
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨214⟩, ⟨6490⟩⟩, ⟨.program ⟨214⟩, ⟨18863⟩⟩], []⟩ [⟨.result 603 .coefficient, true, some 1⟩, ⟨.result 2131 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 603 .coefficient)
      LeftAuthority602.bound (LeftAuthority602.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨6490⟩⟩) (rawTerms := some (Proof.Events002.exact603RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority602.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority602.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 2131 .coefficient)
      LeftAuthority2130.bound (LeftAuthority2130.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨18863⟩⟩) (rawTerms := some (Proof.Events008.exact2131RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority2130.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority2130.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority602.bound [LeftAuthority2130.bound]
def bound : CoeffClass := .finite ⟨228236850212900051643120, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority602.bound, LeftAuthority2130.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority602.actual selector witness) * ([LeftAuthority2130.actual selector witness].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.cons (.intro (input1 selector witness)) (.nil))
end LeftBound35947

namespace LeftBound35948
def owner : Owner := ⟨.program ⟨214⟩, ⟨18876⟩⟩
def transferEvent : Nat := 35948
def frameStart : Nat := 0
def rule : BoundRule := .sum [.transfer 35946, .transfer 35947]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 35946)
      LeftBound35946.bound (LeftBound35946.actual selector witness) := by
  exact .transfer (LeftBound35946.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 35947)
      LeftBound35947.bound (LeftBound35947.actual selector witness) := by
  exact .transfer (LeftBound35947.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound35946.bound, LeftBound35947.bound]
def bound : CoeffClass := .finite ⟨5829664127815216198670160, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound35946.bound, LeftBound35947.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound35946.actual selector witness, LeftBound35947.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound35948

namespace LeftBound35949
def owner : Owner := ⟨.program ⟨214⟩, ⟨18876⟩⟩
def transferEvent : Nat := 35949
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨214⟩, ⟨6494⟩⟩, ⟨.program ⟨214⟩, ⟨17614⟩⟩], []⟩ [⟨.result 613 .coefficient, true, some 1⟩, ⟨.result 2139 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 613 .coefficient)
      LeftAuthority612.bound (LeftAuthority612.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨6494⟩⟩) (rawTerms := some (Proof.Events002.exact613RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority612.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority612.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 2139 .coefficient)
      LeftAuthority2138.bound (LeftAuthority2138.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨17614⟩⟩) (rawTerms := some (Proof.Events008.exact2139RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority2138.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority2138.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority612.bound [LeftAuthority2138.bound]
def bound : CoeffClass := .finite ⟨227009770373045750290200, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority612.bound, LeftAuthority2138.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority612.actual selector witness) * ([LeftAuthority2138.actual selector witness].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.cons (.intro (input1 selector witness)) (.nil))
end LeftBound35949

namespace LeftBound35950
def owner : Owner := ⟨.program ⟨214⟩, ⟨18876⟩⟩
def transferEvent : Nat := 35950
def frameStart : Nat := 0
def rule : BoundRule := .sum [.transfer 35948, .transfer 35949]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 35948)
      LeftBound35948.bound (LeftBound35948.actual selector witness) := by
  exact .transfer (LeftBound35948.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 35949)
      LeftBound35949.bound (LeftBound35949.actual selector witness) := by
  exact .transfer (LeftBound35949.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound35948.bound, LeftBound35949.bound]
def bound : CoeffClass := .finite ⟨6056673898188261948960360, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound35948.bound, LeftBound35949.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound35948.actual selector witness, LeftBound35949.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound35950

namespace LeftBound35951
def owner : Owner := ⟨.program ⟨214⟩, ⟨18876⟩⟩
def transferEvent : Nat := 35951
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨214⟩, ⟨6502⟩⟩, ⟨.program ⟨214⟩, ⟨17670⟩⟩], []⟩ [⟨.result 623 .coefficient, true, some 1⟩, ⟨.result 2147 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 623 .coefficient)
      LeftAuthority622.bound (LeftAuthority622.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨6502⟩⟩) (rawTerms := some (Proof.Events002.exact623RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority622.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority622.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 2147 .coefficient)
      LeftAuthority2146.bound (LeftAuthority2146.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨17670⟩⟩) (rawTerms := some (Proof.Events008.exact2147RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority2146.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority2146.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority622.bound [LeftAuthority2146.bound]
def bound : CoeffClass := .finite ⟨226487908831958288795280, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority622.bound, LeftAuthority2146.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority622.actual selector witness) * ([LeftAuthority2146.actual selector witness].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.cons (.intro (input1 selector witness)) (.nil))
end LeftBound35951

namespace LeftBound35952
def owner : Owner := ⟨.program ⟨214⟩, ⟨18876⟩⟩
def transferEvent : Nat := 35952
def frameStart : Nat := 0
def rule : BoundRule := .sum [.transfer 35950, .transfer 35951]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 35950)
      LeftBound35950.bound (LeftBound35950.actual selector witness) := by
  exact .transfer (LeftBound35950.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 35951)
      LeftBound35951.bound (LeftBound35951.actual selector witness) := by
  exact .transfer (LeftBound35951.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound35950.bound, LeftBound35951.bound]
def bound : CoeffClass := .finite ⟨6283161807020220237755640, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound35950.bound, LeftBound35951.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound35950.actual selector witness, LeftBound35951.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound35952

namespace LeftBound35953
def owner : Owner := ⟨.program ⟨214⟩, ⟨18876⟩⟩
def transferEvent : Nat := 35953
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨214⟩, ⟨6383⟩⟩, ⟨.program ⟨214⟩, ⟨18049⟩⟩], []⟩ [⟨.result 633 .coefficient, true, some 1⟩, ⟨.result 2155 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 633 .coefficient)
      LeftAuthority632.bound (LeftAuthority632.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨6383⟩⟩) (rawTerms := some (Proof.Events002.exact633RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority632.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority632.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 2155 .coefficient)
      LeftAuthority2154.bound (LeftAuthority2154.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨18049⟩⟩) (rawTerms := some (Proof.Events008.exact2155RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority2154.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority2154.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority632.bound [LeftAuthority2154.bound]
def bound : CoeffClass := .finite ⟨224377773035387248837560, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority632.bound, LeftAuthority2154.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority632.actual selector witness) * ([LeftAuthority2154.actual selector witness].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.cons (.intro (input1 selector witness)) (.nil))
end LeftBound35953

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
