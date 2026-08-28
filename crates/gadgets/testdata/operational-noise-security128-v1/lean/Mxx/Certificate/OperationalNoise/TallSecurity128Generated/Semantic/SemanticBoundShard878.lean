import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard863
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard864
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard865
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard866
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard867
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard869
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard870
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard871
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard873
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard877

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound133936
def owner : Owner := ⟨.program ⟨257⟩, ⟨33766⟩⟩
def transferEvent : Nat := 133936
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 133934 .coefficient, .predecessor 1 133935 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 133934 .coefficient)
      LeftBound133931.bound (LeftBound133931.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events523.exact133933RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound133931.bound, RecordedBoundRefines] <;> decide)
      (LeftBound133931.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 133935 .coefficient)
      LeftBound133248.bound (LeftBound133248.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events520.exact133255RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound133248.bound, RecordedBoundRefines] <;> decide)
      (LeftBound133248.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound133931.bound, LeftBound133248.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound133931.bound, LeftBound133248.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound133931.actual selector witness, LeftBound133248.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound133936

namespace LeftBound133937
def owner : Owner := ⟨.program ⟨257⟩, ⟨33766⟩⟩
def transferEvent : Nat := 133937
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 133933 .summary, .result 133255 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 133933 .summary)
      LeftBound133932.bound (LeftBound133932.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨23746⟩⟩) (rawTerms := some (Proof.Events523.exact133933RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound133932.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 133255 .summary)
      LeftBound133250.bound (LeftBound133250.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨33765⟩⟩) (rawTerms := some (Proof.Events520.exact133255RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound133250.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound133932.bound, LeftBound133250.bound]
def bound : CoeffClass := .finite ⟨1382506125545760169441014535464825839943732, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound133932.bound, LeftBound133250.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound133932.actual selector witness, LeftBound133250.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound133937

namespace LeftBound133941
def owner : Owner := ⟨.program ⟨257⟩, ⟨52826⟩⟩
def transferEvent : Nat := 133941
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 133939 .coefficient, .predecessor 1 133940 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 133939 .coefficient)
      LeftBound133936.bound (LeftBound133936.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events523.exact133938RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound133936.bound, RecordedBoundRefines] <;> decide)
      (LeftBound133936.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 133940 .coefficient)
      LeftBound133036.bound (LeftBound133036.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events519.exact133043RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound133036.bound, RecordedBoundRefines] <;> decide)
      (LeftBound133036.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound133936.bound, LeftBound133036.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound133936.bound, LeftBound133036.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound133936.actual selector witness, LeftBound133036.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound133941

namespace LeftBound133942
def owner : Owner := ⟨.program ⟨257⟩, ⟨52826⟩⟩
def transferEvent : Nat := 133942
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 133938 .summary, .result 133043 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 133938 .summary)
      LeftBound133937.bound (LeftBound133937.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨33766⟩⟩) (rawTerms := some (Proof.Events523.exact133938RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound133937.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 133043 .summary)
      LeftBound133038.bound (LeftBound133038.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨52825⟩⟩) (rawTerms := some (Proof.Events519.exact133043RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound133038.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound133937.bound, LeftBound133038.bound]
def bound : CoeffClass := .finite ⟨1728139248715321398594155952187700255129652, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound133937.bound, LeftBound133038.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound133937.actual selector witness, LeftBound133038.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound133942

namespace LeftBound133946
def owner : Owner := ⟨.program ⟨257⟩, ⟨55806⟩⟩
def transferEvent : Nat := 133946
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 133944 .coefficient, .predecessor 1 133945 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 133944 .coefficient)
      LeftBound133941.bound (LeftBound133941.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events523.exact133943RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound133941.bound, RecordedBoundRefines] <;> decide)
      (LeftBound133941.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 133945 .coefficient)
      LeftBound132824.bound (LeftBound132824.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events518.exact132831RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound132824.bound, RecordedBoundRefines] <;> decide)
      (LeftBound132824.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound133941.bound, LeftBound132824.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound133941.bound, LeftBound132824.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound133941.actual selector witness, LeftBound132824.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound133946

namespace LeftBound133947
def owner : Owner := ⟨.program ⟨257⟩, ⟨55806⟩⟩
def transferEvent : Nat := 133947
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 133943 .summary, .result 132831 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 133943 .summary)
      LeftBound133942.bound (LeftBound133942.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨52826⟩⟩) (rawTerms := some (Proof.Events523.exact133943RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound133942.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 132831 .summary)
      LeftBound132826.bound (LeftBound132826.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨55805⟩⟩) (rawTerms := some (Proof.Events518.exact132831RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound132826.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound133942.bound, LeftBound132826.bound]
def bound : CoeffClass := .finite ⟨2073774481255481407521021459424708415979572, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound133942.bound, LeftBound132826.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound133942.actual selector witness, LeftBound132826.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound133947

namespace LeftBound133951
def owner : Owner := ⟨.program ⟨257⟩, ⟨58786⟩⟩
def transferEvent : Nat := 133951
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 133949 .coefficient, .predecessor 1 133950 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 133949 .coefficient)
      LeftBound133946.bound (LeftBound133946.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events523.exact133948RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound133946.bound, RecordedBoundRefines] <;> decide)
      (LeftBound133946.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 133950 .coefficient)
      LeftBound132612.bound (LeftBound132612.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events518.exact132619RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound132612.bound, RecordedBoundRefines] <;> decide)
      (LeftBound132612.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound133946.bound, LeftBound132612.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound133946.bound, LeftBound132612.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound133946.actual selector witness, LeftBound132612.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound133951

namespace LeftBound133952
def owner : Owner := ⟨.program ⟨257⟩, ⟨58786⟩⟩
def transferEvent : Nat := 133952
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 133948 .summary, .result 132619 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 133948 .summary)
      LeftBound133947.bound (LeftBound133947.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨55806⟩⟩) (rawTerms := some (Proof.Events523.exact133948RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound133947.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 132619 .summary)
      LeftBound132614.bound (LeftBound132614.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨58785⟩⟩) (rawTerms := some (Proof.Events518.exact132619RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound132614.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound133947.bound, LeftBound132614.bound]
def bound : CoeffClass := .finite ⟨2419413932536838975995335147689984068157492, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound133947.bound, LeftBound132614.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound133947.actual selector witness, LeftBound132614.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound133952

namespace LeftBound133956
def owner : Owner := ⟨.program ⟨257⟩, ⟨61766⟩⟩
def transferEvent : Nat := 133956
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 133954 .coefficient, .predecessor 1 133955 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 133954 .coefficient)
      LeftBound133951.bound (LeftBound133951.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events523.exact133953RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound133951.bound, RecordedBoundRefines] <;> decide)
      (LeftBound133951.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 133955 .coefficient)
      LeftBound132400.bound (LeftBound132400.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events517.exact132407RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound132400.bound, RecordedBoundRefines] <;> decide)
      (LeftBound132400.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound133951.bound, LeftBound132400.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound133951.bound, LeftBound132400.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound133951.actual selector witness, LeftBound132400.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound133956

namespace LeftBound133957
def owner : Owner := ⟨.program ⟨257⟩, ⟨61766⟩⟩
def transferEvent : Nat := 133957
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 133953 .summary, .result 132407 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 133953 .summary)
      LeftBound133952.bound (LeftBound133952.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨58786⟩⟩) (rawTerms := some (Proof.Events523.exact133953RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound133952.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 132407 .summary)
      LeftBound132402.bound (LeftBound132402.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨61765⟩⟩) (rawTerms := some (Proof.Events517.exact132407RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound132402.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound133952.bound, LeftBound132402.bound]
def bound : CoeffClass := .finite ⟨2765055493188795324243372926469393465999412, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound133952.bound, LeftBound132402.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound133952.actual selector witness, LeftBound132402.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound133957

namespace LeftBound133961
def owner : Owner := ⟨.program ⟨257⟩, ⟨64746⟩⟩
def transferEvent : Nat := 133961
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 133959 .coefficient, .predecessor 1 133960 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 133959 .coefficient)
      LeftBound133956.bound (LeftBound133956.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events523.exact133958RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound133956.bound, RecordedBoundRefines] <;> decide)
      (LeftBound133956.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 133960 .coefficient)
      LeftBound132188.bound (LeftBound132188.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events516.exact132195RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound132188.bound, RecordedBoundRefines] <;> decide)
      (LeftBound132188.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound133956.bound, LeftBound132188.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound133956.bound, LeftBound132188.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound133956.actual selector witness, LeftBound132188.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound133961

namespace LeftBound133962
def owner : Owner := ⟨.program ⟨257⟩, ⟨64746⟩⟩
def transferEvent : Nat := 133962
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 133958 .summary, .result 132195 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 133958 .summary)
      LeftBound133957.bound (LeftBound133957.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨61766⟩⟩) (rawTerms := some (Proof.Events523.exact133958RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound133957.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 132195 .summary)
      LeftBound132190.bound (LeftBound132190.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨64745⟩⟩) (rawTerms := some (Proof.Events516.exact132195RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound132190.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound133957.bound, LeftBound132190.bound]
def bound : CoeffClass := .finite ⟨3110701272581949232038858886277070355169332, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound133957.bound, LeftBound132190.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound133957.actual selector witness, LeftBound132190.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound133962

namespace LeftBound133966
def owner : Owner := ⟨.program ⟨257⟩, ⟨69851⟩⟩
def transferEvent : Nat := 133966
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 133964 .coefficient, .predecessor 1 133965 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 133964 .coefficient)
      LeftBound133961.bound (LeftBound133961.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events523.exact133963RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound133961.bound, RecordedBoundRefines] <;> decide)
      (LeftBound133961.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 133965 .coefficient)
      LeftBound131976.bound (LeftBound131976.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events515.exact131983RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound131976.bound, RecordedBoundRefines] <;> decide)
      (LeftBound131976.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound133961.bound, LeftBound131976.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound133961.bound, LeftBound131976.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound133961.actual selector witness, LeftBound131976.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound133966

namespace LeftBound133967
def owner : Owner := ⟨.program ⟨257⟩, ⟨69851⟩⟩
def transferEvent : Nat := 133967
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 133963 .summary, .result 131983 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 133963 .summary)
      LeftBound133962.bound (LeftBound133962.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨64746⟩⟩) (rawTerms := some (Proof.Events523.exact133963RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound133962.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 131983 .summary)
      LeftBound131978.bound (LeftBound131978.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨69850⟩⟩) (rawTerms := some (Proof.Events515.exact131983RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound131978.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound133962.bound, LeftBound131978.bound]
def bound : CoeffClass := .finite ⟨3456353380086899479155517117627148481331252, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound133962.bound, LeftBound131978.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound133962.actual selector witness, LeftBound131978.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound133967

namespace LeftBound133971
def owner : Owner := ⟨.program ⟨257⟩, ⟨69852⟩⟩
def transferEvent : Nat := 133971
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 133969 .coefficient, .predecessor 1 133970 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 133969 .coefficient)
      LeftBound133966.bound (LeftBound133966.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events523.exact133968RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound133966.bound, RecordedBoundRefines] <;> decide)
      (LeftBound133966.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 133970 .coefficient)
      LeftBound131764.bound (LeftBound131764.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events514.exact131771RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound131764.bound, RecordedBoundRefines] <;> decide)
      (LeftBound131764.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound133966.bound, LeftBound131764.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound133966.bound, LeftBound131764.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound133966.actual selector witness, LeftBound131764.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound133971

namespace LeftBound133972
def owner : Owner := ⟨.program ⟨257⟩, ⟨69852⟩⟩
def transferEvent : Nat := 133972
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 133968 .summary, .result 131771 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 133968 .summary)
      LeftBound133967.bound (LeftBound133967.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨69851⟩⟩) (rawTerms := some (Proof.Events523.exact133968RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound133967.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 131771 .summary)
      LeftBound131766.bound (LeftBound131766.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨28187⟩⟩) (rawTerms := some (Proof.Events514.exact131771RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound131766.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound133967.bound, LeftBound131766.bound]
def bound : CoeffClass := .finite ⟨3802007596962448506045899439491360353157172, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound133967.bound, LeftBound131766.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound133967.actual selector witness, LeftBound131766.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound133972

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
