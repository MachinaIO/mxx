import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard2080
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard2082
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard2083
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard2084
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard2086
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard2087
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard2088
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard2090
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard2091

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound307927
def owner : Owner := ⟨.program ⟨257⟩, ⟨17479⟩⟩
def transferEvent : Nat := 307927
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 307923 .summary, .result 307896 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 307923 .summary)
      LeftBound307922.bound (LeftBound307922.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨9447⟩⟩) (rawTerms := some (Proof.Events1202.exact307923RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound307922.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 307896 .summary)
      LeftBound307891.bound (LeftBound307891.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨17478⟩⟩) (rawTerms := some (Proof.Events1202.exact307896RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound307891.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound307922.bound, LeftBound307891.bound]
def bound : CoeffClass := .finite ⟨345624685687166110058245054666339432529972, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound307922.bound, LeftBound307891.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound307922.actual selector witness, LeftBound307891.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound307927

namespace LeftBound307931
def owner : Owner := ⟨.program ⟨257⟩, ⟨20340⟩⟩
def transferEvent : Nat := 307931
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 307929 .coefficient, .predecessor 1 307930 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 307929 .coefficient)
      LeftBound307926.bound (LeftBound307926.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1202.exact307928RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound307926.bound, RecordedBoundRefines] <;> decide)
      (LeftBound307926.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 307930 .coefficient)
      LeftBound307701.bound (LeftBound307701.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1201.exact307708RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound307701.bound, RecordedBoundRefines] <;> decide)
      (LeftBound307701.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound307926.bound, LeftBound307701.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound307926.bound, LeftBound307701.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound307926.actual selector witness, LeftBound307701.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound307931

namespace LeftBound307932
def owner : Owner := ⟨.program ⟨257⟩, ⟨20340⟩⟩
def transferEvent : Nat := 307932
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 307928 .summary, .result 307708 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 307928 .summary)
      LeftBound307927.bound (LeftBound307927.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨17479⟩⟩) (rawTerms := some (Proof.Events1202.exact307928RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound307927.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 307708 .summary)
      LeftBound307703.bound (LeftBound307703.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨20339⟩⟩) (rawTerms := some (Proof.Events1201.exact307708RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound307703.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound307927.bound, LeftBound307703.bound]
def bound : CoeffClass := .finite ⟨691250426059631610003352154589745737891892, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound307927.bound, LeftBound307703.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound307927.actual selector witness, LeftBound307703.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound307932

namespace LeftBound307936
def owner : Owner := ⟨.program ⟨257⟩, ⟨23560⟩⟩
def transferEvent : Nat := 307936
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 307934 .coefficient, .predecessor 1 307935 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 307934 .coefficient)
      LeftBound307931.bound (LeftBound307931.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1202.exact307933RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound307931.bound, RecordedBoundRefines] <;> decide)
      (LeftBound307931.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 307935 .coefficient)
      LeftBound307513.bound (LeftBound307513.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1201.exact307520RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound307513.bound, RecordedBoundRefines] <;> decide)
      (LeftBound307513.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound307931.bound, LeftBound307513.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound307931.bound, LeftBound307513.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound307931.actual selector witness, LeftBound307513.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound307936

namespace LeftBound307937
def owner : Owner := ⟨.program ⟨257⟩, ⟨23560⟩⟩
def transferEvent : Nat := 307937
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 307933 .summary, .result 307520 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 307933 .summary)
      LeftBound307932.bound (LeftBound307932.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨20340⟩⟩) (rawTerms := some (Proof.Events1202.exact307933RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound307932.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 307520 .summary)
      LeftBound307515.bound (LeftBound307515.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨23559⟩⟩) (rawTerms := some (Proof.Events1201.exact307520RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound307515.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound307932.bound, LeftBound307515.bound]
def bound : CoeffClass := .finite ⟨1036877221117396499835321299770218916085812, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound307932.bound, LeftBound307515.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound307932.actual selector witness, LeftBound307515.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound307937

namespace LeftBound307941
def owner : Owner := ⟨.program ⟨257⟩, ⟨33580⟩⟩
def transferEvent : Nat := 307941
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 307939 .coefficient, .predecessor 1 307940 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 307939 .coefficient)
      LeftBound307936.bound (LeftBound307936.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1202.exact307938RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound307936.bound, RecordedBoundRefines] <;> decide)
      (LeftBound307936.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 307940 .coefficient)
      LeftBound307325.bound (LeftBound307325.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1200.exact307332RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound307325.bound, RecordedBoundRefines] <;> decide)
      (LeftBound307325.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound307936.bound, LeftBound307325.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound307936.bound, LeftBound307325.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound307936.actual selector witness, LeftBound307325.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound307941

namespace LeftBound307942
def owner : Owner := ⟨.program ⟨257⟩, ⟨33580⟩⟩
def transferEvent : Nat := 307942
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 307938 .summary, .result 307332 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 307938 .summary)
      LeftBound307937.bound (LeftBound307937.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨23560⟩⟩) (rawTerms := some (Proof.Events1202.exact307938RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound307937.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 307332 .summary)
      LeftBound307327.bound (LeftBound307327.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨33579⟩⟩) (rawTerms := some (Proof.Events1200.exact307332RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound307327.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound307937.bound, LeftBound307327.bound]
def bound : CoeffClass := .finite ⟨1382506125545760169441014535464825839943732, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound307937.bound, LeftBound307327.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound307937.actual selector witness, LeftBound307327.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound307942

namespace LeftBound307946
def owner : Owner := ⟨.program ⟨257⟩, ⟨52640⟩⟩
def transferEvent : Nat := 307946
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 307944 .coefficient, .predecessor 1 307945 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 307944 .coefficient)
      LeftBound307941.bound (LeftBound307941.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1202.exact307943RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound307941.bound, RecordedBoundRefines] <;> decide)
      (LeftBound307941.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 307945 .coefficient)
      LeftBound307137.bound (LeftBound307137.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1199.exact307144RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound307137.bound, RecordedBoundRefines] <;> decide)
      (LeftBound307137.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound307941.bound, LeftBound307137.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound307941.bound, LeftBound307137.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound307941.actual selector witness, LeftBound307137.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound307946

namespace LeftBound307947
def owner : Owner := ⟨.program ⟨257⟩, ⟨52640⟩⟩
def transferEvent : Nat := 307947
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 307943 .summary, .result 307144 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 307943 .summary)
      LeftBound307942.bound (LeftBound307942.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨33580⟩⟩) (rawTerms := some (Proof.Events1202.exact307943RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound307942.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 307144 .summary)
      LeftBound307139.bound (LeftBound307139.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨52639⟩⟩) (rawTerms := some (Proof.Events1199.exact307144RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound307139.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound307942.bound, LeftBound307139.bound]
def bound : CoeffClass := .finite ⟨1728139248715321398594155952187700255129652, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound307942.bound, LeftBound307139.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound307942.actual selector witness, LeftBound307139.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound307947

namespace LeftBound307951
def owner : Owner := ⟨.program ⟨257⟩, ⟨55620⟩⟩
def transferEvent : Nat := 307951
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 307949 .coefficient, .predecessor 1 307950 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 307949 .coefficient)
      LeftBound307946.bound (LeftBound307946.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1202.exact307948RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound307946.bound, RecordedBoundRefines] <;> decide)
      (LeftBound307946.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 307950 .coefficient)
      LeftBound306949.bound (LeftBound306949.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1199.exact306956RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound306949.bound, RecordedBoundRefines] <;> decide)
      (LeftBound306949.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound307946.bound, LeftBound306949.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound307946.bound, LeftBound306949.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound307946.actual selector witness, LeftBound306949.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound307951

namespace LeftBound307952
def owner : Owner := ⟨.program ⟨257⟩, ⟨55620⟩⟩
def transferEvent : Nat := 307952
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 307948 .summary, .result 306956 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 307948 .summary)
      LeftBound307947.bound (LeftBound307947.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨52640⟩⟩) (rawTerms := some (Proof.Events1202.exact307948RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound307947.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 306956 .summary)
      LeftBound306951.bound (LeftBound306951.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨55619⟩⟩) (rawTerms := some (Proof.Events1199.exact306956RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound306951.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound307947.bound, LeftBound306951.bound]
def bound : CoeffClass := .finite ⟨2073774481255481407521021459424708415979572, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound307947.bound, LeftBound306951.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound307947.actual selector witness, LeftBound306951.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound307952

namespace LeftBound307956
def owner : Owner := ⟨.program ⟨257⟩, ⟨58600⟩⟩
def transferEvent : Nat := 307956
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 307954 .coefficient, .predecessor 1 307955 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 307954 .coefficient)
      LeftBound307951.bound (LeftBound307951.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1202.exact307953RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound307951.bound, RecordedBoundRefines] <;> decide)
      (LeftBound307951.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 307955 .coefficient)
      LeftBound306761.bound (LeftBound306761.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1198.exact306768RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound306761.bound, RecordedBoundRefines] <;> decide)
      (LeftBound306761.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound307951.bound, LeftBound306761.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound307951.bound, LeftBound306761.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound307951.actual selector witness, LeftBound306761.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound307956

namespace LeftBound307957
def owner : Owner := ⟨.program ⟨257⟩, ⟨58600⟩⟩
def transferEvent : Nat := 307957
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 307953 .summary, .result 306768 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 307953 .summary)
      LeftBound307952.bound (LeftBound307952.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨55620⟩⟩) (rawTerms := some (Proof.Events1202.exact307953RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound307952.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 306768 .summary)
      LeftBound306763.bound (LeftBound306763.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨58599⟩⟩) (rawTerms := some (Proof.Events1198.exact306768RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound306763.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound307952.bound, LeftBound306763.bound]
def bound : CoeffClass := .finite ⟨2419413932536838975995335147689984068157492, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound307952.bound, LeftBound306763.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound307952.actual selector witness, LeftBound306763.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound307957

namespace LeftBound307961
def owner : Owner := ⟨.program ⟨257⟩, ⟨61580⟩⟩
def transferEvent : Nat := 307961
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 307959 .coefficient, .predecessor 1 307960 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 307959 .coefficient)
      LeftBound307956.bound (LeftBound307956.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1202.exact307958RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound307956.bound, RecordedBoundRefines] <;> decide)
      (LeftBound307956.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 307960 .coefficient)
      LeftBound306573.bound (LeftBound306573.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1197.exact306580RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound306573.bound, RecordedBoundRefines] <;> decide)
      (LeftBound306573.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound307956.bound, LeftBound306573.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound307956.bound, LeftBound306573.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound307956.actual selector witness, LeftBound306573.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound307961

namespace LeftBound307962
def owner : Owner := ⟨.program ⟨257⟩, ⟨61580⟩⟩
def transferEvent : Nat := 307962
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 307958 .summary, .result 306580 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 307958 .summary)
      LeftBound307957.bound (LeftBound307957.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨58600⟩⟩) (rawTerms := some (Proof.Events1202.exact307958RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound307957.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 306580 .summary)
      LeftBound306575.bound (LeftBound306575.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨61579⟩⟩) (rawTerms := some (Proof.Events1197.exact306580RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound306575.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound307957.bound, LeftBound306575.bound]
def bound : CoeffClass := .finite ⟨2765055493188795324243372926469393465999412, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound307957.bound, LeftBound306575.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound307957.actual selector witness, LeftBound306575.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound307962

namespace LeftBound307966
def owner : Owner := ⟨.program ⟨257⟩, ⟨64560⟩⟩
def transferEvent : Nat := 307966
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 307964 .coefficient, .predecessor 1 307965 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 307964 .coefficient)
      LeftBound307961.bound (LeftBound307961.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1202.exact307963RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound307961.bound, RecordedBoundRefines] <;> decide)
      (LeftBound307961.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 307965 .coefficient)
      LeftBound306385.bound (LeftBound306385.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1196.exact306392RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound306385.bound, RecordedBoundRefines] <;> decide)
      (LeftBound306385.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound307961.bound, LeftBound306385.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound307961.bound, LeftBound306385.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound307961.actual selector witness, LeftBound306385.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound307966

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
