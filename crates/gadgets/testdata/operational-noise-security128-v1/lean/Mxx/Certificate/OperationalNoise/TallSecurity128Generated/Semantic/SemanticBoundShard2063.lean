import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard2011
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard2015
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard2018
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard2022
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard2025
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard2026
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard2029
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard2033
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard2036
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard2040
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard2062

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound302935
def owner : Owner := ⟨.program ⟨257⟩, ⟨58606⟩⟩
def transferEvent : Nat := 302935
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 302931 .summary, .result 300302 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 302931 .summary)
      LeftBound302930.bound (LeftBound302930.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨55626⟩⟩) (rawTerms := some (Proof.Events1183.exact302931RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound302930.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 300302 .summary)
      LeftBound300301.bound (LeftBound300301.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨58605⟩⟩) (rawTerms := some (Proof.Events1173.exact300302RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound300301.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound302930.bound, LeftBound300301.bound]
def bound : CoeffClass := .finite ⟨225325481271076852082771728531456, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound302930.bound, LeftBound300301.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound302930.actual selector witness, LeftBound300301.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound302935

namespace LeftBound302939
def owner : Owner := ⟨.program ⟨257⟩, ⟨61586⟩⟩
def transferEvent : Nat := 302939
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 302937 .coefficient, .predecessor 1 302938 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 302937 .coefficient)
      LeftBound302934.bound (LeftBound302934.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1183.exact302936RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound302934.bound, RecordedBoundRefines] <;> decide)
      (LeftBound302934.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 302938 .coefficient)
      LeftBound299864.bound (LeftBound299864.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1171.exact299868RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound299864.bound, RecordedBoundRefines] <;> decide)
      (LeftBound299864.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound302934.bound, LeftBound299864.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound302934.bound, LeftBound299864.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound302934.actual selector witness, LeftBound299864.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound302939

namespace LeftBound302940
def owner : Owner := ⟨.program ⟨257⟩, ⟨61586⟩⟩
def transferEvent : Nat := 302940
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 302936 .summary, .result 299868 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 302936 .summary)
      LeftBound302935.bound (LeftBound302935.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨58606⟩⟩) (rawTerms := some (Proof.Events1183.exact302936RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound302935.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 299868 .summary)
      LeftBound299867.bound (LeftBound299867.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨61585⟩⟩) (rawTerms := some (Proof.Events1171.exact299868RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound299867.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound302935.bound, LeftBound299867.bound]
def bound : CoeffClass := .finite ⟨257515860087126057990209472036864, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound302935.bound, LeftBound299867.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound302935.actual selector witness, LeftBound299867.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound302940

namespace LeftBound302944
def owner : Owner := ⟨.program ⟨257⟩, ⟨64566⟩⟩
def transferEvent : Nat := 302944
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 302942 .coefficient, .predecessor 1 302943 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 302942 .coefficient)
      LeftBound302939.bound (LeftBound302939.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1183.exact302941RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound302939.bound, RecordedBoundRefines] <;> decide)
      (LeftBound302939.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 302943 .coefficient)
      LeftBound299430.bound (LeftBound299430.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1169.exact299434RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound299430.bound, RecordedBoundRefines] <;> decide)
      (LeftBound299430.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound302939.bound, LeftBound299430.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound302939.bound, LeftBound299430.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound302939.actual selector witness, LeftBound299430.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound302944

namespace LeftBound302945
def owner : Owner := ⟨.program ⟨257⟩, ⟨64566⟩⟩
def transferEvent : Nat := 302945
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 302941 .summary, .result 299434 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 302941 .summary)
      LeftBound302940.bound (LeftBound302940.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨61586⟩⟩) (rawTerms := some (Proof.Events1183.exact302941RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound302940.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 299434 .summary)
      LeftBound299433.bound (LeftBound299433.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨64565⟩⟩) (rawTerms := some (Proof.Events1169.exact299434RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound299433.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound302940.bound, LeftBound299433.bound]
def bound : CoeffClass := .finite ⟨289706631804066638652128995049472, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound302940.bound, LeftBound299433.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound302940.actual selector witness, LeftBound299433.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound302945

namespace LeftBound302949
def owner : Owner := ⟨.program ⟨257⟩, ⟨69391⟩⟩
def transferEvent : Nat := 302949
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 302947 .coefficient, .predecessor 1 302948 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 302947 .coefficient)
      LeftBound302944.bound (LeftBound302944.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1183.exact302946RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound302944.bound, RecordedBoundRefines] <;> decide)
      (LeftBound302944.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 302948 .coefficient)
      LeftBound298996.bound (LeftBound298996.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1167.exact299000RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound298996.bound, RecordedBoundRefines] <;> decide)
      (LeftBound298996.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound302944.bound, LeftBound298996.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound302944.bound, LeftBound298996.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound302944.actual selector witness, LeftBound298996.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound302949

namespace LeftBound302950
def owner : Owner := ⟨.program ⟨257⟩, ⟨69391⟩⟩
def transferEvent : Nat := 302950
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 302946 .summary, .result 299000 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 302946 .summary)
      LeftBound302945.bound (LeftBound302945.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨64566⟩⟩) (rawTerms := some (Proof.Events1183.exact302946RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound302945.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 299000 .summary)
      LeftBound298999.bound (LeftBound298999.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨69390⟩⟩) (rawTerms := some (Proof.Events1167.exact299000RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound298999.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound302945.bound, LeftBound298999.bound]
def bound : CoeffClass := .finite ⟨321897992872344281445771187322880, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound302945.bound, LeftBound298999.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound302945.actual selector witness, LeftBound298999.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound302950

namespace LeftBound302954
def owner : Owner := ⟨.program ⟨257⟩, ⟨69392⟩⟩
def transferEvent : Nat := 302954
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 302952 .coefficient, .predecessor 1 302953 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 302952 .coefficient)
      LeftBound302949.bound (LeftBound302949.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1183.exact302951RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound302949.bound, RecordedBoundRefines] <;> decide)
      (LeftBound302949.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 302953 .coefficient)
      LeftBound298562.bound (LeftBound298562.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1166.exact298566RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound298562.bound, RecordedBoundRefines] <;> decide)
      (LeftBound298562.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound302949.bound, LeftBound298562.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound302949.bound, LeftBound298562.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound302949.actual selector witness, LeftBound298562.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound302954

namespace LeftBound302955
def owner : Owner := ⟨.program ⟨257⟩, ⟨69392⟩⟩
def transferEvent : Nat := 302955
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 302951 .summary, .result 298566 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 302951 .summary)
      LeftBound302950.bound (LeftBound302950.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨69391⟩⟩) (rawTerms := some (Proof.Events1183.exact302951RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound302950.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 298566 .summary)
      LeftBound298565.bound (LeftBound298565.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨28042⟩⟩) (rawTerms := some (Proof.Events1166.exact298566RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound298565.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound302950.bound, LeftBound298565.bound]
def bound : CoeffClass := .finite ⟨354089550391067611616654269349888, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound302950.bound, LeftBound298565.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound302950.actual selector witness, LeftBound298565.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound302955

namespace LeftBound302959
def owner : Owner := ⟨.program ⟨257⟩, ⟨69393⟩⟩
def transferEvent : Nat := 302959
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 302957 .coefficient, .predecessor 1 302958 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 302957 .coefficient)
      LeftBound302954.bound (LeftBound302954.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1183.exact302956RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound302954.bound, RecordedBoundRefines] <;> decide)
      (LeftBound302954.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 302958 .coefficient)
      LeftBound298128.bound (LeftBound298128.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1164.exact298132RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound298128.bound, RecordedBoundRefines] <;> decide)
      (LeftBound298128.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound302954.bound, LeftBound298128.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound302954.bound, LeftBound298128.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound302954.actual selector witness, LeftBound298128.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound302959

namespace LeftBound302960
def owner : Owner := ⟨.program ⟨257⟩, ⟨69393⟩⟩
def transferEvent : Nat := 302960
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 302956 .summary, .result 298132 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 302956 .summary)
      LeftBound302955.bound (LeftBound302955.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨69392⟩⟩) (rawTerms := some (Proof.Events1183.exact302956RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound302955.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 298132 .summary)
      LeftBound298131.bound (LeftBound298131.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨30722⟩⟩) (rawTerms := some (Proof.Events1164.exact298132RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound298131.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound302955.bound, LeftBound298131.bound]
def bound : CoeffClass := .finite ⟨386281697261128003919260020637696, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound302955.bound, LeftBound298131.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound302955.actual selector witness, LeftBound298131.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound302960

namespace LeftBound302964
def owner : Owner := ⟨.program ⟨257⟩, ⟨69394⟩⟩
def transferEvent : Nat := 302964
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 302962 .coefficient, .predecessor 1 302963 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 302962 .coefficient)
      LeftBound302959.bound (LeftBound302959.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1183.exact302961RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound302959.bound, RecordedBoundRefines] <;> decide)
      (LeftBound302959.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 302963 .coefficient)
      LeftBound297694.bound (LeftBound297694.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1162.exact297698RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound297694.bound, RecordedBoundRefines] <;> decide)
      (LeftBound297694.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound302959.bound, LeftBound297694.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound302959.bound, LeftBound297694.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound302959.actual selector witness, LeftBound297694.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound302964

namespace LeftBound302965
def owner : Owner := ⟨.program ⟨257⟩, ⟨69394⟩⟩
def transferEvent : Nat := 302965
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 302961 .summary, .result 297698 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 302961 .summary)
      LeftBound302960.bound (LeftBound302960.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨69393⟩⟩) (rawTerms := some (Proof.Events1183.exact302961RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound302960.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 297698 .summary)
      LeftBound297697.bound (LeftBound297697.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨36382⟩⟩) (rawTerms := some (Proof.Events1162.exact297698RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound297697.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound302960.bound, LeftBound297697.bound]
def bound : CoeffClass := .finite ⟨418474237032079770976347551432704, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound302960.bound, LeftBound297697.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound302960.actual selector witness, LeftBound297697.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound302965

namespace LeftBound302969
def owner : Owner := ⟨.program ⟨257⟩, ⟨69395⟩⟩
def transferEvent : Nat := 302969
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 302967 .coefficient, .predecessor 1 302968 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 302967 .coefficient)
      LeftBound302964.bound (LeftBound302964.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1183.exact302966RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound302964.bound, RecordedBoundRefines] <;> decide)
      (LeftBound302964.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 302968 .coefficient)
      LeftBound297260.bound (LeftBound297260.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1161.exact297264RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound297260.bound, RecordedBoundRefines] <;> decide)
      (LeftBound297260.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound302964.bound, LeftBound297260.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound302964.bound, LeftBound297260.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound302964.actual selector witness, LeftBound297260.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound302969

namespace LeftBound302970
def owner : Owner := ⟨.program ⟨257⟩, ⟨69395⟩⟩
def transferEvent : Nat := 302970
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 302966 .summary, .result 297264 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 302966 .summary)
      LeftBound302965.bound (LeftBound302965.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨69394⟩⟩) (rawTerms := some (Proof.Events1183.exact302966RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound302965.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 297264 .summary)
      LeftBound297263.bound (LeftBound297263.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨39062⟩⟩) (rawTerms := some (Proof.Events1161.exact297264RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound297263.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound302965.bound, LeftBound297263.bound]
def bound : CoeffClass := .finite ⟨450666973253477225410675971981312, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound302965.bound, LeftBound297263.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound302965.actual selector witness, LeftBound297263.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound302970

namespace LeftBound302974
def owner : Owner := ⟨.program ⟨257⟩, ⟨69396⟩⟩
def transferEvent : Nat := 302974
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 302972 .coefficient, .predecessor 1 302973 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 302972 .coefficient)
      LeftBound302969.bound (LeftBound302969.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1183.exact302971RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound302969.bound, RecordedBoundRefines] <;> decide)
      (LeftBound302969.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 302973 .coefficient)
      LeftBound296826.bound (LeftBound296826.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1159.exact296830RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound296826.bound, RecordedBoundRefines] <;> decide)
      (LeftBound296826.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound302969.bound, LeftBound296826.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound302969.bound, LeftBound296826.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound302969.actual selector witness, LeftBound296826.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound302974

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
