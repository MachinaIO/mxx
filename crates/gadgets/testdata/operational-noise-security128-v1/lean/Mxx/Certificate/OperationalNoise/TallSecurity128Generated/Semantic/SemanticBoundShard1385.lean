import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1372
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1373
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1375
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1376
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1377
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1378
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1379
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1380
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1381
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1383
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1384

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound207052
def owner : Owner := ⟨.program ⟨257⟩, ⟨20712⟩⟩
def transferEvent : Nat := 207052
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 207048 .summary, .result 206804 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 207048 .summary)
      LeftBound207047.bound (LeftBound207047.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨17815⟩⟩) (rawTerms := some (Proof.Events808.exact207048RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound207047.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 206804 .summary)
      LeftBound206799.bound (LeftBound206799.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨20711⟩⟩) (rawTerms := some (Proof.Events807.exact206804RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound206799.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound207047.bound, LeftBound206799.bound]
def bound : CoeffClass := .finite ⟨691250426059631610003352154589745737891892, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound207047.bound, LeftBound206799.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound207047.actual selector witness, LeftBound206799.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound207052

namespace LeftBound207056
def owner : Owner := ⟨.program ⟨257⟩, ⟨23932⟩⟩
def transferEvent : Nat := 207056
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 207054 .coefficient, .predecessor 1 207055 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 207054 .coefficient)
      LeftBound207051.bound (LeftBound207051.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events808.exact207053RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound207051.bound, RecordedBoundRefines] <;> decide)
      (LeftBound207051.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 207055 .coefficient)
      LeftBound206585.bound (LeftBound206585.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events807.exact206592RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound206585.bound, RecordedBoundRefines] <;> decide)
      (LeftBound206585.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound207051.bound, LeftBound206585.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound207051.bound, LeftBound206585.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound207051.actual selector witness, LeftBound206585.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound207056

namespace LeftBound207057
def owner : Owner := ⟨.program ⟨257⟩, ⟨23932⟩⟩
def transferEvent : Nat := 207057
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 207053 .summary, .result 206592 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 207053 .summary)
      LeftBound207052.bound (LeftBound207052.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨20712⟩⟩) (rawTerms := some (Proof.Events808.exact207053RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound207052.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 206592 .summary)
      LeftBound206587.bound (LeftBound206587.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨23931⟩⟩) (rawTerms := some (Proof.Events807.exact206592RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound206587.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound207052.bound, LeftBound206587.bound]
def bound : CoeffClass := .finite ⟨1036877221117396499835321299770218916085812, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound207052.bound, LeftBound206587.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound207052.actual selector witness, LeftBound206587.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound207057

namespace LeftBound207061
def owner : Owner := ⟨.program ⟨257⟩, ⟨33952⟩⟩
def transferEvent : Nat := 207061
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 207059 .coefficient, .predecessor 1 207060 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 207059 .coefficient)
      LeftBound207056.bound (LeftBound207056.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events808.exact207058RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound207056.bound, RecordedBoundRefines] <;> decide)
      (LeftBound207056.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 207060 .coefficient)
      LeftBound206373.bound (LeftBound206373.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events806.exact206380RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound206373.bound, RecordedBoundRefines] <;> decide)
      (LeftBound206373.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound207056.bound, LeftBound206373.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound207056.bound, LeftBound206373.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound207056.actual selector witness, LeftBound206373.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound207061

namespace LeftBound207062
def owner : Owner := ⟨.program ⟨257⟩, ⟨33952⟩⟩
def transferEvent : Nat := 207062
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 207058 .summary, .result 206380 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 207058 .summary)
      LeftBound207057.bound (LeftBound207057.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨23932⟩⟩) (rawTerms := some (Proof.Events808.exact207058RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound207057.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 206380 .summary)
      LeftBound206375.bound (LeftBound206375.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨33951⟩⟩) (rawTerms := some (Proof.Events806.exact206380RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound206375.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound207057.bound, LeftBound206375.bound]
def bound : CoeffClass := .finite ⟨1382506125545760169441014535464825839943732, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound207057.bound, LeftBound206375.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound207057.actual selector witness, LeftBound206375.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound207062

namespace LeftBound207066
def owner : Owner := ⟨.program ⟨257⟩, ⟨53012⟩⟩
def transferEvent : Nat := 207066
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 207064 .coefficient, .predecessor 1 207065 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 207064 .coefficient)
      LeftBound207061.bound (LeftBound207061.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events808.exact207063RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound207061.bound, RecordedBoundRefines] <;> decide)
      (LeftBound207061.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 207065 .coefficient)
      LeftBound206161.bound (LeftBound206161.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events805.exact206168RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound206161.bound, RecordedBoundRefines] <;> decide)
      (LeftBound206161.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound207061.bound, LeftBound206161.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound207061.bound, LeftBound206161.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound207061.actual selector witness, LeftBound206161.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound207066

namespace LeftBound207067
def owner : Owner := ⟨.program ⟨257⟩, ⟨53012⟩⟩
def transferEvent : Nat := 207067
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 207063 .summary, .result 206168 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 207063 .summary)
      LeftBound207062.bound (LeftBound207062.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨33952⟩⟩) (rawTerms := some (Proof.Events808.exact207063RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound207062.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 206168 .summary)
      LeftBound206163.bound (LeftBound206163.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨53011⟩⟩) (rawTerms := some (Proof.Events805.exact206168RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound206163.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound207062.bound, LeftBound206163.bound]
def bound : CoeffClass := .finite ⟨1728139248715321398594155952187700255129652, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound207062.bound, LeftBound206163.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound207062.actual selector witness, LeftBound206163.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound207067

namespace LeftBound207071
def owner : Owner := ⟨.program ⟨257⟩, ⟨55992⟩⟩
def transferEvent : Nat := 207071
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 207069 .coefficient, .predecessor 1 207070 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 207069 .coefficient)
      LeftBound207066.bound (LeftBound207066.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events808.exact207068RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound207066.bound, RecordedBoundRefines] <;> decide)
      (LeftBound207066.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 207070 .coefficient)
      LeftBound205949.bound (LeftBound205949.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events804.exact205956RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound205949.bound, RecordedBoundRefines] <;> decide)
      (LeftBound205949.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound207066.bound, LeftBound205949.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound207066.bound, LeftBound205949.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound207066.actual selector witness, LeftBound205949.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound207071

namespace LeftBound207072
def owner : Owner := ⟨.program ⟨257⟩, ⟨55992⟩⟩
def transferEvent : Nat := 207072
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 207068 .summary, .result 205956 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 207068 .summary)
      LeftBound207067.bound (LeftBound207067.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨53012⟩⟩) (rawTerms := some (Proof.Events808.exact207068RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound207067.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 205956 .summary)
      LeftBound205951.bound (LeftBound205951.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨55991⟩⟩) (rawTerms := some (Proof.Events804.exact205956RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound205951.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound207067.bound, LeftBound205951.bound]
def bound : CoeffClass := .finite ⟨2073774481255481407521021459424708415979572, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound207067.bound, LeftBound205951.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound207067.actual selector witness, LeftBound205951.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound207072

namespace LeftBound207076
def owner : Owner := ⟨.program ⟨257⟩, ⟨58972⟩⟩
def transferEvent : Nat := 207076
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 207074 .coefficient, .predecessor 1 207075 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 207074 .coefficient)
      LeftBound207071.bound (LeftBound207071.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events808.exact207073RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound207071.bound, RecordedBoundRefines] <;> decide)
      (LeftBound207071.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 207075 .coefficient)
      LeftBound205737.bound (LeftBound205737.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events803.exact205744RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound205737.bound, RecordedBoundRefines] <;> decide)
      (LeftBound205737.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound207071.bound, LeftBound205737.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound207071.bound, LeftBound205737.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound207071.actual selector witness, LeftBound205737.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound207076

namespace LeftBound207077
def owner : Owner := ⟨.program ⟨257⟩, ⟨58972⟩⟩
def transferEvent : Nat := 207077
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 207073 .summary, .result 205744 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 207073 .summary)
      LeftBound207072.bound (LeftBound207072.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨55992⟩⟩) (rawTerms := some (Proof.Events808.exact207073RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound207072.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 205744 .summary)
      LeftBound205739.bound (LeftBound205739.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨58971⟩⟩) (rawTerms := some (Proof.Events803.exact205744RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound205739.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound207072.bound, LeftBound205739.bound]
def bound : CoeffClass := .finite ⟨2419413932536838975995335147689984068157492, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound207072.bound, LeftBound205739.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound207072.actual selector witness, LeftBound205739.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound207077

namespace LeftBound207081
def owner : Owner := ⟨.program ⟨257⟩, ⟨61952⟩⟩
def transferEvent : Nat := 207081
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 207079 .coefficient, .predecessor 1 207080 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 207079 .coefficient)
      LeftBound207076.bound (LeftBound207076.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events808.exact207078RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound207076.bound, RecordedBoundRefines] <;> decide)
      (LeftBound207076.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 207080 .coefficient)
      LeftBound205525.bound (LeftBound205525.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events802.exact205532RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound205525.bound, RecordedBoundRefines] <;> decide)
      (LeftBound205525.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound207076.bound, LeftBound205525.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound207076.bound, LeftBound205525.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound207076.actual selector witness, LeftBound205525.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound207081

namespace LeftBound207082
def owner : Owner := ⟨.program ⟨257⟩, ⟨61952⟩⟩
def transferEvent : Nat := 207082
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 207078 .summary, .result 205532 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 207078 .summary)
      LeftBound207077.bound (LeftBound207077.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨58972⟩⟩) (rawTerms := some (Proof.Events808.exact207078RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound207077.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 205532 .summary)
      LeftBound205527.bound (LeftBound205527.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨61951⟩⟩) (rawTerms := some (Proof.Events802.exact205532RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound205527.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound207077.bound, LeftBound205527.bound]
def bound : CoeffClass := .finite ⟨2765055493188795324243372926469393465999412, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound207077.bound, LeftBound205527.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound207077.actual selector witness, LeftBound205527.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound207082

namespace LeftBound207086
def owner : Owner := ⟨.program ⟨257⟩, ⟨64932⟩⟩
def transferEvent : Nat := 207086
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 207084 .coefficient, .predecessor 1 207085 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 207084 .coefficient)
      LeftBound207081.bound (LeftBound207081.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events808.exact207083RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound207081.bound, RecordedBoundRefines] <;> decide)
      (LeftBound207081.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 207085 .coefficient)
      LeftBound205313.bound (LeftBound205313.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events802.exact205320RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound205313.bound, RecordedBoundRefines] <;> decide)
      (LeftBound205313.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound207081.bound, LeftBound205313.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound207081.bound, LeftBound205313.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound207081.actual selector witness, LeftBound205313.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound207086

namespace LeftBound207087
def owner : Owner := ⟨.program ⟨257⟩, ⟨64932⟩⟩
def transferEvent : Nat := 207087
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 207083 .summary, .result 205320 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 207083 .summary)
      LeftBound207082.bound (LeftBound207082.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨61952⟩⟩) (rawTerms := some (Proof.Events808.exact207083RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound207082.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 205320 .summary)
      LeftBound205315.bound (LeftBound205315.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨64931⟩⟩) (rawTerms := some (Proof.Events802.exact205320RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound205315.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound207082.bound, LeftBound205315.bound]
def bound : CoeffClass := .finite ⟨3110701272581949232038858886277070355169332, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound207082.bound, LeftBound205315.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound207082.actual selector witness, LeftBound205315.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound207087

namespace LeftBound207091
def owner : Owner := ⟨.program ⟨257⟩, ⟨70325⟩⟩
def transferEvent : Nat := 207091
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 207089 .coefficient, .predecessor 1 207090 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 207089 .coefficient)
      LeftBound207086.bound (LeftBound207086.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events808.exact207088RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound207086.bound, RecordedBoundRefines] <;> decide)
      (LeftBound207086.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 207090 .coefficient)
      LeftBound205101.bound (LeftBound205101.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events801.exact205108RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound205101.bound, RecordedBoundRefines] <;> decide)
      (LeftBound205101.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound207086.bound, LeftBound205101.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound207086.bound, LeftBound205101.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound207086.actual selector witness, LeftBound205101.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound207091

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
