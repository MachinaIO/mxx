import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1871
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1873
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1874
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1875
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1876
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1877
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1878
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1879
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1880
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1881
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1892

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound280211
def owner : Owner := ⟨.program ⟨257⟩, ⟨64613⟩⟩
def transferEvent : Nat := 280211
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 280209 .coefficient, .predecessor 1 280210 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 280209 .coefficient)
      LeftBound280206.bound (LeftBound280206.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1094.exact280208RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound280206.bound, RecordedBoundRefines] <;> decide)
      (LeftBound280206.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 280210 .coefficient)
      LeftBound278438.bound (LeftBound278438.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1087.exact278445RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound278438.bound, RecordedBoundRefines] <;> decide)
      (LeftBound278438.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound280206.bound, LeftBound278438.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound280206.bound, LeftBound278438.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound280206.actual selector witness, LeftBound278438.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound280211

namespace LeftBound280212
def owner : Owner := ⟨.program ⟨257⟩, ⟨64613⟩⟩
def transferEvent : Nat := 280212
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 280208 .summary, .result 278445 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 280208 .summary)
      LeftBound280207.bound (LeftBound280207.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨61633⟩⟩) (rawTerms := some (Proof.Events1094.exact280208RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound280207.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 278445 .summary)
      LeftBound278440.bound (LeftBound278440.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨64612⟩⟩) (rawTerms := some (Proof.Events1087.exact278445RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound278440.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound280207.bound, LeftBound278440.bound]
def bound : CoeffClass := .finite ⟨3110701272581949232038858886277070355169332, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound280207.bound, LeftBound278440.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound280207.actual selector witness, LeftBound278440.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound280212

namespace LeftBound280216
def owner : Owner := ⟨.program ⟨257⟩, ⟨69510⟩⟩
def transferEvent : Nat := 280216
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 280214 .coefficient, .predecessor 1 280215 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 280214 .coefficient)
      LeftBound280211.bound (LeftBound280211.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1094.exact280213RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound280211.bound, RecordedBoundRefines] <;> decide)
      (LeftBound280211.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 280215 .coefficient)
      LeftBound278226.bound (LeftBound278226.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1086.exact278233RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound278226.bound, RecordedBoundRefines] <;> decide)
      (LeftBound278226.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound280211.bound, LeftBound278226.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound280211.bound, LeftBound278226.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound280211.actual selector witness, LeftBound278226.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound280216

namespace LeftBound280217
def owner : Owner := ⟨.program ⟨257⟩, ⟨69510⟩⟩
def transferEvent : Nat := 280217
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 280213 .summary, .result 278233 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 280213 .summary)
      LeftBound280212.bound (LeftBound280212.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨64613⟩⟩) (rawTerms := some (Proof.Events1094.exact280213RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound280212.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 278233 .summary)
      LeftBound278228.bound (LeftBound278228.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨69509⟩⟩) (rawTerms := some (Proof.Events1086.exact278233RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound278228.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound280212.bound, LeftBound278228.bound]
def bound : CoeffClass := .finite ⟨3456353380086899479155517117627148481331252, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound280212.bound, LeftBound278228.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound280212.actual selector witness, LeftBound278228.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound280217

namespace LeftBound280221
def owner : Owner := ⟨.program ⟨257⟩, ⟨69511⟩⟩
def transferEvent : Nat := 280221
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 280219 .coefficient, .predecessor 1 280220 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 280219 .coefficient)
      LeftBound280216.bound (LeftBound280216.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1094.exact280218RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound280216.bound, RecordedBoundRefines] <;> decide)
      (LeftBound280216.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 280220 .coefficient)
      LeftBound278014.bound (LeftBound278014.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1086.exact278021RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound278014.bound, RecordedBoundRefines] <;> decide)
      (LeftBound278014.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound280216.bound, LeftBound278014.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound280216.bound, LeftBound278014.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound280216.actual selector witness, LeftBound278014.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound280221

namespace LeftBound280222
def owner : Owner := ⟨.program ⟨257⟩, ⟨69511⟩⟩
def transferEvent : Nat := 280222
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 280218 .summary, .result 278021 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 280218 .summary)
      LeftBound280217.bound (LeftBound280217.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨69510⟩⟩) (rawTerms := some (Proof.Events1094.exact280218RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound280217.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 278021 .summary)
      LeftBound278016.bound (LeftBound278016.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨28080⟩⟩) (rawTerms := some (Proof.Events1086.exact278021RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound278016.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound280217.bound, LeftBound278016.bound]
def bound : CoeffClass := .finite ⟨3802007596962448506045899439491360353157172, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound280217.bound, LeftBound278016.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound280217.actual selector witness, LeftBound278016.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound280222

namespace LeftBound280226
def owner : Owner := ⟨.program ⟨257⟩, ⟨69512⟩⟩
def transferEvent : Nat := 280226
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 280224 .coefficient, .predecessor 1 280225 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 280224 .coefficient)
      LeftBound280221.bound (LeftBound280221.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1094.exact280223RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound280221.bound, RecordedBoundRefines] <;> decide)
      (LeftBound280221.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 280225 .coefficient)
      LeftBound277802.bound (LeftBound277802.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1085.exact277809RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound277802.bound, RecordedBoundRefines] <;> decide)
      (LeftBound277802.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound280221.bound, LeftBound277802.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound280221.bound, LeftBound277802.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound280221.actual selector witness, LeftBound277802.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound280226

namespace LeftBound280227
def owner : Owner := ⟨.program ⟨257⟩, ⟨69512⟩⟩
def transferEvent : Nat := 280227
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 280223 .summary, .result 277809 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 280223 .summary)
      LeftBound280222.bound (LeftBound280222.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨69511⟩⟩) (rawTerms := some (Proof.Events1094.exact280223RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound280222.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 277809 .summary)
      LeftBound277804.bound (LeftBound277804.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨30760⟩⟩) (rawTerms := some (Proof.Events1085.exact277809RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound277804.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound280222.bound, LeftBound277804.bound]
def bound : CoeffClass := .finite ⟨4147668141949793872257454032897973461975092, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound280222.bound, LeftBound277804.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound280222.actual selector witness, LeftBound277804.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound280227

namespace LeftBound280231
def owner : Owner := ⟨.program ⟨257⟩, ⟨69513⟩⟩
def transferEvent : Nat := 280231
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 280229 .coefficient, .predecessor 1 280230 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 280229 .coefficient)
      LeftBound280226.bound (LeftBound280226.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1094.exact280228RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound280226.bound, RecordedBoundRefines] <;> decide)
      (LeftBound280226.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 280230 .coefficient)
      LeftBound277590.bound (LeftBound277590.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1084.exact277597RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound277590.bound, RecordedBoundRefines] <;> decide)
      (LeftBound277590.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound280226.bound, LeftBound277590.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound280226.bound, LeftBound277590.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound280226.actual selector witness, LeftBound277590.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound280231

namespace LeftBound280232
def owner : Owner := ⟨.program ⟨257⟩, ⟨69513⟩⟩
def transferEvent : Nat := 280232
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 280228 .summary, .result 277597 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 280228 .summary)
      LeftBound280227.bound (LeftBound280227.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨69512⟩⟩) (rawTerms := some (Proof.Events1094.exact280228RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound280227.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 277597 .summary)
      LeftBound277592.bound (LeftBound277592.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨36420⟩⟩) (rawTerms := some (Proof.Events1084.exact277597RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound277592.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound280227.bound, LeftBound277592.bound]
def bound : CoeffClass := .finite ⟨4493332905678336798016456807332854062121012, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound280227.bound, LeftBound277592.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound280227.actual selector witness, LeftBound277592.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound280232

namespace LeftBound280236
def owner : Owner := ⟨.program ⟨257⟩, ⟨69514⟩⟩
def transferEvent : Nat := 280236
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 280234 .coefficient, .predecessor 1 280235 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 280234 .coefficient)
      LeftBound280231.bound (LeftBound280231.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1094.exact280233RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound280231.bound, RecordedBoundRefines] <;> decide)
      (LeftBound280231.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 280235 .coefficient)
      LeftBound277378.bound (LeftBound277378.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1083.exact277385RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound277378.bound, RecordedBoundRefines] <;> decide)
      (LeftBound277378.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound280231.bound, LeftBound277378.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound280231.bound, LeftBound277378.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound280231.actual selector witness, LeftBound277378.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound280236

namespace LeftBound280237
def owner : Owner := ⟨.program ⟨257⟩, ⟨69514⟩⟩
def transferEvent : Nat := 280237
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 280233 .summary, .result 277385 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 280233 .summary)
      LeftBound280232.bound (LeftBound280232.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨69513⟩⟩) (rawTerms := some (Proof.Events1094.exact280233RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound280232.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 277385 .summary)
      LeftBound277380.bound (LeftBound277380.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨39100⟩⟩) (rawTerms := some (Proof.Events1083.exact277385RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound277380.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound280232.bound, LeftBound277380.bound]
def bound : CoeffClass := .finite ⟨4838999778777478503549183672281868407930932, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound280232.bound, LeftBound277380.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound280232.actual selector witness, LeftBound277380.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound280237

namespace LeftBound280241
def owner : Owner := ⟨.program ⟨257⟩, ⟨69515⟩⟩
def transferEvent : Nat := 280241
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 280239 .coefficient, .predecessor 1 280240 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 280239 .coefficient)
      LeftBound280236.bound (LeftBound280236.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1094.exact280238RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound280236.bound, RecordedBoundRefines] <;> decide)
      (LeftBound280236.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 280240 .coefficient)
      LeftBound277166.bound (LeftBound277166.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1082.exact277173RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound277166.bound, RecordedBoundRefines] <;> decide)
      (LeftBound277166.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound280236.bound, LeftBound277166.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound280236.bound, LeftBound277166.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound280236.actual selector witness, LeftBound277166.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound280241

namespace LeftBound280242
def owner : Owner := ⟨.program ⟨257⟩, ⟨69515⟩⟩
def transferEvent : Nat := 280242
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 280238 .summary, .result 277173 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 280238 .summary)
      LeftBound280237.bound (LeftBound280237.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨69514⟩⟩) (rawTerms := some (Proof.Events1094.exact280238RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound280237.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 277173 .summary)
      LeftBound277168.bound (LeftBound277168.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨41780⟩⟩) (rawTerms := some (Proof.Events1082.exact277173RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound277168.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound280237.bound, LeftBound277168.bound]
def bound : CoeffClass := .finite ⟨5184670870617817768629358718259150245068852, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound280237.bound, LeftBound277168.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound280237.actual selector witness, LeftBound277168.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound280242

namespace LeftBound280246
def owner : Owner := ⟨.program ⟨257⟩, ⟨69516⟩⟩
def transferEvent : Nat := 280246
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 280244 .coefficient, .predecessor 1 280245 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 280244 .coefficient)
      LeftBound280241.bound (LeftBound280241.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1094.exact280243RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound280241.bound, RecordedBoundRefines] <;> decide)
      (LeftBound280241.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 280245 .coefficient)
      LeftBound276954.bound (LeftBound276954.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1081.exact276961RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound276954.bound, RecordedBoundRefines] <;> decide)
      (LeftBound276954.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound280241.bound, LeftBound276954.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound280241.bound, LeftBound276954.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound280241.actual selector witness, LeftBound276954.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound280246

namespace LeftBound280247
def owner : Owner := ⟨.program ⟨257⟩, ⟨69516⟩⟩
def transferEvent : Nat := 280247
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 280243 .summary, .result 276961 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 280243 .summary)
      LeftBound280242.bound (LeftBound280242.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨69515⟩⟩) (rawTerms := some (Proof.Events1094.exact280243RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound280242.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 276961 .summary)
      LeftBound276956.bound (LeftBound276956.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨44460⟩⟩) (rawTerms := some (Proof.Events1081.exact276961RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound276956.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound280242.bound, LeftBound276956.bound]
def bound : CoeffClass := .finite ⟨5530348290569953373030706035778833319198772, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound280242.bound, LeftBound276956.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound280242.actual selector witness, LeftBound276956.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound280247

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
