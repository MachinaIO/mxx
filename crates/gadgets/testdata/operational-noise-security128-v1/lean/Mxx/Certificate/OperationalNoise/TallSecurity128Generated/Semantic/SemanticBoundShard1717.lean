import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard090
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1692
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1695
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1716

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound254242
def owner : Owner := ⟨.program ⟨257⟩, ⟨36089⟩⟩
def transferEvent : Nat := 254242
def frameStart : Nat := 254169
def rule : BoundRule := .sum [.predecessor 0 254240 .coefficient, .predecessor 1 254241 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 254240 .coefficient)
      LeftAuthority254238.bound (LeftAuthority254238.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events993.exact254239RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority254238.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority254238.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 254241 .coefficient)
      LeftBound254234.bound (LeftBound254234.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events993.exact254236RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound254234.bound, RecordedBoundRefines] <;> decide)
      (LeftBound254234.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority254238.bound, LeftBound254234.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority254238.bound, LeftBound254234.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority254238.actual selector witness, LeftBound254234.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound254242

namespace LeftBound254246
def owner : Owner := ⟨.program ⟨257⟩, ⟨36505⟩⟩
def transferEvent : Nat := 254246
def frameStart : Nat := 254169
def rule : BoundRule := .product (.predecessor 0 254244 .coefficient) (.predecessor 1 254245 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 254244 .coefficient)
      LeftBound254242.bound (LeftBound254242.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events993.exact254243RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound254242.bound, RecordedBoundRefines] <;> decide)
      (LeftBound254242.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 254245 .coefficient)
      LeftAuthority254219.bound (LeftAuthority254219.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events993.exact254220RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority254219.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority254219.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound254242.bound LeftAuthority254219.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound254242.bound, LeftAuthority254219.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound254242.actual selector witness) * (LeftAuthority254219.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound254246

namespace LeftBound254257
def owner : Owner := ⟨.program ⟨257⟩, ⟨34899⟩⟩
def transferEvent : Nat := 254257
def frameStart : Nat := 254169
def rule : BoundRule := .product (.predecessor 0 254255 .coefficient) (.predecessor 1 254256 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 254255 .coefficient)
      LeftAuthority254230.bound (LeftAuthority254230.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events993.exact254231RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority254230.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority254230.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 254256 .coefficient)
      LeftAuthority254253.bound (LeftAuthority254253.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events993.exact254254RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority254253.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority254253.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority254230.bound LeftAuthority254253.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority254230.bound, LeftAuthority254253.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority254230.actual selector witness) * (LeftAuthority254253.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound254257

namespace LeftBound254265
def owner : Owner := ⟨.program ⟨257⟩, ⟨34900⟩⟩
def transferEvent : Nat := 254265
def frameStart : Nat := 254169
def rule : BoundRule := .sum [.predecessor 0 254263 .coefficient, .predecessor 1 254264 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 254263 .coefficient)
      LeftAuthority254261.bound (LeftAuthority254261.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events993.exact254262RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority254261.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority254261.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 254264 .coefficient)
      LeftBound254257.bound (LeftBound254257.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events993.exact254259RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound254257.bound, RecordedBoundRefines] <;> decide)
      (LeftBound254257.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority254261.bound, LeftBound254257.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority254261.bound, LeftBound254257.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority254261.actual selector witness, LeftBound254257.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound254265

namespace LeftBound254269
def owner : Owner := ⟨.program ⟨257⟩, ⟨36508⟩⟩
def transferEvent : Nat := 254269
def frameStart : Nat := 254169
def rule : BoundRule := .sum [.predecessor 0 254267 .coefficient, .predecessor 1 254268 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 254267 .coefficient)
      LeftBound254265.bound (LeftBound254265.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events993.exact254266RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound254265.bound, RecordedBoundRefines] <;> decide)
      (LeftBound254265.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 254268 .coefficient)
      LeftBound254246.bound (LeftBound254246.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events993.exact254251RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound254246.bound, RecordedBoundRefines] <;> decide)
      (LeftBound254246.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound254265.bound, LeftBound254246.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound254265.bound, LeftBound254246.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound254265.actual selector witness, LeftBound254246.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound254269

namespace LeftBound254282
def owner : Owner := ⟨.program ⟨257⟩, ⟨36507⟩⟩
def transferEvent : Nat := 254282
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 254280 .coefficient, .predecessor 1 254281 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 254280 .coefficient)
      LeftBound254111.bound (LeftBound254111.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events993.exact254279RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound254111.bound, RecordedBoundRefines] <;> decide)
      (LeftBound254111.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 254281 .coefficient)
      LeftBound254094.bound (LeftBound254094.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events992.exact254101RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound254094.bound, RecordedBoundRefines] <;> decide)
      (LeftBound254094.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound254111.bound, LeftBound254094.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound254111.bound, LeftBound254094.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound254111.actual selector witness, LeftBound254094.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound254282

namespace LeftBound254285
def owner : Owner := ⟨.program ⟨257⟩, ⟨36507⟩⟩
def transferEvent : Nat := 254285
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 254279 .summary, .result 254101 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 254279 .summary)
      LeftBound254113.bound (LeftBound254113.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨35399⟩⟩) (rawTerms := some (Proof.Events993.exact254279RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound254113.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 254101 .summary)
      LeftBound254096.bound (LeftBound254096.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨36506⟩⟩) (rawTerms := some (Proof.Events992.exact254101RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound254096.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound254113.bound, LeftBound254096.bound]
def bound : CoeffClass := .finite ⟨32192539770951767057087530795008, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound254113.bound, LeftBound254096.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound254113.actual selector witness, LeftBound254096.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound254285

namespace LeftBound254309
def owner : Owner := ⟨.program ⟨257⟩, ⟨28657⟩⟩
def transferEvent : Nat := 254309
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 254307 .coefficient) (.predecessor 1 254308 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 254307 .coefficient)
      LeftAuthority12199.bound (LeftAuthority12199.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events047.exact12200RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority12199.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority12199.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 254308 .coefficient)
      LeftBound251401.bound (LeftBound251401.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events982.exact251403RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound251401.bound, RecordedBoundRefines] <;> decide)
      (LeftBound251401.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32768 ⟨true, false, none, none, none⟩ LeftAuthority12199.bound LeftBound251401.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority12199.bound, LeftBound251401.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := tensorFactor 32768 ⟨true, false, none, none, none⟩ * (LeftAuthority12199.actual selector witness) * (LeftBound251401.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound254309

namespace LeftBound254314
def owner : Owner := ⟨.program ⟨257⟩, ⟨8015⟩⟩
def transferEvent : Nat := 254314
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 254312 .coefficient) (.predecessor 1 254313 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 254312 .coefficient)
      LeftBound251272.bound (LeftBound251272.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events981.exact251273RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound251272.bound, RecordedBoundRefines] <;> decide)
      (LeftBound251272.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 254313 .coefficient)
      LeftBound20085.bound (LeftBound20085.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events078.exact20086RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound20085.bound, RecordedBoundRefines] <;> decide)
      (LeftBound20085.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftBound251272.bound LeftBound20085.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound251272.bound, LeftBound20085.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftBound251272.actual selector witness) * (LeftBound20085.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 40) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound254314

namespace LeftBound254319
def owner : Owner := ⟨.program ⟨257⟩, ⟨28658⟩⟩
def transferEvent : Nat := 254319
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 254317 .coefficient, .predecessor 1 254318 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 254317 .coefficient)
      LeftBound254314.bound (LeftBound254314.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events993.exact254316RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound254314.bound, RecordedBoundRefines] <;> decide)
      (LeftBound254314.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 254318 .coefficient)
      LeftBound254309.bound (LeftBound254309.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events993.exact254311RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound254309.bound, RecordedBoundRefines] <;> decide)
      (LeftBound254309.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound254314.bound, LeftBound254309.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound254314.bound, LeftBound254309.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound254314.actual selector witness, LeftBound254309.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound254319

namespace LeftBound254323
def owner : Owner := ⟨.program ⟨257⟩, ⟨28659⟩⟩
def transferEvent : Nat := 254323
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 254321 .coefficient, .predecessor 1 254322 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 254321 .coefficient)
      LeftBound254319.bound (LeftBound254319.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events993.exact254320RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound254319.bound, RecordedBoundRefines] <;> decide)
      (LeftBound254319.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 254322 .coefficient)
      LeftBound20077.bound (LeftBound20077.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events078.exact20078RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound20077.bound, RecordedBoundRefines] <;> decide)
      (LeftBound20077.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound254319.bound, LeftBound20077.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound254319.bound, LeftBound20077.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound254319.actual selector witness, LeftBound20077.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound254323

namespace LeftBound254324
def owner : Owner := ⟨.program ⟨257⟩, ⟨28659⟩⟩
def transferEvent : Nat := 254324
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨105⟩⟩]⟩ [⟨.result 20078 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 20078 .coefficient)
      LeftBound20077.bound (LeftBound20077.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨105⟩⟩) (rawTerms := some (Proof.Events078.exact20078RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound20077.bound, RecordedBoundRefines] <;> decide)
      (LeftBound20077.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound20077.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound20077.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftBound20077.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound254324

namespace LeftBound254329
def owner : Owner := ⟨.program ⟨257⟩, ⟨28660⟩⟩
def transferEvent : Nat := 254329
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 254327 .coefficient) (.predecessor 1 254328 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 254327 .coefficient)
      LeftBound254323.bound (LeftBound254323.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events993.exact254326RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound254323.bound, RecordedBoundRefines] <;> decide)
      (LeftBound254323.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 254328 .coefficient)
      LeftAuthority12202.bound (LeftAuthority12202.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events047.exact12203RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority12202.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority12202.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftBound254323.bound LeftAuthority12202.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound254323.bound, LeftAuthority12202.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftBound254323.actual selector witness) * (LeftAuthority12202.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound254329

namespace LeftBound254330
def owner : Owner := ⟨.program ⟨257⟩, ⟨28660⟩⟩
def transferEvent : Nat := 254330
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨257⟩, ⟨13206⟩⟩], []⟩ [⟨.result 12203 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 12203 .coefficient)
      LeftAuthority12202.bound (LeftAuthority12202.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨13206⟩⟩) (rawTerms := some (Proof.Events047.exact12203RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority12202.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority12202.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority12202.bound []
def bound : CoeffClass := .finite ⟨36, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority12202.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority12202.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound254330

namespace LeftBound254331
def owner : Owner := ⟨.program ⟨257⟩, ⟨28660⟩⟩
def transferEvent : Nat := 254331
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 254326 .summary) (.transfer 254330) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 254326 .summary)
      LeftBound254324.bound (LeftBound254324.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨28659⟩⟩) (rawTerms := some (Proof.Events993.exact254326RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound254324.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 254330)
      LeftBound254330.bound (LeftBound254330.actual selector witness) := by
  exact .transfer (LeftBound254330.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftBound254324.bound LeftBound254330.bound
def bound : CoeffClass := .finite ⟨30670848, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound254324.bound, LeftBound254330.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftBound254324.actual selector witness) * (LeftBound254330.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound254331

namespace LeftBound254337
def owner : Owner := ⟨.program ⟨257⟩, ⟨13207⟩⟩
def transferEvent : Nat := 254337
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 254335 .coefficient) (.predecessor 1 254336 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 254335 .coefficient)
      LeftAuthority12202.bound (LeftAuthority12202.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events047.exact12203RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority12202.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority12202.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 254336 .coefficient)
      LeftBound251401.bound (LeftBound251401.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events982.exact251403RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound251401.bound, RecordedBoundRefines] <;> decide)
      (LeftBound251401.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32768 ⟨true, false, none, none, none⟩ LeftAuthority12202.bound LeftBound251401.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority12202.bound, LeftBound251401.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := tensorFactor 32768 ⟨true, false, none, none, none⟩ * (LeftAuthority12202.actual selector witness) * (LeftBound251401.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound254337

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
