import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard086
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard087
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1591
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1594
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1611

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound239139
def owner : Owner := ⟨.program ⟨257⟩, ⟨39260⟩⟩
def transferEvent : Nat := 239139
def frameStart : Nat := 239062
def rule : BoundRule := .product (.predecessor 0 239137 .coefficient) (.predecessor 1 239138 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 239137 .coefficient)
      LeftBound239135.bound (LeftBound239135.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events934.exact239136RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound239135.bound, RecordedBoundRefines] <;> decide)
      (LeftBound239135.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 239138 .coefficient)
      LeftAuthority239112.bound (LeftAuthority239112.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events934.exact239113RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority239112.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority239112.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound239135.bound LeftAuthority239112.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound239135.bound, LeftAuthority239112.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound239135.actual selector witness) * (LeftAuthority239112.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound239139

namespace LeftBound239150
def owner : Owner := ⟨.program ⟨257⟩, ⟨37618⟩⟩
def transferEvent : Nat := 239150
def frameStart : Nat := 239062
def rule : BoundRule := .product (.predecessor 0 239148 .coefficient) (.predecessor 1 239149 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 239148 .coefficient)
      LeftAuthority239123.bound (LeftAuthority239123.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events934.exact239124RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority239123.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority239123.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 239149 .coefficient)
      LeftAuthority239146.bound (LeftAuthority239146.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events934.exact239147RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority239146.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority239146.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority239123.bound LeftAuthority239146.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority239123.bound, LeftAuthority239146.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority239123.actual selector witness) * (LeftAuthority239146.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound239150

namespace LeftBound239158
def owner : Owner := ⟨.program ⟨257⟩, ⟨37619⟩⟩
def transferEvent : Nat := 239158
def frameStart : Nat := 239062
def rule : BoundRule := .sum [.predecessor 0 239156 .coefficient, .predecessor 1 239157 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 239156 .coefficient)
      LeftAuthority239154.bound (LeftAuthority239154.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events934.exact239155RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority239154.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority239154.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 239157 .coefficient)
      LeftBound239150.bound (LeftBound239150.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events934.exact239152RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound239150.bound, RecordedBoundRefines] <;> decide)
      (LeftBound239150.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority239154.bound, LeftBound239150.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority239154.bound, LeftBound239150.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority239154.actual selector witness, LeftBound239150.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound239158

namespace LeftBound239162
def owner : Owner := ⟨.program ⟨257⟩, ⟨39263⟩⟩
def transferEvent : Nat := 239162
def frameStart : Nat := 239062
def rule : BoundRule := .sum [.predecessor 0 239160 .coefficient, .predecessor 1 239161 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 239160 .coefficient)
      LeftBound239158.bound (LeftBound239158.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events934.exact239159RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound239158.bound, RecordedBoundRefines] <;> decide)
      (LeftBound239158.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 239161 .coefficient)
      LeftBound239139.bound (LeftBound239139.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events934.exact239144RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound239139.bound, RecordedBoundRefines] <;> decide)
      (LeftBound239139.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound239158.bound, LeftBound239139.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound239158.bound, LeftBound239139.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound239158.actual selector witness, LeftBound239139.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound239162

namespace LeftBound239175
def owner : Owner := ⟨.program ⟨257⟩, ⟨39262⟩⟩
def transferEvent : Nat := 239175
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 239173 .coefficient, .predecessor 1 239174 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 239173 .coefficient)
      LeftBound239004.bound (LeftBound239004.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events934.exact239172RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound239004.bound, RecordedBoundRefines] <;> decide)
      (LeftBound239004.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 239174 .coefficient)
      LeftBound238987.bound (LeftBound238987.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events933.exact238994RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound238987.bound, RecordedBoundRefines] <;> decide)
      (LeftBound238987.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound239004.bound, LeftBound238987.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound239004.bound, LeftBound238987.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound239004.actual selector witness, LeftBound238987.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound239175

namespace LeftBound239178
def owner : Owner := ⟨.program ⟨257⟩, ⟨39262⟩⟩
def transferEvent : Nat := 239178
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 239172 .summary, .result 238994 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 239172 .summary)
      LeftBound239006.bound (LeftBound239006.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨38139⟩⟩) (rawTerms := some (Proof.Events934.exact239172RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound239006.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 238994 .summary)
      LeftBound238989.bound (LeftBound238989.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨39261⟩⟩) (rawTerms := some (Proof.Events933.exact238994RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound238989.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound239006.bound, LeftBound238989.bound]
def bound : CoeffClass := .finite ⟨32192736221397454434328420548608, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound239006.bound, LeftBound238989.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound239006.actual selector witness, LeftBound238989.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound239178

namespace LeftBound239202
def owner : Owner := ⟨.program ⟨257⟩, ⟨34389⟩⟩
def transferEvent : Nat := 239202
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 239200 .coefficient) (.predecessor 1 239201 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 239200 .coefficient)
      LeftAuthority11428.bound (LeftAuthority11428.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events044.exact11429RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority11428.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority11428.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 239201 .coefficient)
      LeftBound236776.bound (LeftBound236776.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events924.exact236778RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound236776.bound, RecordedBoundRefines] <;> decide)
      (LeftBound236776.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32768 ⟨true, false, none, none, none⟩ LeftAuthority11428.bound LeftBound236776.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority11428.bound, LeftBound236776.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := tensorFactor 32768 ⟨true, false, none, none, none⟩ * (LeftAuthority11428.actual selector witness) * (LeftBound236776.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound239202

namespace LeftBound239207
def owner : Owner := ⟨.program ⟨257⟩, ⟨8358⟩⟩
def transferEvent : Nat := 239207
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 239205 .coefficient) (.predecessor 1 239206 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 239205 .coefficient)
      LeftBound236647.bound (LeftBound236647.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events924.exact236648RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound236647.bound, RecordedBoundRefines] <;> decide)
      (LeftBound236647.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 239206 .coefficient)
      LeftBound19584.bound (LeftBound19584.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events076.exact19585RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound19584.bound, RecordedBoundRefines] <;> decide)
      (LeftBound19584.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftBound236647.bound LeftBound19584.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound236647.bound, LeftBound19584.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftBound236647.actual selector witness) * (LeftBound19584.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 40) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound239207

namespace LeftBound239212
def owner : Owner := ⟨.program ⟨257⟩, ⟨34390⟩⟩
def transferEvent : Nat := 239212
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 239210 .coefficient, .predecessor 1 239211 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 239210 .coefficient)
      LeftBound239207.bound (LeftBound239207.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events934.exact239209RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound239207.bound, RecordedBoundRefines] <;> decide)
      (LeftBound239207.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 239211 .coefficient)
      LeftBound239202.bound (LeftBound239202.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events934.exact239204RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound239202.bound, RecordedBoundRefines] <;> decide)
      (LeftBound239202.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound239207.bound, LeftBound239202.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound239207.bound, LeftBound239202.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound239207.actual selector witness, LeftBound239202.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound239212

namespace LeftBound239216
def owner : Owner := ⟨.program ⟨257⟩, ⟨34391⟩⟩
def transferEvent : Nat := 239216
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 239214 .coefficient, .predecessor 1 239215 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 239214 .coefficient)
      LeftBound239212.bound (LeftBound239212.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events934.exact239213RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound239212.bound, RecordedBoundRefines] <;> decide)
      (LeftBound239212.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 239215 .coefficient)
      LeftBound19576.bound (LeftBound19576.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events076.exact19577RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound19576.bound, RecordedBoundRefines] <;> decide)
      (LeftBound19576.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound239212.bound, LeftBound19576.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound239212.bound, LeftBound19576.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound239212.actual selector witness, LeftBound19576.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound239216

namespace LeftBound239217
def owner : Owner := ⟨.program ⟨257⟩, ⟨34391⟩⟩
def transferEvent : Nat := 239217
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨106⟩⟩]⟩ [⟨.result 19577 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 19577 .coefficient)
      LeftBound19576.bound (LeftBound19576.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨106⟩⟩) (rawTerms := some (Proof.Events076.exact19577RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound19576.bound, RecordedBoundRefines] <;> decide)
      (LeftBound19576.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound19576.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound19576.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftBound19576.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound239217

namespace LeftBound239222
def owner : Owner := ⟨.program ⟨257⟩, ⟨34392⟩⟩
def transferEvent : Nat := 239222
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 239220 .coefficient) (.predecessor 1 239221 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 239220 .coefficient)
      LeftBound239216.bound (LeftBound239216.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events934.exact239219RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound239216.bound, RecordedBoundRefines] <;> decide)
      (LeftBound239216.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 239221 .coefficient)
      LeftAuthority11431.bound (LeftAuthority11431.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events044.exact11432RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority11431.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority11431.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftBound239216.bound LeftAuthority11431.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound239216.bound, LeftAuthority11431.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftBound239216.actual selector witness) * (LeftAuthority11431.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound239222

namespace LeftBound239223
def owner : Owner := ⟨.program ⟨257⟩, ⟨34392⟩⟩
def transferEvent : Nat := 239223
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨257⟩, ⟨13551⟩⟩], []⟩ [⟨.result 11432 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 11432 .coefficient)
      LeftAuthority11431.bound (LeftAuthority11431.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨13551⟩⟩) (rawTerms := some (Proof.Events044.exact11432RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority11431.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority11431.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority11431.bound []
def bound : CoeffClass := .finite ⟨40, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority11431.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority11431.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound239223

namespace LeftBound239224
def owner : Owner := ⟨.program ⟨257⟩, ⟨34392⟩⟩
def transferEvent : Nat := 239224
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 239219 .summary) (.transfer 239223) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 239219 .summary)
      LeftBound239217.bound (LeftBound239217.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨34391⟩⟩) (rawTerms := some (Proof.Events934.exact239219RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound239217.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 239223)
      LeftBound239223.bound (LeftBound239223.actual selector witness) := by
  exact .transfer (LeftBound239223.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftBound239217.bound LeftBound239223.bound
def bound : CoeffClass := .finite ⟨34078720, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound239217.bound, LeftBound239223.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftBound239217.actual selector witness) * (LeftBound239223.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound239224

namespace LeftBound239230
def owner : Owner := ⟨.program ⟨257⟩, ⟨13552⟩⟩
def transferEvent : Nat := 239230
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 239228 .coefficient) (.predecessor 1 239229 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 239228 .coefficient)
      LeftAuthority11431.bound (LeftAuthority11431.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events044.exact11432RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority11431.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority11431.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 239229 .coefficient)
      LeftBound236776.bound (LeftBound236776.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events924.exact236778RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound236776.bound, RecordedBoundRefines] <;> decide)
      (LeftBound236776.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32768 ⟨true, false, none, none, none⟩ LeftAuthority11431.bound LeftBound236776.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority11431.bound, LeftBound236776.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := tensorFactor 32768 ⟨true, false, none, none, none⟩ * (LeftAuthority11431.actual selector witness) * (LeftBound236776.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound239230

namespace LeftBound239235
def owner : Owner := ⟨.program ⟨257⟩, ⟨8375⟩⟩
def transferEvent : Nat := 239235
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 239233 .coefficient) (.predecessor 1 239234 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 239233 .coefficient)
      LeftBound236647.bound (LeftBound236647.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events924.exact236648RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound236647.bound, RecordedBoundRefines] <;> decide)
      (LeftBound236647.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 239234 .coefficient)
      LeftBound19625.bound (LeftBound19625.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events076.exact19626RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound19625.bound, RecordedBoundRefines] <;> decide)
      (LeftBound19625.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftBound236647.bound LeftBound19625.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound236647.bound, LeftBound19625.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftBound236647.actual selector witness) * (LeftBound19625.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 40) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound239235

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
