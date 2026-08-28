import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1899
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1946
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1947

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound287152
def owner : Owner := ⟨.program ⟨257⟩, ⟨52457⟩⟩
def transferEvent : Nat := 287152
def frameStart : Nat := 287040
def rule : BoundRule := .sum [.predecessor 0 287150 .coefficient, .predecessor 1 287151 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 287150 .coefficient)
      LeftBound287148.bound (LeftBound287148.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1121.exact287149RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound287148.bound, RecordedBoundRefines] <;> decide)
      (LeftBound287148.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 287151 .coefficient)
      LeftBound287129.bound (LeftBound287129.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1121.exact287134RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound287129.bound, RecordedBoundRefines] <;> decide)
      (LeftBound287129.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound287148.bound, LeftBound287129.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound287148.bound, LeftBound287129.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound287148.actual selector witness, LeftBound287129.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound287152

namespace LeftBound287165
def owner : Owner := ⟨.program ⟨257⟩, ⟨52455⟩⟩
def transferEvent : Nat := 287165
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 287163 .coefficient, .predecessor 1 287164 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 287163 .coefficient)
      LeftBound286988.bound (LeftBound286988.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1121.exact287162RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound286988.bound, RecordedBoundRefines] <;> decide)
      (LeftBound286988.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 287164 .coefficient)
      LeftBound286971.bound (LeftBound286971.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1121.exact286978RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound286971.bound, RecordedBoundRefines] <;> decide)
      (LeftBound286971.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound286988.bound, LeftBound286971.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound286988.bound, LeftBound286971.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound286988.actual selector witness, LeftBound286971.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound287165

namespace LeftBound287168
def owner : Owner := ⟨.program ⟨257⟩, ⟨52455⟩⟩
def transferEvent : Nat := 287168
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 287162 .summary, .result 286978 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 287162 .summary)
      LeftBound286990.bound (LeftBound286990.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨51392⟩⟩) (rawTerms := some (Proof.Events1121.exact287162RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound286990.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 286978 .summary)
      LeftBound286973.bound (LeftBound286973.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨52454⟩⟩) (rawTerms := some (Proof.Events1121.exact286978RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound286973.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound286990.bound, LeftBound286973.bound]
def bound : CoeffClass := .finite ⟨2997889464187086962688, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound286990.bound, LeftBound286973.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound286990.actual selector witness, LeftBound286973.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound287168

namespace LeftBound287172
def owner : Owner := ⟨.program ⟨257⟩, ⟨52768⟩⟩
def transferEvent : Nat := 287172
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 287170 .coefficient) (.predecessor 1 287171 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 287170 .coefficient)
      LeftBound287165.bound (LeftBound287165.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1121.exact287169RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound287165.bound, RecordedBoundRefines] <;> decide)
      (LeftBound287165.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 287171 .coefficient)
      LeftAuthority286893.bound (LeftAuthority286893.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1120.exact286894RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority286893.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority286893.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound287165.bound LeftAuthority286893.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound287165.bound, LeftAuthority286893.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound287165.actual selector witness) * (LeftAuthority286893.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound287172

namespace LeftBound287173
def owner : Owner := ⟨.program ⟨257⟩, ⟨52768⟩⟩
def transferEvent : Nat := 287173
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨52766⟩⟩]⟩ [⟨.result 286894 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 286894 .coefficient)
      LeftAuthority286893.bound (LeftAuthority286893.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨52766⟩⟩) (rawTerms := some (Proof.Events1120.exact286894RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority286893.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority286893.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority286893.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority286893.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority286893.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound287173

namespace LeftBound287174
def owner : Owner := ⟨.program ⟨257⟩, ⟨52768⟩⟩
def transferEvent : Nat := 287174
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 287169 .summary) (.transfer 287173) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 287169 .summary)
      LeftBound287168.bound (LeftBound287168.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨52455⟩⟩) (rawTerms := some (Proof.Events1121.exact287169RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound287168.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 287173)
      LeftBound287173.bound (LeftBound287173.actual selector witness) := by
  exact .transfer (LeftBound287173.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound287168.bound LeftBound287173.bound
def bound : CoeffClass := .finite ⟨32189593014266254325632330629120, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound287168.bound, LeftBound287173.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound287168.actual selector witness) * (LeftBound287173.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound287174

namespace LeftBound287185
def owner : Owner := ⟨.program ⟨257⟩, ⟨51638⟩⟩
def transferEvent : Nat := 287185
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 287183 .coefficient) (.value (.predecessor 1 287184 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 287183 .coefficient)
      LeftAuthority287181.bound (LeftAuthority287181.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1121.exact287182RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority287181.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority287181.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 287184 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority287181.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority287181.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority287181.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound287185

namespace LeftBound287189
def owner : Owner := ⟨.program ⟨257⟩, ⟨51639⟩⟩
def transferEvent : Nat := 287189
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 287187 .coefficient) (.predecessor 1 287188 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 287187 .coefficient)
      LeftBound280742.bound (LeftBound280742.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1096.exact280745RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound280742.bound, RecordedBoundRefines] <;> decide)
      (LeftBound280742.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 287188 .coefficient)
      LeftBound287185.bound (LeftBound287185.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1121.exact287186RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound287185.bound, RecordedBoundRefines] <;> decide)
      (LeftBound287185.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound280742.bound LeftBound287185.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound280742.bound, LeftBound287185.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound280742.actual selector witness) * (LeftBound287185.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound287189

namespace LeftBound287190
def owner : Owner := ⟨.program ⟨257⟩, ⟨51639⟩⟩
def transferEvent : Nat := 287190
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨51636⟩⟩]⟩ [⟨.result 287182 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 287182 .coefficient)
      LeftAuthority287181.bound (LeftAuthority287181.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨51636⟩⟩) (rawTerms := some (Proof.Events1121.exact287182RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority287181.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority287181.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority287181.bound []
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority287181.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority287181.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound287190

namespace LeftBound287191
def owner : Owner := ⟨.program ⟨257⟩, ⟨51639⟩⟩
def transferEvent : Nat := 287191
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 280745 .summary) (.transfer 287190) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 280745 .summary)
      LeftBound280743.bound (LeftBound280743.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨5491⟩⟩) (rawTerms := some (Proof.Events1096.exact280745RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound280743.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 287190)
      LeftBound287190.bound (LeftBound287190.actual selector witness) := by
  exact .transfer (LeftBound287190.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound280743.bound LeftBound287190.bound
def bound : CoeffClass := .finite ⟨202072841853861888, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound280743.bound, LeftBound287190.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound280743.actual selector witness) * (LeftBound287190.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound287191

namespace LeftBound287286
def owner : Owner := ⟨.program ⟨257⟩, ⟨50841⟩⟩
def transferEvent : Nat := 287286
def frameStart : Nat := 287247
def rule : BoundRule := .identity (.predecessor 0 287285 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 287285 .coefficient)
      LeftAuthority287283.bound (LeftAuthority287283.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1122.exact287284RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority287283.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority287283.derived selector witness)

def rawBound : CoeffClass := LeftAuthority287283.bound
def bound : CoeffClass := .finite ⟨10, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority287283.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority287283.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound287286

namespace LeftBound287303
def owner : Owner := ⟨.program ⟨257⟩, ⟨52342⟩⟩
def transferEvent : Nat := 287303
def frameStart : Nat := 287247
def rule : BoundRule := .sum [.predecessor 0 287301 .coefficient, .predecessor 1 287302 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 287301 .coefficient)
      LeftBound287286.bound (LeftBound287286.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound287286.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 287302 .coefficient)
      LeftAuthority287299.bound (LeftAuthority287299.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority287299.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound287286.bound, LeftAuthority287299.bound]
def bound : CoeffClass := .finite ⟨10, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound287286.bound, LeftAuthority287299.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound287286.actual selector witness, LeftAuthority287299.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound287303

namespace LeftBound287306
def owner : Owner := ⟨.program ⟨257⟩, ⟨52343⟩⟩
def transferEvent : Nat := 287306
def frameStart : Nat := 287247
def rule : BoundRule := .identity (.predecessor 0 287305 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 287305 .coefficient)
      LeftBound287303.bound (LeftBound287303.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound287303.derived selector witness)

def rawBound : CoeffClass := LeftBound287303.bound
def bound : CoeffClass := .finite ⟨10, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound287303.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound287303.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound287306

namespace LeftBound287312
def owner : Owner := ⟨.program ⟨257⟩, ⟨52344⟩⟩
def transferEvent : Nat := 287312
def frameStart : Nat := 287247
def rule : BoundRule := .product (.predecessor 0 287310 .coefficient) (.predecessor 1 287311 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 287310 .coefficient)
      LeftAuthority287308.bound (LeftAuthority287308.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1122.exact287309RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority287308.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority287308.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 287311 .coefficient)
      LeftBound287306.bound (LeftBound287306.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1122.exact287307RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound287306.bound, RecordedBoundRefines] <;> decide)
      (LeftBound287306.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority287308.bound LeftBound287306.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority287308.bound, LeftBound287306.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority287308.actual selector witness) * (LeftBound287306.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound287312

namespace LeftBound287320
def owner : Owner := ⟨.program ⟨257⟩, ⟨52345⟩⟩
def transferEvent : Nat := 287320
def frameStart : Nat := 287247
def rule : BoundRule := .sum [.predecessor 0 287318 .coefficient, .predecessor 1 287319 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 287318 .coefficient)
      LeftAuthority287316.bound (LeftAuthority287316.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1122.exact287317RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority287316.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority287316.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 287319 .coefficient)
      LeftBound287312.bound (LeftBound287312.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1122.exact287314RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound287312.bound, RecordedBoundRefines] <;> decide)
      (LeftBound287312.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority287316.bound, LeftBound287312.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority287316.bound, LeftBound287312.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority287316.actual selector witness, LeftBound287312.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound287320

namespace LeftBound287324
def owner : Owner := ⟨.program ⟨257⟩, ⟨52767⟩⟩
def transferEvent : Nat := 287324
def frameStart : Nat := 287247
def rule : BoundRule := .product (.predecessor 0 287322 .coefficient) (.predecessor 1 287323 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 287322 .coefficient)
      LeftBound287320.bound (LeftBound287320.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1122.exact287321RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound287320.bound, RecordedBoundRefines] <;> decide)
      (LeftBound287320.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 287323 .coefficient)
      LeftAuthority287297.bound (LeftAuthority287297.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1122.exact287298RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority287297.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority287297.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound287320.bound LeftAuthority287297.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound287320.bound, LeftAuthority287297.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound287320.actual selector witness) * (LeftAuthority287297.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound287324

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
