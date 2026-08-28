import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard060
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1591
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1593
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1691

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound251081
def owner : Owner := ⟨.program ⟨257⟩, ⟨71179⟩⟩
def transferEvent : Nat := 251081
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 251058 .summary, .result 251013 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 251058 .summary)
      LeftBound251035.bound (LeftBound251035.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨9628⟩⟩) (rawTerms := some (Proof.Events980.exact251058RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound251035.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 251013 .summary)
      LeftBound251012.bound (LeftBound251012.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨71178⟩⟩) (rawTerms := some (Proof.Events980.exact251013RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound251012.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound251035.bound, LeftBound251012.bound]
def bound : CoeffClass := .finite ⟨66805187227601152574551644069558752530002375679672372, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound251035.bound, LeftBound251012.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound251035.actual selector witness, LeftBound251012.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound251081

namespace LeftBound251085
def owner : Owner := ⟨.program ⟨257⟩, ⟨71180⟩⟩
def transferEvent : Nat := 251085
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 251083 .coefficient) (.predecessor 1 251084 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 251083 .coefficient)
      LeftBound251061.bound (LeftBound251061.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events980.exact251082RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound251061.bound, RecordedBoundRefines] <;> decide)
      (LeftBound251061.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 251084 .coefficient)
      LeftBound16583.bound (LeftBound16583.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events064.exact16584RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound16583.bound, RecordedBoundRefines] <;> decide)
      (LeftBound16583.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound251061.bound LeftBound16583.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound251061.bound, LeftBound16583.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound251061.actual selector witness) * (LeftBound16583.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound251085

namespace LeftBound251086
def owner : Owner := ⟨.program ⟨257⟩, ⟨71180⟩⟩
def transferEvent : Nat := 251086
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨9521⟩⟩]⟩ [⟨.result 16580 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 16580 .coefficient)
      LeftAuthority16579.bound (LeftAuthority16579.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨9521⟩⟩) (rawTerms := some (Proof.Events064.exact16580RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority16579.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority16579.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority16579.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority16579.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority16579.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound251086

namespace LeftBound251087
def owner : Owner := ⟨.program ⟨257⟩, ⟨71180⟩⟩
def transferEvent : Nat := 251087
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 251082 .summary) (.transfer 251086) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 251082 .summary)
      LeftBound251081.bound (LeftBound251081.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨71179⟩⟩) (rawTerms := some (Proof.Events980.exact251082RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound251081.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 251086)
      LeftBound251086.bound (LeftBound251086.actual selector witness) := by
  exact .transfer (LeftBound251086.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound251081.bound LeftBound251086.bound
def bound : CoeffClass := .finite ⟨717315235864259647099013782854467978167293655866246524336865280, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound251081.bound, LeftBound251086.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound251081.actual selector witness) * (LeftBound251086.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound251087

namespace LeftBound251149
def owner : Owner := ⟨.program ⟨257⟩, ⟨71181⟩⟩
def transferEvent : Nat := 251149
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 251147 .coefficient, .predecessor 1 251148 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 251147 .coefficient)
      LeftBound251085.bound (LeftBound251085.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events981.exact251146RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound251085.bound, RecordedBoundRefines] <;> decide)
      (LeftBound251085.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 251148 .coefficient)
      LeftBound236666.bound (LeftBound236666.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events924.exact236743RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound236666.bound, RecordedBoundRefines] <;> decide)
      (LeftBound236666.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound251085.bound, LeftBound236666.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound251085.bound, LeftBound236666.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound251085.actual selector witness, LeftBound236666.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound251149

namespace LeftBound251169
def owner : Owner := ⟨.program ⟨257⟩, ⟨71181⟩⟩
def transferEvent : Nat := 251169
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 251146 .summary, .result 236743 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 251146 .summary)
      LeftBound251087.bound (LeftBound251087.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨71180⟩⟩) (rawTerms := some (Proof.Events981.exact251146RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound251087.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 236743 .summary)
      LeftBound236704.bound (LeftBound236704.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨67423⟩⟩) (rawTerms := some (Proof.Events924.exact236743RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound236704.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound251087.bound, LeftBound236704.bound]
def bound : CoeffClass := .finite ⟨717315235864259647099013782854474880280923984914290088855535616, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound251087.bound, LeftBound236704.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound251087.actual selector witness, LeftBound236704.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound251169

namespace LeftBound251173
def owner : Owner := ⟨.program ⟨257⟩, ⟨71182⟩⟩
def transferEvent : Nat := 251173
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 251171 .coefficient) (.predecessor 1 251172 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 251171 .coefficient)
      LeftBound251149.bound (LeftBound251149.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events981.exact251170RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound251149.bound, RecordedBoundRefines] <;> decide)
      (LeftBound251149.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 251172 .coefficient)
      LeftBound16573.bound (LeftBound16573.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events064.exact16574RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound16573.bound, RecordedBoundRefines] <;> decide)
      (LeftBound16573.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound251149.bound LeftBound16573.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound251149.bound, LeftBound16573.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound251149.actual selector witness) * (LeftBound16573.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound251173

namespace LeftBound251174
def owner : Owner := ⟨.program ⟨257⟩, ⟨71182⟩⟩
def transferEvent : Nat := 251174
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨7137⟩⟩]⟩ [⟨.result 16570 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 16570 .coefficient)
      LeftAuthority16569.bound (LeftAuthority16569.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨7137⟩⟩) (rawTerms := some (Proof.Events064.exact16570RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority16569.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority16569.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority16569.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority16569.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority16569.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound251174

namespace LeftBound251175
def owner : Owner := ⟨.program ⟨257⟩, ⟨71182⟩⟩
def transferEvent : Nat := 251175
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 251170 .summary) (.transfer 251174) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 251170 .summary)
      LeftBound251169.bound (LeftBound251169.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨71181⟩⟩) (rawTerms := some (Proof.Events981.exact251170RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound251169.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 251174)
      LeftBound251174.bound (LeftBound251174.actual selector witness) := by
  exact .transfer (LeftBound251174.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound251169.bound LeftBound251174.bound
def bound : CoeffClass := .finite ⟨7702113697398803698856913678033037845150209519672183236728648848208035840, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound251169.bound, LeftBound251174.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound251169.actual selector witness) * (LeftBound251174.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound251175

namespace LeftBound251256
def owner : Owner := ⟨.program ⟨257⟩, ⟨5705⟩⟩
def transferEvent : Nat := 251256
def frameStart : Nat := 0
def rule : BoundRule := .identity (.predecessor 0 251251 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 251251 .coefficient)
      LeftAuthority19.bound (LeftAuthority19.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events000.exact20RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority19.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority19.derived selector witness)

def rawBound : CoeffClass := LeftAuthority19.bound
def bound : CoeffClass := .finite ⟨1, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority19.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority19.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound251256

namespace LeftBound251260
def owner : Owner := ⟨.program ⟨257⟩, ⟨6974⟩⟩
def transferEvent : Nat := 251260
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 251258 .coefficient) (.predecessor 1 251259 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 251258 .coefficient)
      LeftBound251256.bound (LeftBound251256.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events981.exact251257RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound251256.bound, RecordedBoundRefines] <;> decide)
      (LeftBound251256.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 251259 .coefficient)
      LeftAuthority1.bound (LeftAuthority1.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events000.exact2RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority1.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority1.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftBound251256.bound LeftAuthority1.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound251256.bound, LeftAuthority1.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftBound251256.actual selector witness) * (LeftAuthority1.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 40) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound251260

namespace LeftBound251272
def owner : Owner := ⟨.program ⟨257⟩, ⟨5507⟩⟩
def transferEvent : Nat := 251272
def frameStart : Nat := 0
def rule : BoundRule := .identity (.predecessor 0 251267 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 251267 .coefficient)
      LeftAuthority19.bound (LeftAuthority19.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events000.exact20RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority19.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority19.derived selector witness)

def rawBound : CoeffClass := LeftAuthority19.bound
def bound : CoeffClass := .finite ⟨1, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority19.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority19.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound251272

namespace LeftBound251276
def owner : Owner := ⟨.program ⟨257⟩, ⟨8007⟩⟩
def transferEvent : Nat := 251276
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 251274 .coefficient) (.predecessor 1 251275 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 251274 .coefficient)
      LeftBound251272.bound (LeftBound251272.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events981.exact251273RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound251272.bound, RecordedBoundRefines] <;> decide)
      (LeftBound251272.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 251275 .coefficient)
      LeftAuthority16616.bound (LeftAuthority16616.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events064.exact16617RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority16616.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority16616.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftBound251272.bound LeftAuthority16616.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound251272.bound, LeftAuthority16616.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftBound251272.actual selector witness) * (LeftAuthority16616.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 40) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound251276

namespace LeftBound251281
def owner : Owner := ⟨.program ⟨257⟩, ⟨9327⟩⟩
def transferEvent : Nat := 251281
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 251279 .coefficient, .predecessor 1 251280 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 251279 .coefficient)
      LeftBound251276.bound (LeftBound251276.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events981.exact251278RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound251276.bound, RecordedBoundRefines] <;> decide)
      (LeftBound251276.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 251280 .coefficient)
      LeftBound251260.bound (LeftBound251260.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events981.exact251262RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound251260.bound, RecordedBoundRefines] <;> decide)
      (LeftBound251260.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound251276.bound, LeftBound251260.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound251276.bound, LeftBound251260.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound251276.actual selector witness, LeftBound251260.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound251281

namespace LeftBound251285
def owner : Owner := ⟨.program ⟨257⟩, ⟨9328⟩⟩
def transferEvent : Nat := 251285
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 251283 .coefficient, .predecessor 1 251284 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 251283 .coefficient)
      LeftBound251281.bound (LeftBound251281.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events981.exact251282RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound251281.bound, RecordedBoundRefines] <;> decide)
      (LeftBound251281.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 251284 .coefficient)
      LeftAuthority251235.bound (LeftAuthority251235.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events981.exact251236RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority251235.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority251235.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound251281.bound, LeftAuthority251235.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound251281.bound, LeftAuthority251235.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound251281.actual selector witness, LeftAuthority251235.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound251285

namespace LeftBound251286
def owner : Owner := ⟨.program ⟨257⟩, ⟨9328⟩⟩
def transferEvent : Nat := 251286
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨88⟩⟩]⟩ [⟨.result 251236 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 251236 .coefficient)
      LeftAuthority251235.bound (LeftAuthority251235.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨88⟩⟩) (rawTerms := some (Proof.Events981.exact251236RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority251235.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority251235.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority251235.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority251235.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority251235.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound251286

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
