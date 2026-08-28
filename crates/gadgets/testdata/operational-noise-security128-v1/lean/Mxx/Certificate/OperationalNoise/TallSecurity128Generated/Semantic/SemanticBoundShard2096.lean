import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard170
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard272
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard373
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard475
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard576
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard678
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard779
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard880
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard881
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard2095

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound308163
def owner : Owner := ⟨.program ⟨257⟩, ⟨71379⟩⟩
def transferEvent : Nat := 308163
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 308161 .coefficient, .predecessor 1 308162 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 308161 .coefficient)
      LeftBound308158.bound (LeftBound308158.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1203.exact308160RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound308158.bound, RecordedBoundRefines] <;> decide)
      (LeftBound308158.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 308162 .coefficient)
      LeftBound134173.bound (LeftBound134173.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events524.exact134234RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound134173.bound, RecordedBoundRefines] <;> decide)
      (LeftBound134173.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound308158.bound, LeftBound134173.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound308158.bound, LeftBound134173.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound308158.actual selector witness, LeftBound134173.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound308163

namespace LeftBound308164
def owner : Owner := ⟨.program ⟨257⟩, ⟨71379⟩⟩
def transferEvent : Nat := 308164
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 308160 .summary, .result 134234 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 308160 .summary)
      LeftBound308159.bound (LeftBound308159.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨71378⟩⟩) (rawTerms := some (Proof.Events1203.exact308160RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound308159.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 134234 .summary)
      LeftBound134175.bound (LeftBound134175.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨71123⟩⟩) (rawTerms := some (Proof.Events524.exact134234RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound134175.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound308159.bound, LeftBound134175.bound]
def bound : CoeffClass := .finite ⟨92425364369502959622147223783495467924656982214233489499014120725031157812, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound308159.bound, LeftBound134175.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound308159.actual selector witness, LeftBound134175.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound308164

namespace LeftBound308168
def owner : Owner := ⟨.program ⟨257⟩, ⟨71380⟩⟩
def transferEvent : Nat := 308168
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 308166 .coefficient, .predecessor 1 308167 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 308166 .coefficient)
      LeftBound308163.bound (LeftBound308163.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1203.exact308165RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound308163.bound, RecordedBoundRefines] <;> decide)
      (LeftBound308163.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 308167 .coefficient)
      LeftBound119548.bound (LeftBound119548.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events467.exact119609RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound119548.bound, RecordedBoundRefines] <;> decide)
      (LeftBound119548.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound308163.bound, LeftBound119548.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound308163.bound, LeftBound119548.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound308163.actual selector witness, LeftBound119548.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound308168

namespace LeftBound308169
def owner : Owner := ⟨.program ⟨257⟩, ⟨71380⟩⟩
def transferEvent : Nat := 308169
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 308165 .summary, .result 119609 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 308165 .summary)
      LeftBound308164.bound (LeftBound308164.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨71379⟩⟩) (rawTerms := some (Proof.Events1203.exact308165RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound308164.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 119609 .summary)
      LeftBound119550.bound (LeftBound119550.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨71277⟩⟩) (rawTerms := some (Proof.Events467.exact119609RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound119550.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound308164.bound, LeftBound119550.bound]
def bound : CoeffClass := .finite ⟨100127478066901763321004137461528505769807191733905672735742769573239193652, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound308164.bound, LeftBound119550.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound308164.actual selector witness, LeftBound119550.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound308169

namespace LeftBound308173
def owner : Owner := ⟨.program ⟨257⟩, ⟨71416⟩⟩
def transferEvent : Nat := 308173
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 308171 .coefficient, .predecessor 1 308172 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 308171 .coefficient)
      LeftBound308168.bound (LeftBound308168.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1203.exact308170RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound308168.bound, RecordedBoundRefines] <;> decide)
      (LeftBound308168.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 308172 .coefficient)
      LeftBound104923.bound (LeftBound104923.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events410.exact104984RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound104923.bound, RecordedBoundRefines] <;> decide)
      (LeftBound104923.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound308168.bound, LeftBound104923.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound308168.bound, LeftBound104923.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound308168.actual selector witness, LeftBound104923.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound308173

namespace LeftBound308174
def owner : Owner := ⟨.program ⟨257⟩, ⟨71416⟩⟩
def transferEvent : Nat := 308174
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 308170 .summary, .result 104984 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 308170 .summary)
      LeftBound308169.bound (LeftBound308169.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨71380⟩⟩) (rawTerms := some (Proof.Events1203.exact308170RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound308169.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 104984 .summary)
      LeftBound104925.bound (LeftBound104925.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨71415⟩⟩) (rawTerms := some (Proof.Events410.exact104984RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound104925.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound308169.bound, LeftBound104925.bound]
def bound : CoeffClass := .finite ⟨107829591764300567019861051139561543614957401253577855972471418421447229492, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound308169.bound, LeftBound104925.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound308169.actual selector witness, LeftBound104925.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound308174

namespace LeftBound308178
def owner : Owner := ⟨.program ⟨257⟩, ⟨71448⟩⟩
def transferEvent : Nat := 308178
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 308176 .coefficient, .predecessor 1 308177 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 308176 .coefficient)
      LeftBound308173.bound (LeftBound308173.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1203.exact308175RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound308173.bound, RecordedBoundRefines] <;> decide)
      (LeftBound308173.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 308177 .coefficient)
      LeftBound90298.bound (LeftBound90298.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events352.exact90359RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound90298.bound, RecordedBoundRefines] <;> decide)
      (LeftBound90298.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound308173.bound, LeftBound90298.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound308173.bound, LeftBound90298.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound308173.actual selector witness, LeftBound90298.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound308178

namespace LeftBound308179
def owner : Owner := ⟨.program ⟨257⟩, ⟨71448⟩⟩
def transferEvent : Nat := 308179
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 308175 .summary, .result 90359 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 308175 .summary)
      LeftBound308174.bound (LeftBound308174.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨71416⟩⟩) (rawTerms := some (Proof.Events1203.exact308175RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound308174.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 90359 .summary)
      LeftBound90300.bound (LeftBound90300.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨71447⟩⟩) (rawTerms := some (Proof.Events352.exact90359RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound90300.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound308174.bound, LeftBound90300.bound]
def bound : CoeffClass := .finite ⟨115531705461699370718717964817594581460107610773250039209200067269655265332, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound308174.bound, LeftBound90300.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound308174.actual selector witness, LeftBound90300.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound308179

namespace LeftBound308183
def owner : Owner := ⟨.program ⟨257⟩, ⟨71480⟩⟩
def transferEvent : Nat := 308183
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 308181 .coefficient, .predecessor 1 308182 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 308181 .coefficient)
      LeftBound308178.bound (LeftBound308178.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1203.exact308180RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound308178.bound, RecordedBoundRefines] <;> decide)
      (LeftBound308178.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 308182 .coefficient)
      LeftBound75673.bound (LeftBound75673.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events295.exact75734RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound75673.bound, RecordedBoundRefines] <;> decide)
      (LeftBound75673.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound308178.bound, LeftBound75673.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound308178.bound, LeftBound75673.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound308178.actual selector witness, LeftBound75673.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound308183

namespace LeftBound308184
def owner : Owner := ⟨.program ⟨257⟩, ⟨71480⟩⟩
def transferEvent : Nat := 308184
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 308180 .summary, .result 75734 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 308180 .summary)
      LeftBound308179.bound (LeftBound308179.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨71448⟩⟩) (rawTerms := some (Proof.Events1203.exact308180RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound308179.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 75734 .summary)
      LeftBound75675.bound (LeftBound75675.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨71479⟩⟩) (rawTerms := some (Proof.Events295.exact75734RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound75675.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound308179.bound, LeftBound75675.bound]
def bound : CoeffClass := .finite ⟨123233819159098174417574878495627619305257820292922222445928716117863301172, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound308179.bound, LeftBound75675.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound308179.actual selector witness, LeftBound75675.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound308184

namespace LeftBound308188
def owner : Owner := ⟨.program ⟨257⟩, ⟨71512⟩⟩
def transferEvent : Nat := 308188
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 308186 .coefficient, .predecessor 1 308187 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 308186 .coefficient)
      LeftBound308183.bound (LeftBound308183.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1203.exact308185RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound308183.bound, RecordedBoundRefines] <;> decide)
      (LeftBound308183.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 308187 .coefficient)
      LeftBound61048.bound (LeftBound61048.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events238.exact61109RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound61048.bound, RecordedBoundRefines] <;> decide)
      (LeftBound61048.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound308183.bound, LeftBound61048.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound308183.bound, LeftBound61048.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound308183.actual selector witness, LeftBound61048.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound308188

namespace LeftBound308189
def owner : Owner := ⟨.program ⟨257⟩, ⟨71512⟩⟩
def transferEvent : Nat := 308189
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 308185 .summary, .result 61109 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 308185 .summary)
      LeftBound308184.bound (LeftBound308184.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨71480⟩⟩) (rawTerms := some (Proof.Events1203.exact308185RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound308184.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 61109 .summary)
      LeftBound61050.bound (LeftBound61050.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨71511⟩⟩) (rawTerms := some (Proof.Events238.exact61109RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound61050.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound308184.bound, LeftBound61050.bound]
def bound : CoeffClass := .finite ⟨130935932856496978116431792173660657150408029812594405682657364966071337012, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound308184.bound, LeftBound61050.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound308184.actual selector witness, LeftBound61050.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound308189

namespace LeftBound308193
def owner : Owner := ⟨.program ⟨257⟩, ⟨71545⟩⟩
def transferEvent : Nat := 308193
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 308191 .coefficient, .predecessor 1 308192 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 308191 .coefficient)
      LeftBound308188.bound (LeftBound308188.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1203.exact308190RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound308188.bound, RecordedBoundRefines] <;> decide)
      (LeftBound308188.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 308192 .coefficient)
      LeftBound46423.bound (LeftBound46423.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events181.exact46484RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound46423.bound, RecordedBoundRefines] <;> decide)
      (LeftBound46423.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound308188.bound, LeftBound46423.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound308188.bound, LeftBound46423.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound308188.actual selector witness, LeftBound46423.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound308193

namespace LeftBound308194
def owner : Owner := ⟨.program ⟨257⟩, ⟨71545⟩⟩
def transferEvent : Nat := 308194
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 308190 .summary, .result 46484 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 308190 .summary)
      LeftBound308189.bound (LeftBound308189.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨71512⟩⟩) (rawTerms := some (Proof.Events1203.exact308190RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound308189.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 46484 .summary)
      LeftBound46425.bound (LeftBound46425.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨71544⟩⟩) (rawTerms := some (Proof.Events181.exact46484RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound46425.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound308189.bound, LeftBound46425.bound]
def bound : CoeffClass := .finite ⟨138638046553895781815288705851693694995558239332266588919386013814279372852, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound308189.bound, LeftBound46425.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound308189.actual selector witness, LeftBound46425.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound308194

namespace LeftBound308198
def owner : Owner := ⟨.program ⟨257⟩, ⟨71546⟩⟩
def transferEvent : Nat := 308198
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 308196 .coefficient, .predecessor 1 308197 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 308196 .coefficient)
      LeftBound308193.bound (LeftBound308193.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1203.exact308195RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound308193.bound, RecordedBoundRefines] <;> decide)
      (LeftBound308193.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 308197 .coefficient)
      LeftBound31798.bound (LeftBound31798.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events124.exact31859RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound31798.bound, RecordedBoundRefines] <;> decide)
      (LeftBound31798.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound308193.bound, LeftBound31798.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound308193.bound, LeftBound31798.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound308193.actual selector witness, LeftBound31798.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound308198

namespace LeftBound308199
def owner : Owner := ⟨.program ⟨257⟩, ⟨71546⟩⟩
def transferEvent : Nat := 308199
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 308195 .summary, .result 31859 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 308195 .summary)
      LeftBound308194.bound (LeftBound308194.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨71545⟩⟩) (rawTerms := some (Proof.Events1203.exact308195RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound308194.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 31859 .summary)
      LeftBound31800.bound (LeftBound31800.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70978⟩⟩) (rawTerms := some (Proof.Events124.exact31859RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound31800.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound308194.bound, LeftBound31800.bound]
def bound : CoeffClass := .finite ⟨146340160251294585514145619529726732840708448851938772156114662662487408692, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound308194.bound, LeftBound31800.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound308194.actual selector witness, LeftBound31800.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound308199

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
