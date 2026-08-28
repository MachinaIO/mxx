import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard259
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard260
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard261
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard263
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard264
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard265
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard267
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard268

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound46171
def owner : Owner := ⟨.program ⟨257⟩, ⟨18011⟩⟩
def transferEvent : Nat := 46171
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 46169 .coefficient, .predecessor 1 46170 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 46169 .coefficient)
      LeftBound46164.bound (LeftBound46164.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events180.exact46168RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound46164.bound, RecordedBoundRefines] <;> decide)
      (LeftBound46164.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 46170 .coefficient)
      LeftBound46134.bound (LeftBound46134.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events180.exact46141RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound46134.bound, RecordedBoundRefines] <;> decide)
      (LeftBound46134.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound46164.bound, LeftBound46134.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound46164.bound, LeftBound46134.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound46164.actual selector witness, LeftBound46134.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound46171

namespace LeftBound46172
def owner : Owner := ⟨.program ⟨257⟩, ⟨18011⟩⟩
def transferEvent : Nat := 46172
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 46168 .summary, .result 46141 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 46168 .summary)
      LeftBound46167.bound (LeftBound46167.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨11650⟩⟩) (rawTerms := some (Proof.Events180.exact46168RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound46167.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 46141 .summary)
      LeftBound46136.bound (LeftBound46136.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨18010⟩⟩) (rawTerms := some (Proof.Events180.exact46141RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound46136.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound46167.bound, LeftBound46136.bound]
def bound : CoeffClass := .finite ⟨345624685687166110058245054666339432529972, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound46167.bound, LeftBound46136.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound46167.actual selector witness, LeftBound46136.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound46172

namespace LeftBound46176
def owner : Owner := ⟨.program ⟨257⟩, ⟨20929⟩⟩
def transferEvent : Nat := 46176
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 46174 .coefficient, .predecessor 1 46175 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 46174 .coefficient)
      LeftBound46171.bound (LeftBound46171.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events180.exact46173RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound46171.bound, RecordedBoundRefines] <;> decide)
      (LeftBound46171.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 46175 .coefficient)
      LeftBound45922.bound (LeftBound45922.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events179.exact45929RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound45922.bound, RecordedBoundRefines] <;> decide)
      (LeftBound45922.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound46171.bound, LeftBound45922.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound46171.bound, LeftBound45922.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound46171.actual selector witness, LeftBound45922.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound46176

namespace LeftBound46177
def owner : Owner := ⟨.program ⟨257⟩, ⟨20929⟩⟩
def transferEvent : Nat := 46177
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 46173 .summary, .result 45929 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 46173 .summary)
      LeftBound46172.bound (LeftBound46172.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨18011⟩⟩) (rawTerms := some (Proof.Events180.exact46173RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound46172.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 45929 .summary)
      LeftBound45924.bound (LeftBound45924.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨20928⟩⟩) (rawTerms := some (Proof.Events179.exact45929RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound45924.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound46172.bound, LeftBound45924.bound]
def bound : CoeffClass := .finite ⟨691250426059631610003352154589745737891892, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound46172.bound, LeftBound45924.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound46172.actual selector witness, LeftBound45924.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound46177

namespace LeftBound46181
def owner : Owner := ⟨.program ⟨257⟩, ⟨24149⟩⟩
def transferEvent : Nat := 46181
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 46179 .coefficient, .predecessor 1 46180 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 46179 .coefficient)
      LeftBound46176.bound (LeftBound46176.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events180.exact46178RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound46176.bound, RecordedBoundRefines] <;> decide)
      (LeftBound46176.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 46180 .coefficient)
      LeftBound45710.bound (LeftBound45710.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events178.exact45717RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound45710.bound, RecordedBoundRefines] <;> decide)
      (LeftBound45710.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound46176.bound, LeftBound45710.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound46176.bound, LeftBound45710.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound46176.actual selector witness, LeftBound45710.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound46181

namespace LeftBound46182
def owner : Owner := ⟨.program ⟨257⟩, ⟨24149⟩⟩
def transferEvent : Nat := 46182
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 46178 .summary, .result 45717 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 46178 .summary)
      LeftBound46177.bound (LeftBound46177.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨20929⟩⟩) (rawTerms := some (Proof.Events180.exact46178RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound46177.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 45717 .summary)
      LeftBound45712.bound (LeftBound45712.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨24148⟩⟩) (rawTerms := some (Proof.Events178.exact45717RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound45712.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound46177.bound, LeftBound45712.bound]
def bound : CoeffClass := .finite ⟨1036877221117396499835321299770218916085812, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound46177.bound, LeftBound45712.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound46177.actual selector witness, LeftBound45712.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound46182

namespace LeftBound46186
def owner : Owner := ⟨.program ⟨257⟩, ⟨34169⟩⟩
def transferEvent : Nat := 46186
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 46184 .coefficient, .predecessor 1 46185 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 46184 .coefficient)
      LeftBound46181.bound (LeftBound46181.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events180.exact46183RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound46181.bound, RecordedBoundRefines] <;> decide)
      (LeftBound46181.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 46185 .coefficient)
      LeftBound45498.bound (LeftBound45498.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events177.exact45505RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound45498.bound, RecordedBoundRefines] <;> decide)
      (LeftBound45498.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound46181.bound, LeftBound45498.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound46181.bound, LeftBound45498.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound46181.actual selector witness, LeftBound45498.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound46186

namespace LeftBound46187
def owner : Owner := ⟨.program ⟨257⟩, ⟨34169⟩⟩
def transferEvent : Nat := 46187
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 46183 .summary, .result 45505 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 46183 .summary)
      LeftBound46182.bound (LeftBound46182.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨24149⟩⟩) (rawTerms := some (Proof.Events180.exact46183RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound46182.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 45505 .summary)
      LeftBound45500.bound (LeftBound45500.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨34168⟩⟩) (rawTerms := some (Proof.Events177.exact45505RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound45500.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound46182.bound, LeftBound45500.bound]
def bound : CoeffClass := .finite ⟨1382506125545760169441014535464825839943732, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound46182.bound, LeftBound45500.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound46182.actual selector witness, LeftBound45500.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound46187

namespace LeftBound46191
def owner : Owner := ⟨.program ⟨257⟩, ⟨53229⟩⟩
def transferEvent : Nat := 46191
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 46189 .coefficient, .predecessor 1 46190 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 46189 .coefficient)
      LeftBound46186.bound (LeftBound46186.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events180.exact46188RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound46186.bound, RecordedBoundRefines] <;> decide)
      (LeftBound46186.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 46190 .coefficient)
      LeftBound45286.bound (LeftBound45286.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events176.exact45293RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound45286.bound, RecordedBoundRefines] <;> decide)
      (LeftBound45286.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound46186.bound, LeftBound45286.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound46186.bound, LeftBound45286.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound46186.actual selector witness, LeftBound45286.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound46191

namespace LeftBound46192
def owner : Owner := ⟨.program ⟨257⟩, ⟨53229⟩⟩
def transferEvent : Nat := 46192
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 46188 .summary, .result 45293 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 46188 .summary)
      LeftBound46187.bound (LeftBound46187.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨34169⟩⟩) (rawTerms := some (Proof.Events180.exact46188RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound46187.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 45293 .summary)
      LeftBound45288.bound (LeftBound45288.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨53228⟩⟩) (rawTerms := some (Proof.Events176.exact45293RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound45288.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound46187.bound, LeftBound45288.bound]
def bound : CoeffClass := .finite ⟨1728139248715321398594155952187700255129652, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound46187.bound, LeftBound45288.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound46187.actual selector witness, LeftBound45288.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound46192

namespace LeftBound46196
def owner : Owner := ⟨.program ⟨257⟩, ⟨56209⟩⟩
def transferEvent : Nat := 46196
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 46194 .coefficient, .predecessor 1 46195 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 46194 .coefficient)
      LeftBound46191.bound (LeftBound46191.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events180.exact46193RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound46191.bound, RecordedBoundRefines] <;> decide)
      (LeftBound46191.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 46195 .coefficient)
      LeftBound45074.bound (LeftBound45074.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events176.exact45081RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound45074.bound, RecordedBoundRefines] <;> decide)
      (LeftBound45074.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound46191.bound, LeftBound45074.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound46191.bound, LeftBound45074.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound46191.actual selector witness, LeftBound45074.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound46196

namespace LeftBound46197
def owner : Owner := ⟨.program ⟨257⟩, ⟨56209⟩⟩
def transferEvent : Nat := 46197
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 46193 .summary, .result 45081 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 46193 .summary)
      LeftBound46192.bound (LeftBound46192.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨53229⟩⟩) (rawTerms := some (Proof.Events180.exact46193RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound46192.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 45081 .summary)
      LeftBound45076.bound (LeftBound45076.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨56208⟩⟩) (rawTerms := some (Proof.Events176.exact45081RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound45076.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound46192.bound, LeftBound45076.bound]
def bound : CoeffClass := .finite ⟨2073774481255481407521021459424708415979572, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound46192.bound, LeftBound45076.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound46192.actual selector witness, LeftBound45076.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound46197

namespace LeftBound46201
def owner : Owner := ⟨.program ⟨257⟩, ⟨59189⟩⟩
def transferEvent : Nat := 46201
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 46199 .coefficient, .predecessor 1 46200 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 46199 .coefficient)
      LeftBound46196.bound (LeftBound46196.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events180.exact46198RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound46196.bound, RecordedBoundRefines] <;> decide)
      (LeftBound46196.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 46200 .coefficient)
      LeftBound44862.bound (LeftBound44862.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events175.exact44869RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound44862.bound, RecordedBoundRefines] <;> decide)
      (LeftBound44862.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound46196.bound, LeftBound44862.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound46196.bound, LeftBound44862.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound46196.actual selector witness, LeftBound44862.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound46201

namespace LeftBound46202
def owner : Owner := ⟨.program ⟨257⟩, ⟨59189⟩⟩
def transferEvent : Nat := 46202
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 46198 .summary, .result 44869 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 46198 .summary)
      LeftBound46197.bound (LeftBound46197.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨56209⟩⟩) (rawTerms := some (Proof.Events180.exact46198RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound46197.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 44869 .summary)
      LeftBound44864.bound (LeftBound44864.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨59188⟩⟩) (rawTerms := some (Proof.Events175.exact44869RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound44864.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound46197.bound, LeftBound44864.bound]
def bound : CoeffClass := .finite ⟨2419413932536838975995335147689984068157492, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound46197.bound, LeftBound44864.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound46197.actual selector witness, LeftBound44864.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound46202

namespace LeftBound46206
def owner : Owner := ⟨.program ⟨257⟩, ⟨62169⟩⟩
def transferEvent : Nat := 46206
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 46204 .coefficient, .predecessor 1 46205 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 46204 .coefficient)
      LeftBound46201.bound (LeftBound46201.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events180.exact46203RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound46201.bound, RecordedBoundRefines] <;> decide)
      (LeftBound46201.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 46205 .coefficient)
      LeftBound44650.bound (LeftBound44650.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events174.exact44657RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound44650.bound, RecordedBoundRefines] <;> decide)
      (LeftBound44650.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound46201.bound, LeftBound44650.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound46201.bound, LeftBound44650.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound46201.actual selector witness, LeftBound44650.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound46206

namespace LeftBound46207
def owner : Owner := ⟨.program ⟨257⟩, ⟨62169⟩⟩
def transferEvent : Nat := 46207
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 46203 .summary, .result 44657 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 46203 .summary)
      LeftBound46202.bound (LeftBound46202.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨59189⟩⟩) (rawTerms := some (Proof.Events180.exact46203RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound46202.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 44657 .summary)
      LeftBound44652.bound (LeftBound44652.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨62168⟩⟩) (rawTerms := some (Proof.Events174.exact44657RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound44652.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound46202.bound, LeftBound44652.bound]
def bound : CoeffClass := .finite ⟨2765055493188795324243372926469393465999412, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound46202.bound, LeftBound44652.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound46202.actual selector witness, LeftBound44652.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound46207

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
