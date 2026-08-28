import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard1225
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1223
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1224

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult173540
def owner : Owner := ⟨.program ⟨257⟩, ⟨54218⟩⟩
def rawTerms : List Term := Proof.Events677.exact173540RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 173540
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult173540.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 173537) (rightBinding := 173538)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨51238⟩) (rightExpression := ⟨54217⟩)
    (transferEvent := 173539)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult173536.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult173405.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult173540

namespace SemanticResult173544
def owner : Owner := ⟨.program ⟨257⟩, ⟨57198⟩⟩
def rawTerms : List Term := Proof.Events677.exact173544RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 173544
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult173544.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 173541) (rightBinding := 173542)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨54218⟩) (rightExpression := ⟨57197⟩)
    (transferEvent := 173543)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult173540.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult173382.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult173544

namespace SemanticResult173548
def owner : Owner := ⟨.program ⟨257⟩, ⟨60178⟩⟩
def rawTerms : List Term := Proof.Events677.exact173548RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 173548
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult173548.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 173545) (rightBinding := 173546)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨57198⟩) (rightExpression := ⟨60177⟩)
    (transferEvent := 173547)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult173544.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult173359.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult173548

namespace SemanticResult173552
def owner : Owner := ⟨.program ⟨257⟩, ⟨63158⟩⟩
def rawTerms : List Term := Proof.Events677.exact173552RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 173552
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult173552.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 173549) (rightBinding := 173550)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨60178⟩) (rightExpression := ⟨63157⟩)
    (transferEvent := 173551)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult173548.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult173336.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult173552

namespace SemanticResult173556
def owner : Owner := ⟨.program ⟨257⟩, ⟨66882⟩⟩
def rawTerms : List Term := Proof.Events677.exact173556RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 173556
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult173556.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 173553) (rightBinding := 173554)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨63158⟩) (rightExpression := ⟨66881⟩)
    (transferEvent := 173555)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult173552.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult173313.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult173556

namespace SemanticResult173560
def owner : Owner := ⟨.program ⟨257⟩, ⟨66883⟩⟩
def rawTerms : List Term := Proof.Events677.exact173560RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 173560
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult173560.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 173557) (rightBinding := 173558)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨66882⟩) (rightExpression := ⟨26671⟩)
    (transferEvent := 173559)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult173556.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult173290.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult173560

namespace SemanticResult173564
def owner : Owner := ⟨.program ⟨257⟩, ⟨66884⟩⟩
def rawTerms : List Term := Proof.Events677.exact173564RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 173564
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult173564.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 173561) (rightBinding := 173562)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨66883⟩) (rightExpression := ⟨29351⟩)
    (transferEvent := 173563)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult173560.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult173267.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult173564

namespace SemanticResult173568
def owner : Owner := ⟨.program ⟨257⟩, ⟨66885⟩⟩
def rawTerms : List Term := Proof.Events678.exact173568RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 173568
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult173568.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 173565) (rightBinding := 173566)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨66884⟩) (rightExpression := ⟨35015⟩)
    (transferEvent := 173567)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult173564.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult173244.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult173568

namespace SemanticResult173572
def owner : Owner := ⟨.program ⟨257⟩, ⟨66886⟩⟩
def rawTerms : List Term := Proof.Events678.exact173572RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 173572
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult173572.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 173569) (rightBinding := 173570)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨66885⟩) (rightExpression := ⟨37695⟩)
    (transferEvent := 173571)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult173568.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult173221.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult173572

namespace SemanticResult173576
def owner : Owner := ⟨.program ⟨257⟩, ⟨66887⟩⟩
def rawTerms : List Term := Proof.Events678.exact173576RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 173576
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult173576.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 173573) (rightBinding := 173574)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨66886⟩) (rightExpression := ⟨40371⟩)
    (transferEvent := 173575)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult173572.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult173198.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult173576

namespace SemanticResult173580
def owner : Owner := ⟨.program ⟨257⟩, ⟨66888⟩⟩
def rawTerms : List Term := Proof.Events678.exact173580RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 173580
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult173580.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 173577) (rightBinding := 173578)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨66887⟩) (rightExpression := ⟨43051⟩)
    (transferEvent := 173579)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult173576.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult173175.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult173580

namespace SemanticResult173584
def owner : Owner := ⟨.program ⟨257⟩, ⟨66889⟩⟩
def rawTerms : List Term := Proof.Events678.exact173584RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 173584
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult173584.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 173581) (rightBinding := 173582)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨66888⟩) (rightExpression := ⟨45735⟩)
    (transferEvent := 173583)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult173580.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult173152.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult173584

namespace SemanticResult173588
def owner : Owner := ⟨.program ⟨257⟩, ⟨66890⟩⟩
def rawTerms : List Term := Proof.Events678.exact173588RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 173588
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult173588.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 173585) (rightBinding := 173586)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨66889⟩) (rightExpression := ⟨48415⟩)
    (transferEvent := 173587)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult173584.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult173129.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult173588

namespace SemanticResult173599
def owner : Owner := ⟨.program ⟨257⟩, ⟨68854⟩⟩
def rawTerms : List Term := Proof.Events678.exact173599RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 173599
def producerEvent : Nat := 173598
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult173599.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.operator), 173086, .large, .authorityOperator, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult173599

namespace SemanticResult173602
def owner : Owner := ⟨.program ⟨257⟩, ⟨71365⟩⟩
def rawTerms : List Term := Proof.Events678.exact173602RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 173602
def producerEvent : Nat := 173601
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult173602.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.operator), 173086, .finite 8192, .authorityOperator, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult173602

namespace SemanticResult173611
def owner : Owner := ⟨.program ⟨257⟩, ⟨69104⟩⟩
def rawTerms : List Term := Proof.Events678.exact173611RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 173611
def producerEvent : Nat := 173610
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult173611.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.identity (.predecessor 0 173609 .coefficient), 173086, .finite 1059, .identity (.predecessor 0 173609 .coefficient), ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult173611

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
