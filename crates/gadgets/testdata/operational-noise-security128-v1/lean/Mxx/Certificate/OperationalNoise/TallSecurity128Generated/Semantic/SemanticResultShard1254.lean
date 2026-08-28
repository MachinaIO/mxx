import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard1254
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1233
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1234
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1235
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1237
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1238
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1239
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1241
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1242
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1244
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1245
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1246
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1248
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1249
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1250
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1252
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1253

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult177798
def owner : Owner := ⟨.program ⟨257⟩, ⟨17871⟩⟩
def rawTerms : List Term := Proof.Events694.exact177798RawTerms
def summary : Bound := (.finite 345624685687166110058245054666339432529972)
def resultEvent : Nat := 177798
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult177798.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult177793.owner)
    (rightOwner := SemanticResult177766.owner)
    (leftResult := 177793) (rightResult := 177766)
    (leftActual := SemanticResult177793.actual selector witness)
    (rightActual := SemanticResult177766.actual selector witness)
    (leftRaw := SemanticResult177793.rawTerms)
    (rightRaw := SemanticResult177766.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 52)
    (rightMaximum := 345624685687166110058245054666339432529920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 177794) (rightBinding := 177795)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨9489⟩) (rightExpression := ⟨17870⟩)
    (transferEvent := 177796) (summaryTransferEvent := 177797)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult177793.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult177766.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult177798

namespace SemanticResult177803
def owner : Owner := ⟨.program ⟨257⟩, ⟨20774⟩⟩
def rawTerms : List Term := Proof.Events694.exact177803RawTerms
def summary : Bound := (.finite 691250426059631610003352154589745737891892)
def resultEvent : Nat := 177803
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult177803.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult177798.owner)
    (rightOwner := SemanticResult177554.owner)
    (leftResult := 177798) (rightResult := 177554)
    (leftActual := SemanticResult177798.actual selector witness)
    (rightActual := SemanticResult177554.actual selector witness)
    (leftRaw := SemanticResult177798.rawTerms)
    (rightRaw := SemanticResult177554.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 345624685687166110058245054666339432529972)
    (rightMaximum := 345625740372465499945107099923406305361920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 177799) (rightBinding := 177800)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨17871⟩) (rightExpression := ⟨20773⟩)
    (transferEvent := 177801) (summaryTransferEvent := 177802)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult177798.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult177554.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult177803

namespace SemanticResult177808
def owner : Owner := ⟨.program ⟨257⟩, ⟨23994⟩⟩
def rawTerms : List Term := Proof.Events694.exact177808RawTerms
def summary : Bound := (.finite 1036877221117396499835321299770218916085812)
def resultEvent : Nat := 177808
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult177808.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult177803.owner)
    (rightOwner := SemanticResult177342.owner)
    (leftResult := 177803) (rightResult := 177342)
    (leftActual := SemanticResult177803.actual selector witness)
    (rightActual := SemanticResult177342.actual selector witness)
    (leftRaw := SemanticResult177803.rawTerms)
    (rightRaw := SemanticResult177342.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 691250426059631610003352154589745737891892)
    (rightMaximum := 345626795057764889831969145180473178193920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 177804) (rightBinding := 177805)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨20774⟩) (rightExpression := ⟨23993⟩)
    (transferEvent := 177806) (summaryTransferEvent := 177807)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult177803.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult177342.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult177808

namespace SemanticResult177813
def owner : Owner := ⟨.program ⟨257⟩, ⟨34014⟩⟩
def rawTerms : List Term := Proof.Events694.exact177813RawTerms
def summary : Bound := (.finite 1382506125545760169441014535464825839943732)
def resultEvent : Nat := 177813
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult177813.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult177808.owner)
    (rightOwner := SemanticResult177130.owner)
    (leftResult := 177808) (rightResult := 177130)
    (leftActual := SemanticResult177808.actual selector witness)
    (rightActual := SemanticResult177130.actual selector witness)
    (leftRaw := SemanticResult177808.rawTerms)
    (rightRaw := SemanticResult177130.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 1036877221117396499835321299770218916085812)
    (rightMaximum := 345628904428363669605693235694606923857920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 177809) (rightBinding := 177810)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨23994⟩) (rightExpression := ⟨34013⟩)
    (transferEvent := 177811) (summaryTransferEvent := 177812)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult177808.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult177130.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult177813

namespace SemanticResult177818
def owner : Owner := ⟨.program ⟨257⟩, ⟨53074⟩⟩
def rawTerms : List Term := Proof.Events694.exact177818RawTerms
def summary : Bound := (.finite 1728139248715321398594155952187700255129652)
def resultEvent : Nat := 177818
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult177818.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult177813.owner)
    (rightOwner := SemanticResult176918.owner)
    (leftResult := 177813) (rightResult := 176918)
    (leftActual := SemanticResult177813.actual selector witness)
    (rightActual := SemanticResult176918.actual selector witness)
    (leftRaw := SemanticResult177813.rawTerms)
    (rightRaw := SemanticResult176918.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 1382506125545760169441014535464825839943732)
    (rightMaximum := 345633123169561229153141416722874415185920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 177814) (rightBinding := 177815)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨34014⟩) (rightExpression := ⟨53073⟩)
    (transferEvent := 177816) (summaryTransferEvent := 177817)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult177813.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult176918.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult177818

namespace SemanticResult177823
def owner : Owner := ⟨.program ⟨257⟩, ⟨56054⟩⟩
def rawTerms : List Term := Proof.Events694.exact177823RawTerms
def summary : Bound := (.finite 2073774481255481407521021459424708415979572)
def resultEvent : Nat := 177823
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult177823.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult177818.owner)
    (rightOwner := SemanticResult176706.owner)
    (leftResult := 177818) (rightResult := 176706)
    (leftActual := SemanticResult177818.actual selector witness)
    (rightActual := SemanticResult176706.actual selector witness)
    (leftRaw := SemanticResult177818.rawTerms)
    (rightRaw := SemanticResult176706.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 1728139248715321398594155952187700255129652)
    (rightMaximum := 345635232540160008926865507237008160849920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 177819) (rightBinding := 177820)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨53074⟩) (rightExpression := ⟨56053⟩)
    (transferEvent := 177821) (summaryTransferEvent := 177822)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult177818.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult176706.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult177823

namespace SemanticResult177828
def owner : Owner := ⟨.program ⟨257⟩, ⟨59034⟩⟩
def rawTerms : List Term := Proof.Events694.exact177828RawTerms
def summary : Bound := (.finite 2419413932536838975995335147689984068157492)
def resultEvent : Nat := 177828
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult177828.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult177823.owner)
    (rightOwner := SemanticResult176494.owner)
    (leftResult := 177823) (rightResult := 176494)
    (leftActual := SemanticResult177823.actual selector witness)
    (rightActual := SemanticResult176494.actual selector witness)
    (leftRaw := SemanticResult177823.rawTerms)
    (rightRaw := SemanticResult176494.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 2073774481255481407521021459424708415979572)
    (rightMaximum := 345639451281357568474313688265275652177920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 177824) (rightBinding := 177825)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨56054⟩) (rightExpression := ⟨59033⟩)
    (transferEvent := 177826) (summaryTransferEvent := 177827)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult177823.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult176494.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult177828

namespace SemanticResult177833
def owner : Owner := ⟨.program ⟨257⟩, ⟨62014⟩⟩
def rawTerms : List Term := Proof.Events694.exact177833RawTerms
def summary : Bound := (.finite 2765055493188795324243372926469393465999412)
def resultEvent : Nat := 177833
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult177833.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult177828.owner)
    (rightOwner := SemanticResult176282.owner)
    (leftResult := 177828) (rightResult := 176282)
    (leftActual := SemanticResult177828.actual selector witness)
    (rightActual := SemanticResult176282.actual selector witness)
    (leftRaw := SemanticResult177828.rawTerms)
    (rightRaw := SemanticResult176282.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 2419413932536838975995335147689984068157492)
    (rightMaximum := 345641560651956348248037778779409397841920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 177829) (rightBinding := 177830)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨59034⟩) (rightExpression := ⟨62013⟩)
    (transferEvent := 177831) (summaryTransferEvent := 177832)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult177828.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult176282.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult177833

namespace SemanticResult177838
def owner : Owner := ⟨.program ⟨257⟩, ⟨64994⟩⟩
def rawTerms : List Term := Proof.Events694.exact177838RawTerms
def summary : Bound := (.finite 3110701272581949232038858886277070355169332)
def resultEvent : Nat := 177838
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult177838.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult177833.owner)
    (rightOwner := SemanticResult176070.owner)
    (leftResult := 177833) (rightResult := 176070)
    (leftActual := SemanticResult177833.actual selector witness)
    (rightActual := SemanticResult176070.actual selector witness)
    (leftRaw := SemanticResult177833.rawTerms)
    (rightRaw := SemanticResult176070.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 2765055493188795324243372926469393465999412)
    (rightMaximum := 345645779393153907795485959807676889169920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 177834) (rightBinding := 177835)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨62014⟩) (rightExpression := ⟨64993⟩)
    (transferEvent := 177836) (summaryTransferEvent := 177837)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult177833.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult176070.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult177838

namespace SemanticResult177843
def owner : Owner := ⟨.program ⟨257⟩, ⟨70483⟩⟩
def rawTerms : List Term := Proof.Events694.exact177843RawTerms
def summary : Bound := (.finite 3456353380086899479155517117627148481331252)
def resultEvent : Nat := 177843
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult177843.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult177838.owner)
    (rightOwner := SemanticResult175858.owner)
    (leftResult := 177838) (rightResult := 175858)
    (leftActual := SemanticResult177838.actual selector witness)
    (rightActual := SemanticResult175858.actual selector witness)
    (leftRaw := SemanticResult177838.rawTerms)
    (rightRaw := SemanticResult175858.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 3110701272581949232038858886277070355169332)
    (rightMaximum := 345652107504950247116658231350078126161920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 177839) (rightBinding := 177840)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨64994⟩) (rightExpression := ⟨70482⟩)
    (transferEvent := 177841) (summaryTransferEvent := 177842)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult177838.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult175858.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult177843

namespace SemanticResult177848
def owner : Owner := ⟨.program ⟨257⟩, ⟨70484⟩⟩
def rawTerms : List Term := Proof.Events694.exact177848RawTerms
def summary : Bound := (.finite 3802007596962448506045899439491360353157172)
def resultEvent : Nat := 177848
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult177848.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult177843.owner)
    (rightOwner := SemanticResult175646.owner)
    (leftResult := 177843) (rightResult := 175646)
    (leftActual := SemanticResult177843.actual selector witness)
    (rightActual := SemanticResult175646.actual selector witness)
    (leftRaw := SemanticResult177843.rawTerms)
    (rightRaw := SemanticResult175646.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 3456353380086899479155517117627148481331252)
    (rightMaximum := 345654216875549026890382321864211871825920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 177844) (rightBinding := 177845)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨70483⟩) (rightExpression := ⟨28387⟩)
    (transferEvent := 177846) (summaryTransferEvent := 177847)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult177843.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult175646.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult177848

namespace SemanticResult177853
def owner : Owner := ⟨.program ⟨257⟩, ⟨70485⟩⟩
def rawTerms : List Term := Proof.Events694.exact177853RawTerms
def summary : Bound := (.finite 4147668141949793872257454032897973461975092)
def resultEvent : Nat := 177853
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult177853.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult177848.owner)
    (rightOwner := SemanticResult175434.owner)
    (leftResult := 177848) (rightResult := 175434)
    (leftActual := SemanticResult177848.actual selector witness)
    (rightActual := SemanticResult175434.actual selector witness)
    (leftRaw := SemanticResult177848.rawTerms)
    (rightRaw := SemanticResult175434.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 3802007596962448506045899439491360353157172)
    (rightMaximum := 345660544987345366211554593406613108817920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 177849) (rightBinding := 177850)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨70484⟩) (rightExpression := ⟨31067⟩)
    (transferEvent := 177851) (summaryTransferEvent := 177852)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult177848.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult175434.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult177853

namespace SemanticResult177858
def owner : Owner := ⟨.program ⟨257⟩, ⟨70486⟩⟩
def rawTerms : List Term := Proof.Events694.exact177858RawTerms
def summary : Bound := (.finite 4493332905678336798016456807332854062121012)
def resultEvent : Nat := 177858
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult177858.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult177853.owner)
    (rightOwner := SemanticResult175222.owner)
    (leftResult := 177853) (rightResult := 175222)
    (leftActual := SemanticResult177853.actual selector witness)
    (rightActual := SemanticResult175222.actual selector witness)
    (leftRaw := SemanticResult177853.rawTerms)
    (rightRaw := SemanticResult175222.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 4147668141949793872257454032897973461975092)
    (rightMaximum := 345664763728542925759002774434880600145920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 177854) (rightBinding := 177855)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨70485⟩) (rightExpression := ⟨36727⟩)
    (transferEvent := 177856) (summaryTransferEvent := 177857)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult177853.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult175222.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult177858

namespace SemanticResult177863
def owner : Owner := ⟨.program ⟨257⟩, ⟨70487⟩⟩
def rawTerms : List Term := Proof.Events694.exact177863RawTerms
def summary : Bound := (.finite 4838999778777478503549183672281868407930932)
def resultEvent : Nat := 177863
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult177863.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult177858.owner)
    (rightOwner := SemanticResult175010.owner)
    (leftResult := 177858) (rightResult := 175010)
    (leftActual := SemanticResult177858.actual selector witness)
    (rightActual := SemanticResult175010.actual selector witness)
    (leftRaw := SemanticResult177858.rawTerms)
    (rightRaw := SemanticResult175010.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 4493332905678336798016456807332854062121012)
    (rightMaximum := 345666873099141705532726864949014345809920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 177859) (rightBinding := 177860)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨70486⟩) (rightExpression := ⟨39407⟩)
    (transferEvent := 177861) (summaryTransferEvent := 177862)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult177858.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult175010.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult177863

namespace SemanticResult177868
def owner : Owner := ⟨.program ⟨257⟩, ⟨70488⟩⟩
def rawTerms : List Term := Proof.Events694.exact177868RawTerms
def summary : Bound := (.finite 5184670870617817768629358718259150245068852)
def resultEvent : Nat := 177868
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult177868.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult177863.owner)
    (rightOwner := SemanticResult174798.owner)
    (leftResult := 177863) (rightResult := 174798)
    (leftActual := SemanticResult177863.actual selector witness)
    (rightActual := SemanticResult174798.actual selector witness)
    (leftRaw := SemanticResult177863.rawTerms)
    (rightRaw := SemanticResult174798.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 4838999778777478503549183672281868407930932)
    (rightMaximum := 345671091840339265080175045977281837137920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 177864) (rightBinding := 177865)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨70487⟩) (rightExpression := ⟨42087⟩)
    (transferEvent := 177866) (summaryTransferEvent := 177867)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult177863.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult174798.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult177868

namespace SemanticResult177873
def owner : Owner := ⟨.program ⟨257⟩, ⟨70489⟩⟩
def rawTerms : List Term := Proof.Events694.exact177873RawTerms
def summary : Bound := (.finite 5530348290569953373030706035778833319198772)
def resultEvent : Nat := 177873
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult177873.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult177868.owner)
    (rightOwner := SemanticResult174586.owner)
    (leftResult := 177868) (rightResult := 174586)
    (leftActual := SemanticResult177868.actual selector witness)
    (rightActual := SemanticResult174586.actual selector witness)
    (leftRaw := SemanticResult177868.rawTerms)
    (rightRaw := SemanticResult174586.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 5184670870617817768629358718259150245068852)
    (rightMaximum := 345677419952135604401347317519683074129920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 177869) (rightBinding := 177870)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨70488⟩) (rightExpression := ⟨44767⟩)
    (transferEvent := 177871) (summaryTransferEvent := 177872)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult177868.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult174586.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult177873

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
