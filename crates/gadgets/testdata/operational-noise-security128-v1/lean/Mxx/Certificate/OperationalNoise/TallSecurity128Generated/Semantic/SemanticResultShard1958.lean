import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard1958
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1935
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1936
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1938
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1939
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1941
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1942
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1943
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1945
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1946
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1947
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1949
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1950
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1952
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1953
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1954
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1956
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1957

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult280178
def owner : Owner := ⟨.program ⟨257⟩, ⟨20393⟩⟩
def rawTerms : List Term := Proof.Events1094.exact280178RawTerms
def summary : Bound := (.finite 691250426059631610003352154589745737891892)
def resultEvent : Nat := 280178
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult280178.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult280173.owner)
    (rightOwner := SemanticResult279929.owner)
    (leftResult := 280173) (rightResult := 279929)
    (leftActual := SemanticResult280173.actual selector witness)
    (rightActual := SemanticResult279929.actual selector witness)
    (leftRaw := SemanticResult280173.rawTerms)
    (rightRaw := SemanticResult279929.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 345624685687166110058245054666339432529972)
    (rightMaximum := 345625740372465499945107099923406305361920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 280174) (rightBinding := 280175)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨17527⟩) (rightExpression := ⟨20392⟩)
    (transferEvent := 280176) (summaryTransferEvent := 280177)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult280173.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult279929.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult280178

namespace SemanticResult280183
def owner : Owner := ⟨.program ⟨257⟩, ⟨23613⟩⟩
def rawTerms : List Term := Proof.Events1094.exact280183RawTerms
def summary : Bound := (.finite 1036877221117396499835321299770218916085812)
def resultEvent : Nat := 280183
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult280183.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult280178.owner)
    (rightOwner := SemanticResult279717.owner)
    (leftResult := 280178) (rightResult := 279717)
    (leftActual := SemanticResult280178.actual selector witness)
    (rightActual := SemanticResult279717.actual selector witness)
    (leftRaw := SemanticResult280178.rawTerms)
    (rightRaw := SemanticResult279717.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 691250426059631610003352154589745737891892)
    (rightMaximum := 345626795057764889831969145180473178193920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 280179) (rightBinding := 280180)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨20393⟩) (rightExpression := ⟨23612⟩)
    (transferEvent := 280181) (summaryTransferEvent := 280182)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult280178.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult279717.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult280183

namespace SemanticResult280188
def owner : Owner := ⟨.program ⟨257⟩, ⟨33633⟩⟩
def rawTerms : List Term := Proof.Events1094.exact280188RawTerms
def summary : Bound := (.finite 1382506125545760169441014535464825839943732)
def resultEvent : Nat := 280188
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult280188.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult280183.owner)
    (rightOwner := SemanticResult279505.owner)
    (leftResult := 280183) (rightResult := 279505)
    (leftActual := SemanticResult280183.actual selector witness)
    (rightActual := SemanticResult279505.actual selector witness)
    (leftRaw := SemanticResult280183.rawTerms)
    (rightRaw := SemanticResult279505.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 1036877221117396499835321299770218916085812)
    (rightMaximum := 345628904428363669605693235694606923857920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 280184) (rightBinding := 280185)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨23613⟩) (rightExpression := ⟨33632⟩)
    (transferEvent := 280186) (summaryTransferEvent := 280187)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult280183.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult279505.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult280188

namespace SemanticResult280193
def owner : Owner := ⟨.program ⟨257⟩, ⟨52693⟩⟩
def rawTerms : List Term := Proof.Events1094.exact280193RawTerms
def summary : Bound := (.finite 1728139248715321398594155952187700255129652)
def resultEvent : Nat := 280193
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult280193.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult280188.owner)
    (rightOwner := SemanticResult279293.owner)
    (leftResult := 280188) (rightResult := 279293)
    (leftActual := SemanticResult280188.actual selector witness)
    (rightActual := SemanticResult279293.actual selector witness)
    (leftRaw := SemanticResult280188.rawTerms)
    (rightRaw := SemanticResult279293.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 1382506125545760169441014535464825839943732)
    (rightMaximum := 345633123169561229153141416722874415185920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 280189) (rightBinding := 280190)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨33633⟩) (rightExpression := ⟨52692⟩)
    (transferEvent := 280191) (summaryTransferEvent := 280192)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult280188.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult279293.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult280193

namespace SemanticResult280198
def owner : Owner := ⟨.program ⟨257⟩, ⟨55673⟩⟩
def rawTerms : List Term := Proof.Events1094.exact280198RawTerms
def summary : Bound := (.finite 2073774481255481407521021459424708415979572)
def resultEvent : Nat := 280198
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult280198.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult280193.owner)
    (rightOwner := SemanticResult279081.owner)
    (leftResult := 280193) (rightResult := 279081)
    (leftActual := SemanticResult280193.actual selector witness)
    (rightActual := SemanticResult279081.actual selector witness)
    (leftRaw := SemanticResult280193.rawTerms)
    (rightRaw := SemanticResult279081.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 1728139248715321398594155952187700255129652)
    (rightMaximum := 345635232540160008926865507237008160849920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 280194) (rightBinding := 280195)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨52693⟩) (rightExpression := ⟨55672⟩)
    (transferEvent := 280196) (summaryTransferEvent := 280197)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult280193.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult279081.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult280198

namespace SemanticResult280203
def owner : Owner := ⟨.program ⟨257⟩, ⟨58653⟩⟩
def rawTerms : List Term := Proof.Events1094.exact280203RawTerms
def summary : Bound := (.finite 2419413932536838975995335147689984068157492)
def resultEvent : Nat := 280203
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult280203.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult280198.owner)
    (rightOwner := SemanticResult278869.owner)
    (leftResult := 280198) (rightResult := 278869)
    (leftActual := SemanticResult280198.actual selector witness)
    (rightActual := SemanticResult278869.actual selector witness)
    (leftRaw := SemanticResult280198.rawTerms)
    (rightRaw := SemanticResult278869.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 2073774481255481407521021459424708415979572)
    (rightMaximum := 345639451281357568474313688265275652177920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 280199) (rightBinding := 280200)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨55673⟩) (rightExpression := ⟨58652⟩)
    (transferEvent := 280201) (summaryTransferEvent := 280202)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult280198.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult278869.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult280203

namespace SemanticResult280208
def owner : Owner := ⟨.program ⟨257⟩, ⟨61633⟩⟩
def rawTerms : List Term := Proof.Events1094.exact280208RawTerms
def summary : Bound := (.finite 2765055493188795324243372926469393465999412)
def resultEvent : Nat := 280208
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult280208.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult280203.owner)
    (rightOwner := SemanticResult278657.owner)
    (leftResult := 280203) (rightResult := 278657)
    (leftActual := SemanticResult280203.actual selector witness)
    (rightActual := SemanticResult278657.actual selector witness)
    (leftRaw := SemanticResult280203.rawTerms)
    (rightRaw := SemanticResult278657.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 2419413932536838975995335147689984068157492)
    (rightMaximum := 345641560651956348248037778779409397841920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 280204) (rightBinding := 280205)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨58653⟩) (rightExpression := ⟨61632⟩)
    (transferEvent := 280206) (summaryTransferEvent := 280207)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult280203.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult278657.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult280208

namespace SemanticResult280213
def owner : Owner := ⟨.program ⟨257⟩, ⟨64613⟩⟩
def rawTerms : List Term := Proof.Events1094.exact280213RawTerms
def summary : Bound := (.finite 3110701272581949232038858886277070355169332)
def resultEvent : Nat := 280213
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult280213.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult280208.owner)
    (rightOwner := SemanticResult278445.owner)
    (leftResult := 280208) (rightResult := 278445)
    (leftActual := SemanticResult280208.actual selector witness)
    (rightActual := SemanticResult278445.actual selector witness)
    (leftRaw := SemanticResult280208.rawTerms)
    (rightRaw := SemanticResult278445.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 2765055493188795324243372926469393465999412)
    (rightMaximum := 345645779393153907795485959807676889169920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 280209) (rightBinding := 280210)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨61633⟩) (rightExpression := ⟨64612⟩)
    (transferEvent := 280211) (summaryTransferEvent := 280212)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult280208.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult278445.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult280213

namespace SemanticResult280218
def owner : Owner := ⟨.program ⟨257⟩, ⟨69510⟩⟩
def rawTerms : List Term := Proof.Events1094.exact280218RawTerms
def summary : Bound := (.finite 3456353380086899479155517117627148481331252)
def resultEvent : Nat := 280218
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult280218.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult280213.owner)
    (rightOwner := SemanticResult278233.owner)
    (leftResult := 280213) (rightResult := 278233)
    (leftActual := SemanticResult280213.actual selector witness)
    (rightActual := SemanticResult278233.actual selector witness)
    (leftRaw := SemanticResult280213.rawTerms)
    (rightRaw := SemanticResult278233.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 3110701272581949232038858886277070355169332)
    (rightMaximum := 345652107504950247116658231350078126161920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 280214) (rightBinding := 280215)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨64613⟩) (rightExpression := ⟨69509⟩)
    (transferEvent := 280216) (summaryTransferEvent := 280217)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult280213.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult278233.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult280218

namespace SemanticResult280223
def owner : Owner := ⟨.program ⟨257⟩, ⟨69511⟩⟩
def rawTerms : List Term := Proof.Events1094.exact280223RawTerms
def summary : Bound := (.finite 3802007596962448506045899439491360353157172)
def resultEvent : Nat := 280223
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult280223.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult280218.owner)
    (rightOwner := SemanticResult278021.owner)
    (leftResult := 280218) (rightResult := 278021)
    (leftActual := SemanticResult280218.actual selector witness)
    (rightActual := SemanticResult278021.actual selector witness)
    (leftRaw := SemanticResult280218.rawTerms)
    (rightRaw := SemanticResult278021.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 3456353380086899479155517117627148481331252)
    (rightMaximum := 345654216875549026890382321864211871825920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 280219) (rightBinding := 280220)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨69510⟩) (rightExpression := ⟨28080⟩)
    (transferEvent := 280221) (summaryTransferEvent := 280222)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult280218.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult278021.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult280223

namespace SemanticResult280228
def owner : Owner := ⟨.program ⟨257⟩, ⟨69512⟩⟩
def rawTerms : List Term := Proof.Events1094.exact280228RawTerms
def summary : Bound := (.finite 4147668141949793872257454032897973461975092)
def resultEvent : Nat := 280228
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult280228.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult280223.owner)
    (rightOwner := SemanticResult277809.owner)
    (leftResult := 280223) (rightResult := 277809)
    (leftActual := SemanticResult280223.actual selector witness)
    (rightActual := SemanticResult277809.actual selector witness)
    (leftRaw := SemanticResult280223.rawTerms)
    (rightRaw := SemanticResult277809.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 3802007596962448506045899439491360353157172)
    (rightMaximum := 345660544987345366211554593406613108817920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 280224) (rightBinding := 280225)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨69511⟩) (rightExpression := ⟨30760⟩)
    (transferEvent := 280226) (summaryTransferEvent := 280227)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult280223.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult277809.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult280228

namespace SemanticResult280233
def owner : Owner := ⟨.program ⟨257⟩, ⟨69513⟩⟩
def rawTerms : List Term := Proof.Events1094.exact280233RawTerms
def summary : Bound := (.finite 4493332905678336798016456807332854062121012)
def resultEvent : Nat := 280233
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult280233.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult280228.owner)
    (rightOwner := SemanticResult277597.owner)
    (leftResult := 280228) (rightResult := 277597)
    (leftActual := SemanticResult280228.actual selector witness)
    (rightActual := SemanticResult277597.actual selector witness)
    (leftRaw := SemanticResult280228.rawTerms)
    (rightRaw := SemanticResult277597.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 4147668141949793872257454032897973461975092)
    (rightMaximum := 345664763728542925759002774434880600145920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 280229) (rightBinding := 280230)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨69512⟩) (rightExpression := ⟨36420⟩)
    (transferEvent := 280231) (summaryTransferEvent := 280232)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult280228.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult277597.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult280233

namespace SemanticResult280238
def owner : Owner := ⟨.program ⟨257⟩, ⟨69514⟩⟩
def rawTerms : List Term := Proof.Events1094.exact280238RawTerms
def summary : Bound := (.finite 4838999778777478503549183672281868407930932)
def resultEvent : Nat := 280238
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult280238.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult280233.owner)
    (rightOwner := SemanticResult277385.owner)
    (leftResult := 280233) (rightResult := 277385)
    (leftActual := SemanticResult280233.actual selector witness)
    (rightActual := SemanticResult277385.actual selector witness)
    (leftRaw := SemanticResult280233.rawTerms)
    (rightRaw := SemanticResult277385.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 4493332905678336798016456807332854062121012)
    (rightMaximum := 345666873099141705532726864949014345809920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 280234) (rightBinding := 280235)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨69513⟩) (rightExpression := ⟨39100⟩)
    (transferEvent := 280236) (summaryTransferEvent := 280237)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult280233.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult277385.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult280238

namespace SemanticResult280243
def owner : Owner := ⟨.program ⟨257⟩, ⟨69515⟩⟩
def rawTerms : List Term := Proof.Events1094.exact280243RawTerms
def summary : Bound := (.finite 5184670870617817768629358718259150245068852)
def resultEvent : Nat := 280243
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult280243.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult280238.owner)
    (rightOwner := SemanticResult277173.owner)
    (leftResult := 280238) (rightResult := 277173)
    (leftActual := SemanticResult280238.actual selector witness)
    (rightActual := SemanticResult277173.actual selector witness)
    (leftRaw := SemanticResult280238.rawTerms)
    (rightRaw := SemanticResult277173.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 4838999778777478503549183672281868407930932)
    (rightMaximum := 345671091840339265080175045977281837137920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 280239) (rightBinding := 280240)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨69514⟩) (rightExpression := ⟨41780⟩)
    (transferEvent := 280241) (summaryTransferEvent := 280242)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult280238.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult277173.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult280243

namespace SemanticResult280248
def owner : Owner := ⟨.program ⟨257⟩, ⟨69516⟩⟩
def rawTerms : List Term := Proof.Events1094.exact280248RawTerms
def summary : Bound := (.finite 5530348290569953373030706035778833319198772)
def resultEvent : Nat := 280248
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult280248.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult280243.owner)
    (rightOwner := SemanticResult276961.owner)
    (leftResult := 280243) (rightResult := 276961)
    (leftActual := SemanticResult280243.actual selector witness)
    (rightActual := SemanticResult276961.actual selector witness)
    (leftRaw := SemanticResult280243.rawTerms)
    (rightRaw := SemanticResult276961.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 5184670870617817768629358718259150245068852)
    (rightMaximum := 345677419952135604401347317519683074129920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 280244) (rightBinding := 280245)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨69515⟩) (rightExpression := ⟨44460⟩)
    (transferEvent := 280246) (summaryTransferEvent := 280247)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult280243.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult276961.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult280248

namespace SemanticResult280253
def owner : Owner := ⟨.program ⟨257⟩, ⟨69517⟩⟩
def rawTerms : List Term := Proof.Events1094.exact280253RawTerms
def summary : Bound := (.finite 5876032038633885316753225624840917630320692)
def resultEvent : Nat := 280253
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult280253.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult280248.owner)
    (rightOwner := SemanticResult276749.owner)
    (leftResult := 280248) (rightResult := 276749)
    (leftActual := SemanticResult280248.actual selector witness)
    (rightActual := SemanticResult276749.actual selector witness)
    (leftRaw := SemanticResult280248.rawTerms)
    (rightRaw := SemanticResult276749.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 5530348290569953373030706035778833319198772)
    (rightMaximum := 345683748063931943722519589062084311121920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 280249) (rightBinding := 280250)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨69516⟩) (rightExpression := ⟨47140⟩)
    (transferEvent := 280251) (summaryTransferEvent := 280252)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult280248.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult276749.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult280253

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
