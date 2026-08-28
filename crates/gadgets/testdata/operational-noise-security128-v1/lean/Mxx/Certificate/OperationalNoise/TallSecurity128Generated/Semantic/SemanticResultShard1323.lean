import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard1323
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1260
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1263
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1267
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1271
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1274
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1278
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1282
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1286
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1289
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1293
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1297
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1300
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1304
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1308
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1311
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1315
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1322

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult186955
def owner : Owner := ⟨.program ⟨257⟩, ⟨23969⟩⟩
def rawTerms : List Term := Proof.Events730.exact186955RawTerms
def summary : Bound := (.finite 96566716313119651734393211060224)
def resultEvent : Nat := 186955
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult186955.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult186950.owner)
    (rightOwner := SemanticResult185981.owner)
    (leftResult := 186950) (rightResult := 185981)
    (leftActual := SemanticResult186950.actual selector witness)
    (rightActual := SemanticResult185981.actual selector witness)
    (leftRaw := SemanticResult186950.rawTerms)
    (rightRaw := SemanticResult185981.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 64377712650190257467641695830016)
    (rightMaximum := 32189003662929394266751515230208) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 186951) (rightBinding := 186952)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨20749⟩) (rightExpression := ⟨23968⟩)
    (transferEvent := 186953) (summaryTransferEvent := 186954)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult186950.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult185981.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult186955

namespace SemanticResult186960
def owner : Owner := ⟨.program ⟨257⟩, ⟨33989⟩⟩
def rawTerms : List Term := Proof.Events730.exact186960RawTerms
def summary : Bound := (.finite 128755916426494733378385616044032)
def resultEvent : Nat := 186960
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult186960.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult186955.owner)
    (rightOwner := SemanticResult185499.owner)
    (leftResult := 186955) (rightResult := 185499)
    (leftActual := SemanticResult186955.actual selector witness)
    (rightActual := SemanticResult185499.actual selector witness)
    (leftRaw := SemanticResult186955.rawTerms)
    (rightRaw := SemanticResult185499.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 96566716313119651734393211060224)
    (rightMaximum := 32189200113375081643992404983808) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 186956) (rightBinding := 186957)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨23969⟩) (rightExpression := ⟨33988⟩)
    (transferEvent := 186958) (summaryTransferEvent := 186959)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult186955.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult185499.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult186960

namespace SemanticResult186965
def owner : Owner := ⟨.program ⟨257⟩, ⟨53049⟩⟩
def rawTerms : List Term := Proof.Events730.exact186965RawTerms
def summary : Bound := (.finite 160945509440761189776859800535040)
def resultEvent : Nat := 186965
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult186965.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult186960.owner)
    (rightOwner := SemanticResult185017.owner)
    (leftResult := 186960) (rightResult := 185017)
    (leftActual := SemanticResult186960.actual selector witness)
    (rightActual := SemanticResult185017.actual selector witness)
    (leftRaw := SemanticResult186960.rawTerms)
    (rightRaw := SemanticResult185017.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 128755916426494733378385616044032)
    (rightMaximum := 32189593014266456398474184491008) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 186961) (rightBinding := 186962)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨33989⟩) (rightExpression := ⟨53048⟩)
    (transferEvent := 186963) (summaryTransferEvent := 186964)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult186960.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult185017.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult186965

namespace SemanticResult186970
def owner : Owner := ⟨.program ⟨257⟩, ⟨56029⟩⟩
def rawTerms : List Term := Proof.Events730.exact186970RawTerms
def summary : Bound := (.finite 193135298905473333552574874779648)
def resultEvent : Nat := 186970
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult186970.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult186965.owner)
    (rightOwner := SemanticResult184535.owner)
    (leftResult := 186965) (rightResult := 184535)
    (leftActual := SemanticResult186965.actual selector witness)
    (rightActual := SemanticResult184535.actual selector witness)
    (leftRaw := SemanticResult186965.rawTerms)
    (rightRaw := SemanticResult184535.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 160945509440761189776859800535040)
    (rightMaximum := 32189789464712143775715074244608) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 186966) (rightBinding := 186967)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨53049⟩) (rightExpression := ⟨56028⟩)
    (transferEvent := 186968) (summaryTransferEvent := 186969)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult186965.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult184535.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult186970

namespace SemanticResult186975
def owner : Owner := ⟨.program ⟨257⟩, ⟨59009⟩⟩
def rawTerms : List Term := Proof.Events730.exact186975RawTerms
def summary : Bound := (.finite 225325481271076852082771728531456)
def resultEvent : Nat := 186975
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult186975.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult186970.owner)
    (rightOwner := SemanticResult184053.owner)
    (leftResult := 186970) (rightResult := 184053)
    (leftActual := SemanticResult186970.actual selector witness)
    (rightActual := SemanticResult184053.actual selector witness)
    (leftRaw := SemanticResult186970.rawTerms)
    (rightRaw := SemanticResult184053.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 193135298905473333552574874779648)
    (rightMaximum := 32190182365603518530196853751808) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 186971) (rightBinding := 186972)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨56029⟩) (rightExpression := ⟨59008⟩)
    (transferEvent := 186973) (summaryTransferEvent := 186974)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult186970.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult184053.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult186975

namespace SemanticResult186980
def owner : Owner := ⟨.program ⟨257⟩, ⟨61989⟩⟩
def rawTerms : List Term := Proof.Events730.exact186980RawTerms
def summary : Bound := (.finite 257515860087126057990209472036864)
def resultEvent : Nat := 186980
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult186980.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult186975.owner)
    (rightOwner := SemanticResult183571.owner)
    (leftResult := 186975) (rightResult := 183571)
    (leftActual := SemanticResult186975.actual selector witness)
    (rightActual := SemanticResult183571.actual selector witness)
    (leftRaw := SemanticResult186975.rawTerms)
    (rightRaw := SemanticResult183571.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 225325481271076852082771728531456)
    (rightMaximum := 32190378816049205907437743505408) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 186976) (rightBinding := 186977)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨59009⟩) (rightExpression := ⟨61988⟩)
    (transferEvent := 186978) (summaryTransferEvent := 186979)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult186975.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult183571.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult186980

namespace SemanticResult186985
def owner : Owner := ⟨.program ⟨257⟩, ⟨64969⟩⟩
def rawTerms : List Term := Proof.Events730.exact186985RawTerms
def summary : Bound := (.finite 289706631804066638652128995049472)
def resultEvent : Nat := 186985
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult186985.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult186980.owner)
    (rightOwner := SemanticResult183089.owner)
    (leftResult := 186980) (rightResult := 183089)
    (leftActual := SemanticResult186980.actual selector witness)
    (rightActual := SemanticResult183089.actual selector witness)
    (leftRaw := SemanticResult186980.rawTerms)
    (rightRaw := SemanticResult183089.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 257515860087126057990209472036864)
    (rightMaximum := 32190771716940580661919523012608) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 186981) (rightBinding := 186982)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨61989⟩) (rightExpression := ⟨64968⟩)
    (transferEvent := 186983) (summaryTransferEvent := 186984)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult186980.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult183089.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult186985

namespace SemanticResult186990
def owner : Owner := ⟨.program ⟨257⟩, ⟨70418⟩⟩
def rawTerms : List Term := Proof.Events730.exact186990RawTerms
def summary : Bound := (.finite 321897992872344281445771187322880)
def resultEvent : Nat := 186990
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult186990.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult186985.owner)
    (rightOwner := SemanticResult182607.owner)
    (leftResult := 186985) (rightResult := 182607)
    (leftActual := SemanticResult186985.actual selector witness)
    (rightActual := SemanticResult182607.actual selector witness)
    (leftRaw := SemanticResult186985.rawTerms)
    (rightRaw := SemanticResult182607.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 289706631804066638652128995049472)
    (rightMaximum := 32191361068277642793642192273408) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 186986) (rightBinding := 186987)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨64969⟩) (rightExpression := ⟨70417⟩)
    (transferEvent := 186988) (summaryTransferEvent := 186989)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult186985.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult182607.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult186990

namespace SemanticResult186995
def owner : Owner := ⟨.program ⟨257⟩, ⟨70419⟩⟩
def rawTerms : List Term := Proof.Events730.exact186995RawTerms
def summary : Bound := (.finite 354089550391067611616654269349888)
def resultEvent : Nat := 186995
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult186995.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult186990.owner)
    (rightOwner := SemanticResult182125.owner)
    (leftResult := 186990) (rightResult := 182125)
    (leftActual := SemanticResult186990.actual selector witness)
    (rightActual := SemanticResult182125.actual selector witness)
    (leftRaw := SemanticResult186990.rawTerms)
    (rightRaw := SemanticResult182125.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 321897992872344281445771187322880)
    (rightMaximum := 32191557518723330170883082027008) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 186991) (rightBinding := 186992)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨70418⟩) (rightExpression := ⟨28367⟩)
    (transferEvent := 186993) (summaryTransferEvent := 186994)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult186990.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult182125.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult186995

namespace SemanticResult187000
def owner : Owner := ⟨.program ⟨257⟩, ⟨70420⟩⟩
def rawTerms : List Term := Proof.Events730.exact187000RawTerms
def summary : Bound := (.finite 386281697261128003919260020637696)
def resultEvent : Nat := 187000
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult187000.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult186995.owner)
    (rightOwner := SemanticResult181643.owner)
    (leftResult := 186995) (rightResult := 181643)
    (leftActual := SemanticResult186995.actual selector witness)
    (rightActual := SemanticResult181643.actual selector witness)
    (leftRaw := SemanticResult186995.rawTerms)
    (rightRaw := SemanticResult181643.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 354089550391067611616654269349888)
    (rightMaximum := 32192146870060392302605751287808) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 186996) (rightBinding := 186997)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨70419⟩) (rightExpression := ⟨31047⟩)
    (transferEvent := 186998) (summaryTransferEvent := 186999)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult186995.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult181643.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult187000

namespace SemanticResult187005
def owner : Owner := ⟨.program ⟨257⟩, ⟨70421⟩⟩
def rawTerms : List Term := Proof.Events730.exact187005RawTerms
def summary : Bound := (.finite 418474237032079770976347551432704)
def resultEvent : Nat := 187005
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult187005.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult187000.owner)
    (rightOwner := SemanticResult181161.owner)
    (leftResult := 187000) (rightResult := 181161)
    (leftActual := SemanticResult187000.actual selector witness)
    (rightActual := SemanticResult181161.actual selector witness)
    (leftRaw := SemanticResult187000.rawTerms)
    (rightRaw := SemanticResult181161.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 386281697261128003919260020637696)
    (rightMaximum := 32192539770951767057087530795008) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 187001) (rightBinding := 187002)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨70420⟩) (rightExpression := ⟨36707⟩)
    (transferEvent := 187003) (summaryTransferEvent := 187004)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult187000.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult181161.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult187005

namespace SemanticResult187010
def owner : Owner := ⟨.program ⟨257⟩, ⟨70422⟩⟩
def rawTerms : List Term := Proof.Events730.exact187010RawTerms
def summary : Bound := (.finite 450666973253477225410675971981312)
def resultEvent : Nat := 187010
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult187010.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult187005.owner)
    (rightOwner := SemanticResult180679.owner)
    (leftResult := 187005) (rightResult := 180679)
    (leftActual := SemanticResult187005.actual selector witness)
    (rightActual := SemanticResult180679.actual selector witness)
    (leftRaw := SemanticResult187005.rawTerms)
    (rightRaw := SemanticResult180679.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 418474237032079770976347551432704)
    (rightMaximum := 32192736221397454434328420548608) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 187006) (rightBinding := 187007)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨70421⟩) (rightExpression := ⟨39387⟩)
    (transferEvent := 187008) (summaryTransferEvent := 187009)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult187005.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult180679.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult187010

namespace SemanticResult187015
def owner : Owner := ⟨.program ⟨257⟩, ⟨70423⟩⟩
def rawTerms : List Term := Proof.Events730.exact187015RawTerms
def summary : Bound := (.finite 482860102375766054599486172037120)
def resultEvent : Nat := 187015
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult187015.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult187010.owner)
    (rightOwner := SemanticResult180197.owner)
    (leftResult := 187010) (rightResult := 180197)
    (leftActual := SemanticResult187010.actual selector witness)
    (rightActual := SemanticResult180197.actual selector witness)
    (leftRaw := SemanticResult187010.rawTerms)
    (rightRaw := SemanticResult180197.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 450666973253477225410675971981312)
    (rightMaximum := 32193129122288829188810200055808) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 187011) (rightBinding := 187012)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨70422⟩) (rightExpression := ⟨42067⟩)
    (transferEvent := 187013) (summaryTransferEvent := 187014)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult187010.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult180197.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult187015

namespace SemanticResult187020
def owner : Owner := ⟨.program ⟨257⟩, ⟨70424⟩⟩
def rawTerms : List Term := Proof.Events730.exact187020RawTerms
def summary : Bound := (.finite 515053820849391945920019041353728)
def resultEvent : Nat := 187020
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult187020.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult187015.owner)
    (rightOwner := SemanticResult179715.owner)
    (leftResult := 187015) (rightResult := 179715)
    (leftActual := SemanticResult187015.actual selector witness)
    (rightActual := SemanticResult179715.actual selector witness)
    (leftRaw := SemanticResult187015.rawTerms)
    (rightRaw := SemanticResult179715.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 482860102375766054599486172037120)
    (rightMaximum := 32193718473625891320532869316608) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 187016) (rightBinding := 187017)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨70423⟩) (rightExpression := ⟨44747⟩)
    (transferEvent := 187018) (summaryTransferEvent := 187019)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult187015.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult179715.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult187020

namespace SemanticResult187025
def owner : Owner := ⟨.program ⟨257⟩, ⟨70425⟩⟩
def rawTerms : List Term := Proof.Events730.exact187025RawTerms
def summary : Bound := (.finite 547248128674354899372274579931136)
def resultEvent : Nat := 187025
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult187025.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult187020.owner)
    (rightOwner := SemanticResult179233.owner)
    (leftResult := 187020) (rightResult := 179233)
    (leftActual := SemanticResult187020.actual selector witness)
    (rightActual := SemanticResult179233.actual selector witness)
    (leftRaw := SemanticResult187020.rawTerms)
    (rightRaw := SemanticResult179233.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 515053820849391945920019041353728)
    (rightMaximum := 32194307824962953452255538577408) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 187021) (rightBinding := 187022)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨70424⟩) (rightExpression := ⟨47427⟩)
    (transferEvent := 187023) (summaryTransferEvent := 187024)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult187020.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult179233.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult187025

namespace SemanticResult187030
def owner : Owner := ⟨.program ⟨257⟩, ⟨70426⟩⟩
def rawTerms : List Term := Proof.Events730.exact187030RawTerms
def summary : Bound := (.finite 579442632949763540201771008262144)
def resultEvent : Nat := 187030
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult187030.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult187025.owner)
    (rightOwner := SemanticResult178751.owner)
    (leftResult := 187025) (rightResult := 178751)
    (leftActual := SemanticResult187025.actual selector witness)
    (rightActual := SemanticResult178751.actual selector witness)
    (leftRaw := SemanticResult187025.rawTerms)
    (rightRaw := SemanticResult178751.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 547248128674354899372274579931136)
    (rightMaximum := 32194504275408640829496428331008) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 187026) (rightBinding := 187027)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨70425⟩) (rightExpression := ⟨50107⟩)
    (transferEvent := 187028) (summaryTransferEvent := 187029)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult187025.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult178751.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult187030

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
