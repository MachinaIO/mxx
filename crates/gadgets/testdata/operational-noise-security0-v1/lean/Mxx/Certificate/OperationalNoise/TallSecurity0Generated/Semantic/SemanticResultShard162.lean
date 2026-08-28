import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard162
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard000
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard051
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard056
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard136
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard137
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard139
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard140
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard141
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard143
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard144
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard145
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard147
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard148
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard161

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult20985
def owner : Owner := ⟨.program ⟨214⟩, ⟨28350⟩⟩
def rawTerms : List Term := Proof.Events081.exact20985RawTerms
def summary : Bound := (.finite 47416693230599820876439355444)
def resultEvent : Nat := 20985
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult20985.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult20980.owner)
    (rightOwner := SemanticResult18997.owner)
    (leftResult := 20980) (rightResult := 18997)
    (leftActual := SemanticResult20980.actual selector witness)
    (rightActual := SemanticResult18997.actual selector witness)
    (leftRaw := SemanticResult20980.rawTerms)
    (rightRaw := SemanticResult18997.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 42674369987986832655214706740)
    (rightMaximum := 4742323242612988221224648704) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 20981) (rightBinding := 20982)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨28133⟩) (rightExpression := ⟨28349⟩)
    (transferEvent := 20983) (summaryTransferEvent := 20984)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult20980.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult18997.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult20985

namespace SemanticResult20990
def owner : Owner := ⟨.program ⟨214⟩, ⟨28567⟩⟩
def rawTerms : List Term := Proof.Events081.exact20990RawTerms
def summary : Bound := (.finite 52159098727244633768554659892)
def resultEvent : Nat := 20990
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult20990.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult20985.owner)
    (rightOwner := SemanticResult18785.owner)
    (leftResult := 20985) (rightResult := 18785)
    (leftActual := SemanticResult20985.actual selector witness)
    (rightActual := SemanticResult18785.actual selector witness)
    (leftRaw := SemanticResult20985.rawTerms)
    (rightRaw := SemanticResult18785.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 47416693230599820876439355444)
    (rightMaximum := 4742405496644812892115304448) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 20986) (rightBinding := 20987)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨28350⟩) (rightExpression := ⟨28566⟩)
    (transferEvent := 20988) (summaryTransferEvent := 20989)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult20985.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult18785.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult20990

namespace SemanticResult20995
def owner : Owner := ⟨.program ⟨214⟩, ⟨28784⟩⟩
def rawTerms : List Term := Proof.Events082.exact20995RawTerms
def summary : Bound := (.finite 56901750985984920673341931572)
def resultEvent : Nat := 20995
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult20995.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult20990.owner)
    (rightOwner := SemanticResult18573.owner)
    (leftResult := 20990) (rightResult := 18573)
    (leftActual := SemanticResult20990.actual selector witness)
    (rightActual := SemanticResult18573.actual selector witness)
    (leftRaw := SemanticResult20990.rawTerms)
    (rightRaw := SemanticResult18573.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 52159098727244633768554659892)
    (rightMaximum := 4742652258740286904787271680) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 20991) (rightBinding := 20992)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨28567⟩) (rightExpression := ⟨28783⟩)
    (transferEvent := 20993) (summaryTransferEvent := 20994)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult20990.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult18573.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult20995

namespace SemanticResult21000
def owner : Owner := ⟨.program ⟨214⟩, ⟨29001⟩⟩
def rawTerms : List Term := Proof.Events082.exact21000RawTerms
def summary : Bound := (.finite 61644567752788856919910514740)
def resultEvent : Nat := 21000
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult21000.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult20995.owner)
    (rightOwner := SemanticResult18361.owner)
    (leftResult := 20995) (rightResult := 18361)
    (leftActual := SemanticResult20995.actual selector witness)
    (rightActual := SemanticResult18361.actual selector witness)
    (leftRaw := SemanticResult20995.rawTerms)
    (rightRaw := SemanticResult18361.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 56901750985984920673341931572)
    (rightMaximum := 4742816766803936246568583168) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 20996) (rightBinding := 20997)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨28784⟩) (rightExpression := ⟨29000⟩)
    (transferEvent := 20998) (summaryTransferEvent := 20999)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult20995.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult18361.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult21000

namespace SemanticResult21005
def owner : Owner := ⟨.program ⟨214⟩, ⟨29218⟩⟩
def rawTerms : List Term := Proof.Events082.exact21005RawTerms
def summary : Bound := (.finite 66387466773624617837369753652)
def resultEvent : Nat := 21005
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult21005.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult21000.owner)
    (rightOwner := SemanticResult18149.owner)
    (leftResult := 21000) (rightResult := 18149)
    (leftActual := SemanticResult21000.actual selector witness)
    (rightActual := SemanticResult18149.actual selector witness)
    (leftRaw := SemanticResult21000.rawTerms)
    (rightRaw := SemanticResult18149.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 61644567752788856919910514740)
    (rightMaximum := 4742899020835760917459238912) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 21001) (rightBinding := 21002)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨29001⟩) (rightExpression := ⟨29217⟩)
    (transferEvent := 21003) (summaryTransferEvent := 21004)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult21000.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult18149.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult21005

namespace SemanticResult21010
def owner : Owner := ⟨.program ⟨214⟩, ⟨29435⟩⟩
def rawTerms : List Term := Proof.Events082.exact21010RawTerms
def summary : Bound := (.finite 71130530302524028096610304052)
def resultEvent : Nat := 21010
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult21010.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult21005.owner)
    (rightOwner := SemanticResult17937.owner)
    (leftResult := 21005) (rightResult := 17937)
    (leftActual := SemanticResult21005.actual selector witness)
    (rightActual := SemanticResult17937.actual selector witness)
    (leftRaw := SemanticResult21005.rawTerms)
    (rightRaw := SemanticResult17937.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 66387466773624617837369753652)
    (rightMaximum := 4743063528899410259240550400) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 21006) (rightBinding := 21007)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨29218⟩) (rightExpression := ⟨29434⟩)
    (transferEvent := 21008) (summaryTransferEvent := 21009)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult21005.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult17937.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult21010

namespace SemanticResult21015
def owner : Owner := ⟨.program ⟨214⟩, ⟨29652⟩⟩
def rawTerms : List Term := Proof.Events082.exact21015RawTerms
def summary : Bound := (.finite 75873840593518912368522821684)
def resultEvent : Nat := 21015
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult21015.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult21010.owner)
    (rightOwner := SemanticResult17725.owner)
    (leftResult := 21010) (rightResult := 17725)
    (leftActual := SemanticResult21010.actual selector witness)
    (rightActual := SemanticResult17725.actual selector witness)
    (leftRaw := SemanticResult21010.rawTerms)
    (rightRaw := SemanticResult17725.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 71130530302524028096610304052)
    (rightMaximum := 4743310290994884271912517632) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 21011) (rightBinding := 21012)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨29435⟩) (rightExpression := ⟨29651⟩)
    (transferEvent := 21013) (summaryTransferEvent := 21014)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult21010.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult17725.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult21015

namespace SemanticResult21020
def owner : Owner := ⟨.program ⟨214⟩, ⟨29869⟩⟩
def rawTerms : List Term := Proof.Events082.exact21020RawTerms
def summary : Bound := (.finite 80617397646609270653107306548)
def resultEvent : Nat := 21020
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult21020.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult21015.owner)
    (rightOwner := SemanticResult17513.owner)
    (leftResult := 21015) (rightResult := 17513)
    (leftActual := SemanticResult21015.actual selector witness)
    (rightActual := SemanticResult17513.actual selector witness)
    (leftRaw := SemanticResult21015.rawTerms)
    (rightRaw := SemanticResult17513.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 75873840593518912368522821684)
    (rightMaximum := 4743557053090358284584484864) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 21016) (rightBinding := 21017)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨29652⟩) (rightExpression := ⟨29868⟩)
    (transferEvent := 21018) (summaryTransferEvent := 21019)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult21015.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult17513.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult21020

namespace SemanticResult21025
def owner : Owner := ⟨.program ⟨214⟩, ⟨30203⟩⟩
def rawTerms : List Term := Proof.Events082.exact21025RawTerms
def summary : Bound := (.finite 85361036953731453608582447156)
def resultEvent : Nat := 21025
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult21025.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult21020.owner)
    (rightOwner := SemanticResult17301.owner)
    (leftResult := 21020) (rightResult := 17301)
    (leftActual := SemanticResult21020.actual selector witness)
    (rightActual := SemanticResult17301.actual selector witness)
    (leftRaw := SemanticResult21020.rawTerms)
    (rightRaw := SemanticResult17301.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 80617397646609270653107306548)
    (rightMaximum := 4743639307122182955475140608) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 21021) (rightBinding := 21022)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨29869⟩) (rightExpression := ⟨30202⟩)
    (transferEvent := 21023) (summaryTransferEvent := 21024)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult21020.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult17301.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult21025

namespace SemanticResult21030
def owner : Owner := ⟨.program ⟨214⟩, ⟨30214⟩⟩
def rawTerms : List Term := Proof.Events082.exact21030RawTerms
def summary : Bound := (.finite 313276456757822654825721789388161076)
def resultEvent : Nat := 21030
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult21030.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult21025.owner)
    (rightOwner := SemanticResult17089.owner)
    (leftResult := 21025) (rightResult := 17089)
    (leftActual := SemanticResult21025.actual selector witness)
    (rightActual := SemanticResult17089.actual selector witness)
    (leftRaw := SemanticResult21025.rawTerms)
    (rightRaw := SemanticResult17089.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 85361036953731453608582447156)
    (rightMaximum := 313276371396785701094268180805713920) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 21026) (rightBinding := 21027)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨30203⟩) (rightExpression := ⟨30212⟩)
    (transferEvent := 21028) (summaryTransferEvent := 21029)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult21025.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult17089.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult21030

namespace SemanticResult21032
def owner : Owner := ⟨.program ⟨214⟩, ⟨4⟩⟩
def rawTerms : List Term := Proof.Events082.exact21032RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 21032
def producerEvent : Nat := 21031
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult21032.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.operator), 0, .finite 26, .authorityOperator, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult21032

namespace SemanticResult21037
def owner : Owner := ⟨.program ⟨214⟩, ⟨7089⟩⟩
def rawTerms : List Term := Proof.Events082.exact21037RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 21037
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult21037.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge21036.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge21036.frameStart)
    (transferEvent := 21035) (owner := owner)
    (leftResult := 27) (rightResult := 5964)
    (working := LeftOperatorMerge21036.working)
    (reconstruction := LeftOperatorMerge21036.reconstruction)
    (leftReference := .predecessor 0 21033 .coefficient) (rightReference := .predecessor 1 21034 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := true) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult27.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult5964.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge21036.operationAgreement
  · decide

theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply exactClaimAt_of_mergeClaim
    (mergeClaim selector selectorLower selectorUpper witness)
  · decide +kernel
  · rfl
end SemanticResult21037

namespace SemanticResult21041
def owner : Owner := ⟨.program ⟨214⟩, ⟨7719⟩⟩
def rawTerms : List Term := Proof.Events082.exact21041RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 21041
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult21041.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 21038) (rightBinding := 21039)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7089⟩) (rightExpression := ⟨6571⟩)
    (transferEvent := 21040)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult21037.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult6449.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult21041

namespace SemanticResult21047
def owner : Owner := ⟨.program ⟨214⟩, ⟨7720⟩⟩
def rawTerms : List Term := Proof.Events082.exact21047RawTerms
def summary : Bound := (.finite 26)
def resultEvent : Nat := 21047
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult21047.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddSingletonSurvivorFoldClaimAt
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (witness := witness) (frameStart := 0)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (coefficientTransfer := 21044) (survivorTransfer := 21045)
    (survivorEvent := 21046) (resultEvent := resultEvent)
    (rightCoefficientProducer := 21031)
    (owner := owner) (leftOwner := SemanticResult21041.owner)
    (rightOwner := SemanticResult21032.owner)
    (leftResult := 21041) (rightResult := 21032)
    (leftBinding := 21042) (rightBinding := 21043)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7719⟩) (rightExpression := ⟨4⟩)
    (leftActual := SemanticResult21041.actual selector witness)
    (rightActual := SemanticResult21032.actual selector witness)
    (leftRaw := SemanticResult21041.rawTerms)
    (survivorMonomial := ⟨[], [⟨.program ⟨214⟩, ⟨4⟩⟩]⟩) (maximum := ⟨26, by decide⟩)
    (rightMagnitude := LeftAuthority21031.actual selector witness)
    (survivorMagnitude := LeftBound21045.actual selector witness)
  · decide +kernel
  · rfl
  · rfl
  · rfl
  · exact SemanticResult21041.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult21032.claimSound selector selectorLower selectorUpper witness
  · rfl
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority21031.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority21031.derived selector witness)
  · exact LeftBound21045.derived selector witness
  · rfl
  · rfl
  · decide
end SemanticResult21047

namespace SemanticResult21075
def owner : Owner := ⟨.program ⟨214⟩, ⟨7899⟩⟩
def rawTerms : List Term := Proof.Events082.exact21075RawTerms
def summary : Bound := (.finite 95420416)
def resultEvent : Nat := 21075
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult21075.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 448
      (.finite ⟨26, by decide⟩)
      (.finite ⟨8192, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge21053.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge21053.frameStart)
    (owner := owner) (leftOwner := SemanticResult21047.owner)
    (rightOwner := SemanticResult5961.owner)
    (leftResult := 21047) (rightResult := 5961)
    (leftActual := SemanticResult21047.actual selector witness)
    (rightActual := SemanticResult5961.actual selector witness)
    (leftRaw := SemanticResult21047.rawTerms)
    (rightRaw := SemanticResult5961.rawTerms)
    (working := LeftOperatorMerge21053.working)
    (leftBinding := 21048) (rightBinding := 21049)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7720⟩) (rightExpression := ⟨7886⟩)
    (coefficientTransfer := 21050) (summaryTransfer := 21052)
    (rightCoefficientProducer := 5960)
    (rightSummaryTransfer := 21051)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨8192, by decide⟩)
    (rightRecordedMaximum := 8192)
    (rightSummaryMaximum := ⟨8192, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 448)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge21053.base)
    (coefficientFacts := ⟨false, false, none, none, none⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftBound5960.actual selector witness)
    (summaryMagnitude := LeftBound21052.actual selector witness)
    (reconstruction := LeftOperatorMerge21053.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult21047.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult5961.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5960.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftBound5960.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge21053.operationAgreement
  · exact LeftBound21052.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge21053.working summary) := by
  have claim := computedMergeClaim selector selectorLower selectorUpper witness
  have summaryEq : computedSummary = summary := by
    dsimp [computedSummary, summary, boundOfCoeffClass, EventReplay.productWithFactor]
  simpa only [summaryEq] using claim
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 21054 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6745⟩⟩, ⟨.program ⟨214⟩, ⟨7885⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6708⟩⟩, ⟨.program ⟨214⟩, ⟨6679⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6710⟩⟩, ⟨.program ⟨214⟩, ⟨6671⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6712⟩⟩, ⟨.program ⟨214⟩, ⟨6663⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6714⟩⟩, ⟨.program ⟨214⟩, ⟨6655⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6716⟩⟩, ⟨.program ⟨214⟩, ⟨6649⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6718⟩⟩, ⟨.program ⟨214⟩, ⟨6647⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6720⟩⟩, ⟨.program ⟨214⟩, ⟨6643⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6722⟩⟩, ⟨.program ⟨214⟩, ⟨6641⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6724⟩⟩, ⟨.program ⟨214⟩, ⟨6637⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6726⟩⟩, ⟨.program ⟨214⟩, ⟨6681⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6728⟩⟩, ⟨.program ⟨214⟩, ⟨6677⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6730⟩⟩, ⟨.program ⟨214⟩, ⟨6673⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6732⟩⟩, ⟨.program ⟨214⟩, ⟨6669⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6734⟩⟩, ⟨.program ⟨214⟩, ⟨6667⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6736⟩⟩, ⟨.program ⟨214⟩, ⟨6665⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6738⟩⟩, ⟨.program ⟨214⟩, ⟨6661⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6740⟩⟩, ⟨.program ⟨214⟩, ⟨6659⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6742⟩⟩, ⟨.program ⟨214⟩, ⟨6657⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6744⟩⟩, ⟨.program ⟨214⟩, ⟨6651⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6708⟩⟩, ⟨.program ⟨214⟩, ⟨6679⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6710⟩⟩, ⟨.program ⟨214⟩, ⟨6671⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6712⟩⟩, ⟨.program ⟨214⟩, ⟨6663⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6714⟩⟩, ⟨.program ⟨214⟩, ⟨6655⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6716⟩⟩, ⟨.program ⟨214⟩, ⟨6649⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6718⟩⟩, ⟨.program ⟨214⟩, ⟨6647⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6720⟩⟩, ⟨.program ⟨214⟩, ⟨6643⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6722⟩⟩, ⟨.program ⟨214⟩, ⟨6641⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6724⟩⟩, ⟨.program ⟨214⟩, ⟨6637⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6726⟩⟩, ⟨.program ⟨214⟩, ⟨6681⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6728⟩⟩, ⟨.program ⟨214⟩, ⟨6677⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6730⟩⟩, ⟨.program ⟨214⟩, ⟨6673⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6732⟩⟩, ⟨.program ⟨214⟩, ⟨6669⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6734⟩⟩, ⟨.program ⟨214⟩, ⟨6667⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6736⟩⟩, ⟨.program ⟨214⟩, ⟨6665⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6738⟩⟩, ⟨.program ⟨214⟩, ⟨6661⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6740⟩⟩, ⟨.program ⟨214⟩, ⟨6659⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6742⟩⟩, ⟨.program ⟨214⟩, ⟨6657⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6744⟩⟩, ⟨.program ⟨214⟩, ⟨6651⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge21053.working
    [{ coefficient := (-1), key := LeftRelationMerge21054.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge21054.frameStart
      LeftRelationMerge21054.owner (.relation 21054) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge21054.deltas
    rows := LeftRelationMerge21054.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge21053.working LeftRelationMerge21054.source
        (relationContext LeftRelationMerge21054.source
          LeftRelationMerge21054.source.centralFactors 0 2) (-1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge21053.working, LeftRelationMerge21054.deltas,
    LeftRelationMerge21054.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply gadgetRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (witness := witness) (application := 21054)
    (frameStart := 0) (owner := ⟨.program ⟨214⟩, ⟨7899⟩⟩)
    (source := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7885⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7885⟩⟩]⟩)
    (outerCoefficient := -1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge21053.working) (working := relationWorking0)
    (reconstruction := relationReconstruction0)
    (actual := actual selector witness) (summary := summary)
  · exact relationApplicationAt0 selector selectorLower selectorUpper
  · rfl
  · rfl
  · decide +kernel
  · exact mergeClaim selector selectorLower selectorUpper witness
  · exact relationAgreement0
  · decide +kernel
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply exactClaimAt_of_mergeClaim
    (relationClaim0 selector selectorLower selectorUpper witness)
  · decide +kernel
  · rfl
end SemanticResult21075

namespace SemanticResult21099
def owner : Owner := ⟨.program ⟨214⟩, ⟨30215⟩⟩
def rawTerms : List Term := Proof.Events082.exact21099RawTerms
def summary : Bound := (.finite 313276456757822654825721789483581492)
def resultEvent : Nat := 21099
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult21099.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddFiniteMergeClaimAt
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := LeftOperatorMerge21079.frameStart)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult21075.owner)
    (rightOwner := SemanticResult21030.owner)
    (leftResult := 21075) (rightResult := 21030)
    (leftActual := SemanticResult21075.actual selector witness)
    (rightActual := SemanticResult21030.actual selector witness)
    (leftRaw := SemanticResult21075.rawTerms)
    (rightRaw := SemanticResult21030.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 95420416)
    (rightMaximum := 313276456757822654825721789388161076) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 21076) (rightBinding := 21077)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7899⟩) (rightExpression := ⟨30214⟩)
    (coefficientTransfer := 21078) (summaryTransfer := 21098)
    (base := LeftOperatorMerge21079.base)
    (reconstruction := LeftOperatorMerge21079.reconstruction)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult21075.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult21030.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge21079.operationAgreement
  · rfl
  · decide
end SemanticResult21099

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
