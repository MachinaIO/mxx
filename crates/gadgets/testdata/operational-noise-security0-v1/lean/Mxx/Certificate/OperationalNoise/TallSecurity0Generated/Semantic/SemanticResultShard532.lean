import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard532
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard476
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard480
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard484
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard487
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard491
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard495
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard498
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard502
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard506
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard509
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard513
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard517
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard520
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard524
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard528
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard530
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard531

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult73962
def owner : Owner := ⟨.program ⟨214⟩, ⟨26349⟩⟩
def rawTerms : List Term := Proof.Events288.exact73962RawTerms
def summary : Bound := (.finite 1291889174379421642752)
def resultEvent : Nat := 73962
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult73962.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddFiniteMergeClaimAt
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := LeftOperatorMerge73959.frameStart)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult73955.owner)
    (rightOwner := SemanticResult73777.owner)
    (leftResult := 73955) (rightResult := 73777)
    (leftActual := SemanticResult73955.actual selector witness)
    (rightActual := SemanticResult73777.actual selector witness)
    (leftRaw := SemanticResult73955.rawTerms)
    (rightRaw := SemanticResult73777.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 1811303510016)
    (rightMaximum := 1291889172568118132736) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 73956) (rightBinding := 73957)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨20391⟩) (rightExpression := ⟨26348⟩)
    (coefficientTransfer := 73958) (summaryTransfer := 73961)
    (base := LeftOperatorMerge73959.base)
    (reconstruction := LeftOperatorMerge73959.reconstruction)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult73955.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult73777.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge73959.operationAgreement
  · rfl
  · decide
end SemanticResult73962

namespace SemanticResult73967
def owner : Owner := ⟨.program ⟨214⟩, ⟨26555⟩⟩
def rawTerms : List Term := Proof.Events288.exact73967RawTerms
def summary : Bound := (.finite 2583789554981353578496)
def resultEvent : Nat := 73967
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult73967.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult73962.owner)
    (rightOwner := SemanticResult73480.owner)
    (leftResult := 73962) (rightResult := 73480)
    (leftActual := SemanticResult73962.actual selector witness)
    (rightActual := SemanticResult73480.actual selector witness)
    (leftRaw := SemanticResult73962.rawTerms)
    (rightRaw := SemanticResult73480.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 1291889174379421642752)
    (rightMaximum := 1291900380601931935744) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 73963) (rightBinding := 73964)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨26349⟩) (rightExpression := ⟨26554⟩)
    (transferEvent := 73965) (summaryTransferEvent := 73966)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult73962.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult73480.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult73967

namespace SemanticResult73972
def owner : Owner := ⟨.program ⟨214⟩, ⟨26772⟩⟩
def rawTerms : List Term := Proof.Events288.exact73972RawTerms
def summary : Bound := (.finite 3875701141805795807232)
def resultEvent : Nat := 73972
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult73972.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult73967.owner)
    (rightOwner := SemanticResult72998.owner)
    (leftResult := 73967) (rightResult := 72998)
    (leftActual := SemanticResult73967.actual selector witness)
    (rightActual := SemanticResult72998.actual selector witness)
    (leftRaw := SemanticResult73967.rawTerms)
    (rightRaw := SemanticResult72998.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 2583789554981353578496)
    (rightMaximum := 1291911586824442228736) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 73968) (rightBinding := 73969)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨26555⟩) (rightExpression := ⟨26771⟩)
    (transferEvent := 73970) (summaryTransferEvent := 73971)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult73967.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult72998.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult73972

namespace SemanticResult73977
def owner : Owner := ⟨.program ⟨214⟩, ⟨26989⟩⟩
def rawTerms : List Term := Proof.Events288.exact73977RawTerms
def summary : Bound := (.finite 5167635141075258621952)
def resultEvent : Nat := 73977
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult73977.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult73972.owner)
    (rightOwner := SemanticResult72516.owner)
    (leftResult := 73972) (rightResult := 72516)
    (leftActual := SemanticResult73972.actual selector witness)
    (rightActual := SemanticResult72516.actual selector witness)
    (leftRaw := SemanticResult73972.rawTerms)
    (rightRaw := SemanticResult72516.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 3875701141805795807232)
    (rightMaximum := 1291933999269462814720) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 73973) (rightBinding := 73974)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨26772⟩) (rightExpression := ⟨26988⟩)
    (transferEvent := 73975) (summaryTransferEvent := 73976)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult73972.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult72516.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult73977

namespace SemanticResult73982
def owner : Owner := ⟨.program ⟨214⟩, ⟨27206⟩⟩
def rawTerms : List Term := Proof.Events288.exact73982RawTerms
def summary : Bound := (.finite 6459613965234762608640)
def resultEvent : Nat := 73982
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult73982.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult73977.owner)
    (rightOwner := SemanticResult72034.owner)
    (leftResult := 73977) (rightResult := 72034)
    (leftActual := SemanticResult73977.actual selector witness)
    (rightActual := SemanticResult72034.actual selector witness)
    (leftRaw := SemanticResult73977.rawTerms)
    (rightRaw := SemanticResult72034.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 5167635141075258621952)
    (rightMaximum := 1291978824159503986688) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 73978) (rightBinding := 73979)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨26989⟩) (rightExpression := ⟨27205⟩)
    (transferEvent := 73980) (summaryTransferEvent := 73981)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult73977.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult72034.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult73982

namespace SemanticResult73987
def owner : Owner := ⟨.program ⟨214⟩, ⟨27423⟩⟩
def rawTerms : List Term := Proof.Events289.exact73987RawTerms
def summary : Bound := (.finite 7751615201839287181312)
def resultEvent : Nat := 73987
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult73987.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult73982.owner)
    (rightOwner := SemanticResult71552.owner)
    (leftResult := 73982) (rightResult := 71552)
    (leftActual := SemanticResult73982.actual selector witness)
    (rightActual := SemanticResult71552.actual selector witness)
    (leftRaw := SemanticResult73982.rawTerms)
    (rightRaw := SemanticResult71552.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 6459613965234762608640)
    (rightMaximum := 1292001236604524572672) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 73983) (rightBinding := 73984)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨27206⟩) (rightExpression := ⟨27422⟩)
    (transferEvent := 73985) (summaryTransferEvent := 73986)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult73982.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult71552.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult73987

namespace SemanticResult73992
def owner : Owner := ⟨.program ⟨214⟩, ⟨27640⟩⟩
def rawTerms : List Term := Proof.Events289.exact73992RawTerms
def summary : Bound := (.finite 9043661263333852925952)
def resultEvent : Nat := 73992
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult73992.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult73987.owner)
    (rightOwner := SemanticResult71070.owner)
    (leftResult := 73987) (rightResult := 71070)
    (leftActual := SemanticResult73987.actual selector witness)
    (rightActual := SemanticResult71070.actual selector witness)
    (leftRaw := SemanticResult73987.rawTerms)
    (rightRaw := SemanticResult71070.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 7751615201839287181312)
    (rightMaximum := 1292046061494565744640) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 73988) (rightBinding := 73989)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨27423⟩) (rightExpression := ⟨27639⟩)
    (transferEvent := 73990) (summaryTransferEvent := 73991)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult73987.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult71070.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult73992

namespace SemanticResult73997
def owner : Owner := ⟨.program ⟨214⟩, ⟨27857⟩⟩
def rawTerms : List Term := Proof.Events289.exact73997RawTerms
def summary : Bound := (.finite 10335729737273439256576)
def resultEvent : Nat := 73997
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult73997.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult73992.owner)
    (rightOwner := SemanticResult70588.owner)
    (leftResult := 73992) (rightResult := 70588)
    (leftActual := SemanticResult73992.actual selector witness)
    (rightActual := SemanticResult70588.actual selector witness)
    (leftRaw := SemanticResult73992.rawTerms)
    (rightRaw := SemanticResult70588.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 9043661263333852925952)
    (rightMaximum := 1292068473939586330624) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 73993) (rightBinding := 73994)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨27640⟩) (rightExpression := ⟨27856⟩)
    (transferEvent := 73995) (summaryTransferEvent := 73996)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult73992.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult70588.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult73997

namespace SemanticResult74002
def owner : Owner := ⟨.program ⟨214⟩, ⟨28074⟩⟩
def rawTerms : List Term := Proof.Events289.exact74002RawTerms
def summary : Bound := (.finite 11627843036103066759168)
def resultEvent : Nat := 74002
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult74002.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult73997.owner)
    (rightOwner := SemanticResult70106.owner)
    (leftResult := 73997) (rightResult := 70106)
    (leftActual := SemanticResult73997.actual selector witness)
    (rightActual := SemanticResult70106.actual selector witness)
    (leftRaw := SemanticResult73997.rawTerms)
    (rightRaw := SemanticResult70106.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 10335729737273439256576)
    (rightMaximum := 1292113298829627502592) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 73998) (rightBinding := 73999)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨27857⟩) (rightExpression := ⟨28073⟩)
    (transferEvent := 74000) (summaryTransferEvent := 74001)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult73997.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult70106.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult74002

namespace SemanticResult74007
def owner : Owner := ⟨.program ⟨214⟩, ⟨28291⟩⟩
def rawTerms : List Term := Proof.Events289.exact74007RawTerms
def summary : Bound := (.finite 12920023572267756019712)
def resultEvent : Nat := 74007
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult74007.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult74002.owner)
    (rightOwner := SemanticResult69624.owner)
    (leftResult := 74002) (rightResult := 69624)
    (leftActual := SemanticResult74002.actual selector witness)
    (rightActual := SemanticResult69624.actual selector witness)
    (leftRaw := SemanticResult74002.rawTerms)
    (rightRaw := SemanticResult69624.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 11627843036103066759168)
    (rightMaximum := 1292180536164689260544) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 74003) (rightBinding := 74004)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨28074⟩) (rightExpression := ⟨28290⟩)
    (transferEvent := 74005) (summaryTransferEvent := 74006)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult74002.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult69624.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult74007

namespace SemanticResult74012
def owner : Owner := ⟨.program ⟨214⟩, ⟨28508⟩⟩
def rawTerms : List Term := Proof.Events289.exact74012RawTerms
def summary : Bound := (.finite 14212226520877465866240)
def resultEvent : Nat := 74012
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult74012.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult74007.owner)
    (rightOwner := SemanticResult69142.owner)
    (leftResult := 74007) (rightResult := 69142)
    (leftActual := SemanticResult74007.actual selector witness)
    (rightActual := SemanticResult69142.actual selector witness)
    (leftRaw := SemanticResult74007.rawTerms)
    (rightRaw := SemanticResult69142.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 12920023572267756019712)
    (rightMaximum := 1292202948609709846528) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 74008) (rightBinding := 74009)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨28291⟩) (rightExpression := ⟨28507⟩)
    (transferEvent := 74010) (summaryTransferEvent := 74011)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult74007.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult69142.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult74012

namespace SemanticResult74017
def owner : Owner := ⟨.program ⟨214⟩, ⟨28725⟩⟩
def rawTerms : List Term := Proof.Events289.exact74017RawTerms
def summary : Bound := (.finite 15504496706822237470720)
def resultEvent : Nat := 74017
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult74017.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult74012.owner)
    (rightOwner := SemanticResult68660.owner)
    (leftResult := 74012) (rightResult := 68660)
    (leftActual := SemanticResult74012.actual selector witness)
    (rightActual := SemanticResult68660.actual selector witness)
    (leftRaw := SemanticResult74012.rawTerms)
    (rightRaw := SemanticResult68660.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 14212226520877465866240)
    (rightMaximum := 1292270185944771604480) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 74013) (rightBinding := 74014)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨28508⟩) (rightExpression := ⟨28724⟩)
    (transferEvent := 74015) (summaryTransferEvent := 74016)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult74012.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult68660.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult74017

namespace SemanticResult74022
def owner : Owner := ⟨.program ⟨214⟩, ⟨28942⟩⟩
def rawTerms : List Term := Proof.Events289.exact74022RawTerms
def summary : Bound := (.finite 16796811717657050247168)
def resultEvent : Nat := 74022
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult74022.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult74017.owner)
    (rightOwner := SemanticResult68178.owner)
    (leftResult := 74017) (rightResult := 68178)
    (leftActual := SemanticResult74017.actual selector witness)
    (rightActual := SemanticResult68178.actual selector witness)
    (leftRaw := SemanticResult74017.rawTerms)
    (rightRaw := SemanticResult68178.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 15504496706822237470720)
    (rightMaximum := 1292315010834812776448) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 74018) (rightBinding := 74019)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨28725⟩) (rightExpression := ⟨28941⟩)
    (transferEvent := 74020) (summaryTransferEvent := 74021)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult74017.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult68178.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult74022

namespace SemanticResult74027
def owner : Owner := ⟨.program ⟨214⟩, ⟨29159⟩⟩
def rawTerms : List Term := Proof.Events289.exact74027RawTerms
def summary : Bound := (.finite 18089149140936883609600)
def resultEvent : Nat := 74027
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult74027.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult74022.owner)
    (rightOwner := SemanticResult67696.owner)
    (leftResult := 74022) (rightResult := 67696)
    (leftActual := SemanticResult74022.actual selector witness)
    (rightActual := SemanticResult67696.actual selector witness)
    (leftRaw := SemanticResult74022.rawTerms)
    (rightRaw := SemanticResult67696.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 16796811717657050247168)
    (rightMaximum := 1292337423279833362432) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 74023) (rightBinding := 74024)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨28942⟩) (rightExpression := ⟨29158⟩)
    (transferEvent := 74025) (summaryTransferEvent := 74026)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult74022.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult67696.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult74027

namespace SemanticResult74032
def owner : Owner := ⟨.program ⟨214⟩, ⟨29376⟩⟩
def rawTerms : List Term := Proof.Events289.exact74032RawTerms
def summary : Bound := (.finite 19381531389106758144000)
def resultEvent : Nat := 74032
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult74032.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult74027.owner)
    (rightOwner := SemanticResult67214.owner)
    (leftResult := 74027) (rightResult := 67214)
    (leftActual := SemanticResult74027.actual selector witness)
    (rightActual := SemanticResult67214.actual selector witness)
    (leftRaw := SemanticResult74027.rawTerms)
    (rightRaw := SemanticResult67214.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 18089149140936883609600)
    (rightMaximum := 1292382248169874534400) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 74028) (rightBinding := 74029)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨29159⟩) (rightExpression := ⟨29375⟩)
    (transferEvent := 74030) (summaryTransferEvent := 74031)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult74027.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult67214.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult74032

namespace SemanticResult74037
def owner : Owner := ⟨.program ⟨214⟩, ⟨29593⟩⟩
def rawTerms : List Term := Proof.Events289.exact74037RawTerms
def summary : Bound := (.finite 20673980874611694436352)
def resultEvent : Nat := 74037
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult74037.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult74032.owner)
    (rightOwner := SemanticResult66732.owner)
    (leftResult := 74032) (rightResult := 66732)
    (leftActual := SemanticResult74032.actual selector witness)
    (rightActual := SemanticResult66732.actual selector witness)
    (leftRaw := SemanticResult74032.rawTerms)
    (rightRaw := SemanticResult66732.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 19381531389106758144000)
    (rightMaximum := 1292449485504936292352) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 74033) (rightBinding := 74034)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨29376⟩) (rightExpression := ⟨29592⟩)
    (transferEvent := 74035) (summaryTransferEvent := 74036)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult74032.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult66732.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult74037

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
