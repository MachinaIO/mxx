import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events346

open Mxx.Certificate.OperationalNoise
open CertificateABI

def exact88576RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6709⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6711⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6713⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6715⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6717⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6719⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15265⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15310⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15366⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15629⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15748⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨17327⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact88576RawTermsValid :
    exact88576RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event88576 : Event := .resultExact (⟨.program ⟨214⟩, ⟨27436⟩⟩) exact88576RawTerms .large 88574 (.finite 7751615201839287181312) (some (88575))

def event88577 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27653⟩⟩) 0 ⟨27436⟩ 88576

def event88578 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27653⟩⟩) 1 ⟨27652⟩ 85671

def event88579 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27653⟩⟩) (.sum [.predecessor 0 88577 .coefficient, .predecessor 1 88578 .coefficient])

def event88580 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27653⟩⟩) (.sum [.result 88576 .summary, .result 85671 .summary])

def exact88581RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6709⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6711⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6713⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6715⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6717⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6719⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6721⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15265⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15310⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15366⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15629⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15748⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15867⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨17327⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact88581RawTermsValid :
    exact88581RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event88581 : Event := .resultExact (⟨.program ⟨214⟩, ⟨27653⟩⟩) exact88581RawTerms .large 88579 (.finite 9043661263333852925952) (some (88580))

def event88582 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27870⟩⟩) 0 ⟨27653⟩ 88581

def event88583 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27870⟩⟩) 1 ⟨27869⟩ 85191

def event88584 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27870⟩⟩) (.sum [.predecessor 0 88582 .coefficient, .predecessor 1 88583 .coefficient])

def event88585 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27870⟩⟩) (.sum [.result 88581 .summary, .result 85191 .summary])

def exact88586RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6709⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6711⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6713⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6715⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6717⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6719⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6721⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6723⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15265⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15310⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15366⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15629⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15748⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15867⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15986⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨17327⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact88586RawTermsValid :
    exact88586RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event88586 : Event := .resultExact (⟨.program ⟨214⟩, ⟨27870⟩⟩) exact88586RawTerms .large 88584 (.finite 10335729737273439256576) (some (88585))

def event88587 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28087⟩⟩) 0 ⟨27870⟩ 88586

def event88588 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28087⟩⟩) 1 ⟨28086⟩ 84711

def event88589 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28087⟩⟩) (.sum [.predecessor 0 88587 .coefficient, .predecessor 1 88588 .coefficient])

def event88590 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28087⟩⟩) (.sum [.result 88586 .summary, .result 84711 .summary])

def exact88591RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6709⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6711⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6713⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6715⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6717⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6719⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6721⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6723⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6725⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15265⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15310⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15366⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15629⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15748⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15867⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15986⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16105⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨17327⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact88591RawTermsValid :
    exact88591RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event88591 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28087⟩⟩) exact88591RawTerms .large 88589 (.finite 11627843036103066759168) (some (88590))

def event88592 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28304⟩⟩) 0 ⟨28087⟩ 88591

def event88593 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28304⟩⟩) 1 ⟨28303⟩ 84231

def event88594 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28304⟩⟩) (.sum [.predecessor 0 88592 .coefficient, .predecessor 1 88593 .coefficient])

def event88595 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28304⟩⟩) (.sum [.result 88591 .summary, .result 84231 .summary])

def exact88596RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6709⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6711⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6713⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6715⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6717⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6719⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6721⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6723⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6725⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6727⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15265⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15310⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15366⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15629⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15748⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15867⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15986⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16105⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨17327⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨18340⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact88596RawTermsValid :
    exact88596RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event88596 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28304⟩⟩) exact88596RawTerms .large 88594 (.finite 12920023572267756019712) (some (88595))

def event88597 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28521⟩⟩) 0 ⟨28304⟩ 88596

def event88598 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28521⟩⟩) 1 ⟨28520⟩ 83751

def event88599 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28521⟩⟩) (.sum [.predecessor 0 88597 .coefficient, .predecessor 1 88598 .coefficient])

def event88600 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28521⟩⟩) (.sum [.result 88596 .summary, .result 83751 .summary])

def exact88601RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6709⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6711⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6713⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6715⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6717⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6719⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6721⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6723⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6725⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6727⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6729⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15265⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15310⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15366⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15629⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15748⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15867⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15986⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16105⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16308⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨17327⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨18340⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact88601RawTermsValid :
    exact88601RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event88601 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28521⟩⟩) exact88601RawTerms .large 88599 (.finite 14212226520877465866240) (some (88600))

def event88602 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28738⟩⟩) 0 ⟨28521⟩ 88601

def event88603 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28738⟩⟩) 1 ⟨28737⟩ 83271

def event88604 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28738⟩⟩) (.sum [.predecessor 0 88602 .coefficient, .predecessor 1 88603 .coefficient])

def event88605 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28738⟩⟩) (.sum [.result 88601 .summary, .result 83271 .summary])

def exact88606RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6709⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6711⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6713⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6715⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6717⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6719⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6721⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6723⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6725⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6727⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6729⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6731⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15265⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15310⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15366⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15629⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15748⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15867⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15986⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16105⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16308⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨17120⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨17327⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨18340⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact88606RawTermsValid :
    exact88606RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event88606 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28738⟩⟩) exact88606RawTerms .large 88604 (.finite 15504496706822237470720) (some (88605))

def event88607 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28955⟩⟩) 0 ⟨28738⟩ 88606

def event88608 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28955⟩⟩) 1 ⟨28954⟩ 82791

def event88609 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28955⟩⟩) (.sum [.predecessor 0 88607 .coefficient, .predecessor 1 88608 .coefficient])

def event88610 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28955⟩⟩) (.sum [.result 88606 .summary, .result 82791 .summary])

def exact88611RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6709⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6711⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6713⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6715⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6717⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6719⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6721⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6723⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6725⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6727⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6729⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6731⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6733⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15265⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15310⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15366⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15629⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15748⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15867⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15986⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16105⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16308⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨17120⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨17327⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨17904⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨18340⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact88611RawTermsValid :
    exact88611RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event88611 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28955⟩⟩) exact88611RawTerms .large 88609 (.finite 16796811717657050247168) (some (88610))

def event88612 : Event := .predecessor (⟨.program ⟨214⟩, ⟨29172⟩⟩) 0 ⟨28955⟩ 88611

def event88613 : Event := .predecessor (⟨.program ⟨214⟩, ⟨29172⟩⟩) 1 ⟨29171⟩ 82311

def event88614 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨29172⟩⟩) (.sum [.predecessor 0 88612 .coefficient, .predecessor 1 88613 .coefficient])

def event88615 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨29172⟩⟩) (.sum [.result 88611 .summary, .result 82311 .summary])

def exact88616RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6709⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6711⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6713⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6715⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6717⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6719⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6721⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6723⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6725⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6727⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6729⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6731⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6733⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6735⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15265⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15310⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15366⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15629⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15748⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15867⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15986⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16105⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16308⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨17120⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨17327⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨17904⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨18205⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨18340⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact88616RawTermsValid :
    exact88616RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event88616 : Event := .resultExact (⟨.program ⟨214⟩, ⟨29172⟩⟩) exact88616RawTerms .large 88614 (.finite 18089149140936883609600) (some (88615))

def event88617 : Event := .predecessor (⟨.program ⟨214⟩, ⟨29389⟩⟩) 0 ⟨29172⟩ 88616

def event88618 : Event := .predecessor (⟨.program ⟨214⟩, ⟨29389⟩⟩) 1 ⟨29388⟩ 81831

def event88619 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨29389⟩⟩) (.sum [.predecessor 0 88617 .coefficient, .predecessor 1 88618 .coefficient])

def event88620 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨29389⟩⟩) (.sum [.result 88616 .summary, .result 81831 .summary])

def exact88621RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6709⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6711⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6713⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6715⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6717⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6719⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6721⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6723⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6725⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6727⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6729⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6731⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6733⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6735⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6737⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15265⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15310⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15366⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15629⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15748⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15867⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15986⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16105⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16308⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16679⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨17120⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨17327⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨17904⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨18205⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨18340⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact88621RawTermsValid :
    exact88621RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event88621 : Event := .resultExact (⟨.program ⟨214⟩, ⟨29389⟩⟩) exact88621RawTerms .large 88619 (.finite 19381531389106758144000) (some (88620))

def event88622 : Event := .predecessor (⟨.program ⟨214⟩, ⟨29606⟩⟩) 0 ⟨29389⟩ 88621

def event88623 : Event := .predecessor (⟨.program ⟨214⟩, ⟨29606⟩⟩) 1 ⟨29605⟩ 81351

def event88624 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨29606⟩⟩) (.sum [.predecessor 0 88622 .coefficient, .predecessor 1 88623 .coefficient])

def event88625 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨29606⟩⟩) (.sum [.result 88621 .summary, .result 81351 .summary])

def exact88626RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6709⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6711⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6713⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6715⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6717⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6719⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6721⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6723⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6725⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6727⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6729⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6731⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6733⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6735⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6737⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6739⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15265⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15310⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15366⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15629⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15748⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15867⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15986⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16105⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16308⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16679⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16798⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨17120⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨17327⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨17904⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨18205⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨18340⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact88626RawTermsValid :
    exact88626RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event88626 : Event := .resultExact (⟨.program ⟨214⟩, ⟨29606⟩⟩) exact88626RawTerms .large 88624 (.finite 20673980874611694436352) (some (88625))

def event88627 : Event := .predecessor (⟨.program ⟨214⟩, ⟨29823⟩⟩) 0 ⟨29606⟩ 88626

def event88628 : Event := .predecessor (⟨.program ⟨214⟩, ⟨29823⟩⟩) 1 ⟨29822⟩ 80871

def event88629 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨29823⟩⟩) (.sum [.predecessor 0 88627 .coefficient, .predecessor 1 88628 .coefficient])

def event88630 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨29823⟩⟩) (.sum [.result 88626 .summary, .result 80871 .summary])

def exact88631RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6709⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6711⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6713⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6715⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6717⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6719⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6721⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6723⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6725⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6727⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6729⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6731⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6733⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6735⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6737⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6739⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6741⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15265⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15310⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15366⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15629⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15748⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15867⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15986⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16105⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16308⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16679⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16798⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨17085⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨17120⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨17327⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨17904⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨18205⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨18340⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact88631RawTermsValid :
    exact88631RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event88631 : Event := .resultExact (⟨.program ⟨214⟩, ⟨29823⟩⟩) exact88631RawTerms .large 88629 (.finite 21966497597451692486656) (some (88630))

def event88632 : Event := .predecessor (⟨.program ⟨214⟩, ⟨30120⟩⟩) 0 ⟨29823⟩ 88631

def event88633 : Event := .predecessor (⟨.program ⟨214⟩, ⟨30120⟩⟩) 1 ⟨30119⟩ 80391

def event88634 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨30120⟩⟩) (.sum [.predecessor 0 88632 .coefficient, .predecessor 1 88633 .coefficient])

def event88635 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨30120⟩⟩) (.sum [.result 88631 .summary, .result 80391 .summary])

def exact88636RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6709⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6711⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6713⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6715⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6717⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6719⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6721⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6723⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6725⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6727⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6729⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6731⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6733⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6735⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6737⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6739⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6741⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6743⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15265⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15310⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15366⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15629⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15748⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15867⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15986⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16105⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16308⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16679⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16798⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨17085⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨17120⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨17327⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨17904⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨18170⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨18205⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨18340⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact88636RawTermsValid :
    exact88636RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event88636 : Event := .resultExact (⟨.program ⟨214⟩, ⟨30120⟩⟩) exact88636RawTerms .large 88634 (.finite 23259036732736711122944) (some (88635))

def event88637 : Event := .predecessor (⟨.program ⟨214⟩, ⟨30121⟩⟩) 0 ⟨30120⟩ 88636

def event88638 : Event := .predecessor (⟨.program ⟨214⟩, ⟨30121⟩⟩) 1 ⟨18681⟩ 79895

def event88639 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨30121⟩⟩) (.product (.predecessor 0 88637 .coefficient) (.predecessor 1 88638 .coefficient) (⟨false, false, none, none, none⟩))

def event88640 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨30121⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨18681⟩⟩]⟩) [⟨.result 79895 .coefficient, false, none⟩])

def event88641 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨30121⟩⟩) (.product (.result 88636 .summary) (.transfer 88640) (⟨false, false, none, none, none⟩))

def event88642 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30121⟩⟩, .operator (⟨88636, 17⟩, ⟨79895, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6743⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩]⟩, (1)⟩)

def event88643 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30121⟩⟩, .operator (⟨88636, 33⟩, ⟨79895, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨18170⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩]⟩, (-1)⟩)

def event88644 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨30121⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨18170⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18681⟩⟩) ⟨18618⟩ 79892)

def event88645 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30121⟩⟩, .relation 88644 0, ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨18170⟩⟩], [⟨.program ⟨214⟩, ⟨18618⟩⟩]⟩, (-1)⟩)

def event88646 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30121⟩⟩, .operator (⟨88636, 16⟩, ⟨79895, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6741⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩]⟩, (1)⟩)

def event88647 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30121⟩⟩, .operator (⟨88636, 29⟩, ⟨79895, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨17085⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩]⟩, (-1)⟩)

def event88648 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨30121⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨17085⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18681⟩⟩) ⟨18618⟩ 79892)

def event88649 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30121⟩⟩, .relation 88648 0, ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨17085⟩⟩], [⟨.program ⟨214⟩, ⟨18618⟩⟩]⟩, (-1)⟩)

def event88650 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30121⟩⟩, .operator (⟨88636, 15⟩, ⟨79895, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6739⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩]⟩, (1)⟩)

def event88651 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30121⟩⟩, .operator (⟨88636, 28⟩, ⟨79895, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16798⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩]⟩, (-1)⟩)

def event88652 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨30121⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16798⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18681⟩⟩) ⟨18618⟩ 79892)

def event88653 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30121⟩⟩, .relation 88652 0, ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16798⟩⟩], [⟨.program ⟨214⟩, ⟨18618⟩⟩]⟩, (-1)⟩)

def event88654 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30121⟩⟩, .operator (⟨88636, 14⟩, ⟨79895, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6737⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩]⟩, (1)⟩)

def event88655 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30121⟩⟩, .operator (⟨88636, 27⟩, ⟨79895, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16679⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩]⟩, (-1)⟩)

def event88656 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨30121⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16679⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18681⟩⟩) ⟨18618⟩ 79892)

def event88657 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30121⟩⟩, .relation 88656 0, ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16679⟩⟩], [⟨.program ⟨214⟩, ⟨18618⟩⟩]⟩, (-1)⟩)

def event88658 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30121⟩⟩, .operator (⟨88636, 13⟩, ⟨79895, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6735⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩]⟩, (1)⟩)

def event88659 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30121⟩⟩, .operator (⟨88636, 34⟩, ⟨79895, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨18205⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩]⟩, (-1)⟩)

def event88660 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨30121⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨18205⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18681⟩⟩) ⟨18618⟩ 79892)

def event88661 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30121⟩⟩, .relation 88660 0, ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨18205⟩⟩], [⟨.program ⟨214⟩, ⟨18618⟩⟩]⟩, (-1)⟩)

def event88662 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30121⟩⟩, .operator (⟨88636, 12⟩, ⟨79895, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6733⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩]⟩, (1)⟩)

def event88663 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30121⟩⟩, .operator (⟨88636, 32⟩, ⟨79895, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨17904⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩]⟩, (-1)⟩)

def event88664 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨30121⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨17904⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18681⟩⟩) ⟨18618⟩ 79892)

def event88665 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30121⟩⟩, .relation 88664 0, ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨17904⟩⟩], [⟨.program ⟨214⟩, ⟨18618⟩⟩]⟩, (-1)⟩)

def event88666 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30121⟩⟩, .operator (⟨88636, 11⟩, ⟨79895, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6731⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩]⟩, (1)⟩)

def event88667 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30121⟩⟩, .operator (⟨88636, 30⟩, ⟨79895, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨17120⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩]⟩, (-1)⟩)

def event88668 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨30121⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨17120⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18681⟩⟩) ⟨18618⟩ 79892)

def event88669 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30121⟩⟩, .relation 88668 0, ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨17120⟩⟩], [⟨.program ⟨214⟩, ⟨18618⟩⟩]⟩, (-1)⟩)

def event88670 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30121⟩⟩, .operator (⟨88636, 10⟩, ⟨79895, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6729⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩]⟩, (1)⟩)

def event88671 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30121⟩⟩, .operator (⟨88636, 26⟩, ⟨79895, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16308⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩]⟩, (-1)⟩)

def event88672 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨30121⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16308⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18681⟩⟩) ⟨18618⟩ 79892)

def event88673 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30121⟩⟩, .relation 88672 0, ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16308⟩⟩], [⟨.program ⟨214⟩, ⟨18618⟩⟩]⟩, (-1)⟩)

def event88674 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30121⟩⟩, .operator (⟨88636, 9⟩, ⟨79895, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6727⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩]⟩, (1)⟩)

def event88675 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30121⟩⟩, .operator (⟨88636, 35⟩, ⟨79895, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨18340⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩]⟩, (-1)⟩)

def event88676 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨30121⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨18340⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18681⟩⟩) ⟨18618⟩ 79892)

def event88677 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30121⟩⟩, .relation 88676 0, ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨18340⟩⟩], [⟨.program ⟨214⟩, ⟨18618⟩⟩]⟩, (-1)⟩)

def event88678 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30121⟩⟩, .operator (⟨88636, 8⟩, ⟨79895, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6725⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩]⟩, (1)⟩)

def event88679 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30121⟩⟩, .operator (⟨88636, 25⟩, ⟨79895, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16105⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩]⟩, (-1)⟩)

def event88680 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨30121⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16105⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18681⟩⟩) ⟨18618⟩ 79892)

def event88681 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30121⟩⟩, .relation 88680 0, ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16105⟩⟩], [⟨.program ⟨214⟩, ⟨18618⟩⟩]⟩, (-1)⟩)

def event88682 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30121⟩⟩, .operator (⟨88636, 7⟩, ⟨79895, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6723⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩]⟩, (1)⟩)

def event88683 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30121⟩⟩, .operator (⟨88636, 24⟩, ⟨79895, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15986⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩]⟩, (-1)⟩)

def event88684 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨30121⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15986⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18681⟩⟩) ⟨18618⟩ 79892)

def event88685 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30121⟩⟩, .relation 88684 0, ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15986⟩⟩], [⟨.program ⟨214⟩, ⟨18618⟩⟩]⟩, (-1)⟩)

def event88686 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30121⟩⟩, .operator (⟨88636, 6⟩, ⟨79895, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6721⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩]⟩, (1)⟩)

def event88687 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30121⟩⟩, .operator (⟨88636, 23⟩, ⟨79895, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15867⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩]⟩, (-1)⟩)

def event88688 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨30121⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15867⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18681⟩⟩) ⟨18618⟩ 79892)

def event88689 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30121⟩⟩, .relation 88688 0, ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15867⟩⟩], [⟨.program ⟨214⟩, ⟨18618⟩⟩]⟩, (-1)⟩)

def event88690 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30121⟩⟩, .operator (⟨88636, 5⟩, ⟨79895, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6719⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩]⟩, (1)⟩)

def event88691 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30121⟩⟩, .operator (⟨88636, 22⟩, ⟨79895, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15748⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩]⟩, (-1)⟩)

def event88692 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨30121⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15748⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18681⟩⟩) ⟨18618⟩ 79892)

def event88693 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30121⟩⟩, .relation 88692 0, ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15748⟩⟩], [⟨.program ⟨214⟩, ⟨18618⟩⟩]⟩, (-1)⟩)

def event88694 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30121⟩⟩, .operator (⟨88636, 4⟩, ⟨79895, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6717⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩]⟩, (1)⟩)

def event88695 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30121⟩⟩, .operator (⟨88636, 21⟩, ⟨79895, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15629⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩]⟩, (-1)⟩)

def event88696 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨30121⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15629⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18681⟩⟩) ⟨18618⟩ 79892)

def event88697 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30121⟩⟩, .relation 88696 0, ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15629⟩⟩], [⟨.program ⟨214⟩, ⟨18618⟩⟩]⟩, (-1)⟩)

def event88698 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30121⟩⟩, .operator (⟨88636, 3⟩, ⟨79895, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6715⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩]⟩, (1)⟩)

def event88699 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30121⟩⟩, .operator (⟨88636, 31⟩, ⟨79895, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨17327⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩]⟩, (-1)⟩)

def event88700 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨30121⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨17327⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18681⟩⟩) ⟨18618⟩ 79892)

def event88701 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30121⟩⟩, .relation 88700 0, ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨17327⟩⟩], [⟨.program ⟨214⟩, ⟨18618⟩⟩]⟩, (-1)⟩)

def event88702 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30121⟩⟩, .operator (⟨88636, 2⟩, ⟨79895, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6713⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩]⟩, (1)⟩)

def event88703 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30121⟩⟩, .operator (⟨88636, 20⟩, ⟨79895, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15366⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩]⟩, (-1)⟩)

def event88704 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨30121⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15366⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18681⟩⟩) ⟨18618⟩ 79892)

def event88705 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30121⟩⟩, .relation 88704 0, ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15366⟩⟩], [⟨.program ⟨214⟩, ⟨18618⟩⟩]⟩, (-1)⟩)

def event88706 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30121⟩⟩, .operator (⟨88636, 1⟩, ⟨79895, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6711⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩]⟩, (1)⟩)

def event88707 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30121⟩⟩, .operator (⟨88636, 19⟩, ⟨79895, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15310⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩]⟩, (-1)⟩)

def event88708 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨30121⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15310⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18681⟩⟩) ⟨18618⟩ 79892)

def event88709 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30121⟩⟩, .relation 88708 0, ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15310⟩⟩], [⟨.program ⟨214⟩, ⟨18618⟩⟩]⟩, (-1)⟩)

def event88710 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30121⟩⟩, .operator (⟨88636, 0⟩, ⟨79895, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6709⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩]⟩, (1)⟩)

def event88711 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30121⟩⟩, .operator (⟨88636, 18⟩, ⟨79895, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15265⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩]⟩, (-1)⟩)

def event88712 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨30121⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15265⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18681⟩⟩) ⟨18618⟩ 79892)

def event88713 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30121⟩⟩, .relation 88712 0, ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15265⟩⟩], [⟨.program ⟨214⟩, ⟨18618⟩⟩]⟩, (-1)⟩)

def exact88714RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6709⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6711⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6713⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6715⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6717⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6719⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6721⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6723⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6725⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6727⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6729⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6731⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6733⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6735⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6737⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6739⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6741⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6743⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15265⟩⟩], [⟨.program ⟨214⟩, ⟨18618⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15310⟩⟩], [⟨.program ⟨214⟩, ⟨18618⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15366⟩⟩], [⟨.program ⟨214⟩, ⟨18618⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15629⟩⟩], [⟨.program ⟨214⟩, ⟨18618⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15748⟩⟩], [⟨.program ⟨214⟩, ⟨18618⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15867⟩⟩], [⟨.program ⟨214⟩, ⟨18618⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15986⟩⟩], [⟨.program ⟨214⟩, ⟨18618⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16105⟩⟩], [⟨.program ⟨214⟩, ⟨18618⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16308⟩⟩], [⟨.program ⟨214⟩, ⟨18618⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16679⟩⟩], [⟨.program ⟨214⟩, ⟨18618⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16798⟩⟩], [⟨.program ⟨214⟩, ⟨18618⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨17085⟩⟩], [⟨.program ⟨214⟩, ⟨18618⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨17120⟩⟩], [⟨.program ⟨214⟩, ⟨18618⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨17327⟩⟩], [⟨.program ⟨214⟩, ⟨18618⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨17904⟩⟩], [⟨.program ⟨214⟩, ⟨18618⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨18170⟩⟩], [⟨.program ⟨214⟩, ⟨18618⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨18205⟩⟩], [⟨.program ⟨214⟩, ⟨18618⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨18340⟩⟩], [⟨.program ⟨214⟩, ⟨18618⟩⟩]⟩, (-1)⟩]

theorem exact88714RawTermsValid :
    exact88714RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event88714 : Event := .resultExact (⟨.program ⟨214⟩, ⟨30121⟩⟩) exact88714RawTerms .large 88639 (.finite 85361036953731453608582447104) (some (88641))

def event88715 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18559⟩⟩) 0 ⟨18350⟩ 4310

def event88716 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨18559⟩⟩) (.authority (.relationPreimageSource ⟨67⟩))

def exact88717RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨18559⟩⟩]⟩, (1)⟩]

theorem exact88717RawTermsValid :
    exact88717RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event88717 : Event := .resultExact (⟨.program ⟨214⟩, ⟨18559⟩⟩) exact88717RawTerms (.finite 136065468) 88716 .exactZero (none)

def event88718 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18561⟩⟩) 0 ⟨18559⟩ 88717

def event88719 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18561⟩⟩) 1 ⟨2348⟩ 4

def event88720 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨18561⟩⟩) (.scale (.predecessor 0 88718 .coefficient) (.value (.predecessor 1 88719 .coefficient)))

def exact88721RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨18559⟩⟩]⟩, (1)⟩]

theorem exact88721RawTermsValid :
    exact88721RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event88721 : Event := .resultExact (⟨.program ⟨214⟩, ⟨18561⟩⟩) exact88721RawTerms (.finite 136065468) 88720 .exactZero (none)

def event88722 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18562⟩⟩) 0 ⟨5541⟩ 80012

def event88723 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18562⟩⟩) 1 ⟨18561⟩ 88721

def event88724 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨18562⟩⟩) (.product (.predecessor 0 88722 .coefficient) (.predecessor 1 88723 .coefficient) (⟨false, false, none, none, none⟩))

def event88725 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨18562⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨18559⟩⟩]⟩) [⟨.result 88717 .coefficient, false, none⟩])

def event88726 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨18562⟩⟩) (.product (.result 80012 .summary) (.transfer 88725) (⟨false, false, none, none, none⟩))

def event88727 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18562⟩⟩, .operator (⟨80012, 0⟩, ⟨88721, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18559⟩⟩]⟩, (1)⟩)

def event88728 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨18560⟩⟩)

def event88729 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event88730 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event88731 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨2348⟩⟩) (.authority (.operator))

def event88732 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨2348⟩⟩) (.finite 1)

def event88733 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event88734 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event88735 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event88736 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event88737 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 88736

def event88738 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 88734

def event88739 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 88737 .coefficient) (.value (.predecessor 1 88738 .coefficient)))

def event88740 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event88741 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5505⟩⟩) 0 ⟨5503⟩ 88740

def event88742 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5505⟩⟩) 1 ⟨2348⟩ 88732

def event88743 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5505⟩⟩) (.sum [.predecessor 0 88741 .coefficient, .predecessor 1 88742 .coefficient])

def event88744 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5505⟩⟩) (.finite 218)

def event88745 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5536⟩⟩) 0 ⟨5505⟩ 88744

def event88746 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5536⟩⟩) 1 ⟨961⟩ 88730

def event88747 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5536⟩⟩) (.identity (.predecessor 1 88746 .coefficient))

def event88748 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5536⟩⟩) (.finite 224)

def event88749 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13350⟩⟩) 0 ⟨5536⟩ 88748

def event88750 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13350⟩⟩) (.authority (.programFamilyFact))

def exact88751RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨13350⟩⟩], []⟩, (1)⟩]

theorem exact88751RawTermsValid :
    exact88751RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event88751 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13350⟩⟩) exact88751RawTerms (.finite 60) 88750 .exactZero (none)

def event88752 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10345⟩⟩) 0 ⟨5536⟩ 88748

def event88753 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10345⟩⟩) (.authority (.programFamilyFact))

def exact88754RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨10345⟩⟩], []⟩, (1)⟩]

theorem exact88754RawTermsValid :
    exact88754RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event88754 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10345⟩⟩) exact88754RawTerms (.finite 60) 88753 .exactZero (none)

def event88755 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13351⟩⟩) 0 ⟨10345⟩ 88754

def event88756 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13351⟩⟩) 1 ⟨13350⟩ 88751

def event88757 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13351⟩⟩) (.product (.predecessor 0 88755 .coefficient) (.predecessor 1 88756 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event88758 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13351⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨10345⟩⟩, ⟨.program ⟨214⟩, ⟨13350⟩⟩], []⟩) [⟨.result 88754 .coefficient, true, some 1⟩, ⟨.result 88751 .coefficient, true, some 1⟩])

def event88759 : Event := .survivorFold (1) 88758

def exact88760RawTerms : List Term := []

theorem exact88760RawTermsValid :
    exact88760RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event88760 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13351⟩⟩) exact88760RawTerms (.finite 3600) 88757 (.finite 3600) (some (88758))

def event88761 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13352⟩⟩) 0 ⟨13351⟩ 88760

def event88762 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13352⟩⟩) (.identity (.predecessor 0 88761 .coefficient))

def event88763 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨13352⟩⟩) (.finite 3600)

def event88764 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17011⟩⟩) 0 ⟨13352⟩ 88763

def event88765 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨17011⟩⟩) (.authority (.programFamilyFact))

def exact88766RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨17011⟩⟩], []⟩, (1)⟩]

theorem exact88766RawTermsValid :
    exact88766RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event88766 : Event := .resultExact (⟨.program ⟨214⟩, ⟨17011⟩⟩) exact88766RawTerms (.finite 60) 88765 .exactZero (none)

def event88767 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17012⟩⟩) 0 ⟨17011⟩ 88766

def event88768 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨17012⟩⟩) (.identity (.predecessor 0 88767 .coefficient))

def event88769 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨17012⟩⟩) (.finite 60)

def event88770 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18170⟩⟩) 0 ⟨17012⟩ 88769

def event88771 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨18170⟩⟩) (.authority (.programFamilyFact))

def exact88772RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨18170⟩⟩], []⟩, (1)⟩]

theorem exact88772RawTermsValid :
    exact88772RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event88772 : Event := .resultExact (⟨.program ⟨214⟩, ⟨18170⟩⟩) exact88772RawTerms (.finite 63) 88771 .exactZero (none)

def event88773 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13154⟩⟩) 0 ⟨5536⟩ 88748

def event88774 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13154⟩⟩) (.authority (.programFamilyFact))

def exact88775RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨13154⟩⟩], []⟩, (1)⟩]

theorem exact88775RawTermsValid :
    exact88775RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event88775 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13154⟩⟩) exact88775RawTerms (.finite 58) 88774 .exactZero (none)

def event88776 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10240⟩⟩) 0 ⟨5536⟩ 88748

def event88777 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10240⟩⟩) (.authority (.programFamilyFact))

def exact88778RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨10240⟩⟩], []⟩, (1)⟩]

theorem exact88778RawTermsValid :
    exact88778RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event88778 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10240⟩⟩) exact88778RawTerms (.finite 58) 88777 .exactZero (none)

def event88779 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13155⟩⟩) 0 ⟨10240⟩ 88778

def event88780 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13155⟩⟩) 1 ⟨13154⟩ 88775

def event88781 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13155⟩⟩) (.product (.predecessor 0 88779 .coefficient) (.predecessor 1 88780 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event88782 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13155⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨10240⟩⟩, ⟨.program ⟨214⟩, ⟨13154⟩⟩], []⟩) [⟨.result 88778 .coefficient, true, some 1⟩, ⟨.result 88775 .coefficient, true, some 1⟩])

def event88783 : Event := .survivorFold (1) 88782

def exact88784RawTerms : List Term := []

theorem exact88784RawTermsValid :
    exact88784RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event88784 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13155⟩⟩) exact88784RawTerms (.finite 3364) 88781 (.finite 3364) (some (88782))

def event88785 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13156⟩⟩) 0 ⟨13155⟩ 88784

def event88786 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13156⟩⟩) (.identity (.predecessor 0 88785 .coefficient))

def event88787 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨13156⟩⟩) (.finite 3364)

def event88788 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16871⟩⟩) 0 ⟨13156⟩ 88787

def event88789 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16871⟩⟩) (.authority (.programFamilyFact))

def exact88790RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16871⟩⟩], []⟩, (1)⟩]

theorem exact88790RawTermsValid :
    exact88790RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event88790 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16871⟩⟩) exact88790RawTerms (.finite 58) 88789 .exactZero (none)

def event88791 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16872⟩⟩) 0 ⟨16871⟩ 88790

def event88792 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16872⟩⟩) (.identity (.predecessor 0 88791 .coefficient))

def event88793 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨16872⟩⟩) (.finite 58)

def event88794 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17085⟩⟩) 0 ⟨16872⟩ 88793

def event88795 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨17085⟩⟩) (.authority (.programFamilyFact))

def exact88796RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨17085⟩⟩], []⟩, (1)⟩]

theorem exact88796RawTermsValid :
    exact88796RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event88796 : Event := .resultExact (⟨.program ⟨214⟩, ⟨17085⟩⟩) exact88796RawTerms (.finite 63) 88795 .exactZero (none)

def event88797 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12958⟩⟩) 0 ⟨5536⟩ 88748

def event88798 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12958⟩⟩) (.authority (.programFamilyFact))

def exact88799RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨12958⟩⟩], []⟩, (1)⟩]

theorem exact88799RawTermsValid :
    exact88799RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event88799 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12958⟩⟩) exact88799RawTerms (.finite 52) 88798 .exactZero (none)

def event88800 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10135⟩⟩) 0 ⟨5536⟩ 88748

def event88801 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10135⟩⟩) (.authority (.programFamilyFact))

def exact88802RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨10135⟩⟩], []⟩, (1)⟩]

theorem exact88802RawTermsValid :
    exact88802RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event88802 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10135⟩⟩) exact88802RawTerms (.finite 52) 88801 .exactZero (none)

def event88803 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12959⟩⟩) 0 ⟨10135⟩ 88802

def event88804 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12959⟩⟩) 1 ⟨12958⟩ 88799

def event88805 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12959⟩⟩) (.product (.predecessor 0 88803 .coefficient) (.predecessor 1 88804 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event88806 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12959⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨10135⟩⟩, ⟨.program ⟨214⟩, ⟨12958⟩⟩], []⟩) [⟨.result 88802 .coefficient, true, some 1⟩, ⟨.result 88799 .coefficient, true, some 1⟩])

def event88807 : Event := .survivorFold (1) 88806

def exact88808RawTerms : List Term := []

theorem exact88808RawTermsValid :
    exact88808RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event88808 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12959⟩⟩) exact88808RawTerms (.finite 2704) 88805 (.finite 2704) (some (88806))

def event88809 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12960⟩⟩) 0 ⟨12959⟩ 88808

def event88810 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12960⟩⟩) (.identity (.predecessor 0 88809 .coefficient))

def event88811 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨12960⟩⟩) (.finite 2704)

def event88812 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16752⟩⟩) 0 ⟨12960⟩ 88811

def event88813 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16752⟩⟩) (.authority (.programFamilyFact))

def exact88814RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16752⟩⟩], []⟩, (1)⟩]

theorem exact88814RawTermsValid :
    exact88814RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event88814 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16752⟩⟩) exact88814RawTerms (.finite 52) 88813 .exactZero (none)

def event88815 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16753⟩⟩) 0 ⟨16752⟩ 88814

def event88816 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16753⟩⟩) (.identity (.predecessor 0 88815 .coefficient))

def event88817 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨16753⟩⟩) (.finite 52)

def event88818 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16798⟩⟩) 0 ⟨16753⟩ 88817

def event88819 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16798⟩⟩) (.authority (.programFamilyFact))

def exact88820RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16798⟩⟩], []⟩, (1)⟩]

theorem exact88820RawTermsValid :
    exact88820RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event88820 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16798⟩⟩) exact88820RawTerms (.finite 63) 88819 .exactZero (none)

def event88821 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12762⟩⟩) 0 ⟨5536⟩ 88748

def event88822 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12762⟩⟩) (.authority (.programFamilyFact))

def exact88823RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨12762⟩⟩], []⟩, (1)⟩]

theorem exact88823RawTermsValid :
    exact88823RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event88823 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12762⟩⟩) exact88823RawTerms (.finite 46) 88822 .exactZero (none)

def event88824 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10030⟩⟩) 0 ⟨5536⟩ 88748

def event88825 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10030⟩⟩) (.authority (.programFamilyFact))

def exact88826RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨10030⟩⟩], []⟩, (1)⟩]

theorem exact88826RawTermsValid :
    exact88826RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event88826 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10030⟩⟩) exact88826RawTerms (.finite 46) 88825 .exactZero (none)

def event88827 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12763⟩⟩) 0 ⟨10030⟩ 88826

def event88828 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12763⟩⟩) 1 ⟨12762⟩ 88823

def event88829 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12763⟩⟩) (.product (.predecessor 0 88827 .coefficient) (.predecessor 1 88828 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event88830 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12763⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨10030⟩⟩, ⟨.program ⟨214⟩, ⟨12762⟩⟩], []⟩) [⟨.result 88826 .coefficient, true, some 1⟩, ⟨.result 88823 .coefficient, true, some 1⟩])

def event88831 : Event := .survivorFold (1) 88830

def eventLeaf5536 : Array AnnotatedEvent := #[
  { event := event88576
    frameStart := 0 },
  { event := event88577
    frameStart := 0 },
  { event := event88578
    frameStart := 0 },
  { event := event88579
    frameStart := 0 },
  { event := event88580
    frameStart := 0 },
  { event := event88581
    frameStart := 0 },
  { event := event88582
    frameStart := 0 },
  { event := event88583
    frameStart := 0 },
  { event := event88584
    frameStart := 0 },
  { event := event88585
    frameStart := 0 },
  { event := event88586
    frameStart := 0 },
  { event := event88587
    frameStart := 0 },
  { event := event88588
    frameStart := 0 },
  { event := event88589
    frameStart := 0 },
  { event := event88590
    frameStart := 0 },
  { event := event88591
    frameStart := 0 }
]

def eventLeaf5537 : Array AnnotatedEvent := #[
  { event := event88592
    frameStart := 0 },
  { event := event88593
    frameStart := 0 },
  { event := event88594
    frameStart := 0 },
  { event := event88595
    frameStart := 0 },
  { event := event88596
    frameStart := 0 },
  { event := event88597
    frameStart := 0 },
  { event := event88598
    frameStart := 0 },
  { event := event88599
    frameStart := 0 },
  { event := event88600
    frameStart := 0 },
  { event := event88601
    frameStart := 0 },
  { event := event88602
    frameStart := 0 },
  { event := event88603
    frameStart := 0 },
  { event := event88604
    frameStart := 0 },
  { event := event88605
    frameStart := 0 },
  { event := event88606
    frameStart := 0 },
  { event := event88607
    frameStart := 0 }
]

def eventLeaf5538 : Array AnnotatedEvent := #[
  { event := event88608
    frameStart := 0 },
  { event := event88609
    frameStart := 0 },
  { event := event88610
    frameStart := 0 },
  { event := event88611
    frameStart := 0 },
  { event := event88612
    frameStart := 0 },
  { event := event88613
    frameStart := 0 },
  { event := event88614
    frameStart := 0 },
  { event := event88615
    frameStart := 0 },
  { event := event88616
    frameStart := 0 },
  { event := event88617
    frameStart := 0 },
  { event := event88618
    frameStart := 0 },
  { event := event88619
    frameStart := 0 },
  { event := event88620
    frameStart := 0 },
  { event := event88621
    frameStart := 0 },
  { event := event88622
    frameStart := 0 },
  { event := event88623
    frameStart := 0 }
]

def eventLeaf5539 : Array AnnotatedEvent := #[
  { event := event88624
    frameStart := 0 },
  { event := event88625
    frameStart := 0 },
  { event := event88626
    frameStart := 0 },
  { event := event88627
    frameStart := 0 },
  { event := event88628
    frameStart := 0 },
  { event := event88629
    frameStart := 0 },
  { event := event88630
    frameStart := 0 },
  { event := event88631
    frameStart := 0 },
  { event := event88632
    frameStart := 0 },
  { event := event88633
    frameStart := 0 },
  { event := event88634
    frameStart := 0 },
  { event := event88635
    frameStart := 0 },
  { event := event88636
    frameStart := 0 },
  { event := event88637
    frameStart := 0 },
  { event := event88638
    frameStart := 0 },
  { event := event88639
    frameStart := 0 }
]

def eventLeaf5540 : Array AnnotatedEvent := #[
  { event := event88640
    frameStart := 0 },
  { event := event88641
    frameStart := 0 },
  { event := event88642
    frameStart := 0 },
  { event := event88643
    frameStart := 0 },
  { event := event88644
    frameStart := 0 },
  { event := event88645
    frameStart := 0 },
  { event := event88646
    frameStart := 0 },
  { event := event88647
    frameStart := 0 },
  { event := event88648
    frameStart := 0 },
  { event := event88649
    frameStart := 0 },
  { event := event88650
    frameStart := 0 },
  { event := event88651
    frameStart := 0 },
  { event := event88652
    frameStart := 0 },
  { event := event88653
    frameStart := 0 },
  { event := event88654
    frameStart := 0 },
  { event := event88655
    frameStart := 0 }
]

def eventLeaf5541 : Array AnnotatedEvent := #[
  { event := event88656
    frameStart := 0 },
  { event := event88657
    frameStart := 0 },
  { event := event88658
    frameStart := 0 },
  { event := event88659
    frameStart := 0 },
  { event := event88660
    frameStart := 0 },
  { event := event88661
    frameStart := 0 },
  { event := event88662
    frameStart := 0 },
  { event := event88663
    frameStart := 0 },
  { event := event88664
    frameStart := 0 },
  { event := event88665
    frameStart := 0 },
  { event := event88666
    frameStart := 0 },
  { event := event88667
    frameStart := 0 },
  { event := event88668
    frameStart := 0 },
  { event := event88669
    frameStart := 0 },
  { event := event88670
    frameStart := 0 },
  { event := event88671
    frameStart := 0 }
]

def eventLeaf5542 : Array AnnotatedEvent := #[
  { event := event88672
    frameStart := 0 },
  { event := event88673
    frameStart := 0 },
  { event := event88674
    frameStart := 0 },
  { event := event88675
    frameStart := 0 },
  { event := event88676
    frameStart := 0 },
  { event := event88677
    frameStart := 0 },
  { event := event88678
    frameStart := 0 },
  { event := event88679
    frameStart := 0 },
  { event := event88680
    frameStart := 0 },
  { event := event88681
    frameStart := 0 },
  { event := event88682
    frameStart := 0 },
  { event := event88683
    frameStart := 0 },
  { event := event88684
    frameStart := 0 },
  { event := event88685
    frameStart := 0 },
  { event := event88686
    frameStart := 0 },
  { event := event88687
    frameStart := 0 }
]

def eventLeaf5543 : Array AnnotatedEvent := #[
  { event := event88688
    frameStart := 0 },
  { event := event88689
    frameStart := 0 },
  { event := event88690
    frameStart := 0 },
  { event := event88691
    frameStart := 0 },
  { event := event88692
    frameStart := 0 },
  { event := event88693
    frameStart := 0 },
  { event := event88694
    frameStart := 0 },
  { event := event88695
    frameStart := 0 },
  { event := event88696
    frameStart := 0 },
  { event := event88697
    frameStart := 0 },
  { event := event88698
    frameStart := 0 },
  { event := event88699
    frameStart := 0 },
  { event := event88700
    frameStart := 0 },
  { event := event88701
    frameStart := 0 },
  { event := event88702
    frameStart := 0 },
  { event := event88703
    frameStart := 0 }
]

def eventLeaf5544 : Array AnnotatedEvent := #[
  { event := event88704
    frameStart := 0 },
  { event := event88705
    frameStart := 0 },
  { event := event88706
    frameStart := 0 },
  { event := event88707
    frameStart := 0 },
  { event := event88708
    frameStart := 0 },
  { event := event88709
    frameStart := 0 },
  { event := event88710
    frameStart := 0 },
  { event := event88711
    frameStart := 0 },
  { event := event88712
    frameStart := 0 },
  { event := event88713
    frameStart := 0 },
  { event := event88714
    frameStart := 0 },
  { event := event88715
    frameStart := 0 },
  { event := event88716
    frameStart := 0 },
  { event := event88717
    frameStart := 0 },
  { event := event88718
    frameStart := 0 },
  { event := event88719
    frameStart := 0 }
]

def eventLeaf5545 : Array AnnotatedEvent := #[
  { event := event88720
    frameStart := 0 },
  { event := event88721
    frameStart := 0 },
  { event := event88722
    frameStart := 0 },
  { event := event88723
    frameStart := 0 },
  { event := event88724
    frameStart := 0 },
  { event := event88725
    frameStart := 0 },
  { event := event88726
    frameStart := 0 },
  { event := event88727
    frameStart := 0 },
  { event := event88728
    frameStart := 88728 },
  { event := event88729
    frameStart := 88728 },
  { event := event88730
    frameStart := 88728 },
  { event := event88731
    frameStart := 88728 },
  { event := event88732
    frameStart := 88728 },
  { event := event88733
    frameStart := 88728 },
  { event := event88734
    frameStart := 88728 },
  { event := event88735
    frameStart := 88728 }
]

def eventLeaf5546 : Array AnnotatedEvent := #[
  { event := event88736
    frameStart := 88728 },
  { event := event88737
    frameStart := 88728 },
  { event := event88738
    frameStart := 88728 },
  { event := event88739
    frameStart := 88728 },
  { event := event88740
    frameStart := 88728 },
  { event := event88741
    frameStart := 88728 },
  { event := event88742
    frameStart := 88728 },
  { event := event88743
    frameStart := 88728 },
  { event := event88744
    frameStart := 88728 },
  { event := event88745
    frameStart := 88728 },
  { event := event88746
    frameStart := 88728 },
  { event := event88747
    frameStart := 88728 },
  { event := event88748
    frameStart := 88728 },
  { event := event88749
    frameStart := 88728 },
  { event := event88750
    frameStart := 88728 },
  { event := event88751
    frameStart := 88728 }
]

def eventLeaf5547 : Array AnnotatedEvent := #[
  { event := event88752
    frameStart := 88728 },
  { event := event88753
    frameStart := 88728 },
  { event := event88754
    frameStart := 88728 },
  { event := event88755
    frameStart := 88728 },
  { event := event88756
    frameStart := 88728 },
  { event := event88757
    frameStart := 88728 },
  { event := event88758
    frameStart := 88728 },
  { event := event88759
    frameStart := 88728 },
  { event := event88760
    frameStart := 88728 },
  { event := event88761
    frameStart := 88728 },
  { event := event88762
    frameStart := 88728 },
  { event := event88763
    frameStart := 88728 },
  { event := event88764
    frameStart := 88728 },
  { event := event88765
    frameStart := 88728 },
  { event := event88766
    frameStart := 88728 },
  { event := event88767
    frameStart := 88728 }
]

def eventLeaf5548 : Array AnnotatedEvent := #[
  { event := event88768
    frameStart := 88728 },
  { event := event88769
    frameStart := 88728 },
  { event := event88770
    frameStart := 88728 },
  { event := event88771
    frameStart := 88728 },
  { event := event88772
    frameStart := 88728 },
  { event := event88773
    frameStart := 88728 },
  { event := event88774
    frameStart := 88728 },
  { event := event88775
    frameStart := 88728 },
  { event := event88776
    frameStart := 88728 },
  { event := event88777
    frameStart := 88728 },
  { event := event88778
    frameStart := 88728 },
  { event := event88779
    frameStart := 88728 },
  { event := event88780
    frameStart := 88728 },
  { event := event88781
    frameStart := 88728 },
  { event := event88782
    frameStart := 88728 },
  { event := event88783
    frameStart := 88728 }
]

def eventLeaf5549 : Array AnnotatedEvent := #[
  { event := event88784
    frameStart := 88728 },
  { event := event88785
    frameStart := 88728 },
  { event := event88786
    frameStart := 88728 },
  { event := event88787
    frameStart := 88728 },
  { event := event88788
    frameStart := 88728 },
  { event := event88789
    frameStart := 88728 },
  { event := event88790
    frameStart := 88728 },
  { event := event88791
    frameStart := 88728 },
  { event := event88792
    frameStart := 88728 },
  { event := event88793
    frameStart := 88728 },
  { event := event88794
    frameStart := 88728 },
  { event := event88795
    frameStart := 88728 },
  { event := event88796
    frameStart := 88728 },
  { event := event88797
    frameStart := 88728 },
  { event := event88798
    frameStart := 88728 },
  { event := event88799
    frameStart := 88728 }
]

def eventLeaf5550 : Array AnnotatedEvent := #[
  { event := event88800
    frameStart := 88728 },
  { event := event88801
    frameStart := 88728 },
  { event := event88802
    frameStart := 88728 },
  { event := event88803
    frameStart := 88728 },
  { event := event88804
    frameStart := 88728 },
  { event := event88805
    frameStart := 88728 },
  { event := event88806
    frameStart := 88728 },
  { event := event88807
    frameStart := 88728 },
  { event := event88808
    frameStart := 88728 },
  { event := event88809
    frameStart := 88728 },
  { event := event88810
    frameStart := 88728 },
  { event := event88811
    frameStart := 88728 },
  { event := event88812
    frameStart := 88728 },
  { event := event88813
    frameStart := 88728 },
  { event := event88814
    frameStart := 88728 },
  { event := event88815
    frameStart := 88728 }
]

def eventLeaf5551 : Array AnnotatedEvent := #[
  { event := event88816
    frameStart := 88728 },
  { event := event88817
    frameStart := 88728 },
  { event := event88818
    frameStart := 88728 },
  { event := event88819
    frameStart := 88728 },
  { event := event88820
    frameStart := 88728 },
  { event := event88821
    frameStart := 88728 },
  { event := event88822
    frameStart := 88728 },
  { event := event88823
    frameStart := 88728 },
  { event := event88824
    frameStart := 88728 },
  { event := event88825
    frameStart := 88728 },
  { event := event88826
    frameStart := 88728 },
  { event := event88827
    frameStart := 88728 },
  { event := event88828
    frameStart := 88728 },
  { event := event88829
    frameStart := 88728 },
  { event := event88830
    frameStart := 88728 },
  { event := event88831
    frameStart := 88728 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events346
