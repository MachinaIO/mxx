import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events405

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event103680 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨18488⟩⟩) (.sum [.predecessor 0 103678 .coefficient, .predecessor 1 103679 .coefficient])

def exact103681RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6744⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨18485⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact103681RawTermsValid :
    exact103681RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event103681 : Event := .resultExact (⟨.program ⟨214⟩, ⟨18488⟩⟩) exact103681RawTerms .large 103680 .exactZero (none)

def event103682 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18676⟩⟩) 0 ⟨18488⟩ 103681

def event103683 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18676⟩⟩) 1 ⟨18675⟩ 103666

def event103684 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨18676⟩⟩) (.sum [.predecessor 0 103682 .coefficient, .predecessor 1 103683 .coefficient])

def exact103685RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6709⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6711⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6713⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6715⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6717⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6719⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6721⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6723⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6725⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6727⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6729⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6731⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6733⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6735⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6737⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6739⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6741⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6743⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6744⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15258⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15300⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15356⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15622⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15741⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15860⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15979⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16098⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16301⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16672⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16791⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17078⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17113⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17302⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17897⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨18163⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨18198⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨18303⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨18485⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact103685RawTermsValid :
    exact103685RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event103685 : Event := .resultExact (⟨.program ⟨214⟩, ⟨18676⟩⟩) exact103685RawTerms .large 103684 .exactZero (none)

def event103686 : Event := .preFoldPolynomial 103685 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6709⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6711⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6713⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6715⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6717⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6719⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6721⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6723⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6725⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6727⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6729⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6731⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6733⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6735⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6737⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6739⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6741⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6743⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6744⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15258⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15300⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15356⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15622⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15741⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15860⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15979⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16098⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16301⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16672⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16791⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17078⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17113⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17302⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17897⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨18163⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨18198⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨18303⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨18485⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩] .exactZero none

def exact103687RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6709⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6711⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6713⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6715⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6717⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6719⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6721⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6723⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6725⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6727⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6729⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6731⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6733⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6735⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6737⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6739⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6741⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6743⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6744⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15258⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15300⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15356⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15622⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15741⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15860⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15979⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16098⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16301⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16672⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16791⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17078⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17113⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17302⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17897⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨18163⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨18198⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨18303⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨18485⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

def event103687 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨18676⟩⟩) 103686 exact103687RawTerms .large 103684 .exactZero (none)

def event103688 : Event := .specializationComputed (⟨.program ⟨214⟩, ⟨18313⟩⟩) ⟨⟨1⟩, ⟨67⟩, ⟨109⟩⟩ ⟨102350, 103688⟩

def event103689 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨18551⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18548⟩⟩]⟩) (1) 0 2 (.universal 103688 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18548⟩⟩]⟩) (none) 103687)

def event103690 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18551⟩⟩, .relation 103689 18, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6744⟩⟩]⟩, (1)⟩)

def event103691 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18551⟩⟩, .relation 103689 17, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6743⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (-1)⟩)

def event103692 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18551⟩⟩, .relation 103689 16, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6741⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (-1)⟩)

def event103693 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18551⟩⟩, .relation 103689 15, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6739⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (-1)⟩)

def event103694 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18551⟩⟩, .relation 103689 14, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6737⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (-1)⟩)

def event103695 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18551⟩⟩, .relation 103689 13, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6735⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (-1)⟩)

def event103696 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18551⟩⟩, .relation 103689 12, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6733⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (-1)⟩)

def event103697 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18551⟩⟩, .relation 103689 11, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6731⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (-1)⟩)

def event103698 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18551⟩⟩, .relation 103689 10, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6729⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (-1)⟩)

def event103699 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18551⟩⟩, .relation 103689 9, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6727⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (-1)⟩)

def event103700 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18551⟩⟩, .relation 103689 8, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6725⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (-1)⟩)

def event103701 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18551⟩⟩, .relation 103689 7, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6723⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (-1)⟩)

def event103702 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18551⟩⟩, .relation 103689 6, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6721⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (-1)⟩)

def event103703 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18551⟩⟩, .relation 103689 5, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6719⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (-1)⟩)

def event103704 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18551⟩⟩, .relation 103689 4, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6717⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (-1)⟩)

def event103705 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18551⟩⟩, .relation 103689 3, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6715⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (-1)⟩)

def event103706 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18551⟩⟩, .relation 103689 2, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6713⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (-1)⟩)

def event103707 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18551⟩⟩, .relation 103689 1, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6711⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (-1)⟩)

def event103708 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18551⟩⟩, .relation 103689 0, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6709⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (-1)⟩)

def event103709 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18551⟩⟩, .relation 103689 34, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨18163⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (1)⟩)

def event103710 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18551⟩⟩, .relation 103689 30, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨17078⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (1)⟩)

def event103711 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18551⟩⟩, .relation 103689 29, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16791⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (1)⟩)

def event103712 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18551⟩⟩, .relation 103689 28, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16672⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (1)⟩)

def event103713 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18551⟩⟩, .relation 103689 35, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨18198⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (1)⟩)

def event103714 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18551⟩⟩, .relation 103689 33, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨17897⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (1)⟩)

def event103715 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18551⟩⟩, .relation 103689 31, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨17113⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (1)⟩)

def event103716 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18551⟩⟩, .relation 103689 27, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16301⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (1)⟩)

def event103717 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18551⟩⟩, .relation 103689 36, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨18303⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (1)⟩)

def event103718 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18551⟩⟩, .relation 103689 26, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16098⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (1)⟩)

def event103719 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18551⟩⟩, .relation 103689 25, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15979⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (1)⟩)

def event103720 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18551⟩⟩, .relation 103689 24, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15860⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (1)⟩)

def event103721 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18551⟩⟩, .relation 103689 23, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15741⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (1)⟩)

def event103722 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18551⟩⟩, .relation 103689 22, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15622⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (1)⟩)

def event103723 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18551⟩⟩, .relation 103689 32, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨17302⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (1)⟩)

def event103724 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18551⟩⟩, .relation 103689 21, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15356⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (1)⟩)

def event103725 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18551⟩⟩, .relation 103689 20, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15300⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (1)⟩)

def event103726 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18551⟩⟩, .relation 103689 19, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15258⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (1)⟩)

def event103727 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18551⟩⟩, .relation 103689 37, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨18485⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def exact103728RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6709⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6711⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6713⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6715⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6717⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6719⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6721⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6723⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6725⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6727⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6729⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6731⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6733⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6735⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6737⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6739⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6741⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6743⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6744⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15258⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15300⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15356⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15622⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15741⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15860⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15979⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16098⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16301⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16672⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16791⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨17078⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨17113⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨17302⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨17897⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨18163⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨18198⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨18303⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨18485⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact103728RawTermsValid :
    exact103728RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event103728 : Event := .resultExact (⟨.program ⟨214⟩, ⟨18551⟩⟩) exact103728RawTerms .large 102346 (.finite 1811303510016) (some (102348))

def event103729 : Event := .predecessor (⟨.program ⟨214⟩, ⟨30067⟩⟩) 0 ⟨18551⟩ 103728

def event103730 : Event := .predecessor (⟨.program ⟨214⟩, ⟨30067⟩⟩) 1 ⟨30066⟩ 102336

def event103731 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨30067⟩⟩) (.sum [.predecessor 0 103729 .coefficient, .predecessor 1 103730 .coefficient])

def event103732 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30067⟩⟩, .operator (⟨103728, 17⟩, ⟨102336, 17⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6743⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (1)⟩)

def event103733 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30067⟩⟩, .operator (⟨103728, 34⟩, ⟨102336, 33⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨18163⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (-1)⟩)

def event103734 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30067⟩⟩, .operator (⟨103728, 16⟩, ⟨102336, 16⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6741⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (1)⟩)

def event103735 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30067⟩⟩, .operator (⟨103728, 30⟩, ⟨102336, 29⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨17078⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (-1)⟩)

def event103736 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30067⟩⟩, .operator (⟨103728, 15⟩, ⟨102336, 15⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6739⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (1)⟩)

def event103737 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30067⟩⟩, .operator (⟨103728, 29⟩, ⟨102336, 28⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16791⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (-1)⟩)

def event103738 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30067⟩⟩, .operator (⟨103728, 14⟩, ⟨102336, 14⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6737⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (1)⟩)

def event103739 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30067⟩⟩, .operator (⟨103728, 28⟩, ⟨102336, 27⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16672⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (-1)⟩)

def event103740 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30067⟩⟩, .operator (⟨103728, 13⟩, ⟨102336, 13⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6735⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (1)⟩)

def event103741 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30067⟩⟩, .operator (⟨103728, 35⟩, ⟨102336, 34⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨18198⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (-1)⟩)

def event103742 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30067⟩⟩, .operator (⟨103728, 12⟩, ⟨102336, 12⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6733⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (1)⟩)

def event103743 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30067⟩⟩, .operator (⟨103728, 33⟩, ⟨102336, 32⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨17897⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (-1)⟩)

def event103744 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30067⟩⟩, .operator (⟨103728, 11⟩, ⟨102336, 11⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6731⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (1)⟩)

def event103745 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30067⟩⟩, .operator (⟨103728, 31⟩, ⟨102336, 30⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨17113⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (-1)⟩)

def event103746 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30067⟩⟩, .operator (⟨103728, 10⟩, ⟨102336, 10⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6729⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (1)⟩)

def event103747 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30067⟩⟩, .operator (⟨103728, 27⟩, ⟨102336, 26⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16301⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (-1)⟩)

def event103748 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30067⟩⟩, .operator (⟨103728, 9⟩, ⟨102336, 9⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6727⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (1)⟩)

def event103749 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30067⟩⟩, .operator (⟨103728, 36⟩, ⟨102336, 35⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨18303⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (-1)⟩)

def event103750 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30067⟩⟩, .operator (⟨103728, 8⟩, ⟨102336, 8⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6725⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (1)⟩)

def event103751 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30067⟩⟩, .operator (⟨103728, 26⟩, ⟨102336, 25⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16098⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (-1)⟩)

def event103752 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30067⟩⟩, .operator (⟨103728, 7⟩, ⟨102336, 7⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6723⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (1)⟩)

def event103753 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30067⟩⟩, .operator (⟨103728, 25⟩, ⟨102336, 24⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15979⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (-1)⟩)

def event103754 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30067⟩⟩, .operator (⟨103728, 6⟩, ⟨102336, 6⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6721⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (1)⟩)

def event103755 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30067⟩⟩, .operator (⟨103728, 24⟩, ⟨102336, 23⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15860⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (-1)⟩)

def event103756 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30067⟩⟩, .operator (⟨103728, 5⟩, ⟨102336, 5⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6719⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (1)⟩)

def event103757 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30067⟩⟩, .operator (⟨103728, 23⟩, ⟨102336, 22⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15741⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (-1)⟩)

def event103758 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30067⟩⟩, .operator (⟨103728, 4⟩, ⟨102336, 4⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6717⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (1)⟩)

def event103759 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30067⟩⟩, .operator (⟨103728, 22⟩, ⟨102336, 21⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15622⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (-1)⟩)

def event103760 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30067⟩⟩, .operator (⟨103728, 3⟩, ⟨102336, 3⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6715⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (1)⟩)

def event103761 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30067⟩⟩, .operator (⟨103728, 32⟩, ⟨102336, 31⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨17302⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (-1)⟩)

def event103762 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30067⟩⟩, .operator (⟨103728, 2⟩, ⟨102336, 2⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6713⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (1)⟩)

def event103763 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30067⟩⟩, .operator (⟨103728, 21⟩, ⟨102336, 20⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15356⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (-1)⟩)

def event103764 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30067⟩⟩, .operator (⟨103728, 1⟩, ⟨102336, 1⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6711⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (1)⟩)

def event103765 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30067⟩⟩, .operator (⟨103728, 20⟩, ⟨102336, 19⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15300⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (-1)⟩)

def event103766 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30067⟩⟩, .operator (⟨103728, 0⟩, ⟨102336, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6709⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (1)⟩)

def event103767 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30067⟩⟩, .operator (⟨103728, 19⟩, ⟨102336, 18⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15258⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (-1)⟩)

def event103768 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨30067⟩⟩) (.sum [.result 103728 .summary, .result 102336 .summary])

def exact103769RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6744⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨18485⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact103769RawTermsValid :
    exact103769RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event103769 : Event := .resultExact (⟨.program ⟨214⟩, ⟨30067⟩⟩) exact103769RawTerms .large 103731 (.finite 85361036953731455419885957120) (some (103768))

def event103770 : Event := .predecessor (⟨.program ⟨214⟩, ⟨30068⟩⟩) 0 ⟨30067⟩ 103769

def event103771 : Event := .predecessor (⟨.program ⟨214⟩, ⟨30068⟩⟩) 1 ⟨6652⟩ 5499

def event103772 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨30068⟩⟩) (.product (.predecessor 0 103770 .coefficient) (.predecessor 1 103771 .coefficient) (⟨false, false, none, none, none⟩))

def event103773 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨30068⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨6651⟩⟩]⟩) [⟨.result 5495 .coefficient, false, none⟩])

def event103774 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨30068⟩⟩) (.product (.result 103769 .summary) (.transfer 103773) (⟨false, false, none, none, none⟩))

def event103775 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30068⟩⟩, .operator (⟨103769, 0⟩, ⟨5499, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6744⟩⟩, ⟨.program ⟨214⟩, ⟨6651⟩⟩]⟩, (1)⟩)

def event103776 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30068⟩⟩, .operator (⟨103769, 1⟩, ⟨5499, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨18485⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨6651⟩⟩]⟩, (-1)⟩)

def event103777 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨30068⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨18485⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨6651⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨6651⟩⟩) ⟨6597⟩ 5492)

def event103778 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30068⟩⟩, .relation 103777 0, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6410⟩⟩, ⟨.program ⟨214⟩, ⟨18485⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def exact103779RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6744⟩⟩, ⟨.program ⟨214⟩, ⟨6651⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6410⟩⟩, ⟨.program ⟨214⟩, ⟨18485⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact103779RawTermsValid :
    exact103779RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event103779 : Event := .resultExact (⟨.program ⟨214⟩, ⟨30068⟩⟩) exact103779RawTerms .large 103772 (.finite 313276371396785701094268180805713920) (some (103774))

def event103780 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24782⟩⟩) 0 ⟨6689⟩ 5477

def event103781 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24782⟩⟩) 1 ⟨24781⟩ 94353

def event103782 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24782⟩⟩) (.authority (.operator))

def exact103783RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨24782⟩⟩]⟩, (1)⟩]

theorem exact103783RawTermsValid :
    exact103783RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event103783 : Event := .resultExact (⟨.program ⟨214⟩, ⟨24782⟩⟩) exact103783RawTerms .large 103782 .exactZero (none)

def event103784 : Event := .predecessor (⟨.program ⟨214⟩, ⟨30054⟩⟩) 0 ⟨24782⟩ 103783

def event103785 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨30054⟩⟩) (.authority (.operator))

def exact103786RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨30054⟩⟩]⟩, (1)⟩]

theorem exact103786RawTermsValid :
    exact103786RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event103786 : Event := .resultExact (⟨.program ⟨214⟩, ⟨30054⟩⟩) exact103786RawTerms (.finite 8192) 103785 .exactZero (none)

def event103787 : Event := .predecessor (⟨.program ⟨214⟩, ⟨30056⟩⟩) 0 ⟨25747⟩ 94624

def event103788 : Event := .predecessor (⟨.program ⟨214⟩, ⟨30056⟩⟩) 1 ⟨30054⟩ 103786

def event103789 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨30056⟩⟩) (.product (.predecessor 0 103787 .coefficient) (.predecessor 1 103788 .coefficient) (⟨false, false, none, none, none⟩))

def event103790 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨30056⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨30054⟩⟩]⟩) [⟨.result 103786 .coefficient, false, none⟩])

def event103791 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨30056⟩⟩) (.product (.result 94624 .summary) (.transfer 103790) (⟨false, false, none, none, none⟩))

def event103792 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30056⟩⟩, .operator (⟨94624, 0⟩, ⟨103786, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6707⟩⟩, ⟨.program ⟨214⟩, ⟨30054⟩⟩]⟩, (1)⟩)

def event103793 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30056⟩⟩, .operator (⟨94624, 1⟩, ⟨103786, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨17001⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨30054⟩⟩]⟩, (-1)⟩)

def event103794 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨30056⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨17001⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨30054⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨30054⟩⟩) ⟨24782⟩ 103783)

def event103795 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30056⟩⟩, .relation 103794 0, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨17001⟩⟩], [⟨.program ⟨214⟩, ⟨24782⟩⟩]⟩, (-1)⟩)

def exact103796RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6707⟩⟩, ⟨.program ⟨214⟩, ⟨30054⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨17001⟩⟩], [⟨.program ⟨214⟩, ⟨24782⟩⟩]⟩, (-1)⟩]

theorem exact103796RawTermsValid :
    exact103796RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event103796 : Event := .resultExact (⟨.program ⟨214⟩, ⟨30056⟩⟩) exact103796RawTerms .large 103789 (.finite 1292539133473715126272) (some (103791))

def event103797 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22757⟩⟩) 0 ⟨17002⟩ 4583

def event103798 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨22757⟩⟩) (.authority (.relationPreimageSource ⟨64⟩))

def exact103799RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨22757⟩⟩]⟩, (1)⟩]

theorem exact103799RawTermsValid :
    exact103799RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event103799 : Event := .resultExact (⟨.program ⟨214⟩, ⟨22757⟩⟩) exact103799RawTerms (.finite 136065468) 103798 .exactZero (none)

def event103800 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22759⟩⟩) 0 ⟨22757⟩ 103799

def event103801 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22759⟩⟩) 1 ⟨2348⟩ 4

def event103802 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨22759⟩⟩) (.scale (.predecessor 0 103800 .coefficient) (.value (.predecessor 1 103801 .coefficient)))

def exact103803RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨22757⟩⟩]⟩, (1)⟩]

theorem exact103803RawTermsValid :
    exact103803RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event103803 : Event := .resultExact (⟨.program ⟨214⟩, ⟨22759⟩⟩) exact103803RawTerms (.finite 136065468) 103802 .exactZero (none)

def event103804 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22760⟩⟩) 0 ⟨5509⟩ 94462

def event103805 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22760⟩⟩) 1 ⟨22759⟩ 103803

def event103806 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨22760⟩⟩) (.product (.predecessor 0 103804 .coefficient) (.predecessor 1 103805 .coefficient) (⟨false, false, none, none, none⟩))

def event103807 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨22760⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨22757⟩⟩]⟩) [⟨.result 103799 .coefficient, false, none⟩])

def event103808 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨22760⟩⟩) (.product (.result 94462 .summary) (.transfer 103807) (⟨false, false, none, none, none⟩))

def event103809 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨22760⟩⟩, .operator (⟨94462, 0⟩, ⟨103803, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22757⟩⟩]⟩, (1)⟩)

def event103810 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨22758⟩⟩)

def event103811 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event103812 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event103813 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event103814 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event103815 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 103814

def event103816 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 103812

def event103817 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 103815 .coefficient) (.value (.predecessor 1 103816 .coefficient)))

def event103818 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event103819 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13326⟩⟩) 0 ⟨5503⟩ 103818

def event103820 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13326⟩⟩) (.authority (.programFamilyFact))

def exact103821RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨13326⟩⟩], []⟩, (1)⟩]

theorem exact103821RawTermsValid :
    exact103821RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event103821 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13326⟩⟩) exact103821RawTerms (.finite 60) 103820 .exactZero (none)

def event103822 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10330⟩⟩) 0 ⟨5503⟩ 103818

def event103823 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10330⟩⟩) (.authority (.programFamilyFact))

def exact103824RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨10330⟩⟩], []⟩, (1)⟩]

theorem exact103824RawTermsValid :
    exact103824RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event103824 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10330⟩⟩) exact103824RawTerms (.finite 60) 103823 .exactZero (none)

def event103825 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13327⟩⟩) 0 ⟨10330⟩ 103824

def event103826 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13327⟩⟩) 1 ⟨13326⟩ 103821

def event103827 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13327⟩⟩) (.product (.predecessor 0 103825 .coefficient) (.predecessor 1 103826 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event103828 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13327⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨10330⟩⟩, ⟨.program ⟨214⟩, ⟨13326⟩⟩], []⟩) [⟨.result 103824 .coefficient, true, some 1⟩, ⟨.result 103821 .coefficient, true, some 1⟩])

def event103829 : Event := .survivorFold (1) 103828

def exact103830RawTerms : List Term := []

theorem exact103830RawTermsValid :
    exact103830RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event103830 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13327⟩⟩) exact103830RawTerms (.finite 3600) 103827 (.finite 3600) (some (103828))

def event103831 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13328⟩⟩) 0 ⟨13327⟩ 103830

def event103832 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13328⟩⟩) (.identity (.predecessor 0 103831 .coefficient))

def event103833 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨13328⟩⟩) (.finite 3600)

def event103834 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17001⟩⟩) 0 ⟨13328⟩ 103833

def event103835 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨17001⟩⟩) (.authority (.programFamilyFact))

def exact103836RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨17001⟩⟩], []⟩, (1)⟩]

theorem exact103836RawTermsValid :
    exact103836RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event103836 : Event := .resultExact (⟨.program ⟨214⟩, ⟨17001⟩⟩) exact103836RawTerms (.finite 60) 103835 .exactZero (none)

def event103837 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17002⟩⟩) 0 ⟨17001⟩ 103836

def event103838 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨17002⟩⟩) (.identity (.predecessor 0 103837 .coefficient))

def event103839 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨17002⟩⟩) (.finite 60)

def event103840 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22757⟩⟩) 0 ⟨17002⟩ 103839

def event103841 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨22757⟩⟩) (.authority (.relationPreimageSource ⟨64⟩))

def exact103842RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨22757⟩⟩]⟩, (1)⟩]

theorem exact103842RawTermsValid :
    exact103842RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event103842 : Event := .resultExact (⟨.program ⟨214⟩, ⟨22757⟩⟩) exact103842RawTerms (.finite 136065468) 103841 .exactZero (none)

def event103843 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6⟩⟩) (.authority (.operator))

def exact103844RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩]⟩, (1)⟩]

theorem exact103844RawTermsValid :
    exact103844RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event103844 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6⟩⟩) exact103844RawTerms .large 103843 .exactZero (none)

def event103845 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22758⟩⟩) 0 ⟨6⟩ 103844

def event103846 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22758⟩⟩) 1 ⟨22757⟩ 103842

def event103847 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨22758⟩⟩) (.product (.predecessor 0 103845 .coefficient) (.predecessor 1 103846 .coefficient) (⟨false, false, none, none, none⟩))

def event103848 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨22758⟩⟩, .operator (⟨103844, 0⟩, ⟨103842, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22757⟩⟩]⟩, (1)⟩)

def exact103849RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22757⟩⟩]⟩, (1)⟩]

theorem exact103849RawTermsValid :
    exact103849RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event103849 : Event := .resultExact (⟨.program ⟨214⟩, ⟨22758⟩⟩) exact103849RawTerms .large 103847 .exactZero (none)

def event103850 : Event := .preFoldPolynomial 103849 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22757⟩⟩]⟩, (1)⟩] .exactZero none

def exact103851RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22757⟩⟩]⟩, (1)⟩]

def event103851 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨22758⟩⟩) 103850 exact103851RawTerms .large 103847 .exactZero (none)

def event103852 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨30060⟩⟩)

def event103853 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event103854 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event103855 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event103856 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event103857 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 103856

def event103858 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 103854

def event103859 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 103857 .coefficient) (.value (.predecessor 1 103858 .coefficient)))

def event103860 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event103861 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13326⟩⟩) 0 ⟨5503⟩ 103860

def event103862 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13326⟩⟩) (.authority (.programFamilyFact))

def exact103863RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨13326⟩⟩], []⟩, (1)⟩]

theorem exact103863RawTermsValid :
    exact103863RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event103863 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13326⟩⟩) exact103863RawTerms (.finite 60) 103862 .exactZero (none)

def event103864 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10330⟩⟩) 0 ⟨5503⟩ 103860

def event103865 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10330⟩⟩) (.authority (.programFamilyFact))

def exact103866RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨10330⟩⟩], []⟩, (1)⟩]

theorem exact103866RawTermsValid :
    exact103866RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event103866 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10330⟩⟩) exact103866RawTerms (.finite 60) 103865 .exactZero (none)

def event103867 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13327⟩⟩) 0 ⟨10330⟩ 103866

def event103868 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13327⟩⟩) 1 ⟨13326⟩ 103863

def event103869 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13327⟩⟩) (.product (.predecessor 0 103867 .coefficient) (.predecessor 1 103868 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event103870 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨13327⟩⟩, .operator (⟨103866, 0⟩, ⟨103863, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨10330⟩⟩, ⟨.program ⟨214⟩, ⟨13326⟩⟩], []⟩, (1)⟩)

def exact103871RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨10330⟩⟩, ⟨.program ⟨214⟩, ⟨13326⟩⟩], []⟩, (1)⟩]

theorem exact103871RawTermsValid :
    exact103871RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event103871 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13327⟩⟩) exact103871RawTerms (.finite 3600) 103869 .exactZero (none)

def event103872 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13328⟩⟩) 0 ⟨13327⟩ 103871

def event103873 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13328⟩⟩) (.identity (.predecessor 0 103872 .coefficient))

def event103874 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨13328⟩⟩) (.finite 3600)

def event103875 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17001⟩⟩) 0 ⟨13328⟩ 103874

def event103876 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨17001⟩⟩) (.authority (.programFamilyFact))

def exact103877RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨17001⟩⟩], []⟩, (1)⟩]

theorem exact103877RawTermsValid :
    exact103877RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event103877 : Event := .resultExact (⟨.program ⟨214⟩, ⟨17001⟩⟩) exact103877RawTerms (.finite 60) 103876 .exactZero (none)

def event103878 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17002⟩⟩) 0 ⟨17001⟩ 103877

def event103879 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨17002⟩⟩) (.identity (.predecessor 0 103878 .coefficient))

def event103880 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨17002⟩⟩) (.finite 60)

def event103881 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24781⟩⟩) 0 ⟨17002⟩ 103880

def event103882 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24781⟩⟩) (.authority (.programFamilyFact))

def event103883 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨24781⟩⟩) (.finite 3720)

def event103884 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨6689⟩⟩) .missing

def event103885 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24782⟩⟩) 0 ⟨6689⟩ 103884

def event103886 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24782⟩⟩) 1 ⟨24781⟩ 103883

def event103887 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24782⟩⟩) (.authority (.operator))

def exact103888RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨24782⟩⟩]⟩, (1)⟩]

theorem exact103888RawTermsValid :
    exact103888RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event103888 : Event := .resultExact (⟨.program ⟨214⟩, ⟨24782⟩⟩) exact103888RawTerms .large 103887 .exactZero (none)

def event103889 : Event := .predecessor (⟨.program ⟨214⟩, ⟨30054⟩⟩) 0 ⟨24782⟩ 103888

def event103890 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨30054⟩⟩) (.authority (.operator))

def exact103891RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨30054⟩⟩]⟩, (1)⟩]

theorem exact103891RawTermsValid :
    exact103891RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event103891 : Event := .resultExact (⟨.program ⟨214⟩, ⟨30054⟩⟩) exact103891RawTerms (.finite 8192) 103890 .exactZero (none)

def event103892 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨110⟩⟩) (.authority (.operator))

def event103893 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨110⟩⟩) .exactZero

def event103894 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17043⟩⟩) 0 ⟨17002⟩ 103880

def event103895 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17043⟩⟩) 1 ⟨110⟩ 103893

def event103896 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨17043⟩⟩) (.sum [.predecessor 0 103894 .coefficient, .predecessor 1 103895 .coefficient])

def event103897 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨17043⟩⟩) (.finite 60)

def event103898 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17044⟩⟩) 0 ⟨17043⟩ 103897

def event103899 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨17044⟩⟩) (.identity (.predecessor 0 103898 .coefficient))

def exact103900RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨17001⟩⟩], []⟩, (1)⟩]

theorem exact103900RawTermsValid :
    exact103900RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event103900 : Event := .resultExact (⟨.program ⟨214⟩, ⟨17044⟩⟩) exact103900RawTerms (.finite 60) 103899 .exactZero (none)

def event103901 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6544⟩⟩) (.authority (.factStore))

def exact103902RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact103902RawTermsValid :
    exact103902RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event103902 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6544⟩⟩) exact103902RawTerms .large 103901 .exactZero (none)

def event103903 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17045⟩⟩) 0 ⟨6544⟩ 103902

def event103904 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17045⟩⟩) 1 ⟨17044⟩ 103900

def event103905 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨17045⟩⟩) (.product (.predecessor 0 103903 .coefficient) (.predecessor 1 103904 .coefficient) (⟨false, false, none, none, none⟩))

def event103906 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨17045⟩⟩, .operator (⟨103902, 0⟩, ⟨103900, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨17001⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact103907RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨17001⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact103907RawTermsValid :
    exact103907RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event103907 : Event := .resultExact (⟨.program ⟨214⟩, ⟨17045⟩⟩) exact103907RawTerms .large 103905 .exactZero (none)

def event103908 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6707⟩⟩) 0 ⟨6689⟩ 103884

def event103909 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6707⟩⟩) (.authority (.operator))

def exact103910RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6707⟩⟩]⟩, (1)⟩]

theorem exact103910RawTermsValid :
    exact103910RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event103910 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6707⟩⟩) exact103910RawTerms .large 103909 .exactZero (none)

def event103911 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17046⟩⟩) 0 ⟨6707⟩ 103910

def event103912 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17046⟩⟩) 1 ⟨17045⟩ 103907

def event103913 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨17046⟩⟩) (.sum [.predecessor 0 103911 .coefficient, .predecessor 1 103912 .coefficient])

def exact103914RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6707⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17001⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact103914RawTermsValid :
    exact103914RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event103914 : Event := .resultExact (⟨.program ⟨214⟩, ⟨17046⟩⟩) exact103914RawTerms .large 103913 .exactZero (none)

def event103915 : Event := .predecessor (⟨.program ⟨214⟩, ⟨30055⟩⟩) 0 ⟨17046⟩ 103914

def event103916 : Event := .predecessor (⟨.program ⟨214⟩, ⟨30055⟩⟩) 1 ⟨30054⟩ 103891

def event103917 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨30055⟩⟩) (.product (.predecessor 0 103915 .coefficient) (.predecessor 1 103916 .coefficient) (⟨false, false, none, none, none⟩))

def event103918 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30055⟩⟩, .operator (⟨103914, 0⟩, ⟨103891, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6707⟩⟩, ⟨.program ⟨214⟩, ⟨30054⟩⟩]⟩, (1)⟩)

def event103919 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30055⟩⟩, .operator (⟨103914, 1⟩, ⟨103891, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨17001⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨30054⟩⟩]⟩, (-1)⟩)

def event103920 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨30055⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨17001⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨30054⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨30054⟩⟩) ⟨24782⟩ 103888)

def event103921 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30055⟩⟩, .relation 103920 0, ⟨[⟨.program ⟨214⟩, ⟨17001⟩⟩], [⟨.program ⟨214⟩, ⟨24782⟩⟩]⟩, (-1)⟩)

def exact103922RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6707⟩⟩, ⟨.program ⟨214⟩, ⟨30054⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17001⟩⟩], [⟨.program ⟨214⟩, ⟨24782⟩⟩]⟩, (-1)⟩]

theorem exact103922RawTermsValid :
    exact103922RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event103922 : Event := .resultExact (⟨.program ⟨214⟩, ⟨30055⟩⟩) exact103922RawTerms .large 103917 .exactZero (none)

def event103923 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18114⟩⟩) 0 ⟨17002⟩ 103880

def event103924 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨18114⟩⟩) (.authority (.programFamilyFact))

def exact103925RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨18114⟩⟩], []⟩, (1)⟩]

theorem exact103925RawTermsValid :
    exact103925RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event103925 : Event := .resultExact (⟨.program ⟨214⟩, ⟨18114⟩⟩) exact103925RawTerms (.finite 60) 103924 .exactZero (none)

def event103926 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18116⟩⟩) 0 ⟨6544⟩ 103902

def event103927 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18116⟩⟩) 1 ⟨18114⟩ 103925

def event103928 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨18116⟩⟩) (.product (.predecessor 0 103926 .coefficient) (.predecessor 1 103927 .coefficient) (⟨false, true, none, none, some 1⟩))

def event103929 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18116⟩⟩, .operator (⟨103902, 0⟩, ⟨103925, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨18114⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact103930RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨18114⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact103930RawTermsValid :
    exact103930RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event103930 : Event := .resultExact (⟨.program ⟨214⟩, ⟨18116⟩⟩) exact103930RawTerms .large 103928 .exactZero (none)

def event103931 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6742⟩⟩) 0 ⟨6689⟩ 103884

def event103932 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6742⟩⟩) (.authority (.operator))

def exact103933RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6742⟩⟩]⟩, (1)⟩]

theorem exact103933RawTermsValid :
    exact103933RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event103933 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6742⟩⟩) exact103933RawTerms .large 103932 .exactZero (none)

def event103934 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18117⟩⟩) 0 ⟨6742⟩ 103933

def event103935 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18117⟩⟩) 1 ⟨18116⟩ 103930

def eventLeaf6480 : Array AnnotatedEvent := #[
  { event := event103680
    frameStart := 102927 },
  { event := event103681
    frameStart := 102927 },
  { event := event103682
    frameStart := 102927 },
  { event := event103683
    frameStart := 102927 },
  { event := event103684
    frameStart := 102927 },
  { event := event103685
    frameStart := 102927 },
  { event := event103686
    frameStart := 102927 },
  { event := event103687
    frameStart := 102927 },
  { event := event103688
    frameStart := 0 },
  { event := event103689
    frameStart := 0 },
  { event := event103690
    frameStart := 0 },
  { event := event103691
    frameStart := 0 },
  { event := event103692
    frameStart := 0 },
  { event := event103693
    frameStart := 0 },
  { event := event103694
    frameStart := 0 },
  { event := event103695
    frameStart := 0 }
]

def eventLeaf6481 : Array AnnotatedEvent := #[
  { event := event103696
    frameStart := 0 },
  { event := event103697
    frameStart := 0 },
  { event := event103698
    frameStart := 0 },
  { event := event103699
    frameStart := 0 },
  { event := event103700
    frameStart := 0 },
  { event := event103701
    frameStart := 0 },
  { event := event103702
    frameStart := 0 },
  { event := event103703
    frameStart := 0 },
  { event := event103704
    frameStart := 0 },
  { event := event103705
    frameStart := 0 },
  { event := event103706
    frameStart := 0 },
  { event := event103707
    frameStart := 0 },
  { event := event103708
    frameStart := 0 },
  { event := event103709
    frameStart := 0 },
  { event := event103710
    frameStart := 0 },
  { event := event103711
    frameStart := 0 }
]

def eventLeaf6482 : Array AnnotatedEvent := #[
  { event := event103712
    frameStart := 0 },
  { event := event103713
    frameStart := 0 },
  { event := event103714
    frameStart := 0 },
  { event := event103715
    frameStart := 0 },
  { event := event103716
    frameStart := 0 },
  { event := event103717
    frameStart := 0 },
  { event := event103718
    frameStart := 0 },
  { event := event103719
    frameStart := 0 },
  { event := event103720
    frameStart := 0 },
  { event := event103721
    frameStart := 0 },
  { event := event103722
    frameStart := 0 },
  { event := event103723
    frameStart := 0 },
  { event := event103724
    frameStart := 0 },
  { event := event103725
    frameStart := 0 },
  { event := event103726
    frameStart := 0 },
  { event := event103727
    frameStart := 0 }
]

def eventLeaf6483 : Array AnnotatedEvent := #[
  { event := event103728
    frameStart := 0 },
  { event := event103729
    frameStart := 0 },
  { event := event103730
    frameStart := 0 },
  { event := event103731
    frameStart := 0 },
  { event := event103732
    frameStart := 0 },
  { event := event103733
    frameStart := 0 },
  { event := event103734
    frameStart := 0 },
  { event := event103735
    frameStart := 0 },
  { event := event103736
    frameStart := 0 },
  { event := event103737
    frameStart := 0 },
  { event := event103738
    frameStart := 0 },
  { event := event103739
    frameStart := 0 },
  { event := event103740
    frameStart := 0 },
  { event := event103741
    frameStart := 0 },
  { event := event103742
    frameStart := 0 },
  { event := event103743
    frameStart := 0 }
]

def eventLeaf6484 : Array AnnotatedEvent := #[
  { event := event103744
    frameStart := 0 },
  { event := event103745
    frameStart := 0 },
  { event := event103746
    frameStart := 0 },
  { event := event103747
    frameStart := 0 },
  { event := event103748
    frameStart := 0 },
  { event := event103749
    frameStart := 0 },
  { event := event103750
    frameStart := 0 },
  { event := event103751
    frameStart := 0 },
  { event := event103752
    frameStart := 0 },
  { event := event103753
    frameStart := 0 },
  { event := event103754
    frameStart := 0 },
  { event := event103755
    frameStart := 0 },
  { event := event103756
    frameStart := 0 },
  { event := event103757
    frameStart := 0 },
  { event := event103758
    frameStart := 0 },
  { event := event103759
    frameStart := 0 }
]

def eventLeaf6485 : Array AnnotatedEvent := #[
  { event := event103760
    frameStart := 0 },
  { event := event103761
    frameStart := 0 },
  { event := event103762
    frameStart := 0 },
  { event := event103763
    frameStart := 0 },
  { event := event103764
    frameStart := 0 },
  { event := event103765
    frameStart := 0 },
  { event := event103766
    frameStart := 0 },
  { event := event103767
    frameStart := 0 },
  { event := event103768
    frameStart := 0 },
  { event := event103769
    frameStart := 0 },
  { event := event103770
    frameStart := 0 },
  { event := event103771
    frameStart := 0 },
  { event := event103772
    frameStart := 0 },
  { event := event103773
    frameStart := 0 },
  { event := event103774
    frameStart := 0 },
  { event := event103775
    frameStart := 0 }
]

def eventLeaf6486 : Array AnnotatedEvent := #[
  { event := event103776
    frameStart := 0 },
  { event := event103777
    frameStart := 0 },
  { event := event103778
    frameStart := 0 },
  { event := event103779
    frameStart := 0 },
  { event := event103780
    frameStart := 0 },
  { event := event103781
    frameStart := 0 },
  { event := event103782
    frameStart := 0 },
  { event := event103783
    frameStart := 0 },
  { event := event103784
    frameStart := 0 },
  { event := event103785
    frameStart := 0 },
  { event := event103786
    frameStart := 0 },
  { event := event103787
    frameStart := 0 },
  { event := event103788
    frameStart := 0 },
  { event := event103789
    frameStart := 0 },
  { event := event103790
    frameStart := 0 },
  { event := event103791
    frameStart := 0 }
]

def eventLeaf6487 : Array AnnotatedEvent := #[
  { event := event103792
    frameStart := 0 },
  { event := event103793
    frameStart := 0 },
  { event := event103794
    frameStart := 0 },
  { event := event103795
    frameStart := 0 },
  { event := event103796
    frameStart := 0 },
  { event := event103797
    frameStart := 0 },
  { event := event103798
    frameStart := 0 },
  { event := event103799
    frameStart := 0 },
  { event := event103800
    frameStart := 0 },
  { event := event103801
    frameStart := 0 },
  { event := event103802
    frameStart := 0 },
  { event := event103803
    frameStart := 0 },
  { event := event103804
    frameStart := 0 },
  { event := event103805
    frameStart := 0 },
  { event := event103806
    frameStart := 0 },
  { event := event103807
    frameStart := 0 }
]

def eventLeaf6488 : Array AnnotatedEvent := #[
  { event := event103808
    frameStart := 0 },
  { event := event103809
    frameStart := 0 },
  { event := event103810
    frameStart := 103810 },
  { event := event103811
    frameStart := 103810 },
  { event := event103812
    frameStart := 103810 },
  { event := event103813
    frameStart := 103810 },
  { event := event103814
    frameStart := 103810 },
  { event := event103815
    frameStart := 103810 },
  { event := event103816
    frameStart := 103810 },
  { event := event103817
    frameStart := 103810 },
  { event := event103818
    frameStart := 103810 },
  { event := event103819
    frameStart := 103810 },
  { event := event103820
    frameStart := 103810 },
  { event := event103821
    frameStart := 103810 },
  { event := event103822
    frameStart := 103810 },
  { event := event103823
    frameStart := 103810 }
]

def eventLeaf6489 : Array AnnotatedEvent := #[
  { event := event103824
    frameStart := 103810 },
  { event := event103825
    frameStart := 103810 },
  { event := event103826
    frameStart := 103810 },
  { event := event103827
    frameStart := 103810 },
  { event := event103828
    frameStart := 103810 },
  { event := event103829
    frameStart := 103810 },
  { event := event103830
    frameStart := 103810 },
  { event := event103831
    frameStart := 103810 },
  { event := event103832
    frameStart := 103810 },
  { event := event103833
    frameStart := 103810 },
  { event := event103834
    frameStart := 103810 },
  { event := event103835
    frameStart := 103810 },
  { event := event103836
    frameStart := 103810 },
  { event := event103837
    frameStart := 103810 },
  { event := event103838
    frameStart := 103810 },
  { event := event103839
    frameStart := 103810 }
]

def eventLeaf6490 : Array AnnotatedEvent := #[
  { event := event103840
    frameStart := 103810 },
  { event := event103841
    frameStart := 103810 },
  { event := event103842
    frameStart := 103810 },
  { event := event103843
    frameStart := 103810 },
  { event := event103844
    frameStart := 103810 },
  { event := event103845
    frameStart := 103810 },
  { event := event103846
    frameStart := 103810 },
  { event := event103847
    frameStart := 103810 },
  { event := event103848
    frameStart := 103810 },
  { event := event103849
    frameStart := 103810 },
  { event := event103850
    frameStart := 103810 },
  { event := event103851
    frameStart := 103810 },
  { event := event103852
    frameStart := 103852 },
  { event := event103853
    frameStart := 103852 },
  { event := event103854
    frameStart := 103852 },
  { event := event103855
    frameStart := 103852 }
]

def eventLeaf6491 : Array AnnotatedEvent := #[
  { event := event103856
    frameStart := 103852 },
  { event := event103857
    frameStart := 103852 },
  { event := event103858
    frameStart := 103852 },
  { event := event103859
    frameStart := 103852 },
  { event := event103860
    frameStart := 103852 },
  { event := event103861
    frameStart := 103852 },
  { event := event103862
    frameStart := 103852 },
  { event := event103863
    frameStart := 103852 },
  { event := event103864
    frameStart := 103852 },
  { event := event103865
    frameStart := 103852 },
  { event := event103866
    frameStart := 103852 },
  { event := event103867
    frameStart := 103852 },
  { event := event103868
    frameStart := 103852 },
  { event := event103869
    frameStart := 103852 },
  { event := event103870
    frameStart := 103852 },
  { event := event103871
    frameStart := 103852 }
]

def eventLeaf6492 : Array AnnotatedEvent := #[
  { event := event103872
    frameStart := 103852 },
  { event := event103873
    frameStart := 103852 },
  { event := event103874
    frameStart := 103852 },
  { event := event103875
    frameStart := 103852 },
  { event := event103876
    frameStart := 103852 },
  { event := event103877
    frameStart := 103852 },
  { event := event103878
    frameStart := 103852 },
  { event := event103879
    frameStart := 103852 },
  { event := event103880
    frameStart := 103852 },
  { event := event103881
    frameStart := 103852 },
  { event := event103882
    frameStart := 103852 },
  { event := event103883
    frameStart := 103852 },
  { event := event103884
    frameStart := 103852 },
  { event := event103885
    frameStart := 103852 },
  { event := event103886
    frameStart := 103852 },
  { event := event103887
    frameStart := 103852 }
]

def eventLeaf6493 : Array AnnotatedEvent := #[
  { event := event103888
    frameStart := 103852 },
  { event := event103889
    frameStart := 103852 },
  { event := event103890
    frameStart := 103852 },
  { event := event103891
    frameStart := 103852 },
  { event := event103892
    frameStart := 103852 },
  { event := event103893
    frameStart := 103852 },
  { event := event103894
    frameStart := 103852 },
  { event := event103895
    frameStart := 103852 },
  { event := event103896
    frameStart := 103852 },
  { event := event103897
    frameStart := 103852 },
  { event := event103898
    frameStart := 103852 },
  { event := event103899
    frameStart := 103852 },
  { event := event103900
    frameStart := 103852 },
  { event := event103901
    frameStart := 103852 },
  { event := event103902
    frameStart := 103852 },
  { event := event103903
    frameStart := 103852 }
]

def eventLeaf6494 : Array AnnotatedEvent := #[
  { event := event103904
    frameStart := 103852 },
  { event := event103905
    frameStart := 103852 },
  { event := event103906
    frameStart := 103852 },
  { event := event103907
    frameStart := 103852 },
  { event := event103908
    frameStart := 103852 },
  { event := event103909
    frameStart := 103852 },
  { event := event103910
    frameStart := 103852 },
  { event := event103911
    frameStart := 103852 },
  { event := event103912
    frameStart := 103852 },
  { event := event103913
    frameStart := 103852 },
  { event := event103914
    frameStart := 103852 },
  { event := event103915
    frameStart := 103852 },
  { event := event103916
    frameStart := 103852 },
  { event := event103917
    frameStart := 103852 },
  { event := event103918
    frameStart := 103852 },
  { event := event103919
    frameStart := 103852 }
]

def eventLeaf6495 : Array AnnotatedEvent := #[
  { event := event103920
    frameStart := 103852 },
  { event := event103921
    frameStart := 103852 },
  { event := event103922
    frameStart := 103852 },
  { event := event103923
    frameStart := 103852 },
  { event := event103924
    frameStart := 103852 },
  { event := event103925
    frameStart := 103852 },
  { event := event103926
    frameStart := 103852 },
  { event := event103927
    frameStart := 103852 },
  { event := event103928
    frameStart := 103852 },
  { event := event103929
    frameStart := 103852 },
  { event := event103930
    frameStart := 103852 },
  { event := event103931
    frameStart := 103852 },
  { event := event103932
    frameStart := 103852 },
  { event := event103933
    frameStart := 103852 },
  { event := event103934
    frameStart := 103852 },
  { event := event103935
    frameStart := 103852 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events405
