import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events404

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event103424 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨6689⟩⟩) .missing

def event103425 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18612⟩⟩) 0 ⟨6689⟩ 103424

def event103426 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18612⟩⟩) 1 ⟨18611⟩ 103423

def event103427 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨18612⟩⟩) (.authority (.operator))

def exact103428RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (1)⟩]

theorem exact103428RawTermsValid :
    exact103428RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event103428 : Event := .resultExact (⟨.program ⟨214⟩, ⟨18612⟩⟩) exact103428RawTerms .large 103427 .exactZero (none)

def event103429 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18674⟩⟩) 0 ⟨18612⟩ 103428

def event103430 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨18674⟩⟩) (.authority (.operator))

def exact103431RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (1)⟩]

theorem exact103431RawTermsValid :
    exact103431RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event103431 : Event := .resultExact (⟨.program ⟨214⟩, ⟨18674⟩⟩) exact103431RawTerms (.finite 8192) 103430 .exactZero (none)

def event103432 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨110⟩⟩) (.authority (.operator))

def event103433 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨110⟩⟩) .exactZero

def event103434 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18639⟩⟩) 0 ⟨18313⟩ 103420

def event103435 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18639⟩⟩) 1 ⟨110⟩ 103433

def event103436 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨18639⟩⟩) (.sum [.predecessor 0 103434 .coefficient, .predecessor 1 103435 .coefficient])

def event103437 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨18639⟩⟩) (.finite 1059)

def event103438 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18640⟩⟩) 0 ⟨18639⟩ 103437

def event103439 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨18640⟩⟩) (.identity (.predecessor 0 103438 .coefficient))

def exact103440RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15258⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15300⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15356⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15622⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15741⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15860⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15979⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16098⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16301⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16672⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16791⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17078⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17113⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17302⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17897⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨18163⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨18198⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨18303⟩⟩], []⟩, (1)⟩]

theorem exact103440RawTermsValid :
    exact103440RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event103440 : Event := .resultExact (⟨.program ⟨214⟩, ⟨18640⟩⟩) exact103440RawTerms (.finite 1059) 103439 .exactZero (none)

def event103441 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6544⟩⟩) (.authority (.factStore))

def exact103442RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact103442RawTermsValid :
    exact103442RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event103442 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6544⟩⟩) exact103442RawTerms .large 103441 .exactZero (none)

def event103443 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18641⟩⟩) 0 ⟨6544⟩ 103442

def event103444 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18641⟩⟩) 1 ⟨18640⟩ 103440

def event103445 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨18641⟩⟩) (.product (.predecessor 0 103443 .coefficient) (.predecessor 1 103444 .coefficient) (⟨false, false, none, none, none⟩))

def event103446 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18641⟩⟩, .operator (⟨103442, 0⟩, ⟨103440, 15⟩), ⟨[⟨.program ⟨214⟩, ⟨18163⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def event103447 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18641⟩⟩, .operator (⟨103442, 0⟩, ⟨103440, 11⟩), ⟨[⟨.program ⟨214⟩, ⟨17078⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def event103448 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18641⟩⟩, .operator (⟨103442, 0⟩, ⟨103440, 10⟩), ⟨[⟨.program ⟨214⟩, ⟨16791⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def event103449 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18641⟩⟩, .operator (⟨103442, 0⟩, ⟨103440, 9⟩), ⟨[⟨.program ⟨214⟩, ⟨16672⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def event103450 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18641⟩⟩, .operator (⟨103442, 0⟩, ⟨103440, 16⟩), ⟨[⟨.program ⟨214⟩, ⟨18198⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def event103451 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18641⟩⟩, .operator (⟨103442, 0⟩, ⟨103440, 14⟩), ⟨[⟨.program ⟨214⟩, ⟨17897⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def event103452 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18641⟩⟩, .operator (⟨103442, 0⟩, ⟨103440, 12⟩), ⟨[⟨.program ⟨214⟩, ⟨17113⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def event103453 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18641⟩⟩, .operator (⟨103442, 0⟩, ⟨103440, 8⟩), ⟨[⟨.program ⟨214⟩, ⟨16301⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def event103454 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18641⟩⟩, .operator (⟨103442, 0⟩, ⟨103440, 17⟩), ⟨[⟨.program ⟨214⟩, ⟨18303⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def event103455 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18641⟩⟩, .operator (⟨103442, 0⟩, ⟨103440, 7⟩), ⟨[⟨.program ⟨214⟩, ⟨16098⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def event103456 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18641⟩⟩, .operator (⟨103442, 0⟩, ⟨103440, 6⟩), ⟨[⟨.program ⟨214⟩, ⟨15979⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def event103457 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18641⟩⟩, .operator (⟨103442, 0⟩, ⟨103440, 5⟩), ⟨[⟨.program ⟨214⟩, ⟨15860⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def event103458 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18641⟩⟩, .operator (⟨103442, 0⟩, ⟨103440, 4⟩), ⟨[⟨.program ⟨214⟩, ⟨15741⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def event103459 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18641⟩⟩, .operator (⟨103442, 0⟩, ⟨103440, 3⟩), ⟨[⟨.program ⟨214⟩, ⟨15622⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def event103460 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18641⟩⟩, .operator (⟨103442, 0⟩, ⟨103440, 13⟩), ⟨[⟨.program ⟨214⟩, ⟨17302⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def event103461 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18641⟩⟩, .operator (⟨103442, 0⟩, ⟨103440, 2⟩), ⟨[⟨.program ⟨214⟩, ⟨15356⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def event103462 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18641⟩⟩, .operator (⟨103442, 0⟩, ⟨103440, 1⟩), ⟨[⟨.program ⟨214⟩, ⟨15300⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def event103463 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18641⟩⟩, .operator (⟨103442, 0⟩, ⟨103440, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨15258⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact103464RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15258⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15300⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15356⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15622⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15741⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15860⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15979⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16098⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16301⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16672⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16791⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17078⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17113⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17302⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17897⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨18163⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨18198⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨18303⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact103464RawTermsValid :
    exact103464RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event103464 : Event := .resultExact (⟨.program ⟨214⟩, ⟨18641⟩⟩) exact103464RawTerms .large 103445 .exactZero (none)

def event103465 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6743⟩⟩) 0 ⟨6689⟩ 103424

def event103466 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6743⟩⟩) (.authority (.operator))

def exact103467RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6743⟩⟩]⟩, (1)⟩]

theorem exact103467RawTermsValid :
    exact103467RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event103467 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6743⟩⟩) exact103467RawTerms .large 103466 .exactZero (none)

def event103468 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6741⟩⟩) 0 ⟨6689⟩ 103424

def event103469 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6741⟩⟩) (.authority (.operator))

def exact103470RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6741⟩⟩]⟩, (1)⟩]

theorem exact103470RawTermsValid :
    exact103470RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event103470 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6741⟩⟩) exact103470RawTerms .large 103469 .exactZero (none)

def event103471 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6739⟩⟩) 0 ⟨6689⟩ 103424

def event103472 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6739⟩⟩) (.authority (.operator))

def exact103473RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6739⟩⟩]⟩, (1)⟩]

theorem exact103473RawTermsValid :
    exact103473RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event103473 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6739⟩⟩) exact103473RawTerms .large 103472 .exactZero (none)

def event103474 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6737⟩⟩) 0 ⟨6689⟩ 103424

def event103475 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6737⟩⟩) (.authority (.operator))

def exact103476RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6737⟩⟩]⟩, (1)⟩]

theorem exact103476RawTermsValid :
    exact103476RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event103476 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6737⟩⟩) exact103476RawTerms .large 103475 .exactZero (none)

def event103477 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6735⟩⟩) 0 ⟨6689⟩ 103424

def event103478 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6735⟩⟩) (.authority (.operator))

def exact103479RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6735⟩⟩]⟩, (1)⟩]

theorem exact103479RawTermsValid :
    exact103479RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event103479 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6735⟩⟩) exact103479RawTerms .large 103478 .exactZero (none)

def event103480 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6733⟩⟩) 0 ⟨6689⟩ 103424

def event103481 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6733⟩⟩) (.authority (.operator))

def exact103482RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6733⟩⟩]⟩, (1)⟩]

theorem exact103482RawTermsValid :
    exact103482RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event103482 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6733⟩⟩) exact103482RawTerms .large 103481 .exactZero (none)

def event103483 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6731⟩⟩) 0 ⟨6689⟩ 103424

def event103484 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6731⟩⟩) (.authority (.operator))

def exact103485RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6731⟩⟩]⟩, (1)⟩]

theorem exact103485RawTermsValid :
    exact103485RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event103485 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6731⟩⟩) exact103485RawTerms .large 103484 .exactZero (none)

def event103486 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6729⟩⟩) 0 ⟨6689⟩ 103424

def event103487 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6729⟩⟩) (.authority (.operator))

def exact103488RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6729⟩⟩]⟩, (1)⟩]

theorem exact103488RawTermsValid :
    exact103488RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event103488 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6729⟩⟩) exact103488RawTerms .large 103487 .exactZero (none)

def event103489 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6727⟩⟩) 0 ⟨6689⟩ 103424

def event103490 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6727⟩⟩) (.authority (.operator))

def exact103491RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6727⟩⟩]⟩, (1)⟩]

theorem exact103491RawTermsValid :
    exact103491RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event103491 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6727⟩⟩) exact103491RawTerms .large 103490 .exactZero (none)

def event103492 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6725⟩⟩) 0 ⟨6689⟩ 103424

def event103493 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6725⟩⟩) (.authority (.operator))

def exact103494RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6725⟩⟩]⟩, (1)⟩]

theorem exact103494RawTermsValid :
    exact103494RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event103494 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6725⟩⟩) exact103494RawTerms .large 103493 .exactZero (none)

def event103495 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6723⟩⟩) 0 ⟨6689⟩ 103424

def event103496 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6723⟩⟩) (.authority (.operator))

def exact103497RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6723⟩⟩]⟩, (1)⟩]

theorem exact103497RawTermsValid :
    exact103497RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event103497 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6723⟩⟩) exact103497RawTerms .large 103496 .exactZero (none)

def event103498 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6721⟩⟩) 0 ⟨6689⟩ 103424

def event103499 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6721⟩⟩) (.authority (.operator))

def exact103500RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6721⟩⟩]⟩, (1)⟩]

theorem exact103500RawTermsValid :
    exact103500RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event103500 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6721⟩⟩) exact103500RawTerms .large 103499 .exactZero (none)

def event103501 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6719⟩⟩) 0 ⟨6689⟩ 103424

def event103502 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6719⟩⟩) (.authority (.operator))

def exact103503RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6719⟩⟩]⟩, (1)⟩]

theorem exact103503RawTermsValid :
    exact103503RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event103503 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6719⟩⟩) exact103503RawTerms .large 103502 .exactZero (none)

def event103504 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6717⟩⟩) 0 ⟨6689⟩ 103424

def event103505 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6717⟩⟩) (.authority (.operator))

def exact103506RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6717⟩⟩]⟩, (1)⟩]

theorem exact103506RawTermsValid :
    exact103506RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event103506 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6717⟩⟩) exact103506RawTerms .large 103505 .exactZero (none)

def event103507 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6715⟩⟩) 0 ⟨6689⟩ 103424

def event103508 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6715⟩⟩) (.authority (.operator))

def exact103509RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6715⟩⟩]⟩, (1)⟩]

theorem exact103509RawTermsValid :
    exact103509RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event103509 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6715⟩⟩) exact103509RawTerms .large 103508 .exactZero (none)

def event103510 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6713⟩⟩) 0 ⟨6689⟩ 103424

def event103511 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6713⟩⟩) (.authority (.operator))

def exact103512RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6713⟩⟩]⟩, (1)⟩]

theorem exact103512RawTermsValid :
    exact103512RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event103512 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6713⟩⟩) exact103512RawTerms .large 103511 .exactZero (none)

def event103513 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6711⟩⟩) 0 ⟨6689⟩ 103424

def event103514 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6711⟩⟩) (.authority (.operator))

def exact103515RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6711⟩⟩]⟩, (1)⟩]

theorem exact103515RawTermsValid :
    exact103515RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event103515 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6711⟩⟩) exact103515RawTerms .large 103514 .exactZero (none)

def event103516 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6709⟩⟩) 0 ⟨6689⟩ 103424

def event103517 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6709⟩⟩) (.authority (.operator))

def exact103518RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6709⟩⟩]⟩, (1)⟩]

theorem exact103518RawTermsValid :
    exact103518RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event103518 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6709⟩⟩) exact103518RawTerms .large 103517 .exactZero (none)

def event103519 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6795⟩⟩) 0 ⟨6709⟩ 103518

def event103520 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6795⟩⟩) 1 ⟨6711⟩ 103515

def event103521 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6795⟩⟩) (.sum [.predecessor 0 103519 .coefficient, .predecessor 1 103520 .coefficient])

def exact103522RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6709⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6711⟩⟩]⟩, (1)⟩]

theorem exact103522RawTermsValid :
    exact103522RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event103522 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6795⟩⟩) exact103522RawTerms .large 103521 .exactZero (none)

def event103523 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6796⟩⟩) 0 ⟨6795⟩ 103522

def event103524 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6796⟩⟩) 1 ⟨6713⟩ 103512

def event103525 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6796⟩⟩) (.sum [.predecessor 0 103523 .coefficient, .predecessor 1 103524 .coefficient])

def exact103526RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6709⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6711⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6713⟩⟩]⟩, (1)⟩]

theorem exact103526RawTermsValid :
    exact103526RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event103526 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6796⟩⟩) exact103526RawTerms .large 103525 .exactZero (none)

def event103527 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6797⟩⟩) 0 ⟨6796⟩ 103526

def event103528 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6797⟩⟩) 1 ⟨6715⟩ 103509

def event103529 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6797⟩⟩) (.sum [.predecessor 0 103527 .coefficient, .predecessor 1 103528 .coefficient])

def exact103530RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6709⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6711⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6713⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6715⟩⟩]⟩, (1)⟩]

theorem exact103530RawTermsValid :
    exact103530RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event103530 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6797⟩⟩) exact103530RawTerms .large 103529 .exactZero (none)

def event103531 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6798⟩⟩) 0 ⟨6797⟩ 103530

def event103532 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6798⟩⟩) 1 ⟨6717⟩ 103506

def event103533 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6798⟩⟩) (.sum [.predecessor 0 103531 .coefficient, .predecessor 1 103532 .coefficient])

def exact103534RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6709⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6711⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6713⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6715⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6717⟩⟩]⟩, (1)⟩]

theorem exact103534RawTermsValid :
    exact103534RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event103534 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6798⟩⟩) exact103534RawTerms .large 103533 .exactZero (none)

def event103535 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6799⟩⟩) 0 ⟨6798⟩ 103534

def event103536 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6799⟩⟩) 1 ⟨6719⟩ 103503

def event103537 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6799⟩⟩) (.sum [.predecessor 0 103535 .coefficient, .predecessor 1 103536 .coefficient])

def exact103538RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6709⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6711⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6713⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6715⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6717⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6719⟩⟩]⟩, (1)⟩]

theorem exact103538RawTermsValid :
    exact103538RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event103538 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6799⟩⟩) exact103538RawTerms .large 103537 .exactZero (none)

def event103539 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6800⟩⟩) 0 ⟨6799⟩ 103538

def event103540 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6800⟩⟩) 1 ⟨6721⟩ 103500

def event103541 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6800⟩⟩) (.sum [.predecessor 0 103539 .coefficient, .predecessor 1 103540 .coefficient])

def exact103542RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6709⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6711⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6713⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6715⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6717⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6719⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6721⟩⟩]⟩, (1)⟩]

theorem exact103542RawTermsValid :
    exact103542RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event103542 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6800⟩⟩) exact103542RawTerms .large 103541 .exactZero (none)

def event103543 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6801⟩⟩) 0 ⟨6800⟩ 103542

def event103544 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6801⟩⟩) 1 ⟨6723⟩ 103497

def event103545 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6801⟩⟩) (.sum [.predecessor 0 103543 .coefficient, .predecessor 1 103544 .coefficient])

def exact103546RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6709⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6711⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6713⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6715⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6717⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6719⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6721⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6723⟩⟩]⟩, (1)⟩]

theorem exact103546RawTermsValid :
    exact103546RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event103546 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6801⟩⟩) exact103546RawTerms .large 103545 .exactZero (none)

def event103547 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6802⟩⟩) 0 ⟨6801⟩ 103546

def event103548 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6802⟩⟩) 1 ⟨6725⟩ 103494

def event103549 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6802⟩⟩) (.sum [.predecessor 0 103547 .coefficient, .predecessor 1 103548 .coefficient])

def exact103550RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6709⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6711⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6713⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6715⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6717⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6719⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6721⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6723⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6725⟩⟩]⟩, (1)⟩]

theorem exact103550RawTermsValid :
    exact103550RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event103550 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6802⟩⟩) exact103550RawTerms .large 103549 .exactZero (none)

def event103551 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6803⟩⟩) 0 ⟨6802⟩ 103550

def event103552 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6803⟩⟩) 1 ⟨6727⟩ 103491

def event103553 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6803⟩⟩) (.sum [.predecessor 0 103551 .coefficient, .predecessor 1 103552 .coefficient])

def exact103554RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6709⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6711⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6713⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6715⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6717⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6719⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6721⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6723⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6725⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6727⟩⟩]⟩, (1)⟩]

theorem exact103554RawTermsValid :
    exact103554RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event103554 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6803⟩⟩) exact103554RawTerms .large 103553 .exactZero (none)

def event103555 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6804⟩⟩) 0 ⟨6803⟩ 103554

def event103556 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6804⟩⟩) 1 ⟨6729⟩ 103488

def event103557 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6804⟩⟩) (.sum [.predecessor 0 103555 .coefficient, .predecessor 1 103556 .coefficient])

def exact103558RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6709⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6711⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6713⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6715⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6717⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6719⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6721⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6723⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6725⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6727⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6729⟩⟩]⟩, (1)⟩]

theorem exact103558RawTermsValid :
    exact103558RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event103558 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6804⟩⟩) exact103558RawTerms .large 103557 .exactZero (none)

def event103559 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6805⟩⟩) 0 ⟨6804⟩ 103558

def event103560 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6805⟩⟩) 1 ⟨6731⟩ 103485

def event103561 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6805⟩⟩) (.sum [.predecessor 0 103559 .coefficient, .predecessor 1 103560 .coefficient])

def exact103562RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6709⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6711⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6713⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6715⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6717⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6719⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6721⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6723⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6725⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6727⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6729⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6731⟩⟩]⟩, (1)⟩]

theorem exact103562RawTermsValid :
    exact103562RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event103562 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6805⟩⟩) exact103562RawTerms .large 103561 .exactZero (none)

def event103563 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6806⟩⟩) 0 ⟨6805⟩ 103562

def event103564 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6806⟩⟩) 1 ⟨6733⟩ 103482

def event103565 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6806⟩⟩) (.sum [.predecessor 0 103563 .coefficient, .predecessor 1 103564 .coefficient])

def exact103566RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6709⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6711⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6713⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6715⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6717⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6719⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6721⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6723⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6725⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6727⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6729⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6731⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6733⟩⟩]⟩, (1)⟩]

theorem exact103566RawTermsValid :
    exact103566RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event103566 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6806⟩⟩) exact103566RawTerms .large 103565 .exactZero (none)

def event103567 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6807⟩⟩) 0 ⟨6806⟩ 103566

def event103568 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6807⟩⟩) 1 ⟨6735⟩ 103479

def event103569 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6807⟩⟩) (.sum [.predecessor 0 103567 .coefficient, .predecessor 1 103568 .coefficient])

def exact103570RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6709⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6711⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6713⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6715⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6717⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6719⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6721⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6723⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6725⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6727⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6729⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6731⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6733⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6735⟩⟩]⟩, (1)⟩]

theorem exact103570RawTermsValid :
    exact103570RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event103570 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6807⟩⟩) exact103570RawTerms .large 103569 .exactZero (none)

def event103571 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6808⟩⟩) 0 ⟨6807⟩ 103570

def event103572 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6808⟩⟩) 1 ⟨6737⟩ 103476

def event103573 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6808⟩⟩) (.sum [.predecessor 0 103571 .coefficient, .predecessor 1 103572 .coefficient])

def exact103574RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6709⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6711⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6713⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6715⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6717⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6719⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6721⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6723⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6725⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6727⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6729⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6731⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6733⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6735⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6737⟩⟩]⟩, (1)⟩]

theorem exact103574RawTermsValid :
    exact103574RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event103574 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6808⟩⟩) exact103574RawTerms .large 103573 .exactZero (none)

def event103575 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6809⟩⟩) 0 ⟨6808⟩ 103574

def event103576 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6809⟩⟩) 1 ⟨6739⟩ 103473

def event103577 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6809⟩⟩) (.sum [.predecessor 0 103575 .coefficient, .predecessor 1 103576 .coefficient])

def exact103578RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6709⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6711⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6713⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6715⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6717⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6719⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6721⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6723⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6725⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6727⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6729⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6731⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6733⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6735⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6737⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6739⟩⟩]⟩, (1)⟩]

theorem exact103578RawTermsValid :
    exact103578RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event103578 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6809⟩⟩) exact103578RawTerms .large 103577 .exactZero (none)

def event103579 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6810⟩⟩) 0 ⟨6809⟩ 103578

def event103580 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6810⟩⟩) 1 ⟨6741⟩ 103470

def event103581 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6810⟩⟩) (.sum [.predecessor 0 103579 .coefficient, .predecessor 1 103580 .coefficient])

def exact103582RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6709⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6711⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6713⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6715⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6717⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6719⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6721⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6723⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6725⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6727⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6729⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6731⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6733⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6735⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6737⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6739⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6741⟩⟩]⟩, (1)⟩]

theorem exact103582RawTermsValid :
    exact103582RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event103582 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6810⟩⟩) exact103582RawTerms .large 103581 .exactZero (none)

def event103583 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6811⟩⟩) 0 ⟨6810⟩ 103582

def event103584 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6811⟩⟩) 1 ⟨6743⟩ 103467

def event103585 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6811⟩⟩) (.sum [.predecessor 0 103583 .coefficient, .predecessor 1 103584 .coefficient])

def exact103586RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6709⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6711⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6713⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6715⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6717⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6719⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6721⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6723⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6725⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6727⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6729⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6731⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6733⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6735⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6737⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6739⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6741⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6743⟩⟩]⟩, (1)⟩]

theorem exact103586RawTermsValid :
    exact103586RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event103586 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6811⟩⟩) exact103586RawTerms .large 103585 .exactZero (none)

def event103587 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18642⟩⟩) 0 ⟨6811⟩ 103586

def event103588 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18642⟩⟩) 1 ⟨18641⟩ 103464

def event103589 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨18642⟩⟩) (.sum [.predecessor 0 103587 .coefficient, .predecessor 1 103588 .coefficient])

def exact103590RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6709⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6711⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6713⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6715⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6717⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6719⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6721⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6723⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6725⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6727⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6729⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6731⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6733⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6735⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6737⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6739⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6741⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6743⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15258⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15300⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15356⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15622⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15741⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15860⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15979⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16098⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16301⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16672⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16791⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17078⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17113⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17302⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17897⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨18163⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨18198⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨18303⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact103590RawTermsValid :
    exact103590RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event103590 : Event := .resultExact (⟨.program ⟨214⟩, ⟨18642⟩⟩) exact103590RawTerms .large 103589 .exactZero (none)

def event103591 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18675⟩⟩) 0 ⟨18642⟩ 103590

def event103592 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18675⟩⟩) 1 ⟨18674⟩ 103431

def event103593 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨18675⟩⟩) (.product (.predecessor 0 103591 .coefficient) (.predecessor 1 103592 .coefficient) (⟨false, false, none, none, none⟩))

def event103594 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18675⟩⟩, .operator (⟨103590, 17⟩, ⟨103431, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6743⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (1)⟩)

def event103595 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18675⟩⟩, .operator (⟨103590, 16⟩, ⟨103431, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6741⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (1)⟩)

def event103596 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18675⟩⟩, .operator (⟨103590, 15⟩, ⟨103431, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6739⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (1)⟩)

def event103597 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18675⟩⟩, .operator (⟨103590, 14⟩, ⟨103431, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6737⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (1)⟩)

def event103598 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18675⟩⟩, .operator (⟨103590, 13⟩, ⟨103431, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6735⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (1)⟩)

def event103599 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18675⟩⟩, .operator (⟨103590, 12⟩, ⟨103431, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6733⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (1)⟩)

def event103600 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18675⟩⟩, .operator (⟨103590, 11⟩, ⟨103431, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6731⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (1)⟩)

def event103601 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18675⟩⟩, .operator (⟨103590, 10⟩, ⟨103431, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6729⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (1)⟩)

def event103602 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18675⟩⟩, .operator (⟨103590, 9⟩, ⟨103431, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6727⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (1)⟩)

def event103603 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18675⟩⟩, .operator (⟨103590, 8⟩, ⟨103431, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6725⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (1)⟩)

def event103604 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18675⟩⟩, .operator (⟨103590, 7⟩, ⟨103431, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6723⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (1)⟩)

def event103605 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18675⟩⟩, .operator (⟨103590, 6⟩, ⟨103431, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6721⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (1)⟩)

def event103606 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18675⟩⟩, .operator (⟨103590, 5⟩, ⟨103431, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6719⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (1)⟩)

def event103607 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18675⟩⟩, .operator (⟨103590, 4⟩, ⟨103431, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6717⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (1)⟩)

def event103608 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18675⟩⟩, .operator (⟨103590, 3⟩, ⟨103431, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6715⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (1)⟩)

def event103609 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18675⟩⟩, .operator (⟨103590, 2⟩, ⟨103431, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6713⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (1)⟩)

def event103610 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18675⟩⟩, .operator (⟨103590, 1⟩, ⟨103431, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6711⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (1)⟩)

def event103611 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18675⟩⟩, .operator (⟨103590, 0⟩, ⟨103431, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6709⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (1)⟩)

def event103612 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18675⟩⟩, .operator (⟨103590, 33⟩, ⟨103431, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨18163⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (-1)⟩)

def event103613 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨18675⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨18163⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18674⟩⟩) ⟨18612⟩ 103428)

def event103614 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18675⟩⟩, .relation 103613 0, ⟨[⟨.program ⟨214⟩, ⟨18163⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (-1)⟩)

def event103615 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18675⟩⟩, .operator (⟨103590, 29⟩, ⟨103431, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨17078⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (-1)⟩)

def event103616 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨18675⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨17078⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18674⟩⟩) ⟨18612⟩ 103428)

def event103617 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18675⟩⟩, .relation 103616 0, ⟨[⟨.program ⟨214⟩, ⟨17078⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (-1)⟩)

def event103618 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18675⟩⟩, .operator (⟨103590, 28⟩, ⟨103431, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨16791⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (-1)⟩)

def event103619 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨18675⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨16791⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18674⟩⟩) ⟨18612⟩ 103428)

def event103620 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18675⟩⟩, .relation 103619 0, ⟨[⟨.program ⟨214⟩, ⟨16791⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (-1)⟩)

def event103621 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18675⟩⟩, .operator (⟨103590, 27⟩, ⟨103431, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨16672⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (-1)⟩)

def event103622 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨18675⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨16672⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18674⟩⟩) ⟨18612⟩ 103428)

def event103623 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18675⟩⟩, .relation 103622 0, ⟨[⟨.program ⟨214⟩, ⟨16672⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (-1)⟩)

def event103624 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18675⟩⟩, .operator (⟨103590, 34⟩, ⟨103431, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨18198⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (-1)⟩)

def event103625 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨18675⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨18198⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18674⟩⟩) ⟨18612⟩ 103428)

def event103626 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18675⟩⟩, .relation 103625 0, ⟨[⟨.program ⟨214⟩, ⟨18198⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (-1)⟩)

def event103627 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18675⟩⟩, .operator (⟨103590, 32⟩, ⟨103431, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨17897⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (-1)⟩)

def event103628 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨18675⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨17897⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18674⟩⟩) ⟨18612⟩ 103428)

def event103629 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18675⟩⟩, .relation 103628 0, ⟨[⟨.program ⟨214⟩, ⟨17897⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (-1)⟩)

def event103630 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18675⟩⟩, .operator (⟨103590, 30⟩, ⟨103431, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨17113⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (-1)⟩)

def event103631 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨18675⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨17113⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18674⟩⟩) ⟨18612⟩ 103428)

def event103632 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18675⟩⟩, .relation 103631 0, ⟨[⟨.program ⟨214⟩, ⟨17113⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (-1)⟩)

def event103633 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18675⟩⟩, .operator (⟨103590, 26⟩, ⟨103431, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨16301⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (-1)⟩)

def event103634 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨18675⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨16301⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18674⟩⟩) ⟨18612⟩ 103428)

def event103635 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18675⟩⟩, .relation 103634 0, ⟨[⟨.program ⟨214⟩, ⟨16301⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (-1)⟩)

def event103636 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18675⟩⟩, .operator (⟨103590, 35⟩, ⟨103431, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨18303⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (-1)⟩)

def event103637 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨18675⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨18303⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18674⟩⟩) ⟨18612⟩ 103428)

def event103638 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18675⟩⟩, .relation 103637 0, ⟨[⟨.program ⟨214⟩, ⟨18303⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (-1)⟩)

def event103639 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18675⟩⟩, .operator (⟨103590, 25⟩, ⟨103431, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨16098⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (-1)⟩)

def event103640 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨18675⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨16098⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18674⟩⟩) ⟨18612⟩ 103428)

def event103641 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18675⟩⟩, .relation 103640 0, ⟨[⟨.program ⟨214⟩, ⟨16098⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (-1)⟩)

def event103642 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18675⟩⟩, .operator (⟨103590, 24⟩, ⟨103431, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨15979⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (-1)⟩)

def event103643 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨18675⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨15979⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18674⟩⟩) ⟨18612⟩ 103428)

def event103644 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18675⟩⟩, .relation 103643 0, ⟨[⟨.program ⟨214⟩, ⟨15979⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (-1)⟩)

def event103645 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18675⟩⟩, .operator (⟨103590, 23⟩, ⟨103431, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨15860⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (-1)⟩)

def event103646 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨18675⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨15860⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18674⟩⟩) ⟨18612⟩ 103428)

def event103647 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18675⟩⟩, .relation 103646 0, ⟨[⟨.program ⟨214⟩, ⟨15860⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (-1)⟩)

def event103648 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18675⟩⟩, .operator (⟨103590, 22⟩, ⟨103431, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨15741⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (-1)⟩)

def event103649 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨18675⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨15741⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18674⟩⟩) ⟨18612⟩ 103428)

def event103650 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18675⟩⟩, .relation 103649 0, ⟨[⟨.program ⟨214⟩, ⟨15741⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (-1)⟩)

def event103651 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18675⟩⟩, .operator (⟨103590, 21⟩, ⟨103431, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨15622⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (-1)⟩)

def event103652 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨18675⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨15622⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18674⟩⟩) ⟨18612⟩ 103428)

def event103653 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18675⟩⟩, .relation 103652 0, ⟨[⟨.program ⟨214⟩, ⟨15622⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (-1)⟩)

def event103654 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18675⟩⟩, .operator (⟨103590, 31⟩, ⟨103431, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨17302⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (-1)⟩)

def event103655 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨18675⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨17302⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18674⟩⟩) ⟨18612⟩ 103428)

def event103656 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18675⟩⟩, .relation 103655 0, ⟨[⟨.program ⟨214⟩, ⟨17302⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (-1)⟩)

def event103657 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18675⟩⟩, .operator (⟨103590, 20⟩, ⟨103431, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨15356⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (-1)⟩)

def event103658 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨18675⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨15356⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18674⟩⟩) ⟨18612⟩ 103428)

def event103659 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18675⟩⟩, .relation 103658 0, ⟨[⟨.program ⟨214⟩, ⟨15356⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (-1)⟩)

def event103660 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18675⟩⟩, .operator (⟨103590, 19⟩, ⟨103431, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨15300⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (-1)⟩)

def event103661 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨18675⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨15300⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18674⟩⟩) ⟨18612⟩ 103428)

def event103662 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18675⟩⟩, .relation 103661 0, ⟨[⟨.program ⟨214⟩, ⟨15300⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (-1)⟩)

def event103663 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18675⟩⟩, .operator (⟨103590, 18⟩, ⟨103431, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨15258⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (-1)⟩)

def event103664 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨18675⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨15258⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18674⟩⟩) ⟨18612⟩ 103428)

def event103665 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18675⟩⟩, .relation 103664 0, ⟨[⟨.program ⟨214⟩, ⟨15258⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (-1)⟩)

def exact103666RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6709⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6711⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6713⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6715⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6717⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6719⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6721⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6723⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6725⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6727⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6729⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6731⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6733⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6735⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6737⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6739⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6741⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6743⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15258⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15300⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15356⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15622⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15741⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15860⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15979⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16098⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16301⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16672⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16791⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17078⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17113⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17302⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17897⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨18163⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨18198⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨18303⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (-1)⟩]

theorem exact103666RawTermsValid :
    exact103666RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event103666 : Event := .resultExact (⟨.program ⟨214⟩, ⟨18675⟩⟩) exact103666RawTerms .large 103593 .exactZero (none)

def event103667 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18485⟩⟩) 0 ⟨18313⟩ 103420

def event103668 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨18485⟩⟩) (.authority (.programFamilyFact))

def exact103669RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨18485⟩⟩], []⟩, (1)⟩]

theorem exact103669RawTermsValid :
    exact103669RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event103669 : Event := .resultExact (⟨.program ⟨214⟩, ⟨18485⟩⟩) exact103669RawTerms (.finite 18) 103668 .exactZero (none)

def event103670 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18487⟩⟩) 0 ⟨6544⟩ 103442

def event103671 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18487⟩⟩) 1 ⟨18485⟩ 103669

def event103672 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨18487⟩⟩) (.product (.predecessor 0 103670 .coefficient) (.predecessor 1 103671 .coefficient) (⟨false, true, none, none, some 1⟩))

def event103673 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18487⟩⟩, .operator (⟨103442, 0⟩, ⟨103669, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨18485⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact103674RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨18485⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact103674RawTermsValid :
    exact103674RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event103674 : Event := .resultExact (⟨.program ⟨214⟩, ⟨18487⟩⟩) exact103674RawTerms .large 103672 .exactZero (none)

def event103675 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6744⟩⟩) 0 ⟨6689⟩ 103424

def event103676 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6744⟩⟩) (.authority (.operator))

def exact103677RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6744⟩⟩]⟩, (1)⟩]

theorem exact103677RawTermsValid :
    exact103677RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event103677 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6744⟩⟩) exact103677RawTerms .large 103676 .exactZero (none)

def event103678 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18488⟩⟩) 0 ⟨6744⟩ 103677

def event103679 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18488⟩⟩) 1 ⟨18487⟩ 103674

def eventLeaf6464 : Array AnnotatedEvent := #[
  { event := event103424
    frameStart := 102927 },
  { event := event103425
    frameStart := 102927 },
  { event := event103426
    frameStart := 102927 },
  { event := event103427
    frameStart := 102927 },
  { event := event103428
    frameStart := 102927 },
  { event := event103429
    frameStart := 102927 },
  { event := event103430
    frameStart := 102927 },
  { event := event103431
    frameStart := 102927 },
  { event := event103432
    frameStart := 102927 },
  { event := event103433
    frameStart := 102927 },
  { event := event103434
    frameStart := 102927 },
  { event := event103435
    frameStart := 102927 },
  { event := event103436
    frameStart := 102927 },
  { event := event103437
    frameStart := 102927 },
  { event := event103438
    frameStart := 102927 },
  { event := event103439
    frameStart := 102927 }
]

def eventLeaf6465 : Array AnnotatedEvent := #[
  { event := event103440
    frameStart := 102927 },
  { event := event103441
    frameStart := 102927 },
  { event := event103442
    frameStart := 102927 },
  { event := event103443
    frameStart := 102927 },
  { event := event103444
    frameStart := 102927 },
  { event := event103445
    frameStart := 102927 },
  { event := event103446
    frameStart := 102927 },
  { event := event103447
    frameStart := 102927 },
  { event := event103448
    frameStart := 102927 },
  { event := event103449
    frameStart := 102927 },
  { event := event103450
    frameStart := 102927 },
  { event := event103451
    frameStart := 102927 },
  { event := event103452
    frameStart := 102927 },
  { event := event103453
    frameStart := 102927 },
  { event := event103454
    frameStart := 102927 },
  { event := event103455
    frameStart := 102927 }
]

def eventLeaf6466 : Array AnnotatedEvent := #[
  { event := event103456
    frameStart := 102927 },
  { event := event103457
    frameStart := 102927 },
  { event := event103458
    frameStart := 102927 },
  { event := event103459
    frameStart := 102927 },
  { event := event103460
    frameStart := 102927 },
  { event := event103461
    frameStart := 102927 },
  { event := event103462
    frameStart := 102927 },
  { event := event103463
    frameStart := 102927 },
  { event := event103464
    frameStart := 102927 },
  { event := event103465
    frameStart := 102927 },
  { event := event103466
    frameStart := 102927 },
  { event := event103467
    frameStart := 102927 },
  { event := event103468
    frameStart := 102927 },
  { event := event103469
    frameStart := 102927 },
  { event := event103470
    frameStart := 102927 },
  { event := event103471
    frameStart := 102927 }
]

def eventLeaf6467 : Array AnnotatedEvent := #[
  { event := event103472
    frameStart := 102927 },
  { event := event103473
    frameStart := 102927 },
  { event := event103474
    frameStart := 102927 },
  { event := event103475
    frameStart := 102927 },
  { event := event103476
    frameStart := 102927 },
  { event := event103477
    frameStart := 102927 },
  { event := event103478
    frameStart := 102927 },
  { event := event103479
    frameStart := 102927 },
  { event := event103480
    frameStart := 102927 },
  { event := event103481
    frameStart := 102927 },
  { event := event103482
    frameStart := 102927 },
  { event := event103483
    frameStart := 102927 },
  { event := event103484
    frameStart := 102927 },
  { event := event103485
    frameStart := 102927 },
  { event := event103486
    frameStart := 102927 },
  { event := event103487
    frameStart := 102927 }
]

def eventLeaf6468 : Array AnnotatedEvent := #[
  { event := event103488
    frameStart := 102927 },
  { event := event103489
    frameStart := 102927 },
  { event := event103490
    frameStart := 102927 },
  { event := event103491
    frameStart := 102927 },
  { event := event103492
    frameStart := 102927 },
  { event := event103493
    frameStart := 102927 },
  { event := event103494
    frameStart := 102927 },
  { event := event103495
    frameStart := 102927 },
  { event := event103496
    frameStart := 102927 },
  { event := event103497
    frameStart := 102927 },
  { event := event103498
    frameStart := 102927 },
  { event := event103499
    frameStart := 102927 },
  { event := event103500
    frameStart := 102927 },
  { event := event103501
    frameStart := 102927 },
  { event := event103502
    frameStart := 102927 },
  { event := event103503
    frameStart := 102927 }
]

def eventLeaf6469 : Array AnnotatedEvent := #[
  { event := event103504
    frameStart := 102927 },
  { event := event103505
    frameStart := 102927 },
  { event := event103506
    frameStart := 102927 },
  { event := event103507
    frameStart := 102927 },
  { event := event103508
    frameStart := 102927 },
  { event := event103509
    frameStart := 102927 },
  { event := event103510
    frameStart := 102927 },
  { event := event103511
    frameStart := 102927 },
  { event := event103512
    frameStart := 102927 },
  { event := event103513
    frameStart := 102927 },
  { event := event103514
    frameStart := 102927 },
  { event := event103515
    frameStart := 102927 },
  { event := event103516
    frameStart := 102927 },
  { event := event103517
    frameStart := 102927 },
  { event := event103518
    frameStart := 102927 },
  { event := event103519
    frameStart := 102927 }
]

def eventLeaf6470 : Array AnnotatedEvent := #[
  { event := event103520
    frameStart := 102927 },
  { event := event103521
    frameStart := 102927 },
  { event := event103522
    frameStart := 102927 },
  { event := event103523
    frameStart := 102927 },
  { event := event103524
    frameStart := 102927 },
  { event := event103525
    frameStart := 102927 },
  { event := event103526
    frameStart := 102927 },
  { event := event103527
    frameStart := 102927 },
  { event := event103528
    frameStart := 102927 },
  { event := event103529
    frameStart := 102927 },
  { event := event103530
    frameStart := 102927 },
  { event := event103531
    frameStart := 102927 },
  { event := event103532
    frameStart := 102927 },
  { event := event103533
    frameStart := 102927 },
  { event := event103534
    frameStart := 102927 },
  { event := event103535
    frameStart := 102927 }
]

def eventLeaf6471 : Array AnnotatedEvent := #[
  { event := event103536
    frameStart := 102927 },
  { event := event103537
    frameStart := 102927 },
  { event := event103538
    frameStart := 102927 },
  { event := event103539
    frameStart := 102927 },
  { event := event103540
    frameStart := 102927 },
  { event := event103541
    frameStart := 102927 },
  { event := event103542
    frameStart := 102927 },
  { event := event103543
    frameStart := 102927 },
  { event := event103544
    frameStart := 102927 },
  { event := event103545
    frameStart := 102927 },
  { event := event103546
    frameStart := 102927 },
  { event := event103547
    frameStart := 102927 },
  { event := event103548
    frameStart := 102927 },
  { event := event103549
    frameStart := 102927 },
  { event := event103550
    frameStart := 102927 },
  { event := event103551
    frameStart := 102927 }
]

def eventLeaf6472 : Array AnnotatedEvent := #[
  { event := event103552
    frameStart := 102927 },
  { event := event103553
    frameStart := 102927 },
  { event := event103554
    frameStart := 102927 },
  { event := event103555
    frameStart := 102927 },
  { event := event103556
    frameStart := 102927 },
  { event := event103557
    frameStart := 102927 },
  { event := event103558
    frameStart := 102927 },
  { event := event103559
    frameStart := 102927 },
  { event := event103560
    frameStart := 102927 },
  { event := event103561
    frameStart := 102927 },
  { event := event103562
    frameStart := 102927 },
  { event := event103563
    frameStart := 102927 },
  { event := event103564
    frameStart := 102927 },
  { event := event103565
    frameStart := 102927 },
  { event := event103566
    frameStart := 102927 },
  { event := event103567
    frameStart := 102927 }
]

def eventLeaf6473 : Array AnnotatedEvent := #[
  { event := event103568
    frameStart := 102927 },
  { event := event103569
    frameStart := 102927 },
  { event := event103570
    frameStart := 102927 },
  { event := event103571
    frameStart := 102927 },
  { event := event103572
    frameStart := 102927 },
  { event := event103573
    frameStart := 102927 },
  { event := event103574
    frameStart := 102927 },
  { event := event103575
    frameStart := 102927 },
  { event := event103576
    frameStart := 102927 },
  { event := event103577
    frameStart := 102927 },
  { event := event103578
    frameStart := 102927 },
  { event := event103579
    frameStart := 102927 },
  { event := event103580
    frameStart := 102927 },
  { event := event103581
    frameStart := 102927 },
  { event := event103582
    frameStart := 102927 },
  { event := event103583
    frameStart := 102927 }
]

def eventLeaf6474 : Array AnnotatedEvent := #[
  { event := event103584
    frameStart := 102927 },
  { event := event103585
    frameStart := 102927 },
  { event := event103586
    frameStart := 102927 },
  { event := event103587
    frameStart := 102927 },
  { event := event103588
    frameStart := 102927 },
  { event := event103589
    frameStart := 102927 },
  { event := event103590
    frameStart := 102927 },
  { event := event103591
    frameStart := 102927 },
  { event := event103592
    frameStart := 102927 },
  { event := event103593
    frameStart := 102927 },
  { event := event103594
    frameStart := 102927 },
  { event := event103595
    frameStart := 102927 },
  { event := event103596
    frameStart := 102927 },
  { event := event103597
    frameStart := 102927 },
  { event := event103598
    frameStart := 102927 },
  { event := event103599
    frameStart := 102927 }
]

def eventLeaf6475 : Array AnnotatedEvent := #[
  { event := event103600
    frameStart := 102927 },
  { event := event103601
    frameStart := 102927 },
  { event := event103602
    frameStart := 102927 },
  { event := event103603
    frameStart := 102927 },
  { event := event103604
    frameStart := 102927 },
  { event := event103605
    frameStart := 102927 },
  { event := event103606
    frameStart := 102927 },
  { event := event103607
    frameStart := 102927 },
  { event := event103608
    frameStart := 102927 },
  { event := event103609
    frameStart := 102927 },
  { event := event103610
    frameStart := 102927 },
  { event := event103611
    frameStart := 102927 },
  { event := event103612
    frameStart := 102927 },
  { event := event103613
    frameStart := 102927 },
  { event := event103614
    frameStart := 102927 },
  { event := event103615
    frameStart := 102927 }
]

def eventLeaf6476 : Array AnnotatedEvent := #[
  { event := event103616
    frameStart := 102927 },
  { event := event103617
    frameStart := 102927 },
  { event := event103618
    frameStart := 102927 },
  { event := event103619
    frameStart := 102927 },
  { event := event103620
    frameStart := 102927 },
  { event := event103621
    frameStart := 102927 },
  { event := event103622
    frameStart := 102927 },
  { event := event103623
    frameStart := 102927 },
  { event := event103624
    frameStart := 102927 },
  { event := event103625
    frameStart := 102927 },
  { event := event103626
    frameStart := 102927 },
  { event := event103627
    frameStart := 102927 },
  { event := event103628
    frameStart := 102927 },
  { event := event103629
    frameStart := 102927 },
  { event := event103630
    frameStart := 102927 },
  { event := event103631
    frameStart := 102927 }
]

def eventLeaf6477 : Array AnnotatedEvent := #[
  { event := event103632
    frameStart := 102927 },
  { event := event103633
    frameStart := 102927 },
  { event := event103634
    frameStart := 102927 },
  { event := event103635
    frameStart := 102927 },
  { event := event103636
    frameStart := 102927 },
  { event := event103637
    frameStart := 102927 },
  { event := event103638
    frameStart := 102927 },
  { event := event103639
    frameStart := 102927 },
  { event := event103640
    frameStart := 102927 },
  { event := event103641
    frameStart := 102927 },
  { event := event103642
    frameStart := 102927 },
  { event := event103643
    frameStart := 102927 },
  { event := event103644
    frameStart := 102927 },
  { event := event103645
    frameStart := 102927 },
  { event := event103646
    frameStart := 102927 },
  { event := event103647
    frameStart := 102927 }
]

def eventLeaf6478 : Array AnnotatedEvent := #[
  { event := event103648
    frameStart := 102927 },
  { event := event103649
    frameStart := 102927 },
  { event := event103650
    frameStart := 102927 },
  { event := event103651
    frameStart := 102927 },
  { event := event103652
    frameStart := 102927 },
  { event := event103653
    frameStart := 102927 },
  { event := event103654
    frameStart := 102927 },
  { event := event103655
    frameStart := 102927 },
  { event := event103656
    frameStart := 102927 },
  { event := event103657
    frameStart := 102927 },
  { event := event103658
    frameStart := 102927 },
  { event := event103659
    frameStart := 102927 },
  { event := event103660
    frameStart := 102927 },
  { event := event103661
    frameStart := 102927 },
  { event := event103662
    frameStart := 102927 },
  { event := event103663
    frameStart := 102927 }
]

def eventLeaf6479 : Array AnnotatedEvent := #[
  { event := event103664
    frameStart := 102927 },
  { event := event103665
    frameStart := 102927 },
  { event := event103666
    frameStart := 102927 },
  { event := event103667
    frameStart := 102927 },
  { event := event103668
    frameStart := 102927 },
  { event := event103669
    frameStart := 102927 },
  { event := event103670
    frameStart := 102927 },
  { event := event103671
    frameStart := 102927 },
  { event := event103672
    frameStart := 102927 },
  { event := event103673
    frameStart := 102927 },
  { event := event103674
    frameStart := 102927 },
  { event := event103675
    frameStart := 102927 },
  { event := event103676
    frameStart := 102927 },
  { event := event103677
    frameStart := 102927 },
  { event := event103678
    frameStart := 102927 },
  { event := event103679
    frameStart := 102927 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events404
