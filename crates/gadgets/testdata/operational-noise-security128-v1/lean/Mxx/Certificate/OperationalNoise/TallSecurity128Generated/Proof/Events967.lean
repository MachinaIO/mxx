import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events967

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event247552 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42426⟩⟩) (.authority (.programFamilyFact))

def exact247553RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨42426⟩⟩], []⟩, (1)⟩]

theorem exact247553RawTermsValid :
    exact247553RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event247553 : Event := .resultExact (⟨.program ⟨257⟩, ⟨42426⟩⟩) exact247553RawTerms (.finite 52) 247552 .exactZero (none)

def event247554 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14451⟩⟩) 0 ⟨5559⟩ 247550

def event247555 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14451⟩⟩) (.authority (.programFamilyFact))

def exact247556RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14451⟩⟩], []⟩, (1)⟩]

theorem exact247556RawTermsValid :
    exact247556RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event247556 : Event := .resultExact (⟨.program ⟨257⟩, ⟨14451⟩⟩) exact247556RawTerms (.finite 52) 247555 .exactZero (none)

def event247557 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42427⟩⟩) 0 ⟨14451⟩ 247556

def event247558 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42427⟩⟩) 1 ⟨42426⟩ 247553

def event247559 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42427⟩⟩) (.product (.predecessor 0 247557 .coefficient) (.predecessor 1 247558 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event247560 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42427⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨14451⟩⟩, ⟨.program ⟨257⟩, ⟨42426⟩⟩], []⟩) [⟨.result 247556 .coefficient, true, some 1⟩, ⟨.result 247553 .coefficient, true, some 1⟩])

def event247561 : Event := .survivorFold (1) 247560

def exact247562RawTerms : List Term := []

theorem exact247562RawTermsValid :
    exact247562RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event247562 : Event := .resultExact (⟨.program ⟨257⟩, ⟨42427⟩⟩) exact247562RawTerms (.finite 2704) 247559 (.finite 2704) (some (247560))

def event247563 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42428⟩⟩) 0 ⟨42427⟩ 247562

def event247564 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42428⟩⟩) (.identity (.predecessor 0 247563 .coefficient))

def event247565 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨42428⟩⟩) (.finite 2704)

def event247566 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42772⟩⟩) 0 ⟨42428⟩ 247565

def event247567 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42772⟩⟩) (.authority (.programFamilyFact))

def exact247568RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨42772⟩⟩], []⟩, (1)⟩]

theorem exact247568RawTermsValid :
    exact247568RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event247568 : Event := .resultExact (⟨.program ⟨257⟩, ⟨42772⟩⟩) exact247568RawTerms (.finite 52) 247567 .exactZero (none)

def event247569 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42773⟩⟩) 0 ⟨42772⟩ 247568

def event247570 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42773⟩⟩) (.identity (.predecessor 0 247569 .coefficient))

def event247571 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨42773⟩⟩) (.finite 52)

def event247572 : Event := .predecessor (⟨.program ⟨257⟩, ⟨43492⟩⟩) 0 ⟨42773⟩ 247571

def event247573 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨43492⟩⟩) (.authority (.relationPreimageSource ⟨89⟩))

def exact247574RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨43492⟩⟩]⟩, (1)⟩]

theorem exact247574RawTermsValid :
    exact247574RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event247574 : Event := .resultExact (⟨.program ⟨257⟩, ⟨43492⟩⟩) exact247574RawTerms (.finite 5647228698) 247573 .exactZero (none)

def event247575 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35⟩⟩) (.authority (.operator))

def exact247576RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩]⟩, (1)⟩]

theorem exact247576RawTermsValid :
    exact247576RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event247576 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35⟩⟩) exact247576RawTerms .large 247575 .exactZero (none)

def event247577 : Event := .predecessor (⟨.program ⟨257⟩, ⟨43493⟩⟩) 0 ⟨35⟩ 247576

def event247578 : Event := .predecessor (⟨.program ⟨257⟩, ⟨43493⟩⟩) 1 ⟨43492⟩ 247574

def event247579 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨43493⟩⟩) (.product (.predecessor 0 247577 .coefficient) (.predecessor 1 247578 .coefficient) (⟨false, false, none, none, none⟩))

def event247580 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨43493⟩⟩, .operator (⟨247576, 0⟩, ⟨247574, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43492⟩⟩]⟩, (1)⟩)

def exact247581RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43492⟩⟩]⟩, (1)⟩]

theorem exact247581RawTermsValid :
    exact247581RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event247581 : Event := .resultExact (⟨.program ⟨257⟩, ⟨43493⟩⟩) exact247581RawTerms .large 247579 .exactZero (none)

def event247582 : Event := .preFoldPolynomial 247581 [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43492⟩⟩]⟩, (1)⟩] .exactZero none

def exact247583RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43492⟩⟩]⟩, (1)⟩]

def event247583 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨43493⟩⟩) 247582 exact247583RawTerms .large 247579 .exactZero (none)

def event247584 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨44618⟩⟩)

def event247585 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event247586 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event247587 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨4740⟩⟩) (.authority (.operator))

def event247588 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨4740⟩⟩) (.finite 4)

def event247589 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event247590 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event247591 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event247592 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event247593 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 247592

def event247594 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 247590

def event247595 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 247593 .coefficient) (.value (.predecessor 1 247594 .coefficient)))

def event247596 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event247597 : Event := .predecessor (⟨.program ⟨257⟩, ⟨4742⟩⟩) 0 ⟨392⟩ 247596

def event247598 : Event := .predecessor (⟨.program ⟨257⟩, ⟨4742⟩⟩) 1 ⟨4740⟩ 247588

def event247599 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨4742⟩⟩) (.sum [.predecessor 0 247597 .coefficient, .predecessor 1 247598 .coefficient])

def event247600 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨4742⟩⟩) (.finite 655344)

def event247601 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5559⟩⟩) 0 ⟨4742⟩ 247600

def event247602 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5559⟩⟩) 1 ⟨5426⟩ 247586

def event247603 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5559⟩⟩) (.identity (.predecessor 1 247602 .coefficient))

def event247604 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5559⟩⟩) (.finite 655360)

def event247605 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42426⟩⟩) 0 ⟨5559⟩ 247604

def event247606 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42426⟩⟩) (.authority (.programFamilyFact))

def exact247607RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨42426⟩⟩], []⟩, (1)⟩]

theorem exact247607RawTermsValid :
    exact247607RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event247607 : Event := .resultExact (⟨.program ⟨257⟩, ⟨42426⟩⟩) exact247607RawTerms (.finite 52) 247606 .exactZero (none)

def event247608 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14451⟩⟩) 0 ⟨5559⟩ 247604

def event247609 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14451⟩⟩) (.authority (.programFamilyFact))

def exact247610RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14451⟩⟩], []⟩, (1)⟩]

theorem exact247610RawTermsValid :
    exact247610RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event247610 : Event := .resultExact (⟨.program ⟨257⟩, ⟨14451⟩⟩) exact247610RawTerms (.finite 52) 247609 .exactZero (none)

def event247611 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42427⟩⟩) 0 ⟨14451⟩ 247610

def event247612 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42427⟩⟩) 1 ⟨42426⟩ 247607

def event247613 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42427⟩⟩) (.product (.predecessor 0 247611 .coefficient) (.predecessor 1 247612 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event247614 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨42427⟩⟩, .operator (⟨247610, 0⟩, ⟨247607, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨14451⟩⟩, ⟨.program ⟨257⟩, ⟨42426⟩⟩], []⟩, (1)⟩)

def exact247615RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14451⟩⟩, ⟨.program ⟨257⟩, ⟨42426⟩⟩], []⟩, (1)⟩]

theorem exact247615RawTermsValid :
    exact247615RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event247615 : Event := .resultExact (⟨.program ⟨257⟩, ⟨42427⟩⟩) exact247615RawTerms (.finite 2704) 247613 .exactZero (none)

def event247616 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42428⟩⟩) 0 ⟨42427⟩ 247615

def event247617 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42428⟩⟩) (.identity (.predecessor 0 247616 .coefficient))

def event247618 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨42428⟩⟩) (.finite 2704)

def event247619 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42772⟩⟩) 0 ⟨42428⟩ 247618

def event247620 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42772⟩⟩) (.authority (.programFamilyFact))

def exact247621RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨42772⟩⟩], []⟩, (1)⟩]

theorem exact247621RawTermsValid :
    exact247621RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event247621 : Event := .resultExact (⟨.program ⟨257⟩, ⟨42772⟩⟩) exact247621RawTerms (.finite 52) 247620 .exactZero (none)

def event247622 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42773⟩⟩) 0 ⟨42772⟩ 247621

def event247623 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42773⟩⟩) (.identity (.predecessor 0 247622 .coefficient))

def event247624 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨42773⟩⟩) (.finite 52)

def event247625 : Event := .predecessor (⟨.program ⟨257⟩, ⟨43921⟩⟩) 0 ⟨42773⟩ 247624

def event247626 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨43921⟩⟩) (.authority (.programFamilyFact))

def event247627 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨43921⟩⟩) (.finite 3720)

def event247628 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨7177⟩⟩) .missing

def event247629 : Event := .predecessor (⟨.program ⟨257⟩, ⟨43922⟩⟩) 0 ⟨7177⟩ 247628

def event247630 : Event := .predecessor (⟨.program ⟨257⟩, ⟨43922⟩⟩) 1 ⟨43921⟩ 247627

def event247631 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨43922⟩⟩) (.authority (.operator))

def exact247632RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨43922⟩⟩]⟩, (1)⟩]

theorem exact247632RawTermsValid :
    exact247632RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event247632 : Event := .resultExact (⟨.program ⟨257⟩, ⟨43922⟩⟩) exact247632RawTerms .large 247631 .exactZero (none)

def event247633 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44613⟩⟩) 0 ⟨43922⟩ 247632

def event247634 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44613⟩⟩) (.authority (.operator))

def exact247635RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨44613⟩⟩]⟩, (1)⟩]

theorem exact247635RawTermsValid :
    exact247635RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event247635 : Event := .resultExact (⟨.program ⟨257⟩, ⟨44613⟩⟩) exact247635RawTerms (.finite 8192) 247634 .exactZero (none)

def event247636 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨136⟩⟩) (.authority (.operator))

def event247637 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨136⟩⟩) .exactZero

def event247638 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44138⟩⟩) 0 ⟨42773⟩ 247624

def event247639 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44138⟩⟩) 1 ⟨136⟩ 247637

def event247640 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44138⟩⟩) (.sum [.predecessor 0 247638 .coefficient, .predecessor 1 247639 .coefficient])

def event247641 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨44138⟩⟩) (.finite 52)

def event247642 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44139⟩⟩) 0 ⟨44138⟩ 247641

def event247643 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44139⟩⟩) (.identity (.predecessor 0 247642 .coefficient))

def exact247644RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨42772⟩⟩], []⟩, (1)⟩]

theorem exact247644RawTermsValid :
    exact247644RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event247644 : Event := .resultExact (⟨.program ⟨257⟩, ⟨44139⟩⟩) exact247644RawTerms (.finite 52) 247643 .exactZero (none)

def event247645 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6908⟩⟩) (.authority (.factStore))

def exact247646RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact247646RawTermsValid :
    exact247646RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event247646 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6908⟩⟩) exact247646RawTerms .large 247645 .exactZero (none)

def event247647 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44140⟩⟩) 0 ⟨6908⟩ 247646

def event247648 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44140⟩⟩) 1 ⟨44139⟩ 247644

def event247649 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44140⟩⟩) (.product (.predecessor 0 247647 .coefficient) (.predecessor 1 247648 .coefficient) (⟨false, false, none, none, none⟩))

def event247650 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨44140⟩⟩, .operator (⟨247646, 0⟩, ⟨247644, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨42772⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact247651RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨42772⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact247651RawTermsValid :
    exact247651RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event247651 : Event := .resultExact (⟨.program ⟨257⟩, ⟨44140⟩⟩) exact247651RawTerms .large 247649 .exactZero (none)

def event247652 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7194⟩⟩) 0 ⟨7177⟩ 247628

def event247653 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7194⟩⟩) (.authority (.operator))

def exact247654RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7194⟩⟩]⟩, (1)⟩]

theorem exact247654RawTermsValid :
    exact247654RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event247654 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7194⟩⟩) exact247654RawTerms .large 247653 .exactZero (none)

def event247655 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44141⟩⟩) 0 ⟨7194⟩ 247654

def event247656 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44141⟩⟩) 1 ⟨44140⟩ 247651

def event247657 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44141⟩⟩) (.sum [.predecessor 0 247655 .coefficient, .predecessor 1 247656 .coefficient])

def exact247658RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7194⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨42772⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact247658RawTermsValid :
    exact247658RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event247658 : Event := .resultExact (⟨.program ⟨257⟩, ⟨44141⟩⟩) exact247658RawTerms .large 247657 .exactZero (none)

def event247659 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44614⟩⟩) 0 ⟨44141⟩ 247658

def event247660 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44614⟩⟩) 1 ⟨44613⟩ 247635

def event247661 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44614⟩⟩) (.product (.predecessor 0 247659 .coefficient) (.predecessor 1 247660 .coefficient) (⟨false, false, none, none, none⟩))

def event247662 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨44614⟩⟩, .operator (⟨247658, 0⟩, ⟨247635, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7194⟩⟩, ⟨.program ⟨257⟩, ⟨44613⟩⟩]⟩, (1)⟩)

def event247663 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨44614⟩⟩, .operator (⟨247658, 1⟩, ⟨247635, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨42772⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨44613⟩⟩]⟩, (-1)⟩)

def event247664 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨44614⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨42772⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨44613⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨44613⟩⟩) ⟨43922⟩ 247632)

def event247665 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨44614⟩⟩, .relation 247664 0, ⟨[⟨.program ⟨257⟩, ⟨42772⟩⟩], [⟨.program ⟨257⟩, ⟨43922⟩⟩]⟩, (-1)⟩)

def exact247666RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7194⟩⟩, ⟨.program ⟨257⟩, ⟨44613⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨42772⟩⟩], [⟨.program ⟨257⟩, ⟨43922⟩⟩]⟩, (-1)⟩]

theorem exact247666RawTermsValid :
    exact247666RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event247666 : Event := .resultExact (⟨.program ⟨257⟩, ⟨44614⟩⟩) exact247666RawTerms .large 247661 .exactZero (none)

def event247667 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42976⟩⟩) 0 ⟨42773⟩ 247624

def event247668 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42976⟩⟩) (.authority (.programFamilyFact))

def exact247669RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨42976⟩⟩], []⟩, (1)⟩]

theorem exact247669RawTermsValid :
    exact247669RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event247669 : Event := .resultExact (⟨.program ⟨257⟩, ⟨42976⟩⟩) exact247669RawTerms (.finite 52) 247668 .exactZero (none)

def event247670 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42978⟩⟩) 0 ⟨6908⟩ 247646

def event247671 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42978⟩⟩) 1 ⟨42976⟩ 247669

def event247672 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42978⟩⟩) (.product (.predecessor 0 247670 .coefficient) (.predecessor 1 247671 .coefficient) (⟨false, true, none, none, some 1⟩))

def event247673 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨42978⟩⟩, .operator (⟨247646, 0⟩, ⟨247669, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨42976⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact247674RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨42976⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact247674RawTermsValid :
    exact247674RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event247674 : Event := .resultExact (⟨.program ⟨257⟩, ⟨42978⟩⟩) exact247674RawTerms .large 247672 .exactZero (none)

def event247675 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7227⟩⟩) 0 ⟨7177⟩ 247628

def event247676 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7227⟩⟩) (.authority (.operator))

def exact247677RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7227⟩⟩]⟩, (1)⟩]

theorem exact247677RawTermsValid :
    exact247677RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event247677 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7227⟩⟩) exact247677RawTerms .large 247676 .exactZero (none)

def event247678 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42979⟩⟩) 0 ⟨7227⟩ 247677

def event247679 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42979⟩⟩) 1 ⟨42978⟩ 247674

def event247680 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42979⟩⟩) (.sum [.predecessor 0 247678 .coefficient, .predecessor 1 247679 .coefficient])

def exact247681RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7227⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨42976⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact247681RawTermsValid :
    exact247681RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event247681 : Event := .resultExact (⟨.program ⟨257⟩, ⟨42979⟩⟩) exact247681RawTerms .large 247680 .exactZero (none)

def event247682 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44618⟩⟩) 0 ⟨42979⟩ 247681

def event247683 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44618⟩⟩) 1 ⟨44614⟩ 247666

def event247684 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44618⟩⟩) (.sum [.predecessor 0 247682 .coefficient, .predecessor 1 247683 .coefficient])

def exact247685RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7194⟩⟩, ⟨.program ⟨257⟩, ⟨44613⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7227⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨42772⟩⟩], [⟨.program ⟨257⟩, ⟨43922⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨42976⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact247685RawTermsValid :
    exact247685RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event247685 : Event := .resultExact (⟨.program ⟨257⟩, ⟨44618⟩⟩) exact247685RawTerms .large 247684 .exactZero (none)

def event247686 : Event := .preFoldPolynomial 247685 [⟨⟨[], [⟨.program ⟨257⟩, ⟨7194⟩⟩, ⟨.program ⟨257⟩, ⟨44613⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7227⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨42772⟩⟩], [⟨.program ⟨257⟩, ⟨43922⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨42976⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩] .exactZero none

def exact247687RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7194⟩⟩, ⟨.program ⟨257⟩, ⟨44613⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7227⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨42772⟩⟩], [⟨.program ⟨257⟩, ⟨43922⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨42976⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

def event247687 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨44618⟩⟩) 247686 exact247687RawTerms .large 247684 .exactZero (none)

def event247688 : Event := .specializationComputed (⟨.program ⟨257⟩, ⟨42773⟩⟩) ⟨⟨106⟩, ⟨89⟩, ⟨135⟩⟩ ⟨247530, 247688⟩

def event247689 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨43495⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43492⟩⟩]⟩) (1) 0 2 (.universal 247688 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43492⟩⟩]⟩) (none) 247687)

def event247690 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨43495⟩⟩, .relation 247689 1, ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7227⟩⟩]⟩, (1)⟩)

def event247691 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨43495⟩⟩, .relation 247689 0, ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7194⟩⟩, ⟨.program ⟨257⟩, ⟨44613⟩⟩]⟩, (-1)⟩)

def event247692 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨43495⟩⟩, .relation 247689 2, ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨42772⟩⟩], [⟨.program ⟨257⟩, ⟨43922⟩⟩]⟩, (1)⟩)

def event247693 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨43495⟩⟩, .relation 247689 3, ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨42976⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact247694RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7194⟩⟩, ⟨.program ⟨257⟩, ⟨44613⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7227⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨42772⟩⟩], [⟨.program ⟨257⟩, ⟨43922⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨42976⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact247694RawTermsValid :
    exact247694RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event247694 : Event := .resultExact (⟨.program ⟨257⟩, ⟨43495⟩⟩) exact247694RawTerms .large 247526 (.finite 202072841853861888) (some (247528))

def event247695 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44616⟩⟩) 0 ⟨43495⟩ 247694

def event247696 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44616⟩⟩) 1 ⟨44615⟩ 247516

def event247697 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44616⟩⟩) (.sum [.predecessor 0 247695 .coefficient, .predecessor 1 247696 .coefficient])

def event247698 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨44616⟩⟩, .operator (⟨247694, 0⟩, ⟨247516, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7194⟩⟩, ⟨.program ⟨257⟩, ⟨44613⟩⟩]⟩, (1)⟩)

def event247699 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨44616⟩⟩, .operator (⟨247694, 2⟩, ⟨247516, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨42772⟩⟩], [⟨.program ⟨257⟩, ⟨43922⟩⟩]⟩, (-1)⟩)

def event247700 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44616⟩⟩) (.sum [.result 247694 .summary, .result 247516 .summary])

def exact247701RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7227⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨42976⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact247701RawTermsValid :
    exact247701RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event247701 : Event := .resultExact (⟨.program ⟨257⟩, ⟨44616⟩⟩) exact247701RawTerms .large 247697 (.finite 32193718473625891320532869316608) (some (247700))

def event247702 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44617⟩⟩) 0 ⟨44616⟩ 247701

def event247703 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44617⟩⟩) 1 ⟨7154⟩ 15582

def event247704 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44617⟩⟩) (.product (.predecessor 0 247702 .coefficient) (.predecessor 1 247703 .coefficient) (⟨false, false, none, none, none⟩))

def event247705 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44617⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨7153⟩⟩]⟩) [⟨.result 15578 .coefficient, false, none⟩])

def event247706 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44617⟩⟩) (.product (.result 247701 .summary) (.transfer 247705) (⟨false, false, none, none, none⟩))

def event247707 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨44617⟩⟩, .operator (⟨247701, 0⟩, ⟨15582, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7227⟩⟩, ⟨.program ⟨257⟩, ⟨7153⟩⟩]⟩, (1)⟩)

def event247708 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨44617⟩⟩, .operator (⟨247701, 1⟩, ⟨15582, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨42976⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7153⟩⟩]⟩, (-1)⟩)

def event247709 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨44617⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨42976⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7153⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨7153⟩⟩) ⟨7042⟩ 15575)

def event247710 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨44617⟩⟩, .relation 247709 0, ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨42976⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact247711RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7227⟩⟩, ⟨.program ⟨257⟩, ⟨7153⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨42976⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact247711RawTermsValid :
    exact247711RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event247711 : Event := .resultExact (⟨.program ⟨257⟩, ⟨44617⟩⟩) exact247711RawTerms .large 247704 (.finite 345677419952135604401347317519683074129920) (some (247706))

def event247712 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41242⟩⟩) 0 ⟨7177⟩ 15500

def event247713 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41242⟩⟩) 1 ⟨41241⟩ 238218

def event247714 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41242⟩⟩) (.authority (.operator))

def exact247715RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨41242⟩⟩]⟩, (1)⟩]

theorem exact247715RawTermsValid :
    exact247715RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event247715 : Event := .resultExact (⟨.program ⟨257⟩, ⟨41242⟩⟩) exact247715RawTerms .large 247714 .exactZero (none)

def event247716 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41933⟩⟩) 0 ⟨41242⟩ 247715

def event247717 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41933⟩⟩) (.authority (.operator))

def exact247718RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨41933⟩⟩]⟩, (1)⟩]

theorem exact247718RawTermsValid :
    exact247718RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event247718 : Event := .resultExact (⟨.program ⟨257⟩, ⟨41933⟩⟩) exact247718RawTerms (.finite 8192) 247717 .exactZero (none)

def event247719 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41935⟩⟩) 0 ⟨41599⟩ 238502

def event247720 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41935⟩⟩) 1 ⟨41933⟩ 247718

def event247721 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41935⟩⟩) (.product (.predecessor 0 247719 .coefficient) (.predecessor 1 247720 .coefficient) (⟨false, false, none, none, none⟩))

def event247722 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41935⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨41933⟩⟩]⟩) [⟨.result 247718 .coefficient, false, none⟩])

def event247723 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41935⟩⟩) (.product (.result 238502 .summary) (.transfer 247722) (⟨false, false, none, none, none⟩))

def event247724 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨41935⟩⟩, .operator (⟨238502, 0⟩, ⟨247718, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨41933⟩⟩]⟩, (1)⟩)

def event247725 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨41935⟩⟩, .operator (⟨238502, 1⟩, ⟨247718, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨40092⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨41933⟩⟩]⟩, (-1)⟩)

def event247726 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨41935⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨40092⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨41933⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨41933⟩⟩) ⟨41242⟩ 247715)

def event247727 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨41935⟩⟩, .relation 247726 0, ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨40092⟩⟩], [⟨.program ⟨257⟩, ⟨41242⟩⟩]⟩, (-1)⟩)

def exact247728RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨41933⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨40092⟩⟩], [⟨.program ⟨257⟩, ⟨41242⟩⟩]⟩, (-1)⟩]

theorem exact247728RawTermsValid :
    exact247728RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event247728 : Event := .resultExact (⟨.program ⟨257⟩, ⟨41935⟩⟩) exact247728RawTerms .large 247721 (.finite 32193129122288627115968346193920) (some (247723))

def event247729 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40812⟩⟩) 0 ⟨40093⟩ 11400

def event247730 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨40812⟩⟩) (.authority (.relationPreimageSource ⟨86⟩))

def exact247731RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨40812⟩⟩]⟩, (1)⟩]

theorem exact247731RawTermsValid :
    exact247731RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event247731 : Event := .resultExact (⟨.program ⟨257⟩, ⟨40812⟩⟩) exact247731RawTerms (.finite 5647228698) 247730 .exactZero (none)

def event247732 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40814⟩⟩) 0 ⟨40812⟩ 247731

def event247733 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40814⟩⟩) 1 ⟨2370⟩ 4

def event247734 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨40814⟩⟩) (.scale (.predecessor 0 247732 .coefficient) (.value (.predecessor 1 247733 .coefficient)))

def exact247735RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨40812⟩⟩]⟩, (1)⟩]

theorem exact247735RawTermsValid :
    exact247735RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event247735 : Event := .resultExact (⟨.program ⟨257⟩, ⟨40814⟩⟩) exact247735RawTerms (.finite 5647228698) 247734 .exactZero (none)

def event247736 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40815⟩⟩) 0 ⟨5563⟩ 236870

def event247737 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40815⟩⟩) 1 ⟨40814⟩ 247735

def event247738 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨40815⟩⟩) (.product (.predecessor 0 247736 .coefficient) (.predecessor 1 247737 .coefficient) (⟨false, false, none, none, none⟩))

def event247739 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨40815⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨40812⟩⟩]⟩) [⟨.result 247731 .coefficient, false, none⟩])

def event247740 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨40815⟩⟩) (.product (.result 236870 .summary) (.transfer 247739) (⟨false, false, none, none, none⟩))

def event247741 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨40815⟩⟩, .operator (⟨236870, 0⟩, ⟨247735, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40812⟩⟩]⟩, (1)⟩)

def event247742 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨40813⟩⟩)

def event247743 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event247744 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event247745 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨4740⟩⟩) (.authority (.operator))

def event247746 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨4740⟩⟩) (.finite 4)

def event247747 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event247748 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event247749 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event247750 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event247751 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 247750

def event247752 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 247748

def event247753 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 247751 .coefficient) (.value (.predecessor 1 247752 .coefficient)))

def event247754 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event247755 : Event := .predecessor (⟨.program ⟨257⟩, ⟨4742⟩⟩) 0 ⟨392⟩ 247754

def event247756 : Event := .predecessor (⟨.program ⟨257⟩, ⟨4742⟩⟩) 1 ⟨4740⟩ 247746

def event247757 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨4742⟩⟩) (.sum [.predecessor 0 247755 .coefficient, .predecessor 1 247756 .coefficient])

def event247758 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨4742⟩⟩) (.finite 655344)

def event247759 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5559⟩⟩) 0 ⟨4742⟩ 247758

def event247760 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5559⟩⟩) 1 ⟨5426⟩ 247744

def event247761 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5559⟩⟩) (.identity (.predecessor 1 247760 .coefficient))

def event247762 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5559⟩⟩) (.finite 655360)

def event247763 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39746⟩⟩) 0 ⟨5559⟩ 247762

def event247764 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39746⟩⟩) (.authority (.programFamilyFact))

def exact247765RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨39746⟩⟩], []⟩, (1)⟩]

theorem exact247765RawTermsValid :
    exact247765RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event247765 : Event := .resultExact (⟨.program ⟨257⟩, ⟨39746⟩⟩) exact247765RawTerms (.finite 46) 247764 .exactZero (none)

def event247766 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14151⟩⟩) 0 ⟨5559⟩ 247762

def event247767 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14151⟩⟩) (.authority (.programFamilyFact))

def exact247768RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14151⟩⟩], []⟩, (1)⟩]

theorem exact247768RawTermsValid :
    exact247768RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event247768 : Event := .resultExact (⟨.program ⟨257⟩, ⟨14151⟩⟩) exact247768RawTerms (.finite 46) 247767 .exactZero (none)

def event247769 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39747⟩⟩) 0 ⟨14151⟩ 247768

def event247770 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39747⟩⟩) 1 ⟨39746⟩ 247765

def event247771 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39747⟩⟩) (.product (.predecessor 0 247769 .coefficient) (.predecessor 1 247770 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event247772 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39747⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨14151⟩⟩, ⟨.program ⟨257⟩, ⟨39746⟩⟩], []⟩) [⟨.result 247768 .coefficient, true, some 1⟩, ⟨.result 247765 .coefficient, true, some 1⟩])

def event247773 : Event := .survivorFold (1) 247772

def exact247774RawTerms : List Term := []

theorem exact247774RawTermsValid :
    exact247774RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event247774 : Event := .resultExact (⟨.program ⟨257⟩, ⟨39747⟩⟩) exact247774RawTerms (.finite 2116) 247771 (.finite 2116) (some (247772))

def event247775 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39748⟩⟩) 0 ⟨39747⟩ 247774

def event247776 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39748⟩⟩) (.identity (.predecessor 0 247775 .coefficient))

def event247777 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨39748⟩⟩) (.finite 2116)

def event247778 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40092⟩⟩) 0 ⟨39748⟩ 247777

def event247779 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨40092⟩⟩) (.authority (.programFamilyFact))

def exact247780RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨40092⟩⟩], []⟩, (1)⟩]

theorem exact247780RawTermsValid :
    exact247780RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event247780 : Event := .resultExact (⟨.program ⟨257⟩, ⟨40092⟩⟩) exact247780RawTerms (.finite 46) 247779 .exactZero (none)

def event247781 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40093⟩⟩) 0 ⟨40092⟩ 247780

def event247782 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨40093⟩⟩) (.identity (.predecessor 0 247781 .coefficient))

def event247783 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨40093⟩⟩) (.finite 46)

def event247784 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40812⟩⟩) 0 ⟨40093⟩ 247783

def event247785 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨40812⟩⟩) (.authority (.relationPreimageSource ⟨86⟩))

def exact247786RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨40812⟩⟩]⟩, (1)⟩]

theorem exact247786RawTermsValid :
    exact247786RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event247786 : Event := .resultExact (⟨.program ⟨257⟩, ⟨40812⟩⟩) exact247786RawTerms (.finite 5647228698) 247785 .exactZero (none)

def event247787 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35⟩⟩) (.authority (.operator))

def exact247788RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩]⟩, (1)⟩]

theorem exact247788RawTermsValid :
    exact247788RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event247788 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35⟩⟩) exact247788RawTerms .large 247787 .exactZero (none)

def event247789 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40813⟩⟩) 0 ⟨35⟩ 247788

def event247790 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40813⟩⟩) 1 ⟨40812⟩ 247786

def event247791 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨40813⟩⟩) (.product (.predecessor 0 247789 .coefficient) (.predecessor 1 247790 .coefficient) (⟨false, false, none, none, none⟩))

def event247792 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨40813⟩⟩, .operator (⟨247788, 0⟩, ⟨247786, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40812⟩⟩]⟩, (1)⟩)

def exact247793RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40812⟩⟩]⟩, (1)⟩]

theorem exact247793RawTermsValid :
    exact247793RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event247793 : Event := .resultExact (⟨.program ⟨257⟩, ⟨40813⟩⟩) exact247793RawTerms .large 247791 .exactZero (none)

def event247794 : Event := .preFoldPolynomial 247793 [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40812⟩⟩]⟩, (1)⟩] .exactZero none

def exact247795RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40812⟩⟩]⟩, (1)⟩]

def event247795 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨40813⟩⟩) 247794 exact247795RawTerms .large 247791 .exactZero (none)

def event247796 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨41938⟩⟩)

def event247797 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event247798 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event247799 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨4740⟩⟩) (.authority (.operator))

def event247800 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨4740⟩⟩) (.finite 4)

def event247801 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event247802 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event247803 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event247804 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event247805 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 247804

def event247806 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 247802

def event247807 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 247805 .coefficient) (.value (.predecessor 1 247806 .coefficient)))

def eventLeaf15472 : Array AnnotatedEvent := #[
  { event := event247552
    frameStart := 247530 },
  { event := event247553
    frameStart := 247530 },
  { event := event247554
    frameStart := 247530 },
  { event := event247555
    frameStart := 247530 },
  { event := event247556
    frameStart := 247530 },
  { event := event247557
    frameStart := 247530 },
  { event := event247558
    frameStart := 247530 },
  { event := event247559
    frameStart := 247530 },
  { event := event247560
    frameStart := 247530 },
  { event := event247561
    frameStart := 247530 },
  { event := event247562
    frameStart := 247530 },
  { event := event247563
    frameStart := 247530 },
  { event := event247564
    frameStart := 247530 },
  { event := event247565
    frameStart := 247530 },
  { event := event247566
    frameStart := 247530 },
  { event := event247567
    frameStart := 247530 }
]

def eventLeaf15473 : Array AnnotatedEvent := #[
  { event := event247568
    frameStart := 247530 },
  { event := event247569
    frameStart := 247530 },
  { event := event247570
    frameStart := 247530 },
  { event := event247571
    frameStart := 247530 },
  { event := event247572
    frameStart := 247530 },
  { event := event247573
    frameStart := 247530 },
  { event := event247574
    frameStart := 247530 },
  { event := event247575
    frameStart := 247530 },
  { event := event247576
    frameStart := 247530 },
  { event := event247577
    frameStart := 247530 },
  { event := event247578
    frameStart := 247530 },
  { event := event247579
    frameStart := 247530 },
  { event := event247580
    frameStart := 247530 },
  { event := event247581
    frameStart := 247530 },
  { event := event247582
    frameStart := 247530 },
  { event := event247583
    frameStart := 247530 }
]

def eventLeaf15474 : Array AnnotatedEvent := #[
  { event := event247584
    frameStart := 247584 },
  { event := event247585
    frameStart := 247584 },
  { event := event247586
    frameStart := 247584 },
  { event := event247587
    frameStart := 247584 },
  { event := event247588
    frameStart := 247584 },
  { event := event247589
    frameStart := 247584 },
  { event := event247590
    frameStart := 247584 },
  { event := event247591
    frameStart := 247584 },
  { event := event247592
    frameStart := 247584 },
  { event := event247593
    frameStart := 247584 },
  { event := event247594
    frameStart := 247584 },
  { event := event247595
    frameStart := 247584 },
  { event := event247596
    frameStart := 247584 },
  { event := event247597
    frameStart := 247584 },
  { event := event247598
    frameStart := 247584 },
  { event := event247599
    frameStart := 247584 }
]

def eventLeaf15475 : Array AnnotatedEvent := #[
  { event := event247600
    frameStart := 247584 },
  { event := event247601
    frameStart := 247584 },
  { event := event247602
    frameStart := 247584 },
  { event := event247603
    frameStart := 247584 },
  { event := event247604
    frameStart := 247584 },
  { event := event247605
    frameStart := 247584 },
  { event := event247606
    frameStart := 247584 },
  { event := event247607
    frameStart := 247584 },
  { event := event247608
    frameStart := 247584 },
  { event := event247609
    frameStart := 247584 },
  { event := event247610
    frameStart := 247584 },
  { event := event247611
    frameStart := 247584 },
  { event := event247612
    frameStart := 247584 },
  { event := event247613
    frameStart := 247584 },
  { event := event247614
    frameStart := 247584 },
  { event := event247615
    frameStart := 247584 }
]

def eventLeaf15476 : Array AnnotatedEvent := #[
  { event := event247616
    frameStart := 247584 },
  { event := event247617
    frameStart := 247584 },
  { event := event247618
    frameStart := 247584 },
  { event := event247619
    frameStart := 247584 },
  { event := event247620
    frameStart := 247584 },
  { event := event247621
    frameStart := 247584 },
  { event := event247622
    frameStart := 247584 },
  { event := event247623
    frameStart := 247584 },
  { event := event247624
    frameStart := 247584 },
  { event := event247625
    frameStart := 247584 },
  { event := event247626
    frameStart := 247584 },
  { event := event247627
    frameStart := 247584 },
  { event := event247628
    frameStart := 247584 },
  { event := event247629
    frameStart := 247584 },
  { event := event247630
    frameStart := 247584 },
  { event := event247631
    frameStart := 247584 }
]

def eventLeaf15477 : Array AnnotatedEvent := #[
  { event := event247632
    frameStart := 247584 },
  { event := event247633
    frameStart := 247584 },
  { event := event247634
    frameStart := 247584 },
  { event := event247635
    frameStart := 247584 },
  { event := event247636
    frameStart := 247584 },
  { event := event247637
    frameStart := 247584 },
  { event := event247638
    frameStart := 247584 },
  { event := event247639
    frameStart := 247584 },
  { event := event247640
    frameStart := 247584 },
  { event := event247641
    frameStart := 247584 },
  { event := event247642
    frameStart := 247584 },
  { event := event247643
    frameStart := 247584 },
  { event := event247644
    frameStart := 247584 },
  { event := event247645
    frameStart := 247584 },
  { event := event247646
    frameStart := 247584 },
  { event := event247647
    frameStart := 247584 }
]

def eventLeaf15478 : Array AnnotatedEvent := #[
  { event := event247648
    frameStart := 247584 },
  { event := event247649
    frameStart := 247584 },
  { event := event247650
    frameStart := 247584 },
  { event := event247651
    frameStart := 247584 },
  { event := event247652
    frameStart := 247584 },
  { event := event247653
    frameStart := 247584 },
  { event := event247654
    frameStart := 247584 },
  { event := event247655
    frameStart := 247584 },
  { event := event247656
    frameStart := 247584 },
  { event := event247657
    frameStart := 247584 },
  { event := event247658
    frameStart := 247584 },
  { event := event247659
    frameStart := 247584 },
  { event := event247660
    frameStart := 247584 },
  { event := event247661
    frameStart := 247584 },
  { event := event247662
    frameStart := 247584 },
  { event := event247663
    frameStart := 247584 }
]

def eventLeaf15479 : Array AnnotatedEvent := #[
  { event := event247664
    frameStart := 247584 },
  { event := event247665
    frameStart := 247584 },
  { event := event247666
    frameStart := 247584 },
  { event := event247667
    frameStart := 247584 },
  { event := event247668
    frameStart := 247584 },
  { event := event247669
    frameStart := 247584 },
  { event := event247670
    frameStart := 247584 },
  { event := event247671
    frameStart := 247584 },
  { event := event247672
    frameStart := 247584 },
  { event := event247673
    frameStart := 247584 },
  { event := event247674
    frameStart := 247584 },
  { event := event247675
    frameStart := 247584 },
  { event := event247676
    frameStart := 247584 },
  { event := event247677
    frameStart := 247584 },
  { event := event247678
    frameStart := 247584 },
  { event := event247679
    frameStart := 247584 }
]

def eventLeaf15480 : Array AnnotatedEvent := #[
  { event := event247680
    frameStart := 247584 },
  { event := event247681
    frameStart := 247584 },
  { event := event247682
    frameStart := 247584 },
  { event := event247683
    frameStart := 247584 },
  { event := event247684
    frameStart := 247584 },
  { event := event247685
    frameStart := 247584 },
  { event := event247686
    frameStart := 247584 },
  { event := event247687
    frameStart := 247584 },
  { event := event247688
    frameStart := 0 },
  { event := event247689
    frameStart := 0 },
  { event := event247690
    frameStart := 0 },
  { event := event247691
    frameStart := 0 },
  { event := event247692
    frameStart := 0 },
  { event := event247693
    frameStart := 0 },
  { event := event247694
    frameStart := 0 },
  { event := event247695
    frameStart := 0 }
]

def eventLeaf15481 : Array AnnotatedEvent := #[
  { event := event247696
    frameStart := 0 },
  { event := event247697
    frameStart := 0 },
  { event := event247698
    frameStart := 0 },
  { event := event247699
    frameStart := 0 },
  { event := event247700
    frameStart := 0 },
  { event := event247701
    frameStart := 0 },
  { event := event247702
    frameStart := 0 },
  { event := event247703
    frameStart := 0 },
  { event := event247704
    frameStart := 0 },
  { event := event247705
    frameStart := 0 },
  { event := event247706
    frameStart := 0 },
  { event := event247707
    frameStart := 0 },
  { event := event247708
    frameStart := 0 },
  { event := event247709
    frameStart := 0 },
  { event := event247710
    frameStart := 0 },
  { event := event247711
    frameStart := 0 }
]

def eventLeaf15482 : Array AnnotatedEvent := #[
  { event := event247712
    frameStart := 0 },
  { event := event247713
    frameStart := 0 },
  { event := event247714
    frameStart := 0 },
  { event := event247715
    frameStart := 0 },
  { event := event247716
    frameStart := 0 },
  { event := event247717
    frameStart := 0 },
  { event := event247718
    frameStart := 0 },
  { event := event247719
    frameStart := 0 },
  { event := event247720
    frameStart := 0 },
  { event := event247721
    frameStart := 0 },
  { event := event247722
    frameStart := 0 },
  { event := event247723
    frameStart := 0 },
  { event := event247724
    frameStart := 0 },
  { event := event247725
    frameStart := 0 },
  { event := event247726
    frameStart := 0 },
  { event := event247727
    frameStart := 0 }
]

def eventLeaf15483 : Array AnnotatedEvent := #[
  { event := event247728
    frameStart := 0 },
  { event := event247729
    frameStart := 0 },
  { event := event247730
    frameStart := 0 },
  { event := event247731
    frameStart := 0 },
  { event := event247732
    frameStart := 0 },
  { event := event247733
    frameStart := 0 },
  { event := event247734
    frameStart := 0 },
  { event := event247735
    frameStart := 0 },
  { event := event247736
    frameStart := 0 },
  { event := event247737
    frameStart := 0 },
  { event := event247738
    frameStart := 0 },
  { event := event247739
    frameStart := 0 },
  { event := event247740
    frameStart := 0 },
  { event := event247741
    frameStart := 0 },
  { event := event247742
    frameStart := 247742 },
  { event := event247743
    frameStart := 247742 }
]

def eventLeaf15484 : Array AnnotatedEvent := #[
  { event := event247744
    frameStart := 247742 },
  { event := event247745
    frameStart := 247742 },
  { event := event247746
    frameStart := 247742 },
  { event := event247747
    frameStart := 247742 },
  { event := event247748
    frameStart := 247742 },
  { event := event247749
    frameStart := 247742 },
  { event := event247750
    frameStart := 247742 },
  { event := event247751
    frameStart := 247742 },
  { event := event247752
    frameStart := 247742 },
  { event := event247753
    frameStart := 247742 },
  { event := event247754
    frameStart := 247742 },
  { event := event247755
    frameStart := 247742 },
  { event := event247756
    frameStart := 247742 },
  { event := event247757
    frameStart := 247742 },
  { event := event247758
    frameStart := 247742 },
  { event := event247759
    frameStart := 247742 }
]

def eventLeaf15485 : Array AnnotatedEvent := #[
  { event := event247760
    frameStart := 247742 },
  { event := event247761
    frameStart := 247742 },
  { event := event247762
    frameStart := 247742 },
  { event := event247763
    frameStart := 247742 },
  { event := event247764
    frameStart := 247742 },
  { event := event247765
    frameStart := 247742 },
  { event := event247766
    frameStart := 247742 },
  { event := event247767
    frameStart := 247742 },
  { event := event247768
    frameStart := 247742 },
  { event := event247769
    frameStart := 247742 },
  { event := event247770
    frameStart := 247742 },
  { event := event247771
    frameStart := 247742 },
  { event := event247772
    frameStart := 247742 },
  { event := event247773
    frameStart := 247742 },
  { event := event247774
    frameStart := 247742 },
  { event := event247775
    frameStart := 247742 }
]

def eventLeaf15486 : Array AnnotatedEvent := #[
  { event := event247776
    frameStart := 247742 },
  { event := event247777
    frameStart := 247742 },
  { event := event247778
    frameStart := 247742 },
  { event := event247779
    frameStart := 247742 },
  { event := event247780
    frameStart := 247742 },
  { event := event247781
    frameStart := 247742 },
  { event := event247782
    frameStart := 247742 },
  { event := event247783
    frameStart := 247742 },
  { event := event247784
    frameStart := 247742 },
  { event := event247785
    frameStart := 247742 },
  { event := event247786
    frameStart := 247742 },
  { event := event247787
    frameStart := 247742 },
  { event := event247788
    frameStart := 247742 },
  { event := event247789
    frameStart := 247742 },
  { event := event247790
    frameStart := 247742 },
  { event := event247791
    frameStart := 247742 }
]

def eventLeaf15487 : Array AnnotatedEvent := #[
  { event := event247792
    frameStart := 247742 },
  { event := event247793
    frameStart := 247742 },
  { event := event247794
    frameStart := 247742 },
  { event := event247795
    frameStart := 247742 },
  { event := event247796
    frameStart := 247796 },
  { event := event247797
    frameStart := 247796 },
  { event := event247798
    frameStart := 247796 },
  { event := event247799
    frameStart := 247796 },
  { event := event247800
    frameStart := 247796 },
  { event := event247801
    frameStart := 247796 },
  { event := event247802
    frameStart := 247796 },
  { event := event247803
    frameStart := 247796 },
  { event := event247804
    frameStart := 247796 },
  { event := event247805
    frameStart := 247796 },
  { event := event247806
    frameStart := 247796 },
  { event := event247807
    frameStart := 247796 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events967
