import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events510

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event130560 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42379⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨14421⟩⟩, ⟨.program ⟨257⟩, ⟨42378⟩⟩], []⟩) [⟨.result 130556 .coefficient, true, some 1⟩, ⟨.result 130553 .coefficient, true, some 1⟩])

def event130561 : Event := .survivorFold (1) 130560

def exact130562RawTerms : List Term := []

theorem exact130562RawTermsValid :
    exact130562RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event130562 : Event := .resultExact (⟨.program ⟨257⟩, ⟨42379⟩⟩) exact130562RawTerms (.finite 2704) 130559 (.finite 2704) (some (130560))

def event130563 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42380⟩⟩) 0 ⟨42379⟩ 130562

def event130564 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42380⟩⟩) (.identity (.predecessor 0 130563 .coefficient))

def event130565 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨42380⟩⟩) (.finite 2704)

def event130566 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42756⟩⟩) 0 ⟨42380⟩ 130565

def event130567 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42756⟩⟩) (.authority (.programFamilyFact))

def exact130568RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨42756⟩⟩], []⟩, (1)⟩]

theorem exact130568RawTermsValid :
    exact130568RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event130568 : Event := .resultExact (⟨.program ⟨257⟩, ⟨42756⟩⟩) exact130568RawTerms (.finite 52) 130567 .exactZero (none)

def event130569 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42757⟩⟩) 0 ⟨42756⟩ 130568

def event130570 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42757⟩⟩) (.identity (.predecessor 0 130569 .coefficient))

def event130571 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨42757⟩⟩) (.finite 52)

def event130572 : Event := .predecessor (⟨.program ⟨257⟩, ⟨43452⟩⟩) 0 ⟨42757⟩ 130571

def event130573 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨43452⟩⟩) (.authority (.relationPreimageSource ⟨89⟩))

def exact130574RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨43452⟩⟩]⟩, (1)⟩]

theorem exact130574RawTermsValid :
    exact130574RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event130574 : Event := .resultExact (⟨.program ⟨257⟩, ⟨43452⟩⟩) exact130574RawTerms (.finite 5647228698) 130573 .exactZero (none)

def event130575 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35⟩⟩) (.authority (.operator))

def exact130576RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩]⟩, (1)⟩]

theorem exact130576RawTermsValid :
    exact130576RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event130576 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35⟩⟩) exact130576RawTerms .large 130575 .exactZero (none)

def event130577 : Event := .predecessor (⟨.program ⟨257⟩, ⟨43453⟩⟩) 0 ⟨35⟩ 130576

def event130578 : Event := .predecessor (⟨.program ⟨257⟩, ⟨43453⟩⟩) 1 ⟨43452⟩ 130574

def event130579 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨43453⟩⟩) (.product (.predecessor 0 130577 .coefficient) (.predecessor 1 130578 .coefficient) (⟨false, false, none, none, none⟩))

def event130580 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨43453⟩⟩, .operator (⟨130576, 0⟩, ⟨130574, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43452⟩⟩]⟩, (1)⟩)

def exact130581RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43452⟩⟩]⟩, (1)⟩]

theorem exact130581RawTermsValid :
    exact130581RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event130581 : Event := .resultExact (⟨.program ⟨257⟩, ⟨43453⟩⟩) exact130581RawTerms .large 130579 .exactZero (none)

def event130582 : Event := .preFoldPolynomial 130581 [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43452⟩⟩]⟩, (1)⟩] .exactZero none

def exact130583RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43452⟩⟩]⟩, (1)⟩]

def event130583 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨43453⟩⟩) 130582 exact130583RawTerms .large 130579 .exactZero (none)

def event130584 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨44568⟩⟩)

def event130585 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event130586 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event130587 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨2942⟩⟩) (.authority (.operator))

def event130588 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨2942⟩⟩) (.finite 12)

def event130589 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event130590 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event130591 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event130592 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event130593 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 130592

def event130594 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 130590

def event130595 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 130593 .coefficient) (.value (.predecessor 1 130594 .coefficient)))

def event130596 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event130597 : Event := .predecessor (⟨.program ⟨257⟩, ⟨2944⟩⟩) 0 ⟨392⟩ 130596

def event130598 : Event := .predecessor (⟨.program ⟨257⟩, ⟨2944⟩⟩) 1 ⟨2942⟩ 130588

def event130599 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨2944⟩⟩) (.sum [.predecessor 0 130597 .coefficient, .predecessor 1 130598 .coefficient])

def event130600 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨2944⟩⟩) (.finite 655352)

def event130601 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5523⟩⟩) 0 ⟨2944⟩ 130600

def event130602 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5523⟩⟩) 1 ⟨5426⟩ 130586

def event130603 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5523⟩⟩) (.identity (.predecessor 1 130602 .coefficient))

def event130604 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5523⟩⟩) (.finite 655360)

def event130605 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42378⟩⟩) 0 ⟨5523⟩ 130604

def event130606 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42378⟩⟩) (.authority (.programFamilyFact))

def exact130607RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨42378⟩⟩], []⟩, (1)⟩]

theorem exact130607RawTermsValid :
    exact130607RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event130607 : Event := .resultExact (⟨.program ⟨257⟩, ⟨42378⟩⟩) exact130607RawTerms (.finite 52) 130606 .exactZero (none)

def event130608 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14421⟩⟩) 0 ⟨5523⟩ 130604

def event130609 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14421⟩⟩) (.authority (.programFamilyFact))

def exact130610RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14421⟩⟩], []⟩, (1)⟩]

theorem exact130610RawTermsValid :
    exact130610RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event130610 : Event := .resultExact (⟨.program ⟨257⟩, ⟨14421⟩⟩) exact130610RawTerms (.finite 52) 130609 .exactZero (none)

def event130611 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42379⟩⟩) 0 ⟨14421⟩ 130610

def event130612 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42379⟩⟩) 1 ⟨42378⟩ 130607

def event130613 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42379⟩⟩) (.product (.predecessor 0 130611 .coefficient) (.predecessor 1 130612 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event130614 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨42379⟩⟩, .operator (⟨130610, 0⟩, ⟨130607, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨14421⟩⟩, ⟨.program ⟨257⟩, ⟨42378⟩⟩], []⟩, (1)⟩)

def exact130615RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14421⟩⟩, ⟨.program ⟨257⟩, ⟨42378⟩⟩], []⟩, (1)⟩]

theorem exact130615RawTermsValid :
    exact130615RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event130615 : Event := .resultExact (⟨.program ⟨257⟩, ⟨42379⟩⟩) exact130615RawTerms (.finite 2704) 130613 .exactZero (none)

def event130616 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42380⟩⟩) 0 ⟨42379⟩ 130615

def event130617 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42380⟩⟩) (.identity (.predecessor 0 130616 .coefficient))

def event130618 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨42380⟩⟩) (.finite 2704)

def event130619 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42756⟩⟩) 0 ⟨42380⟩ 130618

def event130620 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42756⟩⟩) (.authority (.programFamilyFact))

def exact130621RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨42756⟩⟩], []⟩, (1)⟩]

theorem exact130621RawTermsValid :
    exact130621RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event130621 : Event := .resultExact (⟨.program ⟨257⟩, ⟨42756⟩⟩) exact130621RawTerms (.finite 52) 130620 .exactZero (none)

def event130622 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42757⟩⟩) 0 ⟨42756⟩ 130621

def event130623 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42757⟩⟩) (.identity (.predecessor 0 130622 .coefficient))

def event130624 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨42757⟩⟩) (.finite 52)

def event130625 : Event := .predecessor (⟨.program ⟨257⟩, ⟨43903⟩⟩) 0 ⟨42757⟩ 130624

def event130626 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨43903⟩⟩) (.authority (.programFamilyFact))

def event130627 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨43903⟩⟩) (.finite 3720)

def event130628 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨7177⟩⟩) .missing

def event130629 : Event := .predecessor (⟨.program ⟨257⟩, ⟨43904⟩⟩) 0 ⟨7177⟩ 130628

def event130630 : Event := .predecessor (⟨.program ⟨257⟩, ⟨43904⟩⟩) 1 ⟨43903⟩ 130627

def event130631 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨43904⟩⟩) (.authority (.operator))

def exact130632RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨43904⟩⟩]⟩, (1)⟩]

theorem exact130632RawTermsValid :
    exact130632RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event130632 : Event := .resultExact (⟨.program ⟨257⟩, ⟨43904⟩⟩) exact130632RawTerms .large 130631 .exactZero (none)

def event130633 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44563⟩⟩) 0 ⟨43904⟩ 130632

def event130634 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44563⟩⟩) (.authority (.operator))

def exact130635RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨44563⟩⟩]⟩, (1)⟩]

theorem exact130635RawTermsValid :
    exact130635RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event130635 : Event := .resultExact (⟨.program ⟨257⟩, ⟨44563⟩⟩) exact130635RawTerms (.finite 8192) 130634 .exactZero (none)

def event130636 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨136⟩⟩) (.authority (.operator))

def event130637 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨136⟩⟩) .exactZero

def event130638 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44130⟩⟩) 0 ⟨42757⟩ 130624

def event130639 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44130⟩⟩) 1 ⟨136⟩ 130637

def event130640 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44130⟩⟩) (.sum [.predecessor 0 130638 .coefficient, .predecessor 1 130639 .coefficient])

def event130641 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨44130⟩⟩) (.finite 52)

def event130642 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44131⟩⟩) 0 ⟨44130⟩ 130641

def event130643 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44131⟩⟩) (.identity (.predecessor 0 130642 .coefficient))

def exact130644RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨42756⟩⟩], []⟩, (1)⟩]

theorem exact130644RawTermsValid :
    exact130644RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event130644 : Event := .resultExact (⟨.program ⟨257⟩, ⟨44131⟩⟩) exact130644RawTerms (.finite 52) 130643 .exactZero (none)

def event130645 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6908⟩⟩) (.authority (.factStore))

def exact130646RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact130646RawTermsValid :
    exact130646RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event130646 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6908⟩⟩) exact130646RawTerms .large 130645 .exactZero (none)

def event130647 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44132⟩⟩) 0 ⟨6908⟩ 130646

def event130648 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44132⟩⟩) 1 ⟨44131⟩ 130644

def event130649 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44132⟩⟩) (.product (.predecessor 0 130647 .coefficient) (.predecessor 1 130648 .coefficient) (⟨false, false, none, none, none⟩))

def event130650 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨44132⟩⟩, .operator (⟨130646, 0⟩, ⟨130644, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨42756⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact130651RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨42756⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact130651RawTermsValid :
    exact130651RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event130651 : Event := .resultExact (⟨.program ⟨257⟩, ⟨44132⟩⟩) exact130651RawTerms .large 130649 .exactZero (none)

def event130652 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7194⟩⟩) 0 ⟨7177⟩ 130628

def event130653 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7194⟩⟩) (.authority (.operator))

def exact130654RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7194⟩⟩]⟩, (1)⟩]

theorem exact130654RawTermsValid :
    exact130654RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event130654 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7194⟩⟩) exact130654RawTerms .large 130653 .exactZero (none)

def event130655 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44133⟩⟩) 0 ⟨7194⟩ 130654

def event130656 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44133⟩⟩) 1 ⟨44132⟩ 130651

def event130657 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44133⟩⟩) (.sum [.predecessor 0 130655 .coefficient, .predecessor 1 130656 .coefficient])

def exact130658RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7194⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨42756⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact130658RawTermsValid :
    exact130658RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event130658 : Event := .resultExact (⟨.program ⟨257⟩, ⟨44133⟩⟩) exact130658RawTerms .large 130657 .exactZero (none)

def event130659 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44564⟩⟩) 0 ⟨44133⟩ 130658

def event130660 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44564⟩⟩) 1 ⟨44563⟩ 130635

def event130661 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44564⟩⟩) (.product (.predecessor 0 130659 .coefficient) (.predecessor 1 130660 .coefficient) (⟨false, false, none, none, none⟩))

def event130662 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨44564⟩⟩, .operator (⟨130658, 0⟩, ⟨130635, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7194⟩⟩, ⟨.program ⟨257⟩, ⟨44563⟩⟩]⟩, (1)⟩)

def event130663 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨44564⟩⟩, .operator (⟨130658, 1⟩, ⟨130635, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨42756⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨44563⟩⟩]⟩, (-1)⟩)

def event130664 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨44564⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨42756⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨44563⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨44563⟩⟩) ⟨43904⟩ 130632)

def event130665 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨44564⟩⟩, .relation 130664 0, ⟨[⟨.program ⟨257⟩, ⟨42756⟩⟩], [⟨.program ⟨257⟩, ⟨43904⟩⟩]⟩, (-1)⟩)

def exact130666RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7194⟩⟩, ⟨.program ⟨257⟩, ⟨44563⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨42756⟩⟩], [⟨.program ⟨257⟩, ⟨43904⟩⟩]⟩, (-1)⟩]

theorem exact130666RawTermsValid :
    exact130666RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event130666 : Event := .resultExact (⟨.program ⟨257⟩, ⟨44564⟩⟩) exact130666RawTerms .large 130661 .exactZero (none)

def event130667 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42950⟩⟩) 0 ⟨42757⟩ 130624

def event130668 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42950⟩⟩) (.authority (.programFamilyFact))

def exact130669RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨42950⟩⟩], []⟩, (1)⟩]

theorem exact130669RawTermsValid :
    exact130669RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event130669 : Event := .resultExact (⟨.program ⟨257⟩, ⟨42950⟩⟩) exact130669RawTerms (.finite 52) 130668 .exactZero (none)

def event130670 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42952⟩⟩) 0 ⟨6908⟩ 130646

def event130671 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42952⟩⟩) 1 ⟨42950⟩ 130669

def event130672 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42952⟩⟩) (.product (.predecessor 0 130670 .coefficient) (.predecessor 1 130671 .coefficient) (⟨false, true, none, none, some 1⟩))

def event130673 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨42952⟩⟩, .operator (⟨130646, 0⟩, ⟨130669, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨42950⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact130674RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨42950⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact130674RawTermsValid :
    exact130674RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event130674 : Event := .resultExact (⟨.program ⟨257⟩, ⟨42952⟩⟩) exact130674RawTerms .large 130672 .exactZero (none)

def event130675 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7227⟩⟩) 0 ⟨7177⟩ 130628

def event130676 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7227⟩⟩) (.authority (.operator))

def exact130677RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7227⟩⟩]⟩, (1)⟩]

theorem exact130677RawTermsValid :
    exact130677RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event130677 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7227⟩⟩) exact130677RawTerms .large 130676 .exactZero (none)

def event130678 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42953⟩⟩) 0 ⟨7227⟩ 130677

def event130679 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42953⟩⟩) 1 ⟨42952⟩ 130674

def event130680 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42953⟩⟩) (.sum [.predecessor 0 130678 .coefficient, .predecessor 1 130679 .coefficient])

def exact130681RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7227⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨42950⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact130681RawTermsValid :
    exact130681RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event130681 : Event := .resultExact (⟨.program ⟨257⟩, ⟨42953⟩⟩) exact130681RawTerms .large 130680 .exactZero (none)

def event130682 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44568⟩⟩) 0 ⟨42953⟩ 130681

def event130683 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44568⟩⟩) 1 ⟨44564⟩ 130666

def event130684 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44568⟩⟩) (.sum [.predecessor 0 130682 .coefficient, .predecessor 1 130683 .coefficient])

def exact130685RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7194⟩⟩, ⟨.program ⟨257⟩, ⟨44563⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7227⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨42756⟩⟩], [⟨.program ⟨257⟩, ⟨43904⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨42950⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact130685RawTermsValid :
    exact130685RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event130685 : Event := .resultExact (⟨.program ⟨257⟩, ⟨44568⟩⟩) exact130685RawTerms .large 130684 .exactZero (none)

def event130686 : Event := .preFoldPolynomial 130685 [⟨⟨[], [⟨.program ⟨257⟩, ⟨7194⟩⟩, ⟨.program ⟨257⟩, ⟨44563⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7227⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨42756⟩⟩], [⟨.program ⟨257⟩, ⟨43904⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨42950⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩] .exactZero none

def exact130687RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7194⟩⟩, ⟨.program ⟨257⟩, ⟨44563⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7227⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨42756⟩⟩], [⟨.program ⟨257⟩, ⟨43904⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨42950⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

def event130687 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨44568⟩⟩) 130686 exact130687RawTerms .large 130684 .exactZero (none)

def event130688 : Event := .specializationComputed (⟨.program ⟨257⟩, ⟨42757⟩⟩) ⟨⟨106⟩, ⟨89⟩, ⟨135⟩⟩ ⟨130530, 130688⟩

def event130689 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨43455⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43452⟩⟩]⟩) (1) 0 2 (.universal 130688 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43452⟩⟩]⟩) (none) 130687)

def event130690 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨43455⟩⟩, .relation 130689 1, ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7227⟩⟩]⟩, (1)⟩)

def event130691 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨43455⟩⟩, .relation 130689 0, ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7194⟩⟩, ⟨.program ⟨257⟩, ⟨44563⟩⟩]⟩, (-1)⟩)

def event130692 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨43455⟩⟩, .relation 130689 2, ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨42756⟩⟩], [⟨.program ⟨257⟩, ⟨43904⟩⟩]⟩, (1)⟩)

def event130693 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨43455⟩⟩, .relation 130689 3, ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨42950⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact130694RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7194⟩⟩, ⟨.program ⟨257⟩, ⟨44563⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7227⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨42756⟩⟩], [⟨.program ⟨257⟩, ⟨43904⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨42950⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact130694RawTermsValid :
    exact130694RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event130694 : Event := .resultExact (⟨.program ⟨257⟩, ⟨43455⟩⟩) exact130694RawTerms .large 130526 (.finite 202072841853861888) (some (130528))

def event130695 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44566⟩⟩) 0 ⟨43455⟩ 130694

def event130696 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44566⟩⟩) 1 ⟨44565⟩ 130516

def event130697 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44566⟩⟩) (.sum [.predecessor 0 130695 .coefficient, .predecessor 1 130696 .coefficient])

def event130698 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨44566⟩⟩, .operator (⟨130694, 0⟩, ⟨130516, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7194⟩⟩, ⟨.program ⟨257⟩, ⟨44563⟩⟩]⟩, (1)⟩)

def event130699 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨44566⟩⟩, .operator (⟨130694, 2⟩, ⟨130516, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨42756⟩⟩], [⟨.program ⟨257⟩, ⟨43904⟩⟩]⟩, (-1)⟩)

def event130700 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44566⟩⟩) (.sum [.result 130694 .summary, .result 130516 .summary])

def exact130701RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7227⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨42950⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact130701RawTermsValid :
    exact130701RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event130701 : Event := .resultExact (⟨.program ⟨257⟩, ⟨44566⟩⟩) exact130701RawTerms .large 130697 (.finite 32193718473625891320532869316608) (some (130700))

def event130702 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44567⟩⟩) 0 ⟨44566⟩ 130701

def event130703 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44567⟩⟩) 1 ⟨7154⟩ 15582

def event130704 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44567⟩⟩) (.product (.predecessor 0 130702 .coefficient) (.predecessor 1 130703 .coefficient) (⟨false, false, none, none, none⟩))

def event130705 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44567⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨7153⟩⟩]⟩) [⟨.result 15578 .coefficient, false, none⟩])

def event130706 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44567⟩⟩) (.product (.result 130701 .summary) (.transfer 130705) (⟨false, false, none, none, none⟩))

def event130707 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨44567⟩⟩, .operator (⟨130701, 0⟩, ⟨15582, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7227⟩⟩, ⟨.program ⟨257⟩, ⟨7153⟩⟩]⟩, (1)⟩)

def event130708 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨44567⟩⟩, .operator (⟨130701, 1⟩, ⟨15582, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨42950⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7153⟩⟩]⟩, (-1)⟩)

def event130709 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨44567⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨42950⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7153⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨7153⟩⟩) ⟨7042⟩ 15575)

def event130710 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨44567⟩⟩, .relation 130709 0, ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨42950⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact130711RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7227⟩⟩, ⟨.program ⟨257⟩, ⟨7153⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨42950⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact130711RawTermsValid :
    exact130711RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event130711 : Event := .resultExact (⟨.program ⟨257⟩, ⟨44567⟩⟩) exact130711RawTerms .large 130704 (.finite 345677419952135604401347317519683074129920) (some (130706))

def event130712 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41224⟩⟩) 0 ⟨7177⟩ 15500

def event130713 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41224⟩⟩) 1 ⟨41223⟩ 121218

def event130714 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41224⟩⟩) (.authority (.operator))

def exact130715RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨41224⟩⟩]⟩, (1)⟩]

theorem exact130715RawTermsValid :
    exact130715RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event130715 : Event := .resultExact (⟨.program ⟨257⟩, ⟨41224⟩⟩) exact130715RawTerms .large 130714 .exactZero (none)

def event130716 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41883⟩⟩) 0 ⟨41224⟩ 130715

def event130717 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41883⟩⟩) (.authority (.operator))

def exact130718RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨41883⟩⟩]⟩, (1)⟩]

theorem exact130718RawTermsValid :
    exact130718RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event130718 : Event := .resultExact (⟨.program ⟨257⟩, ⟨41883⟩⟩) exact130718RawTerms (.finite 8192) 130717 .exactZero (none)

def event130719 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41885⟩⟩) 0 ⟨41577⟩ 121502

def event130720 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41885⟩⟩) 1 ⟨41883⟩ 130718

def event130721 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41885⟩⟩) (.product (.predecessor 0 130719 .coefficient) (.predecessor 1 130720 .coefficient) (⟨false, false, none, none, none⟩))

def event130722 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41885⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨41883⟩⟩]⟩) [⟨.result 130718 .coefficient, false, none⟩])

def event130723 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41885⟩⟩) (.product (.result 121502 .summary) (.transfer 130722) (⟨false, false, none, none, none⟩))

def event130724 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨41885⟩⟩, .operator (⟨121502, 0⟩, ⟨130718, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨41883⟩⟩]⟩, (1)⟩)

def event130725 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨41885⟩⟩, .operator (⟨121502, 1⟩, ⟨130718, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨40076⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨41883⟩⟩]⟩, (-1)⟩)

def event130726 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨41885⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨40076⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨41883⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨41883⟩⟩) ⟨41224⟩ 130715)

def event130727 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨41885⟩⟩, .relation 130726 0, ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨40076⟩⟩], [⟨.program ⟨257⟩, ⟨41224⟩⟩]⟩, (-1)⟩)

def exact130728RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨41883⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨40076⟩⟩], [⟨.program ⟨257⟩, ⟨41224⟩⟩]⟩, (-1)⟩]

theorem exact130728RawTermsValid :
    exact130728RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event130728 : Event := .resultExact (⟨.program ⟨257⟩, ⟨41885⟩⟩) exact130728RawTerms .large 130721 (.finite 32193129122288627115968346193920) (some (130723))

def event130729 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40772⟩⟩) 0 ⟨40077⟩ 5416

def event130730 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨40772⟩⟩) (.authority (.relationPreimageSource ⟨86⟩))

def exact130731RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨40772⟩⟩]⟩, (1)⟩]

theorem exact130731RawTermsValid :
    exact130731RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event130731 : Event := .resultExact (⟨.program ⟨257⟩, ⟨40772⟩⟩) exact130731RawTerms (.finite 5647228698) 130730 .exactZero (none)

def event130732 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40774⟩⟩) 0 ⟨40772⟩ 130731

def event130733 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40774⟩⟩) 1 ⟨2370⟩ 4

def event130734 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨40774⟩⟩) (.scale (.predecessor 0 130732 .coefficient) (.value (.predecessor 1 130733 .coefficient)))

def exact130735RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨40772⟩⟩]⟩, (1)⟩]

theorem exact130735RawTermsValid :
    exact130735RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event130735 : Event := .resultExact (⟨.program ⟨257⟩, ⟨40774⟩⟩) exact130735RawTerms (.finite 5647228698) 130734 .exactZero (none)

def event130736 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40775⟩⟩) 0 ⟨5527⟩ 119870

def event130737 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40775⟩⟩) 1 ⟨40774⟩ 130735

def event130738 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨40775⟩⟩) (.product (.predecessor 0 130736 .coefficient) (.predecessor 1 130737 .coefficient) (⟨false, false, none, none, none⟩))

def event130739 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨40775⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨40772⟩⟩]⟩) [⟨.result 130731 .coefficient, false, none⟩])

def event130740 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨40775⟩⟩) (.product (.result 119870 .summary) (.transfer 130739) (⟨false, false, none, none, none⟩))

def event130741 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨40775⟩⟩, .operator (⟨119870, 0⟩, ⟨130735, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40772⟩⟩]⟩, (1)⟩)

def event130742 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨40773⟩⟩)

def event130743 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event130744 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event130745 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨2942⟩⟩) (.authority (.operator))

def event130746 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨2942⟩⟩) (.finite 12)

def event130747 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event130748 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event130749 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event130750 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event130751 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 130750

def event130752 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 130748

def event130753 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 130751 .coefficient) (.value (.predecessor 1 130752 .coefficient)))

def event130754 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event130755 : Event := .predecessor (⟨.program ⟨257⟩, ⟨2944⟩⟩) 0 ⟨392⟩ 130754

def event130756 : Event := .predecessor (⟨.program ⟨257⟩, ⟨2944⟩⟩) 1 ⟨2942⟩ 130746

def event130757 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨2944⟩⟩) (.sum [.predecessor 0 130755 .coefficient, .predecessor 1 130756 .coefficient])

def event130758 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨2944⟩⟩) (.finite 655352)

def event130759 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5523⟩⟩) 0 ⟨2944⟩ 130758

def event130760 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5523⟩⟩) 1 ⟨5426⟩ 130744

def event130761 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5523⟩⟩) (.identity (.predecessor 1 130760 .coefficient))

def event130762 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5523⟩⟩) (.finite 655360)

def event130763 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39698⟩⟩) 0 ⟨5523⟩ 130762

def event130764 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39698⟩⟩) (.authority (.programFamilyFact))

def exact130765RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨39698⟩⟩], []⟩, (1)⟩]

theorem exact130765RawTermsValid :
    exact130765RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event130765 : Event := .resultExact (⟨.program ⟨257⟩, ⟨39698⟩⟩) exact130765RawTerms (.finite 46) 130764 .exactZero (none)

def event130766 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14121⟩⟩) 0 ⟨5523⟩ 130762

def event130767 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14121⟩⟩) (.authority (.programFamilyFact))

def exact130768RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14121⟩⟩], []⟩, (1)⟩]

theorem exact130768RawTermsValid :
    exact130768RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event130768 : Event := .resultExact (⟨.program ⟨257⟩, ⟨14121⟩⟩) exact130768RawTerms (.finite 46) 130767 .exactZero (none)

def event130769 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39699⟩⟩) 0 ⟨14121⟩ 130768

def event130770 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39699⟩⟩) 1 ⟨39698⟩ 130765

def event130771 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39699⟩⟩) (.product (.predecessor 0 130769 .coefficient) (.predecessor 1 130770 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event130772 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39699⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨14121⟩⟩, ⟨.program ⟨257⟩, ⟨39698⟩⟩], []⟩) [⟨.result 130768 .coefficient, true, some 1⟩, ⟨.result 130765 .coefficient, true, some 1⟩])

def event130773 : Event := .survivorFold (1) 130772

def exact130774RawTerms : List Term := []

theorem exact130774RawTermsValid :
    exact130774RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event130774 : Event := .resultExact (⟨.program ⟨257⟩, ⟨39699⟩⟩) exact130774RawTerms (.finite 2116) 130771 (.finite 2116) (some (130772))

def event130775 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39700⟩⟩) 0 ⟨39699⟩ 130774

def event130776 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39700⟩⟩) (.identity (.predecessor 0 130775 .coefficient))

def event130777 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨39700⟩⟩) (.finite 2116)

def event130778 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40076⟩⟩) 0 ⟨39700⟩ 130777

def event130779 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨40076⟩⟩) (.authority (.programFamilyFact))

def exact130780RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨40076⟩⟩], []⟩, (1)⟩]

theorem exact130780RawTermsValid :
    exact130780RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event130780 : Event := .resultExact (⟨.program ⟨257⟩, ⟨40076⟩⟩) exact130780RawTerms (.finite 46) 130779 .exactZero (none)

def event130781 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40077⟩⟩) 0 ⟨40076⟩ 130780

def event130782 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨40077⟩⟩) (.identity (.predecessor 0 130781 .coefficient))

def event130783 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨40077⟩⟩) (.finite 46)

def event130784 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40772⟩⟩) 0 ⟨40077⟩ 130783

def event130785 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨40772⟩⟩) (.authority (.relationPreimageSource ⟨86⟩))

def exact130786RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨40772⟩⟩]⟩, (1)⟩]

theorem exact130786RawTermsValid :
    exact130786RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event130786 : Event := .resultExact (⟨.program ⟨257⟩, ⟨40772⟩⟩) exact130786RawTerms (.finite 5647228698) 130785 .exactZero (none)

def event130787 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35⟩⟩) (.authority (.operator))

def exact130788RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩]⟩, (1)⟩]

theorem exact130788RawTermsValid :
    exact130788RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event130788 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35⟩⟩) exact130788RawTerms .large 130787 .exactZero (none)

def event130789 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40773⟩⟩) 0 ⟨35⟩ 130788

def event130790 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40773⟩⟩) 1 ⟨40772⟩ 130786

def event130791 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨40773⟩⟩) (.product (.predecessor 0 130789 .coefficient) (.predecessor 1 130790 .coefficient) (⟨false, false, none, none, none⟩))

def event130792 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨40773⟩⟩, .operator (⟨130788, 0⟩, ⟨130786, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40772⟩⟩]⟩, (1)⟩)

def exact130793RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40772⟩⟩]⟩, (1)⟩]

theorem exact130793RawTermsValid :
    exact130793RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event130793 : Event := .resultExact (⟨.program ⟨257⟩, ⟨40773⟩⟩) exact130793RawTerms .large 130791 .exactZero (none)

def event130794 : Event := .preFoldPolynomial 130793 [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40772⟩⟩]⟩, (1)⟩] .exactZero none

def exact130795RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40772⟩⟩]⟩, (1)⟩]

def event130795 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨40773⟩⟩) 130794 exact130795RawTerms .large 130791 .exactZero (none)

def event130796 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨41888⟩⟩)

def event130797 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event130798 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event130799 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨2942⟩⟩) (.authority (.operator))

def event130800 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨2942⟩⟩) (.finite 12)

def event130801 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event130802 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event130803 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event130804 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event130805 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 130804

def event130806 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 130802

def event130807 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 130805 .coefficient) (.value (.predecessor 1 130806 .coefficient)))

def event130808 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event130809 : Event := .predecessor (⟨.program ⟨257⟩, ⟨2944⟩⟩) 0 ⟨392⟩ 130808

def event130810 : Event := .predecessor (⟨.program ⟨257⟩, ⟨2944⟩⟩) 1 ⟨2942⟩ 130800

def event130811 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨2944⟩⟩) (.sum [.predecessor 0 130809 .coefficient, .predecessor 1 130810 .coefficient])

def event130812 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨2944⟩⟩) (.finite 655352)

def event130813 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5523⟩⟩) 0 ⟨2944⟩ 130812

def event130814 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5523⟩⟩) 1 ⟨5426⟩ 130798

def event130815 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5523⟩⟩) (.identity (.predecessor 1 130814 .coefficient))

def eventLeaf8160 : Array AnnotatedEvent := #[
  { event := event130560
    frameStart := 130530 },
  { event := event130561
    frameStart := 130530 },
  { event := event130562
    frameStart := 130530 },
  { event := event130563
    frameStart := 130530 },
  { event := event130564
    frameStart := 130530 },
  { event := event130565
    frameStart := 130530 },
  { event := event130566
    frameStart := 130530 },
  { event := event130567
    frameStart := 130530 },
  { event := event130568
    frameStart := 130530 },
  { event := event130569
    frameStart := 130530 },
  { event := event130570
    frameStart := 130530 },
  { event := event130571
    frameStart := 130530 },
  { event := event130572
    frameStart := 130530 },
  { event := event130573
    frameStart := 130530 },
  { event := event130574
    frameStart := 130530 },
  { event := event130575
    frameStart := 130530 }
]

def eventLeaf8161 : Array AnnotatedEvent := #[
  { event := event130576
    frameStart := 130530 },
  { event := event130577
    frameStart := 130530 },
  { event := event130578
    frameStart := 130530 },
  { event := event130579
    frameStart := 130530 },
  { event := event130580
    frameStart := 130530 },
  { event := event130581
    frameStart := 130530 },
  { event := event130582
    frameStart := 130530 },
  { event := event130583
    frameStart := 130530 },
  { event := event130584
    frameStart := 130584 },
  { event := event130585
    frameStart := 130584 },
  { event := event130586
    frameStart := 130584 },
  { event := event130587
    frameStart := 130584 },
  { event := event130588
    frameStart := 130584 },
  { event := event130589
    frameStart := 130584 },
  { event := event130590
    frameStart := 130584 },
  { event := event130591
    frameStart := 130584 }
]

def eventLeaf8162 : Array AnnotatedEvent := #[
  { event := event130592
    frameStart := 130584 },
  { event := event130593
    frameStart := 130584 },
  { event := event130594
    frameStart := 130584 },
  { event := event130595
    frameStart := 130584 },
  { event := event130596
    frameStart := 130584 },
  { event := event130597
    frameStart := 130584 },
  { event := event130598
    frameStart := 130584 },
  { event := event130599
    frameStart := 130584 },
  { event := event130600
    frameStart := 130584 },
  { event := event130601
    frameStart := 130584 },
  { event := event130602
    frameStart := 130584 },
  { event := event130603
    frameStart := 130584 },
  { event := event130604
    frameStart := 130584 },
  { event := event130605
    frameStart := 130584 },
  { event := event130606
    frameStart := 130584 },
  { event := event130607
    frameStart := 130584 }
]

def eventLeaf8163 : Array AnnotatedEvent := #[
  { event := event130608
    frameStart := 130584 },
  { event := event130609
    frameStart := 130584 },
  { event := event130610
    frameStart := 130584 },
  { event := event130611
    frameStart := 130584 },
  { event := event130612
    frameStart := 130584 },
  { event := event130613
    frameStart := 130584 },
  { event := event130614
    frameStart := 130584 },
  { event := event130615
    frameStart := 130584 },
  { event := event130616
    frameStart := 130584 },
  { event := event130617
    frameStart := 130584 },
  { event := event130618
    frameStart := 130584 },
  { event := event130619
    frameStart := 130584 },
  { event := event130620
    frameStart := 130584 },
  { event := event130621
    frameStart := 130584 },
  { event := event130622
    frameStart := 130584 },
  { event := event130623
    frameStart := 130584 }
]

def eventLeaf8164 : Array AnnotatedEvent := #[
  { event := event130624
    frameStart := 130584 },
  { event := event130625
    frameStart := 130584 },
  { event := event130626
    frameStart := 130584 },
  { event := event130627
    frameStart := 130584 },
  { event := event130628
    frameStart := 130584 },
  { event := event130629
    frameStart := 130584 },
  { event := event130630
    frameStart := 130584 },
  { event := event130631
    frameStart := 130584 },
  { event := event130632
    frameStart := 130584 },
  { event := event130633
    frameStart := 130584 },
  { event := event130634
    frameStart := 130584 },
  { event := event130635
    frameStart := 130584 },
  { event := event130636
    frameStart := 130584 },
  { event := event130637
    frameStart := 130584 },
  { event := event130638
    frameStart := 130584 },
  { event := event130639
    frameStart := 130584 }
]

def eventLeaf8165 : Array AnnotatedEvent := #[
  { event := event130640
    frameStart := 130584 },
  { event := event130641
    frameStart := 130584 },
  { event := event130642
    frameStart := 130584 },
  { event := event130643
    frameStart := 130584 },
  { event := event130644
    frameStart := 130584 },
  { event := event130645
    frameStart := 130584 },
  { event := event130646
    frameStart := 130584 },
  { event := event130647
    frameStart := 130584 },
  { event := event130648
    frameStart := 130584 },
  { event := event130649
    frameStart := 130584 },
  { event := event130650
    frameStart := 130584 },
  { event := event130651
    frameStart := 130584 },
  { event := event130652
    frameStart := 130584 },
  { event := event130653
    frameStart := 130584 },
  { event := event130654
    frameStart := 130584 },
  { event := event130655
    frameStart := 130584 }
]

def eventLeaf8166 : Array AnnotatedEvent := #[
  { event := event130656
    frameStart := 130584 },
  { event := event130657
    frameStart := 130584 },
  { event := event130658
    frameStart := 130584 },
  { event := event130659
    frameStart := 130584 },
  { event := event130660
    frameStart := 130584 },
  { event := event130661
    frameStart := 130584 },
  { event := event130662
    frameStart := 130584 },
  { event := event130663
    frameStart := 130584 },
  { event := event130664
    frameStart := 130584 },
  { event := event130665
    frameStart := 130584 },
  { event := event130666
    frameStart := 130584 },
  { event := event130667
    frameStart := 130584 },
  { event := event130668
    frameStart := 130584 },
  { event := event130669
    frameStart := 130584 },
  { event := event130670
    frameStart := 130584 },
  { event := event130671
    frameStart := 130584 }
]

def eventLeaf8167 : Array AnnotatedEvent := #[
  { event := event130672
    frameStart := 130584 },
  { event := event130673
    frameStart := 130584 },
  { event := event130674
    frameStart := 130584 },
  { event := event130675
    frameStart := 130584 },
  { event := event130676
    frameStart := 130584 },
  { event := event130677
    frameStart := 130584 },
  { event := event130678
    frameStart := 130584 },
  { event := event130679
    frameStart := 130584 },
  { event := event130680
    frameStart := 130584 },
  { event := event130681
    frameStart := 130584 },
  { event := event130682
    frameStart := 130584 },
  { event := event130683
    frameStart := 130584 },
  { event := event130684
    frameStart := 130584 },
  { event := event130685
    frameStart := 130584 },
  { event := event130686
    frameStart := 130584 },
  { event := event130687
    frameStart := 130584 }
]

def eventLeaf8168 : Array AnnotatedEvent := #[
  { event := event130688
    frameStart := 0 },
  { event := event130689
    frameStart := 0 },
  { event := event130690
    frameStart := 0 },
  { event := event130691
    frameStart := 0 },
  { event := event130692
    frameStart := 0 },
  { event := event130693
    frameStart := 0 },
  { event := event130694
    frameStart := 0 },
  { event := event130695
    frameStart := 0 },
  { event := event130696
    frameStart := 0 },
  { event := event130697
    frameStart := 0 },
  { event := event130698
    frameStart := 0 },
  { event := event130699
    frameStart := 0 },
  { event := event130700
    frameStart := 0 },
  { event := event130701
    frameStart := 0 },
  { event := event130702
    frameStart := 0 },
  { event := event130703
    frameStart := 0 }
]

def eventLeaf8169 : Array AnnotatedEvent := #[
  { event := event130704
    frameStart := 0 },
  { event := event130705
    frameStart := 0 },
  { event := event130706
    frameStart := 0 },
  { event := event130707
    frameStart := 0 },
  { event := event130708
    frameStart := 0 },
  { event := event130709
    frameStart := 0 },
  { event := event130710
    frameStart := 0 },
  { event := event130711
    frameStart := 0 },
  { event := event130712
    frameStart := 0 },
  { event := event130713
    frameStart := 0 },
  { event := event130714
    frameStart := 0 },
  { event := event130715
    frameStart := 0 },
  { event := event130716
    frameStart := 0 },
  { event := event130717
    frameStart := 0 },
  { event := event130718
    frameStart := 0 },
  { event := event130719
    frameStart := 0 }
]

def eventLeaf8170 : Array AnnotatedEvent := #[
  { event := event130720
    frameStart := 0 },
  { event := event130721
    frameStart := 0 },
  { event := event130722
    frameStart := 0 },
  { event := event130723
    frameStart := 0 },
  { event := event130724
    frameStart := 0 },
  { event := event130725
    frameStart := 0 },
  { event := event130726
    frameStart := 0 },
  { event := event130727
    frameStart := 0 },
  { event := event130728
    frameStart := 0 },
  { event := event130729
    frameStart := 0 },
  { event := event130730
    frameStart := 0 },
  { event := event130731
    frameStart := 0 },
  { event := event130732
    frameStart := 0 },
  { event := event130733
    frameStart := 0 },
  { event := event130734
    frameStart := 0 },
  { event := event130735
    frameStart := 0 }
]

def eventLeaf8171 : Array AnnotatedEvent := #[
  { event := event130736
    frameStart := 0 },
  { event := event130737
    frameStart := 0 },
  { event := event130738
    frameStart := 0 },
  { event := event130739
    frameStart := 0 },
  { event := event130740
    frameStart := 0 },
  { event := event130741
    frameStart := 0 },
  { event := event130742
    frameStart := 130742 },
  { event := event130743
    frameStart := 130742 },
  { event := event130744
    frameStart := 130742 },
  { event := event130745
    frameStart := 130742 },
  { event := event130746
    frameStart := 130742 },
  { event := event130747
    frameStart := 130742 },
  { event := event130748
    frameStart := 130742 },
  { event := event130749
    frameStart := 130742 },
  { event := event130750
    frameStart := 130742 },
  { event := event130751
    frameStart := 130742 }
]

def eventLeaf8172 : Array AnnotatedEvent := #[
  { event := event130752
    frameStart := 130742 },
  { event := event130753
    frameStart := 130742 },
  { event := event130754
    frameStart := 130742 },
  { event := event130755
    frameStart := 130742 },
  { event := event130756
    frameStart := 130742 },
  { event := event130757
    frameStart := 130742 },
  { event := event130758
    frameStart := 130742 },
  { event := event130759
    frameStart := 130742 },
  { event := event130760
    frameStart := 130742 },
  { event := event130761
    frameStart := 130742 },
  { event := event130762
    frameStart := 130742 },
  { event := event130763
    frameStart := 130742 },
  { event := event130764
    frameStart := 130742 },
  { event := event130765
    frameStart := 130742 },
  { event := event130766
    frameStart := 130742 },
  { event := event130767
    frameStart := 130742 }
]

def eventLeaf8173 : Array AnnotatedEvent := #[
  { event := event130768
    frameStart := 130742 },
  { event := event130769
    frameStart := 130742 },
  { event := event130770
    frameStart := 130742 },
  { event := event130771
    frameStart := 130742 },
  { event := event130772
    frameStart := 130742 },
  { event := event130773
    frameStart := 130742 },
  { event := event130774
    frameStart := 130742 },
  { event := event130775
    frameStart := 130742 },
  { event := event130776
    frameStart := 130742 },
  { event := event130777
    frameStart := 130742 },
  { event := event130778
    frameStart := 130742 },
  { event := event130779
    frameStart := 130742 },
  { event := event130780
    frameStart := 130742 },
  { event := event130781
    frameStart := 130742 },
  { event := event130782
    frameStart := 130742 },
  { event := event130783
    frameStart := 130742 }
]

def eventLeaf8174 : Array AnnotatedEvent := #[
  { event := event130784
    frameStart := 130742 },
  { event := event130785
    frameStart := 130742 },
  { event := event130786
    frameStart := 130742 },
  { event := event130787
    frameStart := 130742 },
  { event := event130788
    frameStart := 130742 },
  { event := event130789
    frameStart := 130742 },
  { event := event130790
    frameStart := 130742 },
  { event := event130791
    frameStart := 130742 },
  { event := event130792
    frameStart := 130742 },
  { event := event130793
    frameStart := 130742 },
  { event := event130794
    frameStart := 130742 },
  { event := event130795
    frameStart := 130742 },
  { event := event130796
    frameStart := 130796 },
  { event := event130797
    frameStart := 130796 },
  { event := event130798
    frameStart := 130796 },
  { event := event130799
    frameStart := 130796 }
]

def eventLeaf8175 : Array AnnotatedEvent := #[
  { event := event130800
    frameStart := 130796 },
  { event := event130801
    frameStart := 130796 },
  { event := event130802
    frameStart := 130796 },
  { event := event130803
    frameStart := 130796 },
  { event := event130804
    frameStart := 130796 },
  { event := event130805
    frameStart := 130796 },
  { event := event130806
    frameStart := 130796 },
  { event := event130807
    frameStart := 130796 },
  { event := event130808
    frameStart := 130796 },
  { event := event130809
    frameStart := 130796 },
  { event := event130810
    frameStart := 130796 },
  { event := event130811
    frameStart := 130796 },
  { event := event130812
    frameStart := 130796 },
  { event := event130813
    frameStart := 130796 },
  { event := event130814
    frameStart := 130796 },
  { event := event130815
    frameStart := 130796 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events510
