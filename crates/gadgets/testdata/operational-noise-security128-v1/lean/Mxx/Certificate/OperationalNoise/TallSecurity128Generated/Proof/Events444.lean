import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events444

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event113664 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5756⟩⟩) (.sum [.predecessor 0 113662 .coefficient, .predecessor 1 113663 .coefficient])

def event113665 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5756⟩⟩) (.finite 655353)

def event113666 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5766⟩⟩) 0 ⟨5756⟩ 113665

def event113667 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5766⟩⟩) 1 ⟨5426⟩ 113651

def event113668 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5766⟩⟩) (.identity (.predecessor 1 113667 .coefficient))

def event113669 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5766⟩⟩) (.finite 655360)

def event113670 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15498⟩⟩) 0 ⟨5766⟩ 113669

def event113671 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15498⟩⟩) (.authority (.programFamilyFact))

def exact113672RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15498⟩⟩], []⟩, (1)⟩]

theorem exact113672RawTermsValid :
    exact113672RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event113672 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15498⟩⟩) exact113672RawTerms (.finite 2) 113671 .exactZero (none)

def event113673 : Event := .predecessor (⟨.program ⟨257⟩, ⟨12396⟩⟩) 0 ⟨5766⟩ 113669

def event113674 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨12396⟩⟩) (.authority (.programFamilyFact))

def exact113675RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨12396⟩⟩], []⟩, (1)⟩]

theorem exact113675RawTermsValid :
    exact113675RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event113675 : Event := .resultExact (⟨.program ⟨257⟩, ⟨12396⟩⟩) exact113675RawTerms (.finite 2) 113674 .exactZero (none)

def event113676 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15499⟩⟩) 0 ⟨12396⟩ 113675

def event113677 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15499⟩⟩) 1 ⟨15498⟩ 113672

def event113678 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15499⟩⟩) (.product (.predecessor 0 113676 .coefficient) (.predecessor 1 113677 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event113679 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15499⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨12396⟩⟩, ⟨.program ⟨257⟩, ⟨15498⟩⟩], []⟩) [⟨.result 113675 .coefficient, true, some 1⟩, ⟨.result 113672 .coefficient, true, some 1⟩])

def event113680 : Event := .survivorFold (1) 113679

def exact113681RawTerms : List Term := []

theorem exact113681RawTermsValid :
    exact113681RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event113681 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15499⟩⟩) exact113681RawTerms (.finite 4) 113678 (.finite 4) (some (113679))

def event113682 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15500⟩⟩) 0 ⟨15499⟩ 113681

def event113683 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15500⟩⟩) (.identity (.predecessor 0 113682 .coefficient))

def event113684 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨15500⟩⟩) (.finite 4)

def event113685 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15796⟩⟩) 0 ⟨15500⟩ 113684

def event113686 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15796⟩⟩) (.authority (.programFamilyFact))

def exact113687RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15796⟩⟩], []⟩, (1)⟩]

theorem exact113687RawTermsValid :
    exact113687RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event113687 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15796⟩⟩) exact113687RawTerms (.finite 2) 113686 .exactZero (none)

def event113688 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15797⟩⟩) 0 ⟨15796⟩ 113687

def event113689 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15797⟩⟩) (.identity (.predecessor 0 113688 .coefficient))

def event113690 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨15797⟩⟩) (.finite 2)

def event113691 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16616⟩⟩) 0 ⟨15797⟩ 113690

def event113692 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨16616⟩⟩) (.authority (.relationPreimageSource ⟨57⟩))

def exact113693RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨16616⟩⟩]⟩, (1)⟩]

theorem exact113693RawTermsValid :
    exact113693RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event113693 : Event := .resultExact (⟨.program ⟨257⟩, ⟨16616⟩⟩) exact113693RawTerms (.finite 5647228698) 113692 .exactZero (none)

def event113694 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35⟩⟩) (.authority (.operator))

def exact113695RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩]⟩, (1)⟩]

theorem exact113695RawTermsValid :
    exact113695RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event113695 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35⟩⟩) exact113695RawTerms .large 113694 .exactZero (none)

def event113696 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16617⟩⟩) 0 ⟨35⟩ 113695

def event113697 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16617⟩⟩) 1 ⟨16616⟩ 113693

def event113698 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨16617⟩⟩) (.product (.predecessor 0 113696 .coefficient) (.predecessor 1 113697 .coefficient) (⟨false, false, none, none, none⟩))

def event113699 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨16617⟩⟩, .operator (⟨113695, 0⟩, ⟨113693, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16616⟩⟩]⟩, (1)⟩)

def exact113700RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16616⟩⟩]⟩, (1)⟩]

theorem exact113700RawTermsValid :
    exact113700RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event113700 : Event := .resultExact (⟨.program ⟨257⟩, ⟨16617⟩⟩) exact113700RawTerms .large 113698 .exactZero (none)

def event113701 : Event := .preFoldPolynomial 113700 [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16616⟩⟩]⟩, (1)⟩] .exactZero none

def exact113702RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16616⟩⟩]⟩, (1)⟩]

def event113702 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨16617⟩⟩) 113701 exact113702RawTerms .large 113698 .exactZero (none)

def event113703 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨17793⟩⟩)

def event113704 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event113705 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event113706 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5754⟩⟩) (.authority (.operator))

def event113707 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5754⟩⟩) (.finite 13)

def event113708 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event113709 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event113710 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event113711 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event113712 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 113711

def event113713 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 113709

def event113714 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 113712 .coefficient) (.value (.predecessor 1 113713 .coefficient)))

def event113715 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event113716 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5756⟩⟩) 0 ⟨392⟩ 113715

def event113717 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5756⟩⟩) 1 ⟨5754⟩ 113707

def event113718 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5756⟩⟩) (.sum [.predecessor 0 113716 .coefficient, .predecessor 1 113717 .coefficient])

def event113719 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5756⟩⟩) (.finite 655353)

def event113720 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5766⟩⟩) 0 ⟨5756⟩ 113719

def event113721 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5766⟩⟩) 1 ⟨5426⟩ 113705

def event113722 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5766⟩⟩) (.identity (.predecessor 1 113721 .coefficient))

def event113723 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5766⟩⟩) (.finite 655360)

def event113724 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15498⟩⟩) 0 ⟨5766⟩ 113723

def event113725 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15498⟩⟩) (.authority (.programFamilyFact))

def exact113726RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15498⟩⟩], []⟩, (1)⟩]

theorem exact113726RawTermsValid :
    exact113726RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event113726 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15498⟩⟩) exact113726RawTerms (.finite 2) 113725 .exactZero (none)

def event113727 : Event := .predecessor (⟨.program ⟨257⟩, ⟨12396⟩⟩) 0 ⟨5766⟩ 113723

def event113728 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨12396⟩⟩) (.authority (.programFamilyFact))

def exact113729RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨12396⟩⟩], []⟩, (1)⟩]

theorem exact113729RawTermsValid :
    exact113729RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event113729 : Event := .resultExact (⟨.program ⟨257⟩, ⟨12396⟩⟩) exact113729RawTerms (.finite 2) 113728 .exactZero (none)

def event113730 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15499⟩⟩) 0 ⟨12396⟩ 113729

def event113731 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15499⟩⟩) 1 ⟨15498⟩ 113726

def event113732 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15499⟩⟩) (.product (.predecessor 0 113730 .coefficient) (.predecessor 1 113731 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event113733 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨15499⟩⟩, .operator (⟨113729, 0⟩, ⟨113726, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨12396⟩⟩, ⟨.program ⟨257⟩, ⟨15498⟩⟩], []⟩, (1)⟩)

def exact113734RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨12396⟩⟩, ⟨.program ⟨257⟩, ⟨15498⟩⟩], []⟩, (1)⟩]

theorem exact113734RawTermsValid :
    exact113734RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event113734 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15499⟩⟩) exact113734RawTerms (.finite 4) 113732 .exactZero (none)

def event113735 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15500⟩⟩) 0 ⟨15499⟩ 113734

def event113736 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15500⟩⟩) (.identity (.predecessor 0 113735 .coefficient))

def event113737 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨15500⟩⟩) (.finite 4)

def event113738 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15796⟩⟩) 0 ⟨15500⟩ 113737

def event113739 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15796⟩⟩) (.authority (.programFamilyFact))

def exact113740RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15796⟩⟩], []⟩, (1)⟩]

theorem exact113740RawTermsValid :
    exact113740RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event113740 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15796⟩⟩) exact113740RawTerms (.finite 2) 113739 .exactZero (none)

def event113741 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15797⟩⟩) 0 ⟨15796⟩ 113740

def event113742 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15797⟩⟩) (.identity (.predecessor 0 113741 .coefficient))

def event113743 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨15797⟩⟩) (.finite 2)

def event113744 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17008⟩⟩) 0 ⟨15797⟩ 113743

def event113745 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17008⟩⟩) (.authority (.programFamilyFact))

def event113746 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨17008⟩⟩) (.finite 3720)

def event113747 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨7177⟩⟩) .missing

def event113748 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17010⟩⟩) 0 ⟨7177⟩ 113747

def event113749 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17010⟩⟩) 1 ⟨17008⟩ 113746

def event113750 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17010⟩⟩) (.authority (.operator))

def exact113751RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨17010⟩⟩]⟩, (1)⟩]

theorem exact113751RawTermsValid :
    exact113751RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event113751 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17010⟩⟩) exact113751RawTerms .large 113750 .exactZero (none)

def event113752 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17789⟩⟩) 0 ⟨17010⟩ 113751

def event113753 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17789⟩⟩) (.authority (.operator))

def exact113754RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨17789⟩⟩]⟩, (1)⟩]

theorem exact113754RawTermsValid :
    exact113754RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event113754 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17789⟩⟩) exact113754RawTerms (.finite 8192) 113753 .exactZero (none)

def event113755 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨136⟩⟩) (.authority (.operator))

def event113756 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨136⟩⟩) .exactZero

def event113757 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17210⟩⟩) 0 ⟨15797⟩ 113743

def event113758 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17210⟩⟩) 1 ⟨136⟩ 113756

def event113759 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17210⟩⟩) (.sum [.predecessor 0 113757 .coefficient, .predecessor 1 113758 .coefficient])

def event113760 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨17210⟩⟩) (.finite 2)

def event113761 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17211⟩⟩) 0 ⟨17210⟩ 113760

def event113762 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17211⟩⟩) (.identity (.predecessor 0 113761 .coefficient))

def exact113763RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15796⟩⟩], []⟩, (1)⟩]

theorem exact113763RawTermsValid :
    exact113763RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event113763 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17211⟩⟩) exact113763RawTerms (.finite 2) 113762 .exactZero (none)

def event113764 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6908⟩⟩) (.authority (.factStore))

def exact113765RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact113765RawTermsValid :
    exact113765RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event113765 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6908⟩⟩) exact113765RawTerms .large 113764 .exactZero (none)

def event113766 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17212⟩⟩) 0 ⟨6908⟩ 113765

def event113767 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17212⟩⟩) 1 ⟨17211⟩ 113763

def event113768 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17212⟩⟩) (.product (.predecessor 0 113766 .coefficient) (.predecessor 1 113767 .coefficient) (⟨false, false, none, none, none⟩))

def event113769 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17212⟩⟩, .operator (⟨113765, 0⟩, ⟨113763, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨15796⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact113770RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15796⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact113770RawTermsValid :
    exact113770RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event113770 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17212⟩⟩) exact113770RawTerms .large 113768 .exactZero (none)

def event113771 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7179⟩⟩) 0 ⟨7177⟩ 113747

def event113772 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7179⟩⟩) (.authority (.operator))

def exact113773RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩]⟩, (1)⟩]

theorem exact113773RawTermsValid :
    exact113773RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event113773 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7179⟩⟩) exact113773RawTerms .large 113772 .exactZero (none)

def event113774 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17213⟩⟩) 0 ⟨7179⟩ 113773

def event113775 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17213⟩⟩) 1 ⟨17212⟩ 113770

def event113776 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17213⟩⟩) (.sum [.predecessor 0 113774 .coefficient, .predecessor 1 113775 .coefficient])

def exact113777RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15796⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact113777RawTermsValid :
    exact113777RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event113777 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17213⟩⟩) exact113777RawTerms .large 113776 .exactZero (none)

def event113778 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17790⟩⟩) 0 ⟨17213⟩ 113777

def event113779 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17790⟩⟩) 1 ⟨17789⟩ 113754

def event113780 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17790⟩⟩) (.product (.predecessor 0 113778 .coefficient) (.predecessor 1 113779 .coefficient) (⟨false, false, none, none, none⟩))

def event113781 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17790⟩⟩, .operator (⟨113777, 0⟩, ⟨113754, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17789⟩⟩]⟩, (1)⟩)

def event113782 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17790⟩⟩, .operator (⟨113777, 1⟩, ⟨113754, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨15796⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17789⟩⟩]⟩, (-1)⟩)

def event113783 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨17790⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨15796⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17789⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨17789⟩⟩) ⟨17010⟩ 113751)

def event113784 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17790⟩⟩, .relation 113783 0, ⟨[⟨.program ⟨257⟩, ⟨15796⟩⟩], [⟨.program ⟨257⟩, ⟨17010⟩⟩]⟩, (-1)⟩)

def exact113785RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17789⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15796⟩⟩], [⟨.program ⟨257⟩, ⟨17010⟩⟩]⟩, (-1)⟩]

theorem exact113785RawTermsValid :
    exact113785RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event113785 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17790⟩⟩) exact113785RawTerms .large 113780 .exactZero (none)

def event113786 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16051⟩⟩) 0 ⟨15797⟩ 113743

def event113787 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨16051⟩⟩) (.authority (.programFamilyFact))

def exact113788RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨16051⟩⟩], []⟩, (1)⟩]

theorem exact113788RawTermsValid :
    exact113788RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event113788 : Event := .resultExact (⟨.program ⟨257⟩, ⟨16051⟩⟩) exact113788RawTerms (.finite 43) 113787 .exactZero (none)

def event113789 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16052⟩⟩) 0 ⟨6908⟩ 113765

def event113790 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16052⟩⟩) 1 ⟨16051⟩ 113788

def event113791 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨16052⟩⟩) (.product (.predecessor 0 113789 .coefficient) (.predecessor 1 113790 .coefficient) (⟨false, true, none, none, some 1⟩))

def event113792 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨16052⟩⟩, .operator (⟨113765, 0⟩, ⟨113788, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨16051⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact113793RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨16051⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact113793RawTermsValid :
    exact113793RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event113793 : Event := .resultExact (⟨.program ⟨257⟩, ⟨16052⟩⟩) exact113793RawTerms .large 113791 .exactZero (none)

def event113794 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7198⟩⟩) 0 ⟨7177⟩ 113747

def event113795 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7198⟩⟩) (.authority (.operator))

def exact113796RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩]

theorem exact113796RawTermsValid :
    exact113796RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event113796 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7198⟩⟩) exact113796RawTerms .large 113795 .exactZero (none)

def event113797 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16053⟩⟩) 0 ⟨7198⟩ 113796

def event113798 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16053⟩⟩) 1 ⟨16052⟩ 113793

def event113799 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨16053⟩⟩) (.sum [.predecessor 0 113797 .coefficient, .predecessor 1 113798 .coefficient])

def exact113800RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨16051⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact113800RawTermsValid :
    exact113800RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event113800 : Event := .resultExact (⟨.program ⟨257⟩, ⟨16053⟩⟩) exact113800RawTerms .large 113799 .exactZero (none)

def event113801 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17793⟩⟩) 0 ⟨16053⟩ 113800

def event113802 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17793⟩⟩) 1 ⟨17790⟩ 113785

def event113803 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17793⟩⟩) (.sum [.predecessor 0 113801 .coefficient, .predecessor 1 113802 .coefficient])

def exact113804RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17789⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15796⟩⟩], [⟨.program ⟨257⟩, ⟨17010⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨16051⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact113804RawTermsValid :
    exact113804RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event113804 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17793⟩⟩) exact113804RawTerms .large 113803 .exactZero (none)

def event113805 : Event := .preFoldPolynomial 113804 [⟨⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17789⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15796⟩⟩], [⟨.program ⟨257⟩, ⟨17010⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨16051⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩] .exactZero none

def exact113806RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17789⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15796⟩⟩], [⟨.program ⟨257⟩, ⟨17010⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨16051⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

def event113806 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨17793⟩⟩) 113805 exact113806RawTerms .large 113803 .exactZero (none)

def event113807 : Event := .specializationComputed (⟨.program ⟨257⟩, ⟨15797⟩⟩) ⟨⟨77⟩, ⟨57⟩, ⟨135⟩⟩ ⟨113649, 113807⟩

def event113808 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨16619⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16616⟩⟩]⟩) (1) 0 2 (.universal 113807 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16616⟩⟩]⟩) (none) 113806)

def event113809 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨16619⟩⟩, .relation 113808 1, ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩)

def event113810 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨16619⟩⟩, .relation 113808 0, ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17789⟩⟩]⟩, (-1)⟩)

def event113811 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨16619⟩⟩, .relation 113808 2, ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨15796⟩⟩], [⟨.program ⟨257⟩, ⟨17010⟩⟩]⟩, (1)⟩)

def event113812 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨16619⟩⟩, .relation 113808 3, ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨16051⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact113813RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17789⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨15796⟩⟩], [⟨.program ⟨257⟩, ⟨17010⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨16051⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact113813RawTermsValid :
    exact113813RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event113813 : Event := .resultExact (⟨.program ⟨257⟩, ⟨16619⟩⟩) exact113813RawTerms .large 113645 (.finite 202072841853861888) (some (113647))

def event113814 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17792⟩⟩) 0 ⟨16619⟩ 113813

def event113815 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17792⟩⟩) 1 ⟨17791⟩ 113635

def event113816 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17792⟩⟩) (.sum [.predecessor 0 113814 .coefficient, .predecessor 1 113815 .coefficient])

def event113817 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17792⟩⟩, .operator (⟨113813, 0⟩, ⟨113635, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17789⟩⟩]⟩, (1)⟩)

def event113818 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17792⟩⟩, .operator (⟨113813, 2⟩, ⟨113635, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨15796⟩⟩], [⟨.program ⟨257⟩, ⟨17010⟩⟩]⟩, (-1)⟩)

def event113819 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17792⟩⟩) (.sum [.result 113813 .summary, .result 113635 .summary])

def exact113820RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨16051⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact113820RawTermsValid :
    exact113820RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event113820 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17792⟩⟩) exact113820RawTerms .large 113816 (.finite 32188807212483706889510625476608) (some (113819))

def event113821 : Event := .predecessor (⟨.program ⟨257⟩, ⟨20687⟩⟩) 0 ⟨17792⟩ 113820

def event113822 : Event := .predecessor (⟨.program ⟨257⟩, ⟨20687⟩⟩) 1 ⟨20686⟩ 113338

def event113823 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨20687⟩⟩) (.sum [.predecessor 0 113821 .coefficient, .predecessor 1 113822 .coefficient])

def event113824 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨20687⟩⟩) (.sum [.result 113820 .summary, .result 113338 .summary])

def exact113825RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨16051⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨18885⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact113825RawTermsValid :
    exact113825RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event113825 : Event := .resultExact (⟨.program ⟨257⟩, ⟨20687⟩⟩) exact113825RawTerms .large 113823 (.finite 64377712650190257467641695830016) (some (113824))

def event113826 : Event := .predecessor (⟨.program ⟨257⟩, ⟨23907⟩⟩) 0 ⟨20687⟩ 113825

def event113827 : Event := .predecessor (⟨.program ⟨257⟩, ⟨23907⟩⟩) 1 ⟨23906⟩ 112856

def event113828 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨23907⟩⟩) (.sum [.predecessor 0 113826 .coefficient, .predecessor 1 113827 .coefficient])

def event113829 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨23907⟩⟩) (.sum [.result 113825 .summary, .result 112856 .summary])

def exact113830RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨16051⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨18885⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨22105⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact113830RawTermsValid :
    exact113830RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event113830 : Event := .resultExact (⟨.program ⟨257⟩, ⟨23907⟩⟩) exact113830RawTerms .large 113828 (.finite 96566716313119651734393211060224) (some (113829))

def event113831 : Event := .predecessor (⟨.program ⟨257⟩, ⟨33927⟩⟩) 0 ⟨23907⟩ 113830

def event113832 : Event := .predecessor (⟨.program ⟨257⟩, ⟨33927⟩⟩) 1 ⟨33926⟩ 112374

def event113833 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨33927⟩⟩) (.sum [.predecessor 0 113831 .coefficient, .predecessor 1 113832 .coefficient])

def event113834 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨33927⟩⟩) (.sum [.result 113830 .summary, .result 112374 .summary])

def exact113835RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨16051⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨18885⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨22105⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨32125⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact113835RawTermsValid :
    exact113835RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event113835 : Event := .resultExact (⟨.program ⟨257⟩, ⟨33927⟩⟩) exact113835RawTerms .large 113833 (.finite 128755916426494733378385616044032) (some (113834))

def event113836 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52987⟩⟩) 0 ⟨33927⟩ 113835

def event113837 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52987⟩⟩) 1 ⟨52986⟩ 111892

def event113838 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨52987⟩⟩) (.sum [.predecessor 0 113836 .coefficient, .predecessor 1 113837 .coefficient])

def event113839 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨52987⟩⟩) (.sum [.result 113835 .summary, .result 111892 .summary])

def exact113840RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨16051⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨18885⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨22105⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨32125⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨51180⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact113840RawTermsValid :
    exact113840RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event113840 : Event := .resultExact (⟨.program ⟨257⟩, ⟨52987⟩⟩) exact113840RawTerms .large 113838 (.finite 160945509440761189776859800535040) (some (113839))

def event113841 : Event := .predecessor (⟨.program ⟨257⟩, ⟨55967⟩⟩) 0 ⟨52987⟩ 113840

def event113842 : Event := .predecessor (⟨.program ⟨257⟩, ⟨55967⟩⟩) 1 ⟨55966⟩ 111410

def event113843 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨55967⟩⟩) (.sum [.predecessor 0 113841 .coefficient, .predecessor 1 113842 .coefficient])

def event113844 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨55967⟩⟩) (.sum [.result 113840 .summary, .result 111410 .summary])

def exact113845RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨16051⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨18885⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨22105⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨32125⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨51180⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨54160⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact113845RawTermsValid :
    exact113845RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event113845 : Event := .resultExact (⟨.program ⟨257⟩, ⟨55967⟩⟩) exact113845RawTerms .large 113843 (.finite 193135298905473333552574874779648) (some (113844))

def event113846 : Event := .predecessor (⟨.program ⟨257⟩, ⟨58947⟩⟩) 0 ⟨55967⟩ 113845

def event113847 : Event := .predecessor (⟨.program ⟨257⟩, ⟨58947⟩⟩) 1 ⟨58946⟩ 110928

def event113848 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨58947⟩⟩) (.sum [.predecessor 0 113846 .coefficient, .predecessor 1 113847 .coefficient])

def event113849 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨58947⟩⟩) (.sum [.result 113845 .summary, .result 110928 .summary])

def exact113850RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨16051⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨18885⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨22105⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨32125⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨51180⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨54160⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨57140⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact113850RawTermsValid :
    exact113850RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event113850 : Event := .resultExact (⟨.program ⟨257⟩, ⟨58947⟩⟩) exact113850RawTerms .large 113848 (.finite 225325481271076852082771728531456) (some (113849))

def event113851 : Event := .predecessor (⟨.program ⟨257⟩, ⟨61927⟩⟩) 0 ⟨58947⟩ 113850

def event113852 : Event := .predecessor (⟨.program ⟨257⟩, ⟨61927⟩⟩) 1 ⟨61926⟩ 110446

def event113853 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨61927⟩⟩) (.sum [.predecessor 0 113851 .coefficient, .predecessor 1 113852 .coefficient])

def event113854 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨61927⟩⟩) (.sum [.result 113850 .summary, .result 110446 .summary])

def exact113855RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨16051⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨18885⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨22105⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨32125⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨51180⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨54160⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨57140⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨60120⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact113855RawTermsValid :
    exact113855RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event113855 : Event := .resultExact (⟨.program ⟨257⟩, ⟨61927⟩⟩) exact113855RawTerms .large 113853 (.finite 257515860087126057990209472036864) (some (113854))

def event113856 : Event := .predecessor (⟨.program ⟨257⟩, ⟨64907⟩⟩) 0 ⟨61927⟩ 113855

def event113857 : Event := .predecessor (⟨.program ⟨257⟩, ⟨64907⟩⟩) 1 ⟨64906⟩ 109964

def event113858 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨64907⟩⟩) (.sum [.predecessor 0 113856 .coefficient, .predecessor 1 113857 .coefficient])

def event113859 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨64907⟩⟩) (.sum [.result 113855 .summary, .result 109964 .summary])

def exact113860RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨16051⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨18885⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨22105⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨32125⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨51180⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨54160⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨57140⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨60120⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨63100⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact113860RawTermsValid :
    exact113860RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event113860 : Event := .resultExact (⟨.program ⟨257⟩, ⟨64907⟩⟩) exact113860RawTerms .large 113858 (.finite 289706631804066638652128995049472) (some (113859))

def event113861 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70260⟩⟩) 0 ⟨64907⟩ 113860

def event113862 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70260⟩⟩) 1 ⟨70259⟩ 109482

def event113863 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70260⟩⟩) (.sum [.predecessor 0 113861 .coefficient, .predecessor 1 113862 .coefficient])

def event113864 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70260⟩⟩) (.sum [.result 113860 .summary, .result 109482 .summary])

def exact113865RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨16051⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨18885⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨22105⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨32125⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨51180⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨54160⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨57140⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨60120⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨63100⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨66671⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact113865RawTermsValid :
    exact113865RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event113865 : Event := .resultExact (⟨.program ⟨257⟩, ⟨70260⟩⟩) exact113865RawTerms .large 113863 (.finite 321897992872344281445771187322880) (some (113864))

def event113866 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70261⟩⟩) 0 ⟨70260⟩ 113865

def event113867 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70261⟩⟩) 1 ⟨28317⟩ 109000

def event113868 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70261⟩⟩) (.sum [.predecessor 0 113866 .coefficient, .predecessor 1 113867 .coefficient])

def event113869 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70261⟩⟩) (.sum [.result 113865 .summary, .result 109000 .summary])

def exact113870RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨16051⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨18885⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨22105⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨26632⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨32125⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨51180⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨54160⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨57140⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨60120⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨63100⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨66671⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact113870RawTermsValid :
    exact113870RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event113870 : Event := .resultExact (⟨.program ⟨257⟩, ⟨70261⟩⟩) exact113870RawTerms .large 113868 (.finite 354089550391067611616654269349888) (some (113869))

def event113871 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70262⟩⟩) 0 ⟨70261⟩ 113870

def event113872 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70262⟩⟩) 1 ⟨30997⟩ 108518

def event113873 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70262⟩⟩) (.sum [.predecessor 0 113871 .coefficient, .predecessor 1 113872 .coefficient])

def event113874 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70262⟩⟩) (.sum [.result 113870 .summary, .result 108518 .summary])

def exact113875RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨16051⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨18885⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨22105⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨26632⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨29312⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨32125⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨51180⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨54160⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨57140⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨60120⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨63100⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨66671⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact113875RawTermsValid :
    exact113875RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event113875 : Event := .resultExact (⟨.program ⟨257⟩, ⟨70262⟩⟩) exact113875RawTerms .large 113873 (.finite 386281697261128003919260020637696) (some (113874))

def event113876 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70263⟩⟩) 0 ⟨70262⟩ 113875

def event113877 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70263⟩⟩) 1 ⟨36657⟩ 108036

def event113878 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70263⟩⟩) (.sum [.predecessor 0 113876 .coefficient, .predecessor 1 113877 .coefficient])

def event113879 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70263⟩⟩) (.sum [.result 113875 .summary, .result 108036 .summary])

def exact113880RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7222⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨16051⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨18885⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨22105⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨26632⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨29312⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨32125⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨34976⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨51180⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨54160⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨57140⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨60120⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨63100⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨66671⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact113880RawTermsValid :
    exact113880RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event113880 : Event := .resultExact (⟨.program ⟨257⟩, ⟨70263⟩⟩) exact113880RawTerms .large 113878 (.finite 418474237032079770976347551432704) (some (113879))

def event113881 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70264⟩⟩) 0 ⟨70263⟩ 113880

def event113882 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70264⟩⟩) 1 ⟨39337⟩ 107554

def event113883 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70264⟩⟩) (.sum [.predecessor 0 113881 .coefficient, .predecessor 1 113882 .coefficient])

def event113884 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70264⟩⟩) (.sum [.result 113880 .summary, .result 107554 .summary])

def exact113885RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7222⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7224⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨16051⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨18885⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨22105⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨26632⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨29312⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨32125⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨34976⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨37656⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨51180⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨54160⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨57140⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨60120⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨63100⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨66671⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact113885RawTermsValid :
    exact113885RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event113885 : Event := .resultExact (⟨.program ⟨257⟩, ⟨70264⟩⟩) exact113885RawTerms .large 113883 (.finite 450666973253477225410675971981312) (some (113884))

def event113886 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70265⟩⟩) 0 ⟨70264⟩ 113885

def event113887 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70265⟩⟩) 1 ⟨42017⟩ 107072

def event113888 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70265⟩⟩) (.sum [.predecessor 0 113886 .coefficient, .predecessor 1 113887 .coefficient])

def event113889 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70265⟩⟩) (.sum [.result 113885 .summary, .result 107072 .summary])

def exact113890RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7222⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7224⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7226⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨16051⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨18885⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨22105⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨26632⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨29312⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨32125⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨34976⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨37656⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨40332⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨51180⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨54160⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨57140⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨60120⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨63100⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨66671⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact113890RawTermsValid :
    exact113890RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event113890 : Event := .resultExact (⟨.program ⟨257⟩, ⟨70265⟩⟩) exact113890RawTerms .large 113888 (.finite 482860102375766054599486172037120) (some (113889))

def event113891 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70266⟩⟩) 0 ⟨70265⟩ 113890

def event113892 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70266⟩⟩) 1 ⟨44697⟩ 106590

def event113893 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70266⟩⟩) (.sum [.predecessor 0 113891 .coefficient, .predecessor 1 113892 .coefficient])

def event113894 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70266⟩⟩) (.sum [.result 113890 .summary, .result 106590 .summary])

def exact113895RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7222⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7224⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7226⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7228⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨16051⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨18885⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨22105⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨26632⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨29312⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨32125⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨34976⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨37656⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨40332⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨43012⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨51180⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨54160⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨57140⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨60120⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨63100⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨66671⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact113895RawTermsValid :
    exact113895RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event113895 : Event := .resultExact (⟨.program ⟨257⟩, ⟨70266⟩⟩) exact113895RawTerms .large 113893 (.finite 515053820849391945920019041353728) (some (113894))

def event113896 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70267⟩⟩) 0 ⟨70266⟩ 113895

def event113897 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70267⟩⟩) 1 ⟨47377⟩ 106108

def event113898 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70267⟩⟩) (.sum [.predecessor 0 113896 .coefficient, .predecessor 1 113897 .coefficient])

def event113899 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70267⟩⟩) (.sum [.result 113895 .summary, .result 106108 .summary])

def exact113900RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7222⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7224⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7226⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7228⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7230⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨16051⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨18885⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨22105⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨26632⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨29312⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨32125⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨34976⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨37656⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨40332⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨43012⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨45696⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨51180⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨54160⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨57140⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨60120⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨63100⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨66671⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact113900RawTermsValid :
    exact113900RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event113900 : Event := .resultExact (⟨.program ⟨257⟩, ⟨70267⟩⟩) exact113900RawTerms .large 113898 (.finite 547248128674354899372274579931136) (some (113899))

def event113901 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70268⟩⟩) 0 ⟨70267⟩ 113900

def event113902 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70268⟩⟩) 1 ⟨50057⟩ 105626

def event113903 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70268⟩⟩) (.sum [.predecessor 0 113901 .coefficient, .predecessor 1 113902 .coefficient])

def event113904 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70268⟩⟩) (.sum [.result 113900 .summary, .result 105626 .summary])

def exact113905RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7222⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7224⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7226⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7228⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7230⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7232⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨16051⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨18885⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨22105⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨26632⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨29312⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨32125⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨34976⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨37656⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨40332⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨43012⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨45696⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨48376⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨51180⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨54160⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨57140⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨60120⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨63100⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨66671⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact113905RawTermsValid :
    exact113905RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event113905 : Event := .resultExact (⟨.program ⟨257⟩, ⟨70268⟩⟩) exact113905RawTerms .large 113903 (.finite 579442632949763540201771008262144) (some (113904))

def event113906 : Event := .predecessor (⟨.program ⟨257⟩, ⟨71269⟩⟩) 0 ⟨70268⟩ 113905

def event113907 : Event := .predecessor (⟨.program ⟨257⟩, ⟨71269⟩⟩) 1 ⟨71267⟩ 105128

def event113908 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨71269⟩⟩) (.product (.predecessor 0 113906 .coefficient) (.predecessor 1 113907 .coefficient) (⟨false, false, none, none, none⟩))

def event113909 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨71269⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨71267⟩⟩]⟩) [⟨.result 105128 .coefficient, false, none⟩])

def event113910 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨71269⟩⟩) (.product (.result 113905 .summary) (.transfer 113909) (⟨false, false, none, none, none⟩))

def event113911 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71269⟩⟩, .operator (⟨113905, 17⟩, ⟨105128, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7232⟩⟩, ⟨.program ⟨257⟩, ⟨71267⟩⟩]⟩, (1)⟩)

def event113912 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71269⟩⟩, .operator (⟨113905, 29⟩, ⟨105128, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨48376⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71267⟩⟩]⟩, (-1)⟩)

def event113913 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71269⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨48376⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71267⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71267⟩⟩) ⟨68836⟩ 105125)

def event113914 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71269⟩⟩, .relation 113913 0, ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨48376⟩⟩], [⟨.program ⟨257⟩, ⟨68836⟩⟩]⟩, (-1)⟩)

def event113915 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71269⟩⟩, .operator (⟨113905, 16⟩, ⟨105128, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7230⟩⟩, ⟨.program ⟨257⟩, ⟨71267⟩⟩]⟩, (1)⟩)

def event113916 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71269⟩⟩, .operator (⟨113905, 28⟩, ⟨105128, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨45696⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71267⟩⟩]⟩, (-1)⟩)

def event113917 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71269⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨45696⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71267⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71267⟩⟩) ⟨68836⟩ 105125)

def event113918 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71269⟩⟩, .relation 113917 0, ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨45696⟩⟩], [⟨.program ⟨257⟩, ⟨68836⟩⟩]⟩, (-1)⟩)

def event113919 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71269⟩⟩, .operator (⟨113905, 15⟩, ⟨105128, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7228⟩⟩, ⟨.program ⟨257⟩, ⟨71267⟩⟩]⟩, (1)⟩)

def eventLeaf7104 : Array AnnotatedEvent := #[
  { event := event113664
    frameStart := 113649 },
  { event := event113665
    frameStart := 113649 },
  { event := event113666
    frameStart := 113649 },
  { event := event113667
    frameStart := 113649 },
  { event := event113668
    frameStart := 113649 },
  { event := event113669
    frameStart := 113649 },
  { event := event113670
    frameStart := 113649 },
  { event := event113671
    frameStart := 113649 },
  { event := event113672
    frameStart := 113649 },
  { event := event113673
    frameStart := 113649 },
  { event := event113674
    frameStart := 113649 },
  { event := event113675
    frameStart := 113649 },
  { event := event113676
    frameStart := 113649 },
  { event := event113677
    frameStart := 113649 },
  { event := event113678
    frameStart := 113649 },
  { event := event113679
    frameStart := 113649 }
]

def eventLeaf7105 : Array AnnotatedEvent := #[
  { event := event113680
    frameStart := 113649 },
  { event := event113681
    frameStart := 113649 },
  { event := event113682
    frameStart := 113649 },
  { event := event113683
    frameStart := 113649 },
  { event := event113684
    frameStart := 113649 },
  { event := event113685
    frameStart := 113649 },
  { event := event113686
    frameStart := 113649 },
  { event := event113687
    frameStart := 113649 },
  { event := event113688
    frameStart := 113649 },
  { event := event113689
    frameStart := 113649 },
  { event := event113690
    frameStart := 113649 },
  { event := event113691
    frameStart := 113649 },
  { event := event113692
    frameStart := 113649 },
  { event := event113693
    frameStart := 113649 },
  { event := event113694
    frameStart := 113649 },
  { event := event113695
    frameStart := 113649 }
]

def eventLeaf7106 : Array AnnotatedEvent := #[
  { event := event113696
    frameStart := 113649 },
  { event := event113697
    frameStart := 113649 },
  { event := event113698
    frameStart := 113649 },
  { event := event113699
    frameStart := 113649 },
  { event := event113700
    frameStart := 113649 },
  { event := event113701
    frameStart := 113649 },
  { event := event113702
    frameStart := 113649 },
  { event := event113703
    frameStart := 113703 },
  { event := event113704
    frameStart := 113703 },
  { event := event113705
    frameStart := 113703 },
  { event := event113706
    frameStart := 113703 },
  { event := event113707
    frameStart := 113703 },
  { event := event113708
    frameStart := 113703 },
  { event := event113709
    frameStart := 113703 },
  { event := event113710
    frameStart := 113703 },
  { event := event113711
    frameStart := 113703 }
]

def eventLeaf7107 : Array AnnotatedEvent := #[
  { event := event113712
    frameStart := 113703 },
  { event := event113713
    frameStart := 113703 },
  { event := event113714
    frameStart := 113703 },
  { event := event113715
    frameStart := 113703 },
  { event := event113716
    frameStart := 113703 },
  { event := event113717
    frameStart := 113703 },
  { event := event113718
    frameStart := 113703 },
  { event := event113719
    frameStart := 113703 },
  { event := event113720
    frameStart := 113703 },
  { event := event113721
    frameStart := 113703 },
  { event := event113722
    frameStart := 113703 },
  { event := event113723
    frameStart := 113703 },
  { event := event113724
    frameStart := 113703 },
  { event := event113725
    frameStart := 113703 },
  { event := event113726
    frameStart := 113703 },
  { event := event113727
    frameStart := 113703 }
]

def eventLeaf7108 : Array AnnotatedEvent := #[
  { event := event113728
    frameStart := 113703 },
  { event := event113729
    frameStart := 113703 },
  { event := event113730
    frameStart := 113703 },
  { event := event113731
    frameStart := 113703 },
  { event := event113732
    frameStart := 113703 },
  { event := event113733
    frameStart := 113703 },
  { event := event113734
    frameStart := 113703 },
  { event := event113735
    frameStart := 113703 },
  { event := event113736
    frameStart := 113703 },
  { event := event113737
    frameStart := 113703 },
  { event := event113738
    frameStart := 113703 },
  { event := event113739
    frameStart := 113703 },
  { event := event113740
    frameStart := 113703 },
  { event := event113741
    frameStart := 113703 },
  { event := event113742
    frameStart := 113703 },
  { event := event113743
    frameStart := 113703 }
]

def eventLeaf7109 : Array AnnotatedEvent := #[
  { event := event113744
    frameStart := 113703 },
  { event := event113745
    frameStart := 113703 },
  { event := event113746
    frameStart := 113703 },
  { event := event113747
    frameStart := 113703 },
  { event := event113748
    frameStart := 113703 },
  { event := event113749
    frameStart := 113703 },
  { event := event113750
    frameStart := 113703 },
  { event := event113751
    frameStart := 113703 },
  { event := event113752
    frameStart := 113703 },
  { event := event113753
    frameStart := 113703 },
  { event := event113754
    frameStart := 113703 },
  { event := event113755
    frameStart := 113703 },
  { event := event113756
    frameStart := 113703 },
  { event := event113757
    frameStart := 113703 },
  { event := event113758
    frameStart := 113703 },
  { event := event113759
    frameStart := 113703 }
]

def eventLeaf7110 : Array AnnotatedEvent := #[
  { event := event113760
    frameStart := 113703 },
  { event := event113761
    frameStart := 113703 },
  { event := event113762
    frameStart := 113703 },
  { event := event113763
    frameStart := 113703 },
  { event := event113764
    frameStart := 113703 },
  { event := event113765
    frameStart := 113703 },
  { event := event113766
    frameStart := 113703 },
  { event := event113767
    frameStart := 113703 },
  { event := event113768
    frameStart := 113703 },
  { event := event113769
    frameStart := 113703 },
  { event := event113770
    frameStart := 113703 },
  { event := event113771
    frameStart := 113703 },
  { event := event113772
    frameStart := 113703 },
  { event := event113773
    frameStart := 113703 },
  { event := event113774
    frameStart := 113703 },
  { event := event113775
    frameStart := 113703 }
]

def eventLeaf7111 : Array AnnotatedEvent := #[
  { event := event113776
    frameStart := 113703 },
  { event := event113777
    frameStart := 113703 },
  { event := event113778
    frameStart := 113703 },
  { event := event113779
    frameStart := 113703 },
  { event := event113780
    frameStart := 113703 },
  { event := event113781
    frameStart := 113703 },
  { event := event113782
    frameStart := 113703 },
  { event := event113783
    frameStart := 113703 },
  { event := event113784
    frameStart := 113703 },
  { event := event113785
    frameStart := 113703 },
  { event := event113786
    frameStart := 113703 },
  { event := event113787
    frameStart := 113703 },
  { event := event113788
    frameStart := 113703 },
  { event := event113789
    frameStart := 113703 },
  { event := event113790
    frameStart := 113703 },
  { event := event113791
    frameStart := 113703 }
]

def eventLeaf7112 : Array AnnotatedEvent := #[
  { event := event113792
    frameStart := 113703 },
  { event := event113793
    frameStart := 113703 },
  { event := event113794
    frameStart := 113703 },
  { event := event113795
    frameStart := 113703 },
  { event := event113796
    frameStart := 113703 },
  { event := event113797
    frameStart := 113703 },
  { event := event113798
    frameStart := 113703 },
  { event := event113799
    frameStart := 113703 },
  { event := event113800
    frameStart := 113703 },
  { event := event113801
    frameStart := 113703 },
  { event := event113802
    frameStart := 113703 },
  { event := event113803
    frameStart := 113703 },
  { event := event113804
    frameStart := 113703 },
  { event := event113805
    frameStart := 113703 },
  { event := event113806
    frameStart := 113703 },
  { event := event113807
    frameStart := 0 }
]

def eventLeaf7113 : Array AnnotatedEvent := #[
  { event := event113808
    frameStart := 0 },
  { event := event113809
    frameStart := 0 },
  { event := event113810
    frameStart := 0 },
  { event := event113811
    frameStart := 0 },
  { event := event113812
    frameStart := 0 },
  { event := event113813
    frameStart := 0 },
  { event := event113814
    frameStart := 0 },
  { event := event113815
    frameStart := 0 },
  { event := event113816
    frameStart := 0 },
  { event := event113817
    frameStart := 0 },
  { event := event113818
    frameStart := 0 },
  { event := event113819
    frameStart := 0 },
  { event := event113820
    frameStart := 0 },
  { event := event113821
    frameStart := 0 },
  { event := event113822
    frameStart := 0 },
  { event := event113823
    frameStart := 0 }
]

def eventLeaf7114 : Array AnnotatedEvent := #[
  { event := event113824
    frameStart := 0 },
  { event := event113825
    frameStart := 0 },
  { event := event113826
    frameStart := 0 },
  { event := event113827
    frameStart := 0 },
  { event := event113828
    frameStart := 0 },
  { event := event113829
    frameStart := 0 },
  { event := event113830
    frameStart := 0 },
  { event := event113831
    frameStart := 0 },
  { event := event113832
    frameStart := 0 },
  { event := event113833
    frameStart := 0 },
  { event := event113834
    frameStart := 0 },
  { event := event113835
    frameStart := 0 },
  { event := event113836
    frameStart := 0 },
  { event := event113837
    frameStart := 0 },
  { event := event113838
    frameStart := 0 },
  { event := event113839
    frameStart := 0 }
]

def eventLeaf7115 : Array AnnotatedEvent := #[
  { event := event113840
    frameStart := 0 },
  { event := event113841
    frameStart := 0 },
  { event := event113842
    frameStart := 0 },
  { event := event113843
    frameStart := 0 },
  { event := event113844
    frameStart := 0 },
  { event := event113845
    frameStart := 0 },
  { event := event113846
    frameStart := 0 },
  { event := event113847
    frameStart := 0 },
  { event := event113848
    frameStart := 0 },
  { event := event113849
    frameStart := 0 },
  { event := event113850
    frameStart := 0 },
  { event := event113851
    frameStart := 0 },
  { event := event113852
    frameStart := 0 },
  { event := event113853
    frameStart := 0 },
  { event := event113854
    frameStart := 0 },
  { event := event113855
    frameStart := 0 }
]

def eventLeaf7116 : Array AnnotatedEvent := #[
  { event := event113856
    frameStart := 0 },
  { event := event113857
    frameStart := 0 },
  { event := event113858
    frameStart := 0 },
  { event := event113859
    frameStart := 0 },
  { event := event113860
    frameStart := 0 },
  { event := event113861
    frameStart := 0 },
  { event := event113862
    frameStart := 0 },
  { event := event113863
    frameStart := 0 },
  { event := event113864
    frameStart := 0 },
  { event := event113865
    frameStart := 0 },
  { event := event113866
    frameStart := 0 },
  { event := event113867
    frameStart := 0 },
  { event := event113868
    frameStart := 0 },
  { event := event113869
    frameStart := 0 },
  { event := event113870
    frameStart := 0 },
  { event := event113871
    frameStart := 0 }
]

def eventLeaf7117 : Array AnnotatedEvent := #[
  { event := event113872
    frameStart := 0 },
  { event := event113873
    frameStart := 0 },
  { event := event113874
    frameStart := 0 },
  { event := event113875
    frameStart := 0 },
  { event := event113876
    frameStart := 0 },
  { event := event113877
    frameStart := 0 },
  { event := event113878
    frameStart := 0 },
  { event := event113879
    frameStart := 0 },
  { event := event113880
    frameStart := 0 },
  { event := event113881
    frameStart := 0 },
  { event := event113882
    frameStart := 0 },
  { event := event113883
    frameStart := 0 },
  { event := event113884
    frameStart := 0 },
  { event := event113885
    frameStart := 0 },
  { event := event113886
    frameStart := 0 },
  { event := event113887
    frameStart := 0 }
]

def eventLeaf7118 : Array AnnotatedEvent := #[
  { event := event113888
    frameStart := 0 },
  { event := event113889
    frameStart := 0 },
  { event := event113890
    frameStart := 0 },
  { event := event113891
    frameStart := 0 },
  { event := event113892
    frameStart := 0 },
  { event := event113893
    frameStart := 0 },
  { event := event113894
    frameStart := 0 },
  { event := event113895
    frameStart := 0 },
  { event := event113896
    frameStart := 0 },
  { event := event113897
    frameStart := 0 },
  { event := event113898
    frameStart := 0 },
  { event := event113899
    frameStart := 0 },
  { event := event113900
    frameStart := 0 },
  { event := event113901
    frameStart := 0 },
  { event := event113902
    frameStart := 0 },
  { event := event113903
    frameStart := 0 }
]

def eventLeaf7119 : Array AnnotatedEvent := #[
  { event := event113904
    frameStart := 0 },
  { event := event113905
    frameStart := 0 },
  { event := event113906
    frameStart := 0 },
  { event := event113907
    frameStart := 0 },
  { event := event113908
    frameStart := 0 },
  { event := event113909
    frameStart := 0 },
  { event := event113910
    frameStart := 0 },
  { event := event113911
    frameStart := 0 },
  { event := event113912
    frameStart := 0 },
  { event := event113913
    frameStart := 0 },
  { event := event113914
    frameStart := 0 },
  { event := event113915
    frameStart := 0 },
  { event := event113916
    frameStart := 0 },
  { event := event113917
    frameStart := 0 },
  { event := event113918
    frameStart := 0 },
  { event := event113919
    frameStart := 0 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events444
