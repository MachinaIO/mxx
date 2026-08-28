import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events280

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event71680 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5530⟩⟩) (.finite 224)

def event71681 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11213⟩⟩) 0 ⟨5530⟩ 71680

def event71682 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11213⟩⟩) (.authority (.programFamilyFact))

def exact71683RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11213⟩⟩], []⟩, (1)⟩]

theorem exact71683RawTermsValid :
    exact71683RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71683 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11213⟩⟩) exact71683RawTerms (.finite 10) 71682 .exactZero (none)

def event71684 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13547⟩⟩) 0 ⟨5530⟩ 71680

def event71685 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13547⟩⟩) (.authority (.programFamilyFact))

def exact71686RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨13547⟩⟩], []⟩, (1)⟩]

theorem exact71686RawTermsValid :
    exact71686RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71686 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13547⟩⟩) exact71686RawTerms (.finite 10) 71685 .exactZero (none)

def event71687 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13548⟩⟩) 0 ⟨13547⟩ 71686

def event71688 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13548⟩⟩) 1 ⟨11213⟩ 71683

def event71689 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13548⟩⟩) (.product (.predecessor 0 71687 .coefficient) (.predecessor 1 71688 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event71690 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13548⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨11213⟩⟩, ⟨.program ⟨214⟩, ⟨13547⟩⟩], []⟩) [⟨.result 71686 .coefficient, true, some 1⟩, ⟨.result 71683 .coefficient, true, some 1⟩])

def event71691 : Event := .survivorFold (1) 71690

def exact71692RawTerms : List Term := []

theorem exact71692RawTermsValid :
    exact71692RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71692 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13548⟩⟩) exact71692RawTerms (.finite 100) 71689 (.finite 100) (some (71690))

def event71693 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13549⟩⟩) 0 ⟨13548⟩ 71692

def event71694 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13549⟩⟩) (.identity (.predecessor 0 71693 .coefficient))

def event71695 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨13549⟩⟩) (.finite 100)

def event71696 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19308⟩⟩) 0 ⟨13549⟩ 71695

def event71697 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19308⟩⟩) (.authority (.relationPreimageSource ⟨12⟩))

def exact71698RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨19308⟩⟩]⟩, (1)⟩]

theorem exact71698RawTermsValid :
    exact71698RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71698 : Event := .resultExact (⟨.program ⟨214⟩, ⟨19308⟩⟩) exact71698RawTerms (.finite 136065468) 71697 .exactZero (none)

def event71699 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6⟩⟩) (.authority (.operator))

def exact71700RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩]⟩, (1)⟩]

theorem exact71700RawTermsValid :
    exact71700RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71700 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6⟩⟩) exact71700RawTerms .large 71699 .exactZero (none)

def event71701 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19309⟩⟩) 0 ⟨6⟩ 71700

def event71702 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19309⟩⟩) 1 ⟨19308⟩ 71698

def event71703 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19309⟩⟩) (.product (.predecessor 0 71701 .coefficient) (.predecessor 1 71702 .coefficient) (⟨false, false, none, none, none⟩))

def event71704 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19309⟩⟩, .operator (⟨71700, 0⟩, ⟨71698, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19308⟩⟩]⟩, (1)⟩)

def exact71705RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19308⟩⟩]⟩, (1)⟩]

theorem exact71705RawTermsValid :
    exact71705RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71705 : Event := .resultExact (⟨.program ⟨214⟩, ⟨19309⟩⟩) exact71705RawTerms .large 71703 .exactZero (none)

def event71706 : Event := .preFoldPolynomial 71705 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19308⟩⟩]⟩, (1)⟩] .exactZero none

def exact71707RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19308⟩⟩]⟩, (1)⟩]

def event71707 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨19309⟩⟩) 71706 exact71707RawTerms .large 71703 .exactZero (none)

def event71708 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨25834⟩⟩)

def event71709 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event71710 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event71711 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨360⟩⟩) (.authority (.operator))

def event71712 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨360⟩⟩) (.finite 2)

def event71713 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event71714 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event71715 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event71716 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event71717 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 71716

def event71718 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 71714

def event71719 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 71717 .coefficient) (.value (.predecessor 1 71718 .coefficient)))

def event71720 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event71721 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5504⟩⟩) 0 ⟨5503⟩ 71720

def event71722 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5504⟩⟩) 1 ⟨360⟩ 71712

def event71723 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5504⟩⟩) (.sum [.predecessor 0 71721 .coefficient, .predecessor 1 71722 .coefficient])

def event71724 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5504⟩⟩) (.finite 219)

def event71725 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5530⟩⟩) 0 ⟨5504⟩ 71724

def event71726 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5530⟩⟩) 1 ⟨961⟩ 71710

def event71727 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5530⟩⟩) (.identity (.predecessor 1 71726 .coefficient))

def event71728 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5530⟩⟩) (.finite 224)

def event71729 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11213⟩⟩) 0 ⟨5530⟩ 71728

def event71730 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11213⟩⟩) (.authority (.programFamilyFact))

def exact71731RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11213⟩⟩], []⟩, (1)⟩]

theorem exact71731RawTermsValid :
    exact71731RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71731 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11213⟩⟩) exact71731RawTerms (.finite 10) 71730 .exactZero (none)

def event71732 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13547⟩⟩) 0 ⟨5530⟩ 71728

def event71733 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13547⟩⟩) (.authority (.programFamilyFact))

def exact71734RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨13547⟩⟩], []⟩, (1)⟩]

theorem exact71734RawTermsValid :
    exact71734RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71734 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13547⟩⟩) exact71734RawTerms (.finite 10) 71733 .exactZero (none)

def event71735 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13548⟩⟩) 0 ⟨13547⟩ 71734

def event71736 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13548⟩⟩) 1 ⟨11213⟩ 71731

def event71737 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13548⟩⟩) (.product (.predecessor 0 71735 .coefficient) (.predecessor 1 71736 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event71738 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨13548⟩⟩, .operator (⟨71734, 0⟩, ⟨71731, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨11213⟩⟩, ⟨.program ⟨214⟩, ⟨13547⟩⟩], []⟩, (1)⟩)

def exact71739RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11213⟩⟩, ⟨.program ⟨214⟩, ⟨13547⟩⟩], []⟩, (1)⟩]

theorem exact71739RawTermsValid :
    exact71739RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71739 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13548⟩⟩) exact71739RawTerms (.finite 100) 71737 .exactZero (none)

def event71740 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13549⟩⟩) 0 ⟨13548⟩ 71739

def event71741 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13549⟩⟩) (.identity (.predecessor 0 71740 .coefficient))

def event71742 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨13549⟩⟩) (.finite 100)

def event71743 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23455⟩⟩) 0 ⟨13549⟩ 71742

def event71744 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23455⟩⟩) (.authority (.programFamilyFact))

def event71745 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨23455⟩⟩) (.finite 3720)

def event71746 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨6689⟩⟩) .missing

def event71747 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23456⟩⟩) 0 ⟨6689⟩ 71746

def event71748 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23456⟩⟩) 1 ⟨23455⟩ 71745

def event71749 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23456⟩⟩) (.authority (.operator))

def exact71750RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨23456⟩⟩]⟩, (1)⟩]

theorem exact71750RawTermsValid :
    exact71750RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71750 : Event := .resultExact (⟨.program ⟨214⟩, ⟨23456⟩⟩) exact71750RawTerms .large 71749 .exactZero (none)

def event71751 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25830⟩⟩) 0 ⟨23456⟩ 71750

def event71752 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25830⟩⟩) (.authority (.operator))

def exact71753RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨25830⟩⟩]⟩, (1)⟩]

theorem exact71753RawTermsValid :
    exact71753RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71753 : Event := .resultExact (⟨.program ⟨214⟩, ⟨25830⟩⟩) exact71753RawTerms (.finite 8192) 71752 .exactZero (none)

def event71754 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨110⟩⟩) (.authority (.operator))

def event71755 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨110⟩⟩) .exactZero

def event71756 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13659⟩⟩) 0 ⟨13549⟩ 71742

def event71757 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13659⟩⟩) 1 ⟨110⟩ 71755

def event71758 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13659⟩⟩) (.sum [.predecessor 0 71756 .coefficient, .predecessor 1 71757 .coefficient])

def event71759 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨13659⟩⟩) (.finite 100)

def event71760 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13660⟩⟩) 0 ⟨13659⟩ 71759

def event71761 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13660⟩⟩) (.identity (.predecessor 0 71760 .coefficient))

def exact71762RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11213⟩⟩, ⟨.program ⟨214⟩, ⟨13547⟩⟩], []⟩, (1)⟩]

theorem exact71762RawTermsValid :
    exact71762RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71762 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13660⟩⟩) exact71762RawTerms (.finite 100) 71761 .exactZero (none)

def event71763 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6544⟩⟩) (.authority (.factStore))

def exact71764RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact71764RawTermsValid :
    exact71764RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71764 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6544⟩⟩) exact71764RawTerms .large 71763 .exactZero (none)

def event71765 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13661⟩⟩) 0 ⟨6544⟩ 71764

def event71766 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13661⟩⟩) 1 ⟨13660⟩ 71762

def event71767 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13661⟩⟩) (.product (.predecessor 0 71765 .coefficient) (.predecessor 1 71766 .coefficient) (⟨false, false, none, none, none⟩))

def event71768 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨13661⟩⟩, .operator (⟨71764, 0⟩, ⟨71762, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨11213⟩⟩, ⟨.program ⟨214⟩, ⟨13547⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact71769RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11213⟩⟩, ⟨.program ⟨214⟩, ⟨13547⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact71769RawTermsValid :
    exact71769RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71769 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13661⟩⟩) exact71769RawTerms .large 71767 .exactZero (none)

def event71770 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨2348⟩⟩) (.authority (.operator))

def event71771 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨2348⟩⟩) (.finite 1)

def event71772 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6757⟩⟩) 0 ⟨6689⟩ 71746

def event71773 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6757⟩⟩) (.authority (.operator))

def exact71774RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6757⟩⟩]⟩, (1)⟩]

theorem exact71774RawTermsValid :
    exact71774RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71774 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6757⟩⟩) exact71774RawTerms .large 71773 .exactZero (none)

def event71775 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6776⟩⟩) 0 ⟨6757⟩ 71774

def event71776 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6776⟩⟩) (.identity (.predecessor 0 71775 .coefficient))

def exact71777RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6776⟩⟩]⟩, (1)⟩]

theorem exact71777RawTermsValid :
    exact71777RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71777 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6776⟩⟩) exact71777RawTerms .large 71776 .exactZero (none)

def event71778 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7843⟩⟩) 0 ⟨6776⟩ 71777

def event71779 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7843⟩⟩) (.authority (.operator))

def exact71780RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨7843⟩⟩]⟩, (1)⟩]

theorem exact71780RawTermsValid :
    exact71780RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71780 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7843⟩⟩) exact71780RawTerms (.finite 8192) 71779 .exactZero (none)

def event71781 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7844⟩⟩) 0 ⟨7843⟩ 71780

def event71782 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7844⟩⟩) 1 ⟨2348⟩ 71771

def event71783 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7844⟩⟩) (.scale (.predecessor 0 71781 .coefficient) (.value (.predecessor 1 71782 .coefficient)))

def exact71784RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨7843⟩⟩]⟩, (1)⟩]

theorem exact71784RawTermsValid :
    exact71784RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71784 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7844⟩⟩) exact71784RawTerms (.finite 8192) 71783 .exactZero (none)

def event71785 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6793⟩⟩) 0 ⟨6757⟩ 71774

def event71786 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6793⟩⟩) (.identity (.predecessor 0 71785 .coefficient))

def exact71787RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6793⟩⟩]⟩, (1)⟩]

theorem exact71787RawTermsValid :
    exact71787RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71787 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6793⟩⟩) exact71787RawTerms .large 71786 .exactZero (none)

def event71788 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7845⟩⟩) 0 ⟨6793⟩ 71787

def event71789 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7845⟩⟩) 1 ⟨7844⟩ 71784

def event71790 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7845⟩⟩) (.product (.predecessor 0 71788 .coefficient) (.predecessor 1 71789 .coefficient) (⟨false, false, none, none, none⟩))

def event71791 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7845⟩⟩, .operator (⟨71787, 0⟩, ⟨71784, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6793⟩⟩, ⟨.program ⟨214⟩, ⟨7843⟩⟩]⟩, (1)⟩)

def exact71792RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6793⟩⟩, ⟨.program ⟨214⟩, ⟨7843⟩⟩]⟩, (1)⟩]

theorem exact71792RawTermsValid :
    exact71792RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71792 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7845⟩⟩) exact71792RawTerms .large 71790 .exactZero (none)

def event71793 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13662⟩⟩) 0 ⟨7845⟩ 71792

def event71794 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13662⟩⟩) 1 ⟨13661⟩ 71769

def event71795 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13662⟩⟩) (.sum [.predecessor 0 71793 .coefficient, .predecessor 1 71794 .coefficient])

def exact71796RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6793⟩⟩, ⟨.program ⟨214⟩, ⟨7843⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨11213⟩⟩, ⟨.program ⟨214⟩, ⟨13547⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact71796RawTermsValid :
    exact71796RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71796 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13662⟩⟩) exact71796RawTerms .large 71795 .exactZero (none)

def event71797 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25833⟩⟩) 0 ⟨13662⟩ 71796

def event71798 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25833⟩⟩) 1 ⟨25830⟩ 71753

def event71799 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25833⟩⟩) (.product (.predecessor 0 71797 .coefficient) (.predecessor 1 71798 .coefficient) (⟨false, false, none, none, none⟩))

def event71800 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25833⟩⟩, .operator (⟨71796, 0⟩, ⟨71753, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6793⟩⟩, ⟨.program ⟨214⟩, ⟨7843⟩⟩, ⟨.program ⟨214⟩, ⟨25830⟩⟩]⟩, (1)⟩)

def event71801 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25833⟩⟩, .operator (⟨71796, 1⟩, ⟨71753, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨11213⟩⟩, ⟨.program ⟨214⟩, ⟨13547⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25830⟩⟩]⟩, (-1)⟩)

def event71802 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨25833⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨11213⟩⟩, ⟨.program ⟨214⟩, ⟨13547⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25830⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨25830⟩⟩) ⟨23456⟩ 71750)

def event71803 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25833⟩⟩, .relation 71802 0, ⟨[⟨.program ⟨214⟩, ⟨11213⟩⟩, ⟨.program ⟨214⟩, ⟨13547⟩⟩], [⟨.program ⟨214⟩, ⟨23456⟩⟩]⟩, (-1)⟩)

def exact71804RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6793⟩⟩, ⟨.program ⟨214⟩, ⟨7843⟩⟩, ⟨.program ⟨214⟩, ⟨25830⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨11213⟩⟩, ⟨.program ⟨214⟩, ⟨13547⟩⟩], [⟨.program ⟨214⟩, ⟨23456⟩⟩]⟩, (-1)⟩]

theorem exact71804RawTermsValid :
    exact71804RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71804 : Event := .resultExact (⟨.program ⟨214⟩, ⟨25833⟩⟩) exact71804RawTerms .large 71799 .exactZero (none)

def event71805 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15579⟩⟩) 0 ⟨13549⟩ 71742

def event71806 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15579⟩⟩) (.authority (.programFamilyFact))

def exact71807RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15579⟩⟩], []⟩, (1)⟩]

theorem exact71807RawTermsValid :
    exact71807RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71807 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15579⟩⟩) exact71807RawTerms (.finite 10) 71806 .exactZero (none)

def event71808 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15581⟩⟩) 0 ⟨6544⟩ 71764

def event71809 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15581⟩⟩) 1 ⟨15579⟩ 71807

def event71810 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15581⟩⟩) (.product (.predecessor 0 71808 .coefficient) (.predecessor 1 71809 .coefficient) (⟨false, true, none, none, some 1⟩))

def event71811 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨15581⟩⟩, .operator (⟨71764, 0⟩, ⟨71807, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨15579⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact71812RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15579⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact71812RawTermsValid :
    exact71812RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71812 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15581⟩⟩) exact71812RawTerms .large 71810 .exactZero (none)

def event71813 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6694⟩⟩) 0 ⟨6689⟩ 71746

def event71814 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6694⟩⟩) (.authority (.operator))

def exact71815RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6694⟩⟩]⟩, (1)⟩]

theorem exact71815RawTermsValid :
    exact71815RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71815 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6694⟩⟩) exact71815RawTerms .large 71814 .exactZero (none)

def event71816 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15582⟩⟩) 0 ⟨6694⟩ 71815

def event71817 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15582⟩⟩) 1 ⟨15581⟩ 71812

def event71818 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15582⟩⟩) (.sum [.predecessor 0 71816 .coefficient, .predecessor 1 71817 .coefficient])

def exact71819RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6694⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15579⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact71819RawTermsValid :
    exact71819RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71819 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15582⟩⟩) exact71819RawTerms .large 71818 .exactZero (none)

def event71820 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25834⟩⟩) 0 ⟨15582⟩ 71819

def event71821 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25834⟩⟩) 1 ⟨25833⟩ 71804

def event71822 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25834⟩⟩) (.sum [.predecessor 0 71820 .coefficient, .predecessor 1 71821 .coefficient])

def exact71823RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6694⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6793⟩⟩, ⟨.program ⟨214⟩, ⟨7843⟩⟩, ⟨.program ⟨214⟩, ⟨25830⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨11213⟩⟩, ⟨.program ⟨214⟩, ⟨13547⟩⟩], [⟨.program ⟨214⟩, ⟨23456⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15579⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact71823RawTermsValid :
    exact71823RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71823 : Event := .resultExact (⟨.program ⟨214⟩, ⟨25834⟩⟩) exact71823RawTerms .large 71822 .exactZero (none)

def event71824 : Event := .preFoldPolynomial 71823 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6694⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6793⟩⟩, ⟨.program ⟨214⟩, ⟨7843⟩⟩, ⟨.program ⟨214⟩, ⟨25830⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨11213⟩⟩, ⟨.program ⟨214⟩, ⟨13547⟩⟩], [⟨.program ⟨214⟩, ⟨23456⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15579⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩] .exactZero none

def exact71825RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6694⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6793⟩⟩, ⟨.program ⟨214⟩, ⟨7843⟩⟩, ⟨.program ⟨214⟩, ⟨25830⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨11213⟩⟩, ⟨.program ⟨214⟩, ⟨13547⟩⟩], [⟨.program ⟨214⟩, ⟨23456⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15579⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

def event71825 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨25834⟩⟩) 71824 exact71825RawTerms .large 71822 .exactZero (none)

def event71826 : Event := .specializationComputed (⟨.program ⟨214⟩, ⟨13549⟩⟩) ⟨⟨107⟩, ⟨12⟩, ⟨109⟩⟩ ⟨71660, 71826⟩

def event71827 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨19311⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19308⟩⟩]⟩) (1) 0 2 (.universal 71826 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19308⟩⟩]⟩) (none) 71825)

def event71828 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19311⟩⟩, .relation 71827 0, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6694⟩⟩]⟩, (1)⟩)

def event71829 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19311⟩⟩, .relation 71827 1, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6793⟩⟩, ⟨.program ⟨214⟩, ⟨7843⟩⟩, ⟨.program ⟨214⟩, ⟨25830⟩⟩]⟩, (-1)⟩)

def event71830 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19311⟩⟩, .relation 71827 2, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨11213⟩⟩, ⟨.program ⟨214⟩, ⟨13547⟩⟩], [⟨.program ⟨214⟩, ⟨23456⟩⟩]⟩, (1)⟩)

def event71831 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19311⟩⟩, .relation 71827 3, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15579⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def exact71832RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6694⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6793⟩⟩, ⟨.program ⟨214⟩, ⟨7843⟩⟩, ⟨.program ⟨214⟩, ⟨25830⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨11213⟩⟩, ⟨.program ⟨214⟩, ⟨13547⟩⟩], [⟨.program ⟨214⟩, ⟨23456⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15579⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact71832RawTermsValid :
    exact71832RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71832 : Event := .resultExact (⟨.program ⟨214⟩, ⟨19311⟩⟩) exact71832RawTerms .large 71656 (.finite 1811303510016) (some (71658))

def event71833 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25832⟩⟩) 0 ⟨19311⟩ 71832

def event71834 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25832⟩⟩) 1 ⟨25831⟩ 71646

def event71835 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25832⟩⟩) (.sum [.predecessor 0 71833 .coefficient, .predecessor 1 71834 .coefficient])

def event71836 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25832⟩⟩, .operator (⟨71832, 2⟩, ⟨71646, 1⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨11213⟩⟩, ⟨.program ⟨214⟩, ⟨13547⟩⟩], [⟨.program ⟨214⟩, ⟨23456⟩⟩]⟩, (-1)⟩)

def event71837 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25832⟩⟩, .operator (⟨71832, 1⟩, ⟨71646, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6793⟩⟩, ⟨.program ⟨214⟩, ⟨7843⟩⟩, ⟨.program ⟨214⟩, ⟨25830⟩⟩]⟩, (1)⟩)

def event71838 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25832⟩⟩) (.sum [.result 71832 .summary, .result 71646 .summary])

def exact71839RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6694⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15579⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact71839RawTermsValid :
    exact71839RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71839 : Event := .resultExact (⟨.program ⟨214⟩, ⟨25832⟩⟩) exact71839RawTerms .large 71835 (.finite 352036291489792) (some (71838))

def event71840 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27204⟩⟩) 0 ⟨25832⟩ 71839

def event71841 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27204⟩⟩) 1 ⟨27202⟩ 71562

def event71842 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27204⟩⟩) (.product (.predecessor 0 71840 .coefficient) (.predecessor 1 71841 .coefficient) (⟨false, false, none, none, none⟩))

def event71843 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27204⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨27202⟩⟩]⟩) [⟨.result 71562 .coefficient, false, none⟩])

def event71844 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27204⟩⟩) (.product (.result 71839 .summary) (.transfer 71843) (⟨false, false, none, none, none⟩))

def event71845 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27204⟩⟩, .operator (⟨71839, 0⟩, ⟨71562, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6694⟩⟩, ⟨.program ⟨214⟩, ⟨27202⟩⟩]⟩, (1)⟩)

def event71846 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27204⟩⟩, .operator (⟨71839, 1⟩, ⟨71562, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15579⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨27202⟩⟩]⟩, (-1)⟩)

def event71847 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨27204⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15579⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨27202⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨27202⟩⟩) ⟨23970⟩ 71559)

def event71848 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27204⟩⟩, .relation 71847 0, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15579⟩⟩], [⟨.program ⟨214⟩, ⟨23970⟩⟩]⟩, (-1)⟩)

def exact71849RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6694⟩⟩, ⟨.program ⟨214⟩, ⟨27202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15579⟩⟩], [⟨.program ⟨214⟩, ⟨23970⟩⟩]⟩, (-1)⟩]

theorem exact71849RawTermsValid :
    exact71849RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71849 : Event := .resultExact (⟨.program ⟨214⟩, ⟨27204⟩⟩) exact71849RawTerms .large 71842 (.finite 1291978822348200476672) (some (71844))

def event71850 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20964⟩⟩) 0 ⟨15580⟩ 3402

def event71851 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20964⟩⟩) (.authority (.relationPreimageSource ⟨37⟩))

def exact71852RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨20964⟩⟩]⟩, (1)⟩]

theorem exact71852RawTermsValid :
    exact71852RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71852 : Event := .resultExact (⟨.program ⟨214⟩, ⟨20964⟩⟩) exact71852RawTerms (.finite 136065468) 71851 .exactZero (none)

def event71853 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20966⟩⟩) 0 ⟨20964⟩ 71852

def event71854 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20966⟩⟩) 1 ⟨2348⟩ 4

def event71855 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20966⟩⟩) (.scale (.predecessor 0 71853 .coefficient) (.value (.predecessor 1 71854 .coefficient)))

def exact71856RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨20964⟩⟩]⟩, (1)⟩]

theorem exact71856RawTermsValid :
    exact71856RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71856 : Event := .resultExact (⟨.program ⟨214⟩, ⟨20966⟩⟩) exact71856RawTerms (.finite 136065468) 71855 .exactZero (none)

def event71857 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20967⟩⟩) 0 ⟨5535⟩ 65387

def event71858 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20967⟩⟩) 1 ⟨20966⟩ 71856

def event71859 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20967⟩⟩) (.product (.predecessor 0 71857 .coefficient) (.predecessor 1 71858 .coefficient) (⟨false, false, none, none, none⟩))

def event71860 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20967⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨20964⟩⟩]⟩) [⟨.result 71852 .coefficient, false, none⟩])

def event71861 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20967⟩⟩) (.product (.result 65387 .summary) (.transfer 71860) (⟨false, false, none, none, none⟩))

def event71862 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20967⟩⟩, .operator (⟨65387, 0⟩, ⟨71856, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20964⟩⟩]⟩, (1)⟩)

def event71863 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨20965⟩⟩)

def event71864 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event71865 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event71866 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨360⟩⟩) (.authority (.operator))

def event71867 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨360⟩⟩) (.finite 2)

def event71868 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event71869 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event71870 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event71871 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event71872 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 71871

def event71873 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 71869

def event71874 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 71872 .coefficient) (.value (.predecessor 1 71873 .coefficient)))

def event71875 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event71876 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5504⟩⟩) 0 ⟨5503⟩ 71875

def event71877 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5504⟩⟩) 1 ⟨360⟩ 71867

def event71878 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5504⟩⟩) (.sum [.predecessor 0 71876 .coefficient, .predecessor 1 71877 .coefficient])

def event71879 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5504⟩⟩) (.finite 219)

def event71880 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5530⟩⟩) 0 ⟨5504⟩ 71879

def event71881 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5530⟩⟩) 1 ⟨961⟩ 71865

def event71882 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5530⟩⟩) (.identity (.predecessor 1 71881 .coefficient))

def event71883 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5530⟩⟩) (.finite 224)

def event71884 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11213⟩⟩) 0 ⟨5530⟩ 71883

def event71885 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11213⟩⟩) (.authority (.programFamilyFact))

def exact71886RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11213⟩⟩], []⟩, (1)⟩]

theorem exact71886RawTermsValid :
    exact71886RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71886 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11213⟩⟩) exact71886RawTerms (.finite 10) 71885 .exactZero (none)

def event71887 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13547⟩⟩) 0 ⟨5530⟩ 71883

def event71888 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13547⟩⟩) (.authority (.programFamilyFact))

def exact71889RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨13547⟩⟩], []⟩, (1)⟩]

theorem exact71889RawTermsValid :
    exact71889RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71889 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13547⟩⟩) exact71889RawTerms (.finite 10) 71888 .exactZero (none)

def event71890 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13548⟩⟩) 0 ⟨13547⟩ 71889

def event71891 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13548⟩⟩) 1 ⟨11213⟩ 71886

def event71892 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13548⟩⟩) (.product (.predecessor 0 71890 .coefficient) (.predecessor 1 71891 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event71893 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13548⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨11213⟩⟩, ⟨.program ⟨214⟩, ⟨13547⟩⟩], []⟩) [⟨.result 71889 .coefficient, true, some 1⟩, ⟨.result 71886 .coefficient, true, some 1⟩])

def event71894 : Event := .survivorFold (1) 71893

def exact71895RawTerms : List Term := []

theorem exact71895RawTermsValid :
    exact71895RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71895 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13548⟩⟩) exact71895RawTerms (.finite 100) 71892 (.finite 100) (some (71893))

def event71896 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13549⟩⟩) 0 ⟨13548⟩ 71895

def event71897 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13549⟩⟩) (.identity (.predecessor 0 71896 .coefficient))

def event71898 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨13549⟩⟩) (.finite 100)

def event71899 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15579⟩⟩) 0 ⟨13549⟩ 71898

def event71900 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15579⟩⟩) (.authority (.programFamilyFact))

def exact71901RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15579⟩⟩], []⟩, (1)⟩]

theorem exact71901RawTermsValid :
    exact71901RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71901 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15579⟩⟩) exact71901RawTerms (.finite 10) 71900 .exactZero (none)

def event71902 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15580⟩⟩) 0 ⟨15579⟩ 71901

def event71903 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15580⟩⟩) (.identity (.predecessor 0 71902 .coefficient))

def event71904 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨15580⟩⟩) (.finite 10)

def event71905 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20964⟩⟩) 0 ⟨15580⟩ 71904

def event71906 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20964⟩⟩) (.authority (.relationPreimageSource ⟨37⟩))

def exact71907RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨20964⟩⟩]⟩, (1)⟩]

theorem exact71907RawTermsValid :
    exact71907RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71907 : Event := .resultExact (⟨.program ⟨214⟩, ⟨20964⟩⟩) exact71907RawTerms (.finite 136065468) 71906 .exactZero (none)

def event71908 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6⟩⟩) (.authority (.operator))

def exact71909RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩]⟩, (1)⟩]

theorem exact71909RawTermsValid :
    exact71909RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71909 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6⟩⟩) exact71909RawTerms .large 71908 .exactZero (none)

def event71910 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20965⟩⟩) 0 ⟨6⟩ 71909

def event71911 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20965⟩⟩) 1 ⟨20964⟩ 71907

def event71912 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20965⟩⟩) (.product (.predecessor 0 71910 .coefficient) (.predecessor 1 71911 .coefficient) (⟨false, false, none, none, none⟩))

def event71913 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20965⟩⟩, .operator (⟨71909, 0⟩, ⟨71907, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20964⟩⟩]⟩, (1)⟩)

def exact71914RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20964⟩⟩]⟩, (1)⟩]

theorem exact71914RawTermsValid :
    exact71914RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71914 : Event := .resultExact (⟨.program ⟨214⟩, ⟨20965⟩⟩) exact71914RawTerms .large 71912 .exactZero (none)

def event71915 : Event := .preFoldPolynomial 71914 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20964⟩⟩]⟩, (1)⟩] .exactZero none

def exact71916RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20964⟩⟩]⟩, (1)⟩]

def event71916 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨20965⟩⟩) 71915 exact71916RawTerms .large 71912 .exactZero (none)

def event71917 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨27207⟩⟩)

def event71918 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event71919 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event71920 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨360⟩⟩) (.authority (.operator))

def event71921 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨360⟩⟩) (.finite 2)

def event71922 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event71923 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event71924 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event71925 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event71926 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 71925

def event71927 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 71923

def event71928 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 71926 .coefficient) (.value (.predecessor 1 71927 .coefficient)))

def event71929 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event71930 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5504⟩⟩) 0 ⟨5503⟩ 71929

def event71931 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5504⟩⟩) 1 ⟨360⟩ 71921

def event71932 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5504⟩⟩) (.sum [.predecessor 0 71930 .coefficient, .predecessor 1 71931 .coefficient])

def event71933 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5504⟩⟩) (.finite 219)

def event71934 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5530⟩⟩) 0 ⟨5504⟩ 71933

def event71935 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5530⟩⟩) 1 ⟨961⟩ 71919

def eventLeaf4480 : Array AnnotatedEvent := #[
  { event := event71680
    frameStart := 71660 },
  { event := event71681
    frameStart := 71660 },
  { event := event71682
    frameStart := 71660 },
  { event := event71683
    frameStart := 71660 },
  { event := event71684
    frameStart := 71660 },
  { event := event71685
    frameStart := 71660 },
  { event := event71686
    frameStart := 71660 },
  { event := event71687
    frameStart := 71660 },
  { event := event71688
    frameStart := 71660 },
  { event := event71689
    frameStart := 71660 },
  { event := event71690
    frameStart := 71660 },
  { event := event71691
    frameStart := 71660 },
  { event := event71692
    frameStart := 71660 },
  { event := event71693
    frameStart := 71660 },
  { event := event71694
    frameStart := 71660 },
  { event := event71695
    frameStart := 71660 }
]

def eventLeaf4481 : Array AnnotatedEvent := #[
  { event := event71696
    frameStart := 71660 },
  { event := event71697
    frameStart := 71660 },
  { event := event71698
    frameStart := 71660 },
  { event := event71699
    frameStart := 71660 },
  { event := event71700
    frameStart := 71660 },
  { event := event71701
    frameStart := 71660 },
  { event := event71702
    frameStart := 71660 },
  { event := event71703
    frameStart := 71660 },
  { event := event71704
    frameStart := 71660 },
  { event := event71705
    frameStart := 71660 },
  { event := event71706
    frameStart := 71660 },
  { event := event71707
    frameStart := 71660 },
  { event := event71708
    frameStart := 71708 },
  { event := event71709
    frameStart := 71708 },
  { event := event71710
    frameStart := 71708 },
  { event := event71711
    frameStart := 71708 }
]

def eventLeaf4482 : Array AnnotatedEvent := #[
  { event := event71712
    frameStart := 71708 },
  { event := event71713
    frameStart := 71708 },
  { event := event71714
    frameStart := 71708 },
  { event := event71715
    frameStart := 71708 },
  { event := event71716
    frameStart := 71708 },
  { event := event71717
    frameStart := 71708 },
  { event := event71718
    frameStart := 71708 },
  { event := event71719
    frameStart := 71708 },
  { event := event71720
    frameStart := 71708 },
  { event := event71721
    frameStart := 71708 },
  { event := event71722
    frameStart := 71708 },
  { event := event71723
    frameStart := 71708 },
  { event := event71724
    frameStart := 71708 },
  { event := event71725
    frameStart := 71708 },
  { event := event71726
    frameStart := 71708 },
  { event := event71727
    frameStart := 71708 }
]

def eventLeaf4483 : Array AnnotatedEvent := #[
  { event := event71728
    frameStart := 71708 },
  { event := event71729
    frameStart := 71708 },
  { event := event71730
    frameStart := 71708 },
  { event := event71731
    frameStart := 71708 },
  { event := event71732
    frameStart := 71708 },
  { event := event71733
    frameStart := 71708 },
  { event := event71734
    frameStart := 71708 },
  { event := event71735
    frameStart := 71708 },
  { event := event71736
    frameStart := 71708 },
  { event := event71737
    frameStart := 71708 },
  { event := event71738
    frameStart := 71708 },
  { event := event71739
    frameStart := 71708 },
  { event := event71740
    frameStart := 71708 },
  { event := event71741
    frameStart := 71708 },
  { event := event71742
    frameStart := 71708 },
  { event := event71743
    frameStart := 71708 }
]

def eventLeaf4484 : Array AnnotatedEvent := #[
  { event := event71744
    frameStart := 71708 },
  { event := event71745
    frameStart := 71708 },
  { event := event71746
    frameStart := 71708 },
  { event := event71747
    frameStart := 71708 },
  { event := event71748
    frameStart := 71708 },
  { event := event71749
    frameStart := 71708 },
  { event := event71750
    frameStart := 71708 },
  { event := event71751
    frameStart := 71708 },
  { event := event71752
    frameStart := 71708 },
  { event := event71753
    frameStart := 71708 },
  { event := event71754
    frameStart := 71708 },
  { event := event71755
    frameStart := 71708 },
  { event := event71756
    frameStart := 71708 },
  { event := event71757
    frameStart := 71708 },
  { event := event71758
    frameStart := 71708 },
  { event := event71759
    frameStart := 71708 }
]

def eventLeaf4485 : Array AnnotatedEvent := #[
  { event := event71760
    frameStart := 71708 },
  { event := event71761
    frameStart := 71708 },
  { event := event71762
    frameStart := 71708 },
  { event := event71763
    frameStart := 71708 },
  { event := event71764
    frameStart := 71708 },
  { event := event71765
    frameStart := 71708 },
  { event := event71766
    frameStart := 71708 },
  { event := event71767
    frameStart := 71708 },
  { event := event71768
    frameStart := 71708 },
  { event := event71769
    frameStart := 71708 },
  { event := event71770
    frameStart := 71708 },
  { event := event71771
    frameStart := 71708 },
  { event := event71772
    frameStart := 71708 },
  { event := event71773
    frameStart := 71708 },
  { event := event71774
    frameStart := 71708 },
  { event := event71775
    frameStart := 71708 }
]

def eventLeaf4486 : Array AnnotatedEvent := #[
  { event := event71776
    frameStart := 71708 },
  { event := event71777
    frameStart := 71708 },
  { event := event71778
    frameStart := 71708 },
  { event := event71779
    frameStart := 71708 },
  { event := event71780
    frameStart := 71708 },
  { event := event71781
    frameStart := 71708 },
  { event := event71782
    frameStart := 71708 },
  { event := event71783
    frameStart := 71708 },
  { event := event71784
    frameStart := 71708 },
  { event := event71785
    frameStart := 71708 },
  { event := event71786
    frameStart := 71708 },
  { event := event71787
    frameStart := 71708 },
  { event := event71788
    frameStart := 71708 },
  { event := event71789
    frameStart := 71708 },
  { event := event71790
    frameStart := 71708 },
  { event := event71791
    frameStart := 71708 }
]

def eventLeaf4487 : Array AnnotatedEvent := #[
  { event := event71792
    frameStart := 71708 },
  { event := event71793
    frameStart := 71708 },
  { event := event71794
    frameStart := 71708 },
  { event := event71795
    frameStart := 71708 },
  { event := event71796
    frameStart := 71708 },
  { event := event71797
    frameStart := 71708 },
  { event := event71798
    frameStart := 71708 },
  { event := event71799
    frameStart := 71708 },
  { event := event71800
    frameStart := 71708 },
  { event := event71801
    frameStart := 71708 },
  { event := event71802
    frameStart := 71708 },
  { event := event71803
    frameStart := 71708 },
  { event := event71804
    frameStart := 71708 },
  { event := event71805
    frameStart := 71708 },
  { event := event71806
    frameStart := 71708 },
  { event := event71807
    frameStart := 71708 }
]

def eventLeaf4488 : Array AnnotatedEvent := #[
  { event := event71808
    frameStart := 71708 },
  { event := event71809
    frameStart := 71708 },
  { event := event71810
    frameStart := 71708 },
  { event := event71811
    frameStart := 71708 },
  { event := event71812
    frameStart := 71708 },
  { event := event71813
    frameStart := 71708 },
  { event := event71814
    frameStart := 71708 },
  { event := event71815
    frameStart := 71708 },
  { event := event71816
    frameStart := 71708 },
  { event := event71817
    frameStart := 71708 },
  { event := event71818
    frameStart := 71708 },
  { event := event71819
    frameStart := 71708 },
  { event := event71820
    frameStart := 71708 },
  { event := event71821
    frameStart := 71708 },
  { event := event71822
    frameStart := 71708 },
  { event := event71823
    frameStart := 71708 }
]

def eventLeaf4489 : Array AnnotatedEvent := #[
  { event := event71824
    frameStart := 71708 },
  { event := event71825
    frameStart := 71708 },
  { event := event71826
    frameStart := 0 },
  { event := event71827
    frameStart := 0 },
  { event := event71828
    frameStart := 0 },
  { event := event71829
    frameStart := 0 },
  { event := event71830
    frameStart := 0 },
  { event := event71831
    frameStart := 0 },
  { event := event71832
    frameStart := 0 },
  { event := event71833
    frameStart := 0 },
  { event := event71834
    frameStart := 0 },
  { event := event71835
    frameStart := 0 },
  { event := event71836
    frameStart := 0 },
  { event := event71837
    frameStart := 0 },
  { event := event71838
    frameStart := 0 },
  { event := event71839
    frameStart := 0 }
]

def eventLeaf4490 : Array AnnotatedEvent := #[
  { event := event71840
    frameStart := 0 },
  { event := event71841
    frameStart := 0 },
  { event := event71842
    frameStart := 0 },
  { event := event71843
    frameStart := 0 },
  { event := event71844
    frameStart := 0 },
  { event := event71845
    frameStart := 0 },
  { event := event71846
    frameStart := 0 },
  { event := event71847
    frameStart := 0 },
  { event := event71848
    frameStart := 0 },
  { event := event71849
    frameStart := 0 },
  { event := event71850
    frameStart := 0 },
  { event := event71851
    frameStart := 0 },
  { event := event71852
    frameStart := 0 },
  { event := event71853
    frameStart := 0 },
  { event := event71854
    frameStart := 0 },
  { event := event71855
    frameStart := 0 }
]

def eventLeaf4491 : Array AnnotatedEvent := #[
  { event := event71856
    frameStart := 0 },
  { event := event71857
    frameStart := 0 },
  { event := event71858
    frameStart := 0 },
  { event := event71859
    frameStart := 0 },
  { event := event71860
    frameStart := 0 },
  { event := event71861
    frameStart := 0 },
  { event := event71862
    frameStart := 0 },
  { event := event71863
    frameStart := 71863 },
  { event := event71864
    frameStart := 71863 },
  { event := event71865
    frameStart := 71863 },
  { event := event71866
    frameStart := 71863 },
  { event := event71867
    frameStart := 71863 },
  { event := event71868
    frameStart := 71863 },
  { event := event71869
    frameStart := 71863 },
  { event := event71870
    frameStart := 71863 },
  { event := event71871
    frameStart := 71863 }
]

def eventLeaf4492 : Array AnnotatedEvent := #[
  { event := event71872
    frameStart := 71863 },
  { event := event71873
    frameStart := 71863 },
  { event := event71874
    frameStart := 71863 },
  { event := event71875
    frameStart := 71863 },
  { event := event71876
    frameStart := 71863 },
  { event := event71877
    frameStart := 71863 },
  { event := event71878
    frameStart := 71863 },
  { event := event71879
    frameStart := 71863 },
  { event := event71880
    frameStart := 71863 },
  { event := event71881
    frameStart := 71863 },
  { event := event71882
    frameStart := 71863 },
  { event := event71883
    frameStart := 71863 },
  { event := event71884
    frameStart := 71863 },
  { event := event71885
    frameStart := 71863 },
  { event := event71886
    frameStart := 71863 },
  { event := event71887
    frameStart := 71863 }
]

def eventLeaf4493 : Array AnnotatedEvent := #[
  { event := event71888
    frameStart := 71863 },
  { event := event71889
    frameStart := 71863 },
  { event := event71890
    frameStart := 71863 },
  { event := event71891
    frameStart := 71863 },
  { event := event71892
    frameStart := 71863 },
  { event := event71893
    frameStart := 71863 },
  { event := event71894
    frameStart := 71863 },
  { event := event71895
    frameStart := 71863 },
  { event := event71896
    frameStart := 71863 },
  { event := event71897
    frameStart := 71863 },
  { event := event71898
    frameStart := 71863 },
  { event := event71899
    frameStart := 71863 },
  { event := event71900
    frameStart := 71863 },
  { event := event71901
    frameStart := 71863 },
  { event := event71902
    frameStart := 71863 },
  { event := event71903
    frameStart := 71863 }
]

def eventLeaf4494 : Array AnnotatedEvent := #[
  { event := event71904
    frameStart := 71863 },
  { event := event71905
    frameStart := 71863 },
  { event := event71906
    frameStart := 71863 },
  { event := event71907
    frameStart := 71863 },
  { event := event71908
    frameStart := 71863 },
  { event := event71909
    frameStart := 71863 },
  { event := event71910
    frameStart := 71863 },
  { event := event71911
    frameStart := 71863 },
  { event := event71912
    frameStart := 71863 },
  { event := event71913
    frameStart := 71863 },
  { event := event71914
    frameStart := 71863 },
  { event := event71915
    frameStart := 71863 },
  { event := event71916
    frameStart := 71863 },
  { event := event71917
    frameStart := 71917 },
  { event := event71918
    frameStart := 71917 },
  { event := event71919
    frameStart := 71917 }
]

def eventLeaf4495 : Array AnnotatedEvent := #[
  { event := event71920
    frameStart := 71917 },
  { event := event71921
    frameStart := 71917 },
  { event := event71922
    frameStart := 71917 },
  { event := event71923
    frameStart := 71917 },
  { event := event71924
    frameStart := 71917 },
  { event := event71925
    frameStart := 71917 },
  { event := event71926
    frameStart := 71917 },
  { event := event71927
    frameStart := 71917 },
  { event := event71928
    frameStart := 71917 },
  { event := event71929
    frameStart := 71917 },
  { event := event71930
    frameStart := 71917 },
  { event := event71931
    frameStart := 71917 },
  { event := event71932
    frameStart := 71917 },
  { event := event71933
    frameStart := 71917 },
  { event := event71934
    frameStart := 71917 },
  { event := event71935
    frameStart := 71917 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events280
