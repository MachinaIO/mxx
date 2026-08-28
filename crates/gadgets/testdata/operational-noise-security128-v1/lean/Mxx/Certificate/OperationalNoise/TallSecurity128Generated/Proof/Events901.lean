import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events901

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event230656 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event230657 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event230658 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 230657

def event230659 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 230655

def event230660 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 230658 .coefficient) (.value (.predecessor 1 230659 .coefficient)))

def event230661 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event230662 : Event := .predecessor (⟨.program ⟨257⟩, ⟨4992⟩⟩) 0 ⟨392⟩ 230661

def event230663 : Event := .predecessor (⟨.program ⟨257⟩, ⟨4992⟩⟩) 1 ⟨4990⟩ 230653

def event230664 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨4992⟩⟩) (.sum [.predecessor 0 230662 .coefficient, .predecessor 1 230663 .coefficient])

def event230665 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨4992⟩⟩) (.finite 655345)

def event230666 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5577⟩⟩) 0 ⟨4992⟩ 230665

def event230667 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5577⟩⟩) 1 ⟨5426⟩ 230651

def event230668 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5577⟩⟩) (.identity (.predecessor 1 230667 .coefficient))

def event230669 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5577⟩⟩) (.finite 655360)

def event230670 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15450⟩⟩) 0 ⟨5577⟩ 230669

def event230671 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15450⟩⟩) (.authority (.programFamilyFact))

def exact230672RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15450⟩⟩], []⟩, (1)⟩]

theorem exact230672RawTermsValid :
    exact230672RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event230672 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15450⟩⟩) exact230672RawTerms (.finite 2) 230671 .exactZero (none)

def event230673 : Event := .predecessor (⟨.program ⟨257⟩, ⟨12366⟩⟩) 0 ⟨5577⟩ 230669

def event230674 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨12366⟩⟩) (.authority (.programFamilyFact))

def exact230675RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨12366⟩⟩], []⟩, (1)⟩]

theorem exact230675RawTermsValid :
    exact230675RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event230675 : Event := .resultExact (⟨.program ⟨257⟩, ⟨12366⟩⟩) exact230675RawTerms (.finite 2) 230674 .exactZero (none)

def event230676 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15451⟩⟩) 0 ⟨12366⟩ 230675

def event230677 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15451⟩⟩) 1 ⟨15450⟩ 230672

def event230678 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15451⟩⟩) (.product (.predecessor 0 230676 .coefficient) (.predecessor 1 230677 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event230679 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15451⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨12366⟩⟩, ⟨.program ⟨257⟩, ⟨15450⟩⟩], []⟩) [⟨.result 230675 .coefficient, true, some 1⟩, ⟨.result 230672 .coefficient, true, some 1⟩])

def event230680 : Event := .survivorFold (1) 230679

def exact230681RawTerms : List Term := []

theorem exact230681RawTermsValid :
    exact230681RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event230681 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15451⟩⟩) exact230681RawTerms (.finite 4) 230678 (.finite 4) (some (230679))

def event230682 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15452⟩⟩) 0 ⟨15451⟩ 230681

def event230683 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15452⟩⟩) (.identity (.predecessor 0 230682 .coefficient))

def event230684 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨15452⟩⟩) (.finite 4)

def event230685 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15780⟩⟩) 0 ⟨15452⟩ 230684

def event230686 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15780⟩⟩) (.authority (.programFamilyFact))

def exact230687RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15780⟩⟩], []⟩, (1)⟩]

theorem exact230687RawTermsValid :
    exact230687RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event230687 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15780⟩⟩) exact230687RawTerms (.finite 2) 230686 .exactZero (none)

def event230688 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15781⟩⟩) 0 ⟨15780⟩ 230687

def event230689 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15781⟩⟩) (.identity (.predecessor 0 230688 .coefficient))

def event230690 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨15781⟩⟩) (.finite 2)

def event230691 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16576⟩⟩) 0 ⟨15781⟩ 230690

def event230692 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨16576⟩⟩) (.authority (.relationPreimageSource ⟨57⟩))

def exact230693RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨16576⟩⟩]⟩, (1)⟩]

theorem exact230693RawTermsValid :
    exact230693RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event230693 : Event := .resultExact (⟨.program ⟨257⟩, ⟨16576⟩⟩) exact230693RawTerms (.finite 5647228698) 230692 .exactZero (none)

def event230694 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35⟩⟩) (.authority (.operator))

def exact230695RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩]⟩, (1)⟩]

theorem exact230695RawTermsValid :
    exact230695RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event230695 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35⟩⟩) exact230695RawTerms .large 230694 .exactZero (none)

def event230696 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16577⟩⟩) 0 ⟨35⟩ 230695

def event230697 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16577⟩⟩) 1 ⟨16576⟩ 230693

def event230698 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨16577⟩⟩) (.product (.predecessor 0 230696 .coefficient) (.predecessor 1 230697 .coefficient) (⟨false, false, none, none, none⟩))

def event230699 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨16577⟩⟩, .operator (⟨230695, 0⟩, ⟨230693, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16576⟩⟩]⟩, (1)⟩)

def exact230700RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16576⟩⟩]⟩, (1)⟩]

theorem exact230700RawTermsValid :
    exact230700RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event230700 : Event := .resultExact (⟨.program ⟨257⟩, ⟨16577⟩⟩) exact230700RawTerms .large 230698 .exactZero (none)

def event230701 : Event := .preFoldPolynomial 230700 [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16576⟩⟩]⟩, (1)⟩] .exactZero none

def exact230702RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16576⟩⟩]⟩, (1)⟩]

def event230702 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨16577⟩⟩) 230701 exact230702RawTerms .large 230698 .exactZero (none)

def event230703 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨17737⟩⟩)

def event230704 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event230705 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event230706 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨4990⟩⟩) (.authority (.operator))

def event230707 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨4990⟩⟩) (.finite 5)

def event230708 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event230709 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event230710 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event230711 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event230712 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 230711

def event230713 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 230709

def event230714 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 230712 .coefficient) (.value (.predecessor 1 230713 .coefficient)))

def event230715 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event230716 : Event := .predecessor (⟨.program ⟨257⟩, ⟨4992⟩⟩) 0 ⟨392⟩ 230715

def event230717 : Event := .predecessor (⟨.program ⟨257⟩, ⟨4992⟩⟩) 1 ⟨4990⟩ 230707

def event230718 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨4992⟩⟩) (.sum [.predecessor 0 230716 .coefficient, .predecessor 1 230717 .coefficient])

def event230719 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨4992⟩⟩) (.finite 655345)

def event230720 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5577⟩⟩) 0 ⟨4992⟩ 230719

def event230721 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5577⟩⟩) 1 ⟨5426⟩ 230705

def event230722 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5577⟩⟩) (.identity (.predecessor 1 230721 .coefficient))

def event230723 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5577⟩⟩) (.finite 655360)

def event230724 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15450⟩⟩) 0 ⟨5577⟩ 230723

def event230725 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15450⟩⟩) (.authority (.programFamilyFact))

def exact230726RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15450⟩⟩], []⟩, (1)⟩]

theorem exact230726RawTermsValid :
    exact230726RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event230726 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15450⟩⟩) exact230726RawTerms (.finite 2) 230725 .exactZero (none)

def event230727 : Event := .predecessor (⟨.program ⟨257⟩, ⟨12366⟩⟩) 0 ⟨5577⟩ 230723

def event230728 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨12366⟩⟩) (.authority (.programFamilyFact))

def exact230729RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨12366⟩⟩], []⟩, (1)⟩]

theorem exact230729RawTermsValid :
    exact230729RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event230729 : Event := .resultExact (⟨.program ⟨257⟩, ⟨12366⟩⟩) exact230729RawTerms (.finite 2) 230728 .exactZero (none)

def event230730 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15451⟩⟩) 0 ⟨12366⟩ 230729

def event230731 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15451⟩⟩) 1 ⟨15450⟩ 230726

def event230732 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15451⟩⟩) (.product (.predecessor 0 230730 .coefficient) (.predecessor 1 230731 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event230733 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨15451⟩⟩, .operator (⟨230729, 0⟩, ⟨230726, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨12366⟩⟩, ⟨.program ⟨257⟩, ⟨15450⟩⟩], []⟩, (1)⟩)

def exact230734RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨12366⟩⟩, ⟨.program ⟨257⟩, ⟨15450⟩⟩], []⟩, (1)⟩]

theorem exact230734RawTermsValid :
    exact230734RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event230734 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15451⟩⟩) exact230734RawTerms (.finite 4) 230732 .exactZero (none)

def event230735 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15452⟩⟩) 0 ⟨15451⟩ 230734

def event230736 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15452⟩⟩) (.identity (.predecessor 0 230735 .coefficient))

def event230737 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨15452⟩⟩) (.finite 4)

def event230738 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15780⟩⟩) 0 ⟨15452⟩ 230737

def event230739 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15780⟩⟩) (.authority (.programFamilyFact))

def exact230740RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15780⟩⟩], []⟩, (1)⟩]

theorem exact230740RawTermsValid :
    exact230740RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event230740 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15780⟩⟩) exact230740RawTerms (.finite 2) 230739 .exactZero (none)

def event230741 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15781⟩⟩) 0 ⟨15780⟩ 230740

def event230742 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15781⟩⟩) (.identity (.predecessor 0 230741 .coefficient))

def event230743 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨15781⟩⟩) (.finite 2)

def event230744 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16990⟩⟩) 0 ⟨15781⟩ 230743

def event230745 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨16990⟩⟩) (.authority (.programFamilyFact))

def event230746 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨16990⟩⟩) (.finite 3720)

def event230747 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨7177⟩⟩) .missing

def event230748 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16992⟩⟩) 0 ⟨7177⟩ 230747

def event230749 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16992⟩⟩) 1 ⟨16990⟩ 230746

def event230750 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨16992⟩⟩) (.authority (.operator))

def exact230751RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨16992⟩⟩]⟩, (1)⟩]

theorem exact230751RawTermsValid :
    exact230751RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event230751 : Event := .resultExact (⟨.program ⟨257⟩, ⟨16992⟩⟩) exact230751RawTerms .large 230750 .exactZero (none)

def event230752 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17733⟩⟩) 0 ⟨16992⟩ 230751

def event230753 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17733⟩⟩) (.authority (.operator))

def exact230754RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨17733⟩⟩]⟩, (1)⟩]

theorem exact230754RawTermsValid :
    exact230754RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event230754 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17733⟩⟩) exact230754RawTerms (.finite 8192) 230753 .exactZero (none)

def event230755 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨136⟩⟩) (.authority (.operator))

def event230756 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨136⟩⟩) .exactZero

def event230757 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17202⟩⟩) 0 ⟨15781⟩ 230743

def event230758 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17202⟩⟩) 1 ⟨136⟩ 230756

def event230759 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17202⟩⟩) (.sum [.predecessor 0 230757 .coefficient, .predecessor 1 230758 .coefficient])

def event230760 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨17202⟩⟩) (.finite 2)

def event230761 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17203⟩⟩) 0 ⟨17202⟩ 230760

def event230762 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17203⟩⟩) (.identity (.predecessor 0 230761 .coefficient))

def exact230763RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15780⟩⟩], []⟩, (1)⟩]

theorem exact230763RawTermsValid :
    exact230763RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event230763 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17203⟩⟩) exact230763RawTerms (.finite 2) 230762 .exactZero (none)

def event230764 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6908⟩⟩) (.authority (.factStore))

def exact230765RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact230765RawTermsValid :
    exact230765RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event230765 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6908⟩⟩) exact230765RawTerms .large 230764 .exactZero (none)

def event230766 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17204⟩⟩) 0 ⟨6908⟩ 230765

def event230767 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17204⟩⟩) 1 ⟨17203⟩ 230763

def event230768 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17204⟩⟩) (.product (.predecessor 0 230766 .coefficient) (.predecessor 1 230767 .coefficient) (⟨false, false, none, none, none⟩))

def event230769 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17204⟩⟩, .operator (⟨230765, 0⟩, ⟨230763, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨15780⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact230770RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15780⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact230770RawTermsValid :
    exact230770RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event230770 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17204⟩⟩) exact230770RawTerms .large 230768 .exactZero (none)

def event230771 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7179⟩⟩) 0 ⟨7177⟩ 230747

def event230772 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7179⟩⟩) (.authority (.operator))

def exact230773RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩]⟩, (1)⟩]

theorem exact230773RawTermsValid :
    exact230773RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event230773 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7179⟩⟩) exact230773RawTerms .large 230772 .exactZero (none)

def event230774 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17205⟩⟩) 0 ⟨7179⟩ 230773

def event230775 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17205⟩⟩) 1 ⟨17204⟩ 230770

def event230776 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17205⟩⟩) (.sum [.predecessor 0 230774 .coefficient, .predecessor 1 230775 .coefficient])

def exact230777RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15780⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact230777RawTermsValid :
    exact230777RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event230777 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17205⟩⟩) exact230777RawTerms .large 230776 .exactZero (none)

def event230778 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17734⟩⟩) 0 ⟨17205⟩ 230777

def event230779 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17734⟩⟩) 1 ⟨17733⟩ 230754

def event230780 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17734⟩⟩) (.product (.predecessor 0 230778 .coefficient) (.predecessor 1 230779 .coefficient) (⟨false, false, none, none, none⟩))

def event230781 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17734⟩⟩, .operator (⟨230777, 0⟩, ⟨230754, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17733⟩⟩]⟩, (1)⟩)

def event230782 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17734⟩⟩, .operator (⟨230777, 1⟩, ⟨230754, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨15780⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17733⟩⟩]⟩, (-1)⟩)

def event230783 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨17734⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨15780⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17733⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨17733⟩⟩) ⟨16992⟩ 230751)

def event230784 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17734⟩⟩, .relation 230783 0, ⟨[⟨.program ⟨257⟩, ⟨15780⟩⟩], [⟨.program ⟨257⟩, ⟨16992⟩⟩]⟩, (-1)⟩)

def exact230785RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17733⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15780⟩⟩], [⟨.program ⟨257⟩, ⟨16992⟩⟩]⟩, (-1)⟩]

theorem exact230785RawTermsValid :
    exact230785RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event230785 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17734⟩⟩) exact230785RawTerms .large 230780 .exactZero (none)

def event230786 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16019⟩⟩) 0 ⟨15781⟩ 230743

def event230787 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨16019⟩⟩) (.authority (.programFamilyFact))

def exact230788RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨16019⟩⟩], []⟩, (1)⟩]

theorem exact230788RawTermsValid :
    exact230788RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event230788 : Event := .resultExact (⟨.program ⟨257⟩, ⟨16019⟩⟩) exact230788RawTerms (.finite 43) 230787 .exactZero (none)

def event230789 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16020⟩⟩) 0 ⟨6908⟩ 230765

def event230790 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16020⟩⟩) 1 ⟨16019⟩ 230788

def event230791 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨16020⟩⟩) (.product (.predecessor 0 230789 .coefficient) (.predecessor 1 230790 .coefficient) (⟨false, true, none, none, some 1⟩))

def event230792 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨16020⟩⟩, .operator (⟨230765, 0⟩, ⟨230788, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨16019⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact230793RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨16019⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact230793RawTermsValid :
    exact230793RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event230793 : Event := .resultExact (⟨.program ⟨257⟩, ⟨16020⟩⟩) exact230793RawTerms .large 230791 .exactZero (none)

def event230794 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7198⟩⟩) 0 ⟨7177⟩ 230747

def event230795 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7198⟩⟩) (.authority (.operator))

def exact230796RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩]

theorem exact230796RawTermsValid :
    exact230796RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event230796 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7198⟩⟩) exact230796RawTerms .large 230795 .exactZero (none)

def event230797 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16021⟩⟩) 0 ⟨7198⟩ 230796

def event230798 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16021⟩⟩) 1 ⟨16020⟩ 230793

def event230799 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨16021⟩⟩) (.sum [.predecessor 0 230797 .coefficient, .predecessor 1 230798 .coefficient])

def exact230800RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨16019⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact230800RawTermsValid :
    exact230800RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event230800 : Event := .resultExact (⟨.program ⟨257⟩, ⟨16021⟩⟩) exact230800RawTerms .large 230799 .exactZero (none)

def event230801 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17737⟩⟩) 0 ⟨16021⟩ 230800

def event230802 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17737⟩⟩) 1 ⟨17734⟩ 230785

def event230803 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17737⟩⟩) (.sum [.predecessor 0 230801 .coefficient, .predecessor 1 230802 .coefficient])

def exact230804RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17733⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15780⟩⟩], [⟨.program ⟨257⟩, ⟨16992⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨16019⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact230804RawTermsValid :
    exact230804RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event230804 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17737⟩⟩) exact230804RawTerms .large 230803 .exactZero (none)

def event230805 : Event := .preFoldPolynomial 230804 [⟨⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17733⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15780⟩⟩], [⟨.program ⟨257⟩, ⟨16992⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨16019⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩] .exactZero none

def exact230806RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17733⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15780⟩⟩], [⟨.program ⟨257⟩, ⟨16992⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨16019⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

def event230806 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨17737⟩⟩) 230805 exact230806RawTerms .large 230803 .exactZero (none)

def event230807 : Event := .specializationComputed (⟨.program ⟨257⟩, ⟨15781⟩⟩) ⟨⟨77⟩, ⟨57⟩, ⟨135⟩⟩ ⟨230649, 230807⟩

def event230808 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨16579⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16576⟩⟩]⟩) (1) 0 2 (.universal 230807 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16576⟩⟩]⟩) (none) 230806)

def event230809 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨16579⟩⟩, .relation 230808 1, ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩)

def event230810 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨16579⟩⟩, .relation 230808 0, ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17733⟩⟩]⟩, (-1)⟩)

def event230811 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨16579⟩⟩, .relation 230808 2, ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨15780⟩⟩], [⟨.program ⟨257⟩, ⟨16992⟩⟩]⟩, (1)⟩)

def event230812 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨16579⟩⟩, .relation 230808 3, ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨16019⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact230813RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17733⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨15780⟩⟩], [⟨.program ⟨257⟩, ⟨16992⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨16019⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact230813RawTermsValid :
    exact230813RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event230813 : Event := .resultExact (⟨.program ⟨257⟩, ⟨16579⟩⟩) exact230813RawTerms .large 230645 (.finite 202072841853861888) (some (230647))

def event230814 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17736⟩⟩) 0 ⟨16579⟩ 230813

def event230815 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17736⟩⟩) 1 ⟨17735⟩ 230635

def event230816 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17736⟩⟩) (.sum [.predecessor 0 230814 .coefficient, .predecessor 1 230815 .coefficient])

def event230817 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17736⟩⟩, .operator (⟨230813, 0⟩, ⟨230635, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17733⟩⟩]⟩, (1)⟩)

def event230818 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17736⟩⟩, .operator (⟨230813, 2⟩, ⟨230635, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨15780⟩⟩], [⟨.program ⟨257⟩, ⟨16992⟩⟩]⟩, (-1)⟩)

def event230819 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17736⟩⟩) (.sum [.result 230813 .summary, .result 230635 .summary])

def exact230820RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨16019⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact230820RawTermsValid :
    exact230820RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event230820 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17736⟩⟩) exact230820RawTerms .large 230816 (.finite 32188807212483706889510625476608) (some (230819))

def event230821 : Event := .predecessor (⟨.program ⟨257⟩, ⟨20625⟩⟩) 0 ⟨17736⟩ 230820

def event230822 : Event := .predecessor (⟨.program ⟨257⟩, ⟨20625⟩⟩) 1 ⟨20624⟩ 230338

def event230823 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨20625⟩⟩) (.sum [.predecessor 0 230821 .coefficient, .predecessor 1 230822 .coefficient])

def event230824 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨20625⟩⟩) (.sum [.result 230820 .summary, .result 230338 .summary])

def exact230825RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨16019⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨18847⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact230825RawTermsValid :
    exact230825RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event230825 : Event := .resultExact (⟨.program ⟨257⟩, ⟨20625⟩⟩) exact230825RawTerms .large 230823 (.finite 64377712650190257467641695830016) (some (230824))

def event230826 : Event := .predecessor (⟨.program ⟨257⟩, ⟨23845⟩⟩) 0 ⟨20625⟩ 230825

def event230827 : Event := .predecessor (⟨.program ⟨257⟩, ⟨23845⟩⟩) 1 ⟨23844⟩ 229856

def event230828 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨23845⟩⟩) (.sum [.predecessor 0 230826 .coefficient, .predecessor 1 230827 .coefficient])

def event230829 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨23845⟩⟩) (.sum [.result 230825 .summary, .result 229856 .summary])

def exact230830RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨16019⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨18847⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨22067⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact230830RawTermsValid :
    exact230830RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event230830 : Event := .resultExact (⟨.program ⟨257⟩, ⟨23845⟩⟩) exact230830RawTerms .large 230828 (.finite 96566716313119651734393211060224) (some (230829))

def event230831 : Event := .predecessor (⟨.program ⟨257⟩, ⟨33865⟩⟩) 0 ⟨23845⟩ 230830

def event230832 : Event := .predecessor (⟨.program ⟨257⟩, ⟨33865⟩⟩) 1 ⟨33864⟩ 229374

def event230833 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨33865⟩⟩) (.sum [.predecessor 0 230831 .coefficient, .predecessor 1 230832 .coefficient])

def event230834 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨33865⟩⟩) (.sum [.result 230830 .summary, .result 229374 .summary])

def exact230835RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨16019⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨18847⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨22067⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨32087⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact230835RawTermsValid :
    exact230835RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event230835 : Event := .resultExact (⟨.program ⟨257⟩, ⟨33865⟩⟩) exact230835RawTerms .large 230833 (.finite 128755916426494733378385616044032) (some (230834))

def event230836 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52925⟩⟩) 0 ⟨33865⟩ 230835

def event230837 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52925⟩⟩) 1 ⟨52924⟩ 228892

def event230838 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨52925⟩⟩) (.sum [.predecessor 0 230836 .coefficient, .predecessor 1 230837 .coefficient])

def event230839 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨52925⟩⟩) (.sum [.result 230835 .summary, .result 228892 .summary])

def exact230840RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨16019⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨18847⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨22067⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨32087⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨51142⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact230840RawTermsValid :
    exact230840RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event230840 : Event := .resultExact (⟨.program ⟨257⟩, ⟨52925⟩⟩) exact230840RawTerms .large 230838 (.finite 160945509440761189776859800535040) (some (230839))

def event230841 : Event := .predecessor (⟨.program ⟨257⟩, ⟨55905⟩⟩) 0 ⟨52925⟩ 230840

def event230842 : Event := .predecessor (⟨.program ⟨257⟩, ⟨55905⟩⟩) 1 ⟨55904⟩ 228410

def event230843 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨55905⟩⟩) (.sum [.predecessor 0 230841 .coefficient, .predecessor 1 230842 .coefficient])

def event230844 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨55905⟩⟩) (.sum [.result 230840 .summary, .result 228410 .summary])

def exact230845RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨16019⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨18847⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨22067⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨32087⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨51142⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨54122⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact230845RawTermsValid :
    exact230845RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event230845 : Event := .resultExact (⟨.program ⟨257⟩, ⟨55905⟩⟩) exact230845RawTerms .large 230843 (.finite 193135298905473333552574874779648) (some (230844))

def event230846 : Event := .predecessor (⟨.program ⟨257⟩, ⟨58885⟩⟩) 0 ⟨55905⟩ 230845

def event230847 : Event := .predecessor (⟨.program ⟨257⟩, ⟨58885⟩⟩) 1 ⟨58884⟩ 227928

def event230848 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨58885⟩⟩) (.sum [.predecessor 0 230846 .coefficient, .predecessor 1 230847 .coefficient])

def event230849 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨58885⟩⟩) (.sum [.result 230845 .summary, .result 227928 .summary])

def exact230850RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨16019⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨18847⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨22067⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨32087⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨51142⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨54122⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨57102⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact230850RawTermsValid :
    exact230850RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event230850 : Event := .resultExact (⟨.program ⟨257⟩, ⟨58885⟩⟩) exact230850RawTerms .large 230848 (.finite 225325481271076852082771728531456) (some (230849))

def event230851 : Event := .predecessor (⟨.program ⟨257⟩, ⟨61865⟩⟩) 0 ⟨58885⟩ 230850

def event230852 : Event := .predecessor (⟨.program ⟨257⟩, ⟨61865⟩⟩) 1 ⟨61864⟩ 227446

def event230853 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨61865⟩⟩) (.sum [.predecessor 0 230851 .coefficient, .predecessor 1 230852 .coefficient])

def event230854 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨61865⟩⟩) (.sum [.result 230850 .summary, .result 227446 .summary])

def exact230855RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨16019⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨18847⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨22067⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨32087⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨51142⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨54122⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨57102⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨60082⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact230855RawTermsValid :
    exact230855RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event230855 : Event := .resultExact (⟨.program ⟨257⟩, ⟨61865⟩⟩) exact230855RawTerms .large 230853 (.finite 257515860087126057990209472036864) (some (230854))

def event230856 : Event := .predecessor (⟨.program ⟨257⟩, ⟨64845⟩⟩) 0 ⟨61865⟩ 230855

def event230857 : Event := .predecessor (⟨.program ⟨257⟩, ⟨64845⟩⟩) 1 ⟨64844⟩ 226964

def event230858 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨64845⟩⟩) (.sum [.predecessor 0 230856 .coefficient, .predecessor 1 230857 .coefficient])

def event230859 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨64845⟩⟩) (.sum [.result 230855 .summary, .result 226964 .summary])

def exact230860RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨16019⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨18847⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨22067⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨32087⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨51142⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨54122⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨57102⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨60082⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨63062⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact230860RawTermsValid :
    exact230860RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event230860 : Event := .resultExact (⟨.program ⟨257⟩, ⟨64845⟩⟩) exact230860RawTerms .large 230858 (.finite 289706631804066638652128995049472) (some (230859))

def event230861 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70102⟩⟩) 0 ⟨64845⟩ 230860

def event230862 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70102⟩⟩) 1 ⟨70101⟩ 226482

def event230863 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70102⟩⟩) (.sum [.predecessor 0 230861 .coefficient, .predecessor 1 230862 .coefficient])

def event230864 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70102⟩⟩) (.sum [.result 230860 .summary, .result 226482 .summary])

def exact230865RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨16019⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨18847⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨22067⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨32087⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨51142⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨54122⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨57102⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨60082⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨63062⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨66531⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact230865RawTermsValid :
    exact230865RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event230865 : Event := .resultExact (⟨.program ⟨257⟩, ⟨70102⟩⟩) exact230865RawTerms .large 230863 (.finite 321897992872344281445771187322880) (some (230864))

def event230866 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70103⟩⟩) 0 ⟨70102⟩ 230865

def event230867 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70103⟩⟩) 1 ⟨28267⟩ 226000

def event230868 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70103⟩⟩) (.sum [.predecessor 0 230866 .coefficient, .predecessor 1 230867 .coefficient])

def event230869 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70103⟩⟩) (.sum [.result 230865 .summary, .result 226000 .summary])

def exact230870RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨16019⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨18847⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨22067⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨26606⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨32087⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨51142⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨54122⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨57102⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨60082⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨63062⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨66531⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact230870RawTermsValid :
    exact230870RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event230870 : Event := .resultExact (⟨.program ⟨257⟩, ⟨70103⟩⟩) exact230870RawTerms .large 230868 (.finite 354089550391067611616654269349888) (some (230869))

def event230871 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70104⟩⟩) 0 ⟨70103⟩ 230870

def event230872 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70104⟩⟩) 1 ⟨30947⟩ 225518

def event230873 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70104⟩⟩) (.sum [.predecessor 0 230871 .coefficient, .predecessor 1 230872 .coefficient])

def event230874 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70104⟩⟩) (.sum [.result 230870 .summary, .result 225518 .summary])

def exact230875RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨16019⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨18847⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨22067⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨26606⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨29286⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨32087⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨51142⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨54122⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨57102⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨60082⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨63062⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨66531⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact230875RawTermsValid :
    exact230875RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event230875 : Event := .resultExact (⟨.program ⟨257⟩, ⟨70104⟩⟩) exact230875RawTerms .large 230873 (.finite 386281697261128003919260020637696) (some (230874))

def event230876 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70105⟩⟩) 0 ⟨70104⟩ 230875

def event230877 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70105⟩⟩) 1 ⟨36607⟩ 225036

def event230878 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70105⟩⟩) (.sum [.predecessor 0 230876 .coefficient, .predecessor 1 230877 .coefficient])

def event230879 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70105⟩⟩) (.sum [.result 230875 .summary, .result 225036 .summary])

def exact230880RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7222⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨16019⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨18847⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨22067⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨26606⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨29286⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨32087⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨34950⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨51142⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨54122⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨57102⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨60082⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨63062⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨66531⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact230880RawTermsValid :
    exact230880RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event230880 : Event := .resultExact (⟨.program ⟨257⟩, ⟨70105⟩⟩) exact230880RawTerms .large 230878 (.finite 418474237032079770976347551432704) (some (230879))

def event230881 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70106⟩⟩) 0 ⟨70105⟩ 230880

def event230882 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70106⟩⟩) 1 ⟨39287⟩ 224554

def event230883 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70106⟩⟩) (.sum [.predecessor 0 230881 .coefficient, .predecessor 1 230882 .coefficient])

def event230884 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70106⟩⟩) (.sum [.result 230880 .summary, .result 224554 .summary])

def exact230885RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7222⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7224⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨16019⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨18847⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨22067⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨26606⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨29286⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨32087⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨34950⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨37630⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨51142⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨54122⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨57102⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨60082⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨63062⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨66531⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact230885RawTermsValid :
    exact230885RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event230885 : Event := .resultExact (⟨.program ⟨257⟩, ⟨70106⟩⟩) exact230885RawTerms .large 230883 (.finite 450666973253477225410675971981312) (some (230884))

def event230886 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70107⟩⟩) 0 ⟨70106⟩ 230885

def event230887 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70107⟩⟩) 1 ⟨41967⟩ 224072

def event230888 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70107⟩⟩) (.sum [.predecessor 0 230886 .coefficient, .predecessor 1 230887 .coefficient])

def event230889 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70107⟩⟩) (.sum [.result 230885 .summary, .result 224072 .summary])

def exact230890RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7222⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7224⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7226⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨16019⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨18847⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨22067⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨26606⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨29286⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨32087⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨34950⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨37630⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨40306⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨51142⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨54122⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨57102⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨60082⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨63062⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨66531⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact230890RawTermsValid :
    exact230890RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event230890 : Event := .resultExact (⟨.program ⟨257⟩, ⟨70107⟩⟩) exact230890RawTerms .large 230888 (.finite 482860102375766054599486172037120) (some (230889))

def event230891 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70108⟩⟩) 0 ⟨70107⟩ 230890

def event230892 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70108⟩⟩) 1 ⟨44647⟩ 223590

def event230893 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70108⟩⟩) (.sum [.predecessor 0 230891 .coefficient, .predecessor 1 230892 .coefficient])

def event230894 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70108⟩⟩) (.sum [.result 230890 .summary, .result 223590 .summary])

def exact230895RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7222⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7224⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7226⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7228⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨16019⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨18847⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨22067⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨26606⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨29286⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨32087⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨34950⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨37630⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨40306⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨42986⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨51142⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨54122⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨57102⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨60082⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨63062⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨66531⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact230895RawTermsValid :
    exact230895RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event230895 : Event := .resultExact (⟨.program ⟨257⟩, ⟨70108⟩⟩) exact230895RawTerms .large 230893 (.finite 515053820849391945920019041353728) (some (230894))

def event230896 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70109⟩⟩) 0 ⟨70108⟩ 230895

def event230897 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70109⟩⟩) 1 ⟨47327⟩ 223108

def event230898 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70109⟩⟩) (.sum [.predecessor 0 230896 .coefficient, .predecessor 1 230897 .coefficient])

def event230899 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70109⟩⟩) (.sum [.result 230895 .summary, .result 223108 .summary])

def exact230900RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7222⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7224⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7226⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7228⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7230⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨16019⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨18847⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨22067⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨26606⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨29286⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨32087⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨34950⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨37630⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨40306⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨42986⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨45670⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨51142⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨54122⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨57102⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨60082⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨63062⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨66531⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact230900RawTermsValid :
    exact230900RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event230900 : Event := .resultExact (⟨.program ⟨257⟩, ⟨70109⟩⟩) exact230900RawTerms .large 230898 (.finite 547248128674354899372274579931136) (some (230899))

def event230901 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70110⟩⟩) 0 ⟨70109⟩ 230900

def event230902 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70110⟩⟩) 1 ⟨50007⟩ 222626

def event230903 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70110⟩⟩) (.sum [.predecessor 0 230901 .coefficient, .predecessor 1 230902 .coefficient])

def event230904 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70110⟩⟩) (.sum [.result 230900 .summary, .result 222626 .summary])

def exact230905RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7222⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7224⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7226⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7228⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7230⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7232⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨16019⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨18847⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨22067⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨26606⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨29286⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨32087⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨34950⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨37630⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨40306⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨42986⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨45670⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨48350⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨51142⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨54122⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨57102⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨60082⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨63062⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨66531⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact230905RawTermsValid :
    exact230905RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event230905 : Event := .resultExact (⟨.program ⟨257⟩, ⟨70110⟩⟩) exact230905RawTerms .large 230903 (.finite 579442632949763540201771008262144) (some (230904))

def event230906 : Event := .predecessor (⟨.program ⟨257⟩, ⟨71206⟩⟩) 0 ⟨70110⟩ 230905

def event230907 : Event := .predecessor (⟨.program ⟨257⟩, ⟨71206⟩⟩) 1 ⟨71204⟩ 222128

def event230908 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨71206⟩⟩) (.product (.predecessor 0 230906 .coefficient) (.predecessor 1 230907 .coefficient) (⟨false, false, none, none, none⟩))

def event230909 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨71206⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨71204⟩⟩]⟩) [⟨.result 222128 .coefficient, false, none⟩])

def event230910 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨71206⟩⟩) (.product (.result 230905 .summary) (.transfer 230909) (⟨false, false, none, none, none⟩))

def event230911 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71206⟩⟩, .operator (⟨230905, 17⟩, ⟨222128, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7232⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩]⟩, (1)⟩)

def eventLeaf14416 : Array AnnotatedEvent := #[
  { event := event230656
    frameStart := 230649 },
  { event := event230657
    frameStart := 230649 },
  { event := event230658
    frameStart := 230649 },
  { event := event230659
    frameStart := 230649 },
  { event := event230660
    frameStart := 230649 },
  { event := event230661
    frameStart := 230649 },
  { event := event230662
    frameStart := 230649 },
  { event := event230663
    frameStart := 230649 },
  { event := event230664
    frameStart := 230649 },
  { event := event230665
    frameStart := 230649 },
  { event := event230666
    frameStart := 230649 },
  { event := event230667
    frameStart := 230649 },
  { event := event230668
    frameStart := 230649 },
  { event := event230669
    frameStart := 230649 },
  { event := event230670
    frameStart := 230649 },
  { event := event230671
    frameStart := 230649 }
]

def eventLeaf14417 : Array AnnotatedEvent := #[
  { event := event230672
    frameStart := 230649 },
  { event := event230673
    frameStart := 230649 },
  { event := event230674
    frameStart := 230649 },
  { event := event230675
    frameStart := 230649 },
  { event := event230676
    frameStart := 230649 },
  { event := event230677
    frameStart := 230649 },
  { event := event230678
    frameStart := 230649 },
  { event := event230679
    frameStart := 230649 },
  { event := event230680
    frameStart := 230649 },
  { event := event230681
    frameStart := 230649 },
  { event := event230682
    frameStart := 230649 },
  { event := event230683
    frameStart := 230649 },
  { event := event230684
    frameStart := 230649 },
  { event := event230685
    frameStart := 230649 },
  { event := event230686
    frameStart := 230649 },
  { event := event230687
    frameStart := 230649 }
]

def eventLeaf14418 : Array AnnotatedEvent := #[
  { event := event230688
    frameStart := 230649 },
  { event := event230689
    frameStart := 230649 },
  { event := event230690
    frameStart := 230649 },
  { event := event230691
    frameStart := 230649 },
  { event := event230692
    frameStart := 230649 },
  { event := event230693
    frameStart := 230649 },
  { event := event230694
    frameStart := 230649 },
  { event := event230695
    frameStart := 230649 },
  { event := event230696
    frameStart := 230649 },
  { event := event230697
    frameStart := 230649 },
  { event := event230698
    frameStart := 230649 },
  { event := event230699
    frameStart := 230649 },
  { event := event230700
    frameStart := 230649 },
  { event := event230701
    frameStart := 230649 },
  { event := event230702
    frameStart := 230649 },
  { event := event230703
    frameStart := 230703 }
]

def eventLeaf14419 : Array AnnotatedEvent := #[
  { event := event230704
    frameStart := 230703 },
  { event := event230705
    frameStart := 230703 },
  { event := event230706
    frameStart := 230703 },
  { event := event230707
    frameStart := 230703 },
  { event := event230708
    frameStart := 230703 },
  { event := event230709
    frameStart := 230703 },
  { event := event230710
    frameStart := 230703 },
  { event := event230711
    frameStart := 230703 },
  { event := event230712
    frameStart := 230703 },
  { event := event230713
    frameStart := 230703 },
  { event := event230714
    frameStart := 230703 },
  { event := event230715
    frameStart := 230703 },
  { event := event230716
    frameStart := 230703 },
  { event := event230717
    frameStart := 230703 },
  { event := event230718
    frameStart := 230703 },
  { event := event230719
    frameStart := 230703 }
]

def eventLeaf14420 : Array AnnotatedEvent := #[
  { event := event230720
    frameStart := 230703 },
  { event := event230721
    frameStart := 230703 },
  { event := event230722
    frameStart := 230703 },
  { event := event230723
    frameStart := 230703 },
  { event := event230724
    frameStart := 230703 },
  { event := event230725
    frameStart := 230703 },
  { event := event230726
    frameStart := 230703 },
  { event := event230727
    frameStart := 230703 },
  { event := event230728
    frameStart := 230703 },
  { event := event230729
    frameStart := 230703 },
  { event := event230730
    frameStart := 230703 },
  { event := event230731
    frameStart := 230703 },
  { event := event230732
    frameStart := 230703 },
  { event := event230733
    frameStart := 230703 },
  { event := event230734
    frameStart := 230703 },
  { event := event230735
    frameStart := 230703 }
]

def eventLeaf14421 : Array AnnotatedEvent := #[
  { event := event230736
    frameStart := 230703 },
  { event := event230737
    frameStart := 230703 },
  { event := event230738
    frameStart := 230703 },
  { event := event230739
    frameStart := 230703 },
  { event := event230740
    frameStart := 230703 },
  { event := event230741
    frameStart := 230703 },
  { event := event230742
    frameStart := 230703 },
  { event := event230743
    frameStart := 230703 },
  { event := event230744
    frameStart := 230703 },
  { event := event230745
    frameStart := 230703 },
  { event := event230746
    frameStart := 230703 },
  { event := event230747
    frameStart := 230703 },
  { event := event230748
    frameStart := 230703 },
  { event := event230749
    frameStart := 230703 },
  { event := event230750
    frameStart := 230703 },
  { event := event230751
    frameStart := 230703 }
]

def eventLeaf14422 : Array AnnotatedEvent := #[
  { event := event230752
    frameStart := 230703 },
  { event := event230753
    frameStart := 230703 },
  { event := event230754
    frameStart := 230703 },
  { event := event230755
    frameStart := 230703 },
  { event := event230756
    frameStart := 230703 },
  { event := event230757
    frameStart := 230703 },
  { event := event230758
    frameStart := 230703 },
  { event := event230759
    frameStart := 230703 },
  { event := event230760
    frameStart := 230703 },
  { event := event230761
    frameStart := 230703 },
  { event := event230762
    frameStart := 230703 },
  { event := event230763
    frameStart := 230703 },
  { event := event230764
    frameStart := 230703 },
  { event := event230765
    frameStart := 230703 },
  { event := event230766
    frameStart := 230703 },
  { event := event230767
    frameStart := 230703 }
]

def eventLeaf14423 : Array AnnotatedEvent := #[
  { event := event230768
    frameStart := 230703 },
  { event := event230769
    frameStart := 230703 },
  { event := event230770
    frameStart := 230703 },
  { event := event230771
    frameStart := 230703 },
  { event := event230772
    frameStart := 230703 },
  { event := event230773
    frameStart := 230703 },
  { event := event230774
    frameStart := 230703 },
  { event := event230775
    frameStart := 230703 },
  { event := event230776
    frameStart := 230703 },
  { event := event230777
    frameStart := 230703 },
  { event := event230778
    frameStart := 230703 },
  { event := event230779
    frameStart := 230703 },
  { event := event230780
    frameStart := 230703 },
  { event := event230781
    frameStart := 230703 },
  { event := event230782
    frameStart := 230703 },
  { event := event230783
    frameStart := 230703 }
]

def eventLeaf14424 : Array AnnotatedEvent := #[
  { event := event230784
    frameStart := 230703 },
  { event := event230785
    frameStart := 230703 },
  { event := event230786
    frameStart := 230703 },
  { event := event230787
    frameStart := 230703 },
  { event := event230788
    frameStart := 230703 },
  { event := event230789
    frameStart := 230703 },
  { event := event230790
    frameStart := 230703 },
  { event := event230791
    frameStart := 230703 },
  { event := event230792
    frameStart := 230703 },
  { event := event230793
    frameStart := 230703 },
  { event := event230794
    frameStart := 230703 },
  { event := event230795
    frameStart := 230703 },
  { event := event230796
    frameStart := 230703 },
  { event := event230797
    frameStart := 230703 },
  { event := event230798
    frameStart := 230703 },
  { event := event230799
    frameStart := 230703 }
]

def eventLeaf14425 : Array AnnotatedEvent := #[
  { event := event230800
    frameStart := 230703 },
  { event := event230801
    frameStart := 230703 },
  { event := event230802
    frameStart := 230703 },
  { event := event230803
    frameStart := 230703 },
  { event := event230804
    frameStart := 230703 },
  { event := event230805
    frameStart := 230703 },
  { event := event230806
    frameStart := 230703 },
  { event := event230807
    frameStart := 0 },
  { event := event230808
    frameStart := 0 },
  { event := event230809
    frameStart := 0 },
  { event := event230810
    frameStart := 0 },
  { event := event230811
    frameStart := 0 },
  { event := event230812
    frameStart := 0 },
  { event := event230813
    frameStart := 0 },
  { event := event230814
    frameStart := 0 },
  { event := event230815
    frameStart := 0 }
]

def eventLeaf14426 : Array AnnotatedEvent := #[
  { event := event230816
    frameStart := 0 },
  { event := event230817
    frameStart := 0 },
  { event := event230818
    frameStart := 0 },
  { event := event230819
    frameStart := 0 },
  { event := event230820
    frameStart := 0 },
  { event := event230821
    frameStart := 0 },
  { event := event230822
    frameStart := 0 },
  { event := event230823
    frameStart := 0 },
  { event := event230824
    frameStart := 0 },
  { event := event230825
    frameStart := 0 },
  { event := event230826
    frameStart := 0 },
  { event := event230827
    frameStart := 0 },
  { event := event230828
    frameStart := 0 },
  { event := event230829
    frameStart := 0 },
  { event := event230830
    frameStart := 0 },
  { event := event230831
    frameStart := 0 }
]

def eventLeaf14427 : Array AnnotatedEvent := #[
  { event := event230832
    frameStart := 0 },
  { event := event230833
    frameStart := 0 },
  { event := event230834
    frameStart := 0 },
  { event := event230835
    frameStart := 0 },
  { event := event230836
    frameStart := 0 },
  { event := event230837
    frameStart := 0 },
  { event := event230838
    frameStart := 0 },
  { event := event230839
    frameStart := 0 },
  { event := event230840
    frameStart := 0 },
  { event := event230841
    frameStart := 0 },
  { event := event230842
    frameStart := 0 },
  { event := event230843
    frameStart := 0 },
  { event := event230844
    frameStart := 0 },
  { event := event230845
    frameStart := 0 },
  { event := event230846
    frameStart := 0 },
  { event := event230847
    frameStart := 0 }
]

def eventLeaf14428 : Array AnnotatedEvent := #[
  { event := event230848
    frameStart := 0 },
  { event := event230849
    frameStart := 0 },
  { event := event230850
    frameStart := 0 },
  { event := event230851
    frameStart := 0 },
  { event := event230852
    frameStart := 0 },
  { event := event230853
    frameStart := 0 },
  { event := event230854
    frameStart := 0 },
  { event := event230855
    frameStart := 0 },
  { event := event230856
    frameStart := 0 },
  { event := event230857
    frameStart := 0 },
  { event := event230858
    frameStart := 0 },
  { event := event230859
    frameStart := 0 },
  { event := event230860
    frameStart := 0 },
  { event := event230861
    frameStart := 0 },
  { event := event230862
    frameStart := 0 },
  { event := event230863
    frameStart := 0 }
]

def eventLeaf14429 : Array AnnotatedEvent := #[
  { event := event230864
    frameStart := 0 },
  { event := event230865
    frameStart := 0 },
  { event := event230866
    frameStart := 0 },
  { event := event230867
    frameStart := 0 },
  { event := event230868
    frameStart := 0 },
  { event := event230869
    frameStart := 0 },
  { event := event230870
    frameStart := 0 },
  { event := event230871
    frameStart := 0 },
  { event := event230872
    frameStart := 0 },
  { event := event230873
    frameStart := 0 },
  { event := event230874
    frameStart := 0 },
  { event := event230875
    frameStart := 0 },
  { event := event230876
    frameStart := 0 },
  { event := event230877
    frameStart := 0 },
  { event := event230878
    frameStart := 0 },
  { event := event230879
    frameStart := 0 }
]

def eventLeaf14430 : Array AnnotatedEvent := #[
  { event := event230880
    frameStart := 0 },
  { event := event230881
    frameStart := 0 },
  { event := event230882
    frameStart := 0 },
  { event := event230883
    frameStart := 0 },
  { event := event230884
    frameStart := 0 },
  { event := event230885
    frameStart := 0 },
  { event := event230886
    frameStart := 0 },
  { event := event230887
    frameStart := 0 },
  { event := event230888
    frameStart := 0 },
  { event := event230889
    frameStart := 0 },
  { event := event230890
    frameStart := 0 },
  { event := event230891
    frameStart := 0 },
  { event := event230892
    frameStart := 0 },
  { event := event230893
    frameStart := 0 },
  { event := event230894
    frameStart := 0 },
  { event := event230895
    frameStart := 0 }
]

def eventLeaf14431 : Array AnnotatedEvent := #[
  { event := event230896
    frameStart := 0 },
  { event := event230897
    frameStart := 0 },
  { event := event230898
    frameStart := 0 },
  { event := event230899
    frameStart := 0 },
  { event := event230900
    frameStart := 0 },
  { event := event230901
    frameStart := 0 },
  { event := event230902
    frameStart := 0 },
  { event := event230903
    frameStart := 0 },
  { event := event230904
    frameStart := 0 },
  { event := event230905
    frameStart := 0 },
  { event := event230906
    frameStart := 0 },
  { event := event230907
    frameStart := 0 },
  { event := event230908
    frameStart := 0 },
  { event := event230909
    frameStart := 0 },
  { event := event230910
    frameStart := 0 },
  { event := event230911
    frameStart := 0 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events901
