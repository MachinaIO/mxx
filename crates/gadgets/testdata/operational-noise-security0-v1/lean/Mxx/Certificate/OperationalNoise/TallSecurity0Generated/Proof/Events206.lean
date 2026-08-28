import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events206

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event52736 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6⟩⟩) (.authority (.operator))

def exact52737RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩]⟩, (1)⟩]

theorem exact52737RawTermsValid :
    exact52737RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event52737 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6⟩⟩) exact52737RawTerms .large 52736 .exactZero (none)

def event52738 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19965⟩⟩) 0 ⟨6⟩ 52737

def event52739 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19965⟩⟩) 1 ⟨19964⟩ 52735

def event52740 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19965⟩⟩) (.product (.predecessor 0 52738 .coefficient) (.predecessor 1 52739 .coefficient) (⟨false, false, none, none, none⟩))

def event52741 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19965⟩⟩, .operator (⟨52737, 0⟩, ⟨52735, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19964⟩⟩]⟩, (1)⟩)

def exact52742RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19964⟩⟩]⟩, (1)⟩]

theorem exact52742RawTermsValid :
    exact52742RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event52742 : Event := .resultExact (⟨.program ⟨214⟩, ⟨19965⟩⟩) exact52742RawTerms .large 52740 .exactZero (none)

def event52743 : Event := .preFoldPolynomial 52742 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19964⟩⟩]⟩, (1)⟩] .exactZero none

def exact52744RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19964⟩⟩]⟩, (1)⟩]

def event52744 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨19965⟩⟩) 52743 exact52744RawTerms .large 52740 .exactZero (none)

def event52745 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨25459⟩⟩)

def event52746 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event52747 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event52748 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨2875⟩⟩) (.authority (.operator))

def event52749 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨2875⟩⟩) (.finite 3)

def event52750 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event52751 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event52752 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event52753 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event52754 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 52753

def event52755 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 52751

def event52756 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 52754 .coefficient) (.value (.predecessor 1 52755 .coefficient)))

def event52757 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event52758 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5510⟩⟩) 0 ⟨5503⟩ 52757

def event52759 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5510⟩⟩) 1 ⟨2875⟩ 52749

def event52760 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5510⟩⟩) (.sum [.predecessor 0 52758 .coefficient, .predecessor 1 52759 .coefficient])

def event52761 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5510⟩⟩) (.finite 220)

def event52762 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5542⟩⟩) 0 ⟨5510⟩ 52761

def event52763 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5542⟩⟩) 1 ⟨961⟩ 52747

def event52764 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5542⟩⟩) (.identity (.predecessor 1 52763 .coefficient))

def event52765 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5542⟩⟩) (.finite 224)

def event52766 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12574⟩⟩) 0 ⟨5542⟩ 52765

def event52767 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12574⟩⟩) (.authority (.programFamilyFact))

def exact52768RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨12574⟩⟩], []⟩, (1)⟩]

theorem exact52768RawTermsValid :
    exact52768RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event52768 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12574⟩⟩) exact52768RawTerms (.finite 42) 52767 .exactZero (none)

def event52769 : Event := .predecessor (⟨.program ⟨214⟩, ⟨9930⟩⟩) 0 ⟨5542⟩ 52765

def event52770 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨9930⟩⟩) (.authority (.programFamilyFact))

def exact52771RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨9930⟩⟩], []⟩, (1)⟩]

theorem exact52771RawTermsValid :
    exact52771RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event52771 : Event := .resultExact (⟨.program ⟨214⟩, ⟨9930⟩⟩) exact52771RawTerms (.finite 42) 52770 .exactZero (none)

def event52772 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12575⟩⟩) 0 ⟨9930⟩ 52771

def event52773 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12575⟩⟩) 1 ⟨12574⟩ 52768

def event52774 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12575⟩⟩) (.product (.predecessor 0 52772 .coefficient) (.predecessor 1 52773 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event52775 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨12575⟩⟩, .operator (⟨52771, 0⟩, ⟨52768, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨9930⟩⟩, ⟨.program ⟨214⟩, ⟨12574⟩⟩], []⟩, (1)⟩)

def exact52776RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨9930⟩⟩, ⟨.program ⟨214⟩, ⟨12574⟩⟩], []⟩, (1)⟩]

theorem exact52776RawTermsValid :
    exact52776RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event52776 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12575⟩⟩) exact52776RawTerms (.finite 1764) 52774 .exactZero (none)

def event52777 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12576⟩⟩) 0 ⟨12575⟩ 52776

def event52778 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12576⟩⟩) (.identity (.predecessor 0 52777 .coefficient))

def event52779 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨12576⟩⟩) (.finite 1764)

def event52780 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23249⟩⟩) 0 ⟨12576⟩ 52779

def event52781 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23249⟩⟩) (.authority (.programFamilyFact))

def event52782 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨23249⟩⟩) (.finite 3720)

def event52783 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨6689⟩⟩) .missing

def event52784 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23250⟩⟩) 0 ⟨6689⟩ 52783

def event52785 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23250⟩⟩) 1 ⟨23249⟩ 52782

def event52786 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23250⟩⟩) (.authority (.operator))

def exact52787RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨23250⟩⟩]⟩, (1)⟩]

theorem exact52787RawTermsValid :
    exact52787RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event52787 : Event := .resultExact (⟨.program ⟨214⟩, ⟨23250⟩⟩) exact52787RawTerms .large 52786 .exactZero (none)

def event52788 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25455⟩⟩) 0 ⟨23250⟩ 52787

def event52789 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25455⟩⟩) (.authority (.operator))

def exact52790RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨25455⟩⟩]⟩, (1)⟩]

theorem exact52790RawTermsValid :
    exact52790RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event52790 : Event := .resultExact (⟨.program ⟨214⟩, ⟨25455⟩⟩) exact52790RawTerms (.finite 8192) 52789 .exactZero (none)

def event52791 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨110⟩⟩) (.authority (.operator))

def event52792 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨110⟩⟩) .exactZero

def event52793 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12666⟩⟩) 0 ⟨12576⟩ 52779

def event52794 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12666⟩⟩) 1 ⟨110⟩ 52792

def event52795 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12666⟩⟩) (.sum [.predecessor 0 52793 .coefficient, .predecessor 1 52794 .coefficient])

def event52796 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨12666⟩⟩) (.finite 1764)

def event52797 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12667⟩⟩) 0 ⟨12666⟩ 52796

def event52798 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12667⟩⟩) (.identity (.predecessor 0 52797 .coefficient))

def exact52799RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨9930⟩⟩, ⟨.program ⟨214⟩, ⟨12574⟩⟩], []⟩, (1)⟩]

theorem exact52799RawTermsValid :
    exact52799RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event52799 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12667⟩⟩) exact52799RawTerms (.finite 1764) 52798 .exactZero (none)

def event52800 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6544⟩⟩) (.authority (.factStore))

def exact52801RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact52801RawTermsValid :
    exact52801RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event52801 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6544⟩⟩) exact52801RawTerms .large 52800 .exactZero (none)

def event52802 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12668⟩⟩) 0 ⟨6544⟩ 52801

def event52803 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12668⟩⟩) 1 ⟨12667⟩ 52799

def event52804 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12668⟩⟩) (.product (.predecessor 0 52802 .coefficient) (.predecessor 1 52803 .coefficient) (⟨false, false, none, none, none⟩))

def event52805 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨12668⟩⟩, .operator (⟨52801, 0⟩, ⟨52799, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨9930⟩⟩, ⟨.program ⟨214⟩, ⟨12574⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact52806RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨9930⟩⟩, ⟨.program ⟨214⟩, ⟨12574⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact52806RawTermsValid :
    exact52806RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event52806 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12668⟩⟩) exact52806RawTerms .large 52804 .exactZero (none)

def event52807 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨2348⟩⟩) (.authority (.operator))

def event52808 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨2348⟩⟩) (.finite 1)

def event52809 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6757⟩⟩) 0 ⟨6689⟩ 52783

def event52810 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6757⟩⟩) (.authority (.operator))

def exact52811RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6757⟩⟩]⟩, (1)⟩]

theorem exact52811RawTermsValid :
    exact52811RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event52811 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6757⟩⟩) exact52811RawTerms .large 52810 .exactZero (none)

def event52812 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6786⟩⟩) 0 ⟨6757⟩ 52811

def event52813 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6786⟩⟩) (.identity (.predecessor 0 52812 .coefficient))

def exact52814RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6786⟩⟩]⟩, (1)⟩]

theorem exact52814RawTermsValid :
    exact52814RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event52814 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6786⟩⟩) exact52814RawTerms .large 52813 .exactZero (none)

def event52815 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7870⟩⟩) 0 ⟨6786⟩ 52814

def event52816 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7870⟩⟩) (.authority (.operator))

def exact52817RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨7870⟩⟩]⟩, (1)⟩]

theorem exact52817RawTermsValid :
    exact52817RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event52817 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7870⟩⟩) exact52817RawTerms (.finite 8192) 52816 .exactZero (none)

def event52818 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7871⟩⟩) 0 ⟨7870⟩ 52817

def event52819 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7871⟩⟩) 1 ⟨2348⟩ 52808

def event52820 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7871⟩⟩) (.scale (.predecessor 0 52818 .coefficient) (.value (.predecessor 1 52819 .coefficient)))

def exact52821RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨7870⟩⟩]⟩, (1)⟩]

theorem exact52821RawTermsValid :
    exact52821RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event52821 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7871⟩⟩) exact52821RawTerms (.finite 8192) 52820 .exactZero (none)

def event52822 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6766⟩⟩) 0 ⟨6757⟩ 52811

def event52823 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6766⟩⟩) (.identity (.predecessor 0 52822 .coefficient))

def exact52824RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6766⟩⟩]⟩, (1)⟩]

theorem exact52824RawTermsValid :
    exact52824RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event52824 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6766⟩⟩) exact52824RawTerms .large 52823 .exactZero (none)

def event52825 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7872⟩⟩) 0 ⟨6766⟩ 52824

def event52826 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7872⟩⟩) 1 ⟨7871⟩ 52821

def event52827 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7872⟩⟩) (.product (.predecessor 0 52825 .coefficient) (.predecessor 1 52826 .coefficient) (⟨false, false, none, none, none⟩))

def event52828 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7872⟩⟩, .operator (⟨52824, 0⟩, ⟨52821, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6766⟩⟩, ⟨.program ⟨214⟩, ⟨7870⟩⟩]⟩, (1)⟩)

def exact52829RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6766⟩⟩, ⟨.program ⟨214⟩, ⟨7870⟩⟩]⟩, (1)⟩]

theorem exact52829RawTermsValid :
    exact52829RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event52829 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7872⟩⟩) exact52829RawTerms .large 52827 .exactZero (none)

def event52830 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12669⟩⟩) 0 ⟨7872⟩ 52829

def event52831 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12669⟩⟩) 1 ⟨12668⟩ 52806

def event52832 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12669⟩⟩) (.sum [.predecessor 0 52830 .coefficient, .predecessor 1 52831 .coefficient])

def exact52833RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6766⟩⟩, ⟨.program ⟨214⟩, ⟨7870⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨9930⟩⟩, ⟨.program ⟨214⟩, ⟨12574⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact52833RawTermsValid :
    exact52833RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event52833 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12669⟩⟩) exact52833RawTerms .large 52832 .exactZero (none)

def event52834 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25458⟩⟩) 0 ⟨12669⟩ 52833

def event52835 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25458⟩⟩) 1 ⟨25455⟩ 52790

def event52836 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25458⟩⟩) (.product (.predecessor 0 52834 .coefficient) (.predecessor 1 52835 .coefficient) (⟨false, false, none, none, none⟩))

def event52837 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25458⟩⟩, .operator (⟨52833, 0⟩, ⟨52790, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6766⟩⟩, ⟨.program ⟨214⟩, ⟨7870⟩⟩, ⟨.program ⟨214⟩, ⟨25455⟩⟩]⟩, (1)⟩)

def event52838 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25458⟩⟩, .operator (⟨52833, 1⟩, ⟨52790, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨9930⟩⟩, ⟨.program ⟨214⟩, ⟨12574⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25455⟩⟩]⟩, (-1)⟩)

def event52839 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨25458⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨9930⟩⟩, ⟨.program ⟨214⟩, ⟨12574⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25455⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨25455⟩⟩) ⟨23250⟩ 52787)

def event52840 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25458⟩⟩, .relation 52839 0, ⟨[⟨.program ⟨214⟩, ⟨9930⟩⟩, ⟨.program ⟨214⟩, ⟨12574⟩⟩], [⟨.program ⟨214⟩, ⟨23250⟩⟩]⟩, (-1)⟩)

def exact52841RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6766⟩⟩, ⟨.program ⟨214⟩, ⟨7870⟩⟩, ⟨.program ⟨214⟩, ⟨25455⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨9930⟩⟩, ⟨.program ⟨214⟩, ⟨12574⟩⟩], [⟨.program ⟨214⟩, ⟨23250⟩⟩]⟩, (-1)⟩]

theorem exact52841RawTermsValid :
    exact52841RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event52841 : Event := .resultExact (⟨.program ⟨214⟩, ⟨25458⟩⟩) exact52841RawTerms .large 52836 .exactZero (none)

def event52842 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16553⟩⟩) 0 ⟨12576⟩ 52779

def event52843 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16553⟩⟩) (.authority (.programFamilyFact))

def exact52844RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16553⟩⟩], []⟩, (1)⟩]

theorem exact52844RawTermsValid :
    exact52844RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event52844 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16553⟩⟩) exact52844RawTerms (.finite 42) 52843 .exactZero (none)

def event52845 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16555⟩⟩) 0 ⟨6544⟩ 52801

def event52846 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16555⟩⟩) 1 ⟨16553⟩ 52844

def event52847 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16555⟩⟩) (.product (.predecessor 0 52845 .coefficient) (.predecessor 1 52846 .coefficient) (⟨false, true, none, none, some 1⟩))

def event52848 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨16555⟩⟩, .operator (⟨52801, 0⟩, ⟨52844, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨16553⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact52849RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16553⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact52849RawTermsValid :
    exact52849RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event52849 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16555⟩⟩) exact52849RawTerms .large 52847 .exactZero (none)

def event52850 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6703⟩⟩) 0 ⟨6689⟩ 52783

def event52851 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6703⟩⟩) (.authority (.operator))

def exact52852RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6703⟩⟩]⟩, (1)⟩]

theorem exact52852RawTermsValid :
    exact52852RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event52852 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6703⟩⟩) exact52852RawTerms .large 52851 .exactZero (none)

def event52853 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16556⟩⟩) 0 ⟨6703⟩ 52852

def event52854 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16556⟩⟩) 1 ⟨16555⟩ 52849

def event52855 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16556⟩⟩) (.sum [.predecessor 0 52853 .coefficient, .predecessor 1 52854 .coefficient])

def exact52856RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6703⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16553⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact52856RawTermsValid :
    exact52856RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event52856 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16556⟩⟩) exact52856RawTerms .large 52855 .exactZero (none)

def event52857 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25459⟩⟩) 0 ⟨16556⟩ 52856

def event52858 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25459⟩⟩) 1 ⟨25458⟩ 52841

def event52859 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25459⟩⟩) (.sum [.predecessor 0 52857 .coefficient, .predecessor 1 52858 .coefficient])

def exact52860RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6703⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6766⟩⟩, ⟨.program ⟨214⟩, ⟨7870⟩⟩, ⟨.program ⟨214⟩, ⟨25455⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨9930⟩⟩, ⟨.program ⟨214⟩, ⟨12574⟩⟩], [⟨.program ⟨214⟩, ⟨23250⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16553⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact52860RawTermsValid :
    exact52860RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event52860 : Event := .resultExact (⟨.program ⟨214⟩, ⟨25459⟩⟩) exact52860RawTerms .large 52859 .exactZero (none)

def event52861 : Event := .preFoldPolynomial 52860 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6703⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6766⟩⟩, ⟨.program ⟨214⟩, ⟨7870⟩⟩, ⟨.program ⟨214⟩, ⟨25455⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨9930⟩⟩, ⟨.program ⟨214⟩, ⟨12574⟩⟩], [⟨.program ⟨214⟩, ⟨23250⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16553⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩] .exactZero none

def exact52862RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6703⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6766⟩⟩, ⟨.program ⟨214⟩, ⟨7870⟩⟩, ⟨.program ⟨214⟩, ⟨25455⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨9930⟩⟩, ⟨.program ⟨214⟩, ⟨12574⟩⟩], [⟨.program ⟨214⟩, ⟨23250⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16553⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

def event52862 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨25459⟩⟩) 52861 exact52862RawTerms .large 52859 .exactZero (none)

def event52863 : Event := .specializationComputed (⟨.program ⟨214⟩, ⟨12576⟩⟩) ⟨⟨116⟩, ⟨21⟩, ⟨109⟩⟩ ⟨52697, 52863⟩

def event52864 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨19967⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19964⟩⟩]⟩) (1) 0 2 (.universal 52863 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19964⟩⟩]⟩) (none) 52862)

def event52865 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19967⟩⟩, .relation 52864 0, ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6703⟩⟩]⟩, (1)⟩)

def event52866 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19967⟩⟩, .relation 52864 1, ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6766⟩⟩, ⟨.program ⟨214⟩, ⟨7870⟩⟩, ⟨.program ⟨214⟩, ⟨25455⟩⟩]⟩, (-1)⟩)

def event52867 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19967⟩⟩, .relation 52864 2, ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨9930⟩⟩, ⟨.program ⟨214⟩, ⟨12574⟩⟩], [⟨.program ⟨214⟩, ⟨23250⟩⟩]⟩, (1)⟩)

def event52868 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19967⟩⟩, .relation 52864 3, ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨16553⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def exact52869RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6703⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6766⟩⟩, ⟨.program ⟨214⟩, ⟨7870⟩⟩, ⟨.program ⟨214⟩, ⟨25455⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨9930⟩⟩, ⟨.program ⟨214⟩, ⟨12574⟩⟩], [⟨.program ⟨214⟩, ⟨23250⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨16553⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact52869RawTermsValid :
    exact52869RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event52869 : Event := .resultExact (⟨.program ⟨214⟩, ⟨19967⟩⟩) exact52869RawTerms .large 52693 (.finite 1811303510016) (some (52695))

def event52870 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25457⟩⟩) 0 ⟨19967⟩ 52869

def event52871 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25457⟩⟩) 1 ⟨25456⟩ 52683

def event52872 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25457⟩⟩) (.sum [.predecessor 0 52870 .coefficient, .predecessor 1 52871 .coefficient])

def event52873 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25457⟩⟩, .operator (⟨52869, 2⟩, ⟨52683, 1⟩), ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨9930⟩⟩, ⟨.program ⟨214⟩, ⟨12574⟩⟩], [⟨.program ⟨214⟩, ⟨23250⟩⟩]⟩, (-1)⟩)

def event52874 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25457⟩⟩, .operator (⟨52869, 1⟩, ⟨52683, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6766⟩⟩, ⟨.program ⟨214⟩, ⟨7870⟩⟩, ⟨.program ⟨214⟩, ⟨25455⟩⟩]⟩, (1)⟩)

def event52875 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25457⟩⟩) (.sum [.result 52869 .summary, .result 52683 .summary])

def exact52876RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6703⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨16553⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact52876RawTermsValid :
    exact52876RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event52876 : Event := .resultExact (⟨.program ⟨214⟩, ⟨25457⟩⟩) exact52876RawTerms .large 52872 (.finite 352134001995776) (some (52875))

def event52877 : Event := .predecessor (⟨.program ⟨214⟩, ⟨29183⟩⟩) 0 ⟨25457⟩ 52876

def event52878 : Event := .predecessor (⟨.program ⟨214⟩, ⟨29183⟩⟩) 1 ⟨29181⟩ 52599

def event52879 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨29183⟩⟩) (.product (.predecessor 0 52877 .coefficient) (.predecessor 1 52878 .coefficient) (⟨false, false, none, none, none⟩))

def event52880 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨29183⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨29181⟩⟩]⟩) [⟨.result 52599 .coefficient, false, none⟩])

def event52881 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨29183⟩⟩) (.product (.result 52876 .summary) (.transfer 52880) (⟨false, false, none, none, none⟩))

def event52882 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨29183⟩⟩, .operator (⟨52876, 0⟩, ⟨52599, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6703⟩⟩, ⟨.program ⟨214⟩, ⟨29181⟩⟩]⟩, (1)⟩)

def event52883 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨29183⟩⟩, .operator (⟨52876, 1⟩, ⟨52599, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨16553⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29181⟩⟩]⟩, (-1)⟩)

def event52884 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨29183⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨16553⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29181⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨29181⟩⟩) ⟨24543⟩ 52596)

def event52885 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨29183⟩⟩, .relation 52884 0, ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨16553⟩⟩], [⟨.program ⟨214⟩, ⟨24543⟩⟩]⟩, (-1)⟩)

def exact52886RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6703⟩⟩, ⟨.program ⟨214⟩, ⟨29181⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨16553⟩⟩], [⟨.program ⟨214⟩, ⟨24543⟩⟩]⟩, (-1)⟩]

theorem exact52886RawTermsValid :
    exact52886RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event52886 : Event := .resultExact (⟨.program ⟨214⟩, ⟨29183⟩⟩) exact52886RawTerms .large 52879 (.finite 1292337421468529852416) (some (52881))

def event52887 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22268⟩⟩) 0 ⟨16554⟩ 2447

def event52888 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨22268⟩⟩) (.authority (.relationPreimageSource ⟨57⟩))

def exact52889RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨22268⟩⟩]⟩, (1)⟩]

theorem exact52889RawTermsValid :
    exact52889RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event52889 : Event := .resultExact (⟨.program ⟨214⟩, ⟨22268⟩⟩) exact52889RawTerms (.finite 136065468) 52888 .exactZero (none)

def event52890 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22270⟩⟩) 0 ⟨22268⟩ 52889

def event52891 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22270⟩⟩) 1 ⟨2348⟩ 4

def event52892 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨22270⟩⟩) (.scale (.predecessor 0 52890 .coefficient) (.value (.predecessor 1 52891 .coefficient)))

def exact52893RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨22268⟩⟩]⟩, (1)⟩]

theorem exact52893RawTermsValid :
    exact52893RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event52893 : Event := .resultExact (⟨.program ⟨214⟩, ⟨22270⟩⟩) exact52893RawTerms (.finite 136065468) 52892 .exactZero (none)

def event52894 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22271⟩⟩) 0 ⟨5547⟩ 50762

def event52895 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22271⟩⟩) 1 ⟨22270⟩ 52893

def event52896 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨22271⟩⟩) (.product (.predecessor 0 52894 .coefficient) (.predecessor 1 52895 .coefficient) (⟨false, false, none, none, none⟩))

def event52897 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨22271⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨22268⟩⟩]⟩) [⟨.result 52889 .coefficient, false, none⟩])

def event52898 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨22271⟩⟩) (.product (.result 50762 .summary) (.transfer 52897) (⟨false, false, none, none, none⟩))

def event52899 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨22271⟩⟩, .operator (⟨50762, 0⟩, ⟨52893, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22268⟩⟩]⟩, (1)⟩)

def event52900 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨22269⟩⟩)

def event52901 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event52902 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event52903 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨2875⟩⟩) (.authority (.operator))

def event52904 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨2875⟩⟩) (.finite 3)

def event52905 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event52906 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event52907 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event52908 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event52909 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 52908

def event52910 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 52906

def event52911 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 52909 .coefficient) (.value (.predecessor 1 52910 .coefficient)))

def event52912 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event52913 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5510⟩⟩) 0 ⟨5503⟩ 52912

def event52914 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5510⟩⟩) 1 ⟨2875⟩ 52904

def event52915 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5510⟩⟩) (.sum [.predecessor 0 52913 .coefficient, .predecessor 1 52914 .coefficient])

def event52916 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5510⟩⟩) (.finite 220)

def event52917 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5542⟩⟩) 0 ⟨5510⟩ 52916

def event52918 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5542⟩⟩) 1 ⟨961⟩ 52902

def event52919 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5542⟩⟩) (.identity (.predecessor 1 52918 .coefficient))

def event52920 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5542⟩⟩) (.finite 224)

def event52921 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12574⟩⟩) 0 ⟨5542⟩ 52920

def event52922 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12574⟩⟩) (.authority (.programFamilyFact))

def exact52923RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨12574⟩⟩], []⟩, (1)⟩]

theorem exact52923RawTermsValid :
    exact52923RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event52923 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12574⟩⟩) exact52923RawTerms (.finite 42) 52922 .exactZero (none)

def event52924 : Event := .predecessor (⟨.program ⟨214⟩, ⟨9930⟩⟩) 0 ⟨5542⟩ 52920

def event52925 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨9930⟩⟩) (.authority (.programFamilyFact))

def exact52926RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨9930⟩⟩], []⟩, (1)⟩]

theorem exact52926RawTermsValid :
    exact52926RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event52926 : Event := .resultExact (⟨.program ⟨214⟩, ⟨9930⟩⟩) exact52926RawTerms (.finite 42) 52925 .exactZero (none)

def event52927 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12575⟩⟩) 0 ⟨9930⟩ 52926

def event52928 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12575⟩⟩) 1 ⟨12574⟩ 52923

def event52929 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12575⟩⟩) (.product (.predecessor 0 52927 .coefficient) (.predecessor 1 52928 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event52930 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12575⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨9930⟩⟩, ⟨.program ⟨214⟩, ⟨12574⟩⟩], []⟩) [⟨.result 52926 .coefficient, true, some 1⟩, ⟨.result 52923 .coefficient, true, some 1⟩])

def event52931 : Event := .survivorFold (1) 52930

def exact52932RawTerms : List Term := []

theorem exact52932RawTermsValid :
    exact52932RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event52932 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12575⟩⟩) exact52932RawTerms (.finite 1764) 52929 (.finite 1764) (some (52930))

def event52933 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12576⟩⟩) 0 ⟨12575⟩ 52932

def event52934 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12576⟩⟩) (.identity (.predecessor 0 52933 .coefficient))

def event52935 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨12576⟩⟩) (.finite 1764)

def event52936 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16553⟩⟩) 0 ⟨12576⟩ 52935

def event52937 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16553⟩⟩) (.authority (.programFamilyFact))

def exact52938RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16553⟩⟩], []⟩, (1)⟩]

theorem exact52938RawTermsValid :
    exact52938RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event52938 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16553⟩⟩) exact52938RawTerms (.finite 42) 52937 .exactZero (none)

def event52939 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16554⟩⟩) 0 ⟨16553⟩ 52938

def event52940 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16554⟩⟩) (.identity (.predecessor 0 52939 .coefficient))

def event52941 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨16554⟩⟩) (.finite 42)

def event52942 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22268⟩⟩) 0 ⟨16554⟩ 52941

def event52943 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨22268⟩⟩) (.authority (.relationPreimageSource ⟨57⟩))

def exact52944RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨22268⟩⟩]⟩, (1)⟩]

theorem exact52944RawTermsValid :
    exact52944RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event52944 : Event := .resultExact (⟨.program ⟨214⟩, ⟨22268⟩⟩) exact52944RawTerms (.finite 136065468) 52943 .exactZero (none)

def event52945 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6⟩⟩) (.authority (.operator))

def exact52946RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩]⟩, (1)⟩]

theorem exact52946RawTermsValid :
    exact52946RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event52946 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6⟩⟩) exact52946RawTerms .large 52945 .exactZero (none)

def event52947 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22269⟩⟩) 0 ⟨6⟩ 52946

def event52948 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22269⟩⟩) 1 ⟨22268⟩ 52944

def event52949 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨22269⟩⟩) (.product (.predecessor 0 52947 .coefficient) (.predecessor 1 52948 .coefficient) (⟨false, false, none, none, none⟩))

def event52950 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨22269⟩⟩, .operator (⟨52946, 0⟩, ⟨52944, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22268⟩⟩]⟩, (1)⟩)

def exact52951RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22268⟩⟩]⟩, (1)⟩]

theorem exact52951RawTermsValid :
    exact52951RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event52951 : Event := .resultExact (⟨.program ⟨214⟩, ⟨22269⟩⟩) exact52951RawTerms .large 52949 .exactZero (none)

def event52952 : Event := .preFoldPolynomial 52951 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22268⟩⟩]⟩, (1)⟩] .exactZero none

def exact52953RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22268⟩⟩]⟩, (1)⟩]

def event52953 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨22269⟩⟩) 52952 exact52953RawTerms .large 52949 .exactZero (none)

def event52954 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨29186⟩⟩)

def event52955 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event52956 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event52957 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨2875⟩⟩) (.authority (.operator))

def event52958 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨2875⟩⟩) (.finite 3)

def event52959 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event52960 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event52961 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event52962 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event52963 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 52962

def event52964 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 52960

def event52965 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 52963 .coefficient) (.value (.predecessor 1 52964 .coefficient)))

def event52966 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event52967 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5510⟩⟩) 0 ⟨5503⟩ 52966

def event52968 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5510⟩⟩) 1 ⟨2875⟩ 52958

def event52969 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5510⟩⟩) (.sum [.predecessor 0 52967 .coefficient, .predecessor 1 52968 .coefficient])

def event52970 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5510⟩⟩) (.finite 220)

def event52971 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5542⟩⟩) 0 ⟨5510⟩ 52970

def event52972 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5542⟩⟩) 1 ⟨961⟩ 52956

def event52973 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5542⟩⟩) (.identity (.predecessor 1 52972 .coefficient))

def event52974 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5542⟩⟩) (.finite 224)

def event52975 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12574⟩⟩) 0 ⟨5542⟩ 52974

def event52976 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12574⟩⟩) (.authority (.programFamilyFact))

def exact52977RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨12574⟩⟩], []⟩, (1)⟩]

theorem exact52977RawTermsValid :
    exact52977RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event52977 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12574⟩⟩) exact52977RawTerms (.finite 42) 52976 .exactZero (none)

def event52978 : Event := .predecessor (⟨.program ⟨214⟩, ⟨9930⟩⟩) 0 ⟨5542⟩ 52974

def event52979 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨9930⟩⟩) (.authority (.programFamilyFact))

def exact52980RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨9930⟩⟩], []⟩, (1)⟩]

theorem exact52980RawTermsValid :
    exact52980RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event52980 : Event := .resultExact (⟨.program ⟨214⟩, ⟨9930⟩⟩) exact52980RawTerms (.finite 42) 52979 .exactZero (none)

def event52981 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12575⟩⟩) 0 ⟨9930⟩ 52980

def event52982 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12575⟩⟩) 1 ⟨12574⟩ 52977

def event52983 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12575⟩⟩) (.product (.predecessor 0 52981 .coefficient) (.predecessor 1 52982 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event52984 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨12575⟩⟩, .operator (⟨52980, 0⟩, ⟨52977, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨9930⟩⟩, ⟨.program ⟨214⟩, ⟨12574⟩⟩], []⟩, (1)⟩)

def exact52985RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨9930⟩⟩, ⟨.program ⟨214⟩, ⟨12574⟩⟩], []⟩, (1)⟩]

theorem exact52985RawTermsValid :
    exact52985RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event52985 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12575⟩⟩) exact52985RawTerms (.finite 1764) 52983 .exactZero (none)

def event52986 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12576⟩⟩) 0 ⟨12575⟩ 52985

def event52987 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12576⟩⟩) (.identity (.predecessor 0 52986 .coefficient))

def event52988 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨12576⟩⟩) (.finite 1764)

def event52989 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16553⟩⟩) 0 ⟨12576⟩ 52988

def event52990 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16553⟩⟩) (.authority (.programFamilyFact))

def exact52991RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16553⟩⟩], []⟩, (1)⟩]

theorem exact52991RawTermsValid :
    exact52991RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event52991 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16553⟩⟩) exact52991RawTerms (.finite 42) 52990 .exactZero (none)

def eventLeaf3296 : Array AnnotatedEvent := #[
  { event := event52736
    frameStart := 52697 },
  { event := event52737
    frameStart := 52697 },
  { event := event52738
    frameStart := 52697 },
  { event := event52739
    frameStart := 52697 },
  { event := event52740
    frameStart := 52697 },
  { event := event52741
    frameStart := 52697 },
  { event := event52742
    frameStart := 52697 },
  { event := event52743
    frameStart := 52697 },
  { event := event52744
    frameStart := 52697 },
  { event := event52745
    frameStart := 52745 },
  { event := event52746
    frameStart := 52745 },
  { event := event52747
    frameStart := 52745 },
  { event := event52748
    frameStart := 52745 },
  { event := event52749
    frameStart := 52745 },
  { event := event52750
    frameStart := 52745 },
  { event := event52751
    frameStart := 52745 }
]

def eventLeaf3297 : Array AnnotatedEvent := #[
  { event := event52752
    frameStart := 52745 },
  { event := event52753
    frameStart := 52745 },
  { event := event52754
    frameStart := 52745 },
  { event := event52755
    frameStart := 52745 },
  { event := event52756
    frameStart := 52745 },
  { event := event52757
    frameStart := 52745 },
  { event := event52758
    frameStart := 52745 },
  { event := event52759
    frameStart := 52745 },
  { event := event52760
    frameStart := 52745 },
  { event := event52761
    frameStart := 52745 },
  { event := event52762
    frameStart := 52745 },
  { event := event52763
    frameStart := 52745 },
  { event := event52764
    frameStart := 52745 },
  { event := event52765
    frameStart := 52745 },
  { event := event52766
    frameStart := 52745 },
  { event := event52767
    frameStart := 52745 }
]

def eventLeaf3298 : Array AnnotatedEvent := #[
  { event := event52768
    frameStart := 52745 },
  { event := event52769
    frameStart := 52745 },
  { event := event52770
    frameStart := 52745 },
  { event := event52771
    frameStart := 52745 },
  { event := event52772
    frameStart := 52745 },
  { event := event52773
    frameStart := 52745 },
  { event := event52774
    frameStart := 52745 },
  { event := event52775
    frameStart := 52745 },
  { event := event52776
    frameStart := 52745 },
  { event := event52777
    frameStart := 52745 },
  { event := event52778
    frameStart := 52745 },
  { event := event52779
    frameStart := 52745 },
  { event := event52780
    frameStart := 52745 },
  { event := event52781
    frameStart := 52745 },
  { event := event52782
    frameStart := 52745 },
  { event := event52783
    frameStart := 52745 }
]

def eventLeaf3299 : Array AnnotatedEvent := #[
  { event := event52784
    frameStart := 52745 },
  { event := event52785
    frameStart := 52745 },
  { event := event52786
    frameStart := 52745 },
  { event := event52787
    frameStart := 52745 },
  { event := event52788
    frameStart := 52745 },
  { event := event52789
    frameStart := 52745 },
  { event := event52790
    frameStart := 52745 },
  { event := event52791
    frameStart := 52745 },
  { event := event52792
    frameStart := 52745 },
  { event := event52793
    frameStart := 52745 },
  { event := event52794
    frameStart := 52745 },
  { event := event52795
    frameStart := 52745 },
  { event := event52796
    frameStart := 52745 },
  { event := event52797
    frameStart := 52745 },
  { event := event52798
    frameStart := 52745 },
  { event := event52799
    frameStart := 52745 }
]

def eventLeaf3300 : Array AnnotatedEvent := #[
  { event := event52800
    frameStart := 52745 },
  { event := event52801
    frameStart := 52745 },
  { event := event52802
    frameStart := 52745 },
  { event := event52803
    frameStart := 52745 },
  { event := event52804
    frameStart := 52745 },
  { event := event52805
    frameStart := 52745 },
  { event := event52806
    frameStart := 52745 },
  { event := event52807
    frameStart := 52745 },
  { event := event52808
    frameStart := 52745 },
  { event := event52809
    frameStart := 52745 },
  { event := event52810
    frameStart := 52745 },
  { event := event52811
    frameStart := 52745 },
  { event := event52812
    frameStart := 52745 },
  { event := event52813
    frameStart := 52745 },
  { event := event52814
    frameStart := 52745 },
  { event := event52815
    frameStart := 52745 }
]

def eventLeaf3301 : Array AnnotatedEvent := #[
  { event := event52816
    frameStart := 52745 },
  { event := event52817
    frameStart := 52745 },
  { event := event52818
    frameStart := 52745 },
  { event := event52819
    frameStart := 52745 },
  { event := event52820
    frameStart := 52745 },
  { event := event52821
    frameStart := 52745 },
  { event := event52822
    frameStart := 52745 },
  { event := event52823
    frameStart := 52745 },
  { event := event52824
    frameStart := 52745 },
  { event := event52825
    frameStart := 52745 },
  { event := event52826
    frameStart := 52745 },
  { event := event52827
    frameStart := 52745 },
  { event := event52828
    frameStart := 52745 },
  { event := event52829
    frameStart := 52745 },
  { event := event52830
    frameStart := 52745 },
  { event := event52831
    frameStart := 52745 }
]

def eventLeaf3302 : Array AnnotatedEvent := #[
  { event := event52832
    frameStart := 52745 },
  { event := event52833
    frameStart := 52745 },
  { event := event52834
    frameStart := 52745 },
  { event := event52835
    frameStart := 52745 },
  { event := event52836
    frameStart := 52745 },
  { event := event52837
    frameStart := 52745 },
  { event := event52838
    frameStart := 52745 },
  { event := event52839
    frameStart := 52745 },
  { event := event52840
    frameStart := 52745 },
  { event := event52841
    frameStart := 52745 },
  { event := event52842
    frameStart := 52745 },
  { event := event52843
    frameStart := 52745 },
  { event := event52844
    frameStart := 52745 },
  { event := event52845
    frameStart := 52745 },
  { event := event52846
    frameStart := 52745 },
  { event := event52847
    frameStart := 52745 }
]

def eventLeaf3303 : Array AnnotatedEvent := #[
  { event := event52848
    frameStart := 52745 },
  { event := event52849
    frameStart := 52745 },
  { event := event52850
    frameStart := 52745 },
  { event := event52851
    frameStart := 52745 },
  { event := event52852
    frameStart := 52745 },
  { event := event52853
    frameStart := 52745 },
  { event := event52854
    frameStart := 52745 },
  { event := event52855
    frameStart := 52745 },
  { event := event52856
    frameStart := 52745 },
  { event := event52857
    frameStart := 52745 },
  { event := event52858
    frameStart := 52745 },
  { event := event52859
    frameStart := 52745 },
  { event := event52860
    frameStart := 52745 },
  { event := event52861
    frameStart := 52745 },
  { event := event52862
    frameStart := 52745 },
  { event := event52863
    frameStart := 0 }
]

def eventLeaf3304 : Array AnnotatedEvent := #[
  { event := event52864
    frameStart := 0 },
  { event := event52865
    frameStart := 0 },
  { event := event52866
    frameStart := 0 },
  { event := event52867
    frameStart := 0 },
  { event := event52868
    frameStart := 0 },
  { event := event52869
    frameStart := 0 },
  { event := event52870
    frameStart := 0 },
  { event := event52871
    frameStart := 0 },
  { event := event52872
    frameStart := 0 },
  { event := event52873
    frameStart := 0 },
  { event := event52874
    frameStart := 0 },
  { event := event52875
    frameStart := 0 },
  { event := event52876
    frameStart := 0 },
  { event := event52877
    frameStart := 0 },
  { event := event52878
    frameStart := 0 },
  { event := event52879
    frameStart := 0 }
]

def eventLeaf3305 : Array AnnotatedEvent := #[
  { event := event52880
    frameStart := 0 },
  { event := event52881
    frameStart := 0 },
  { event := event52882
    frameStart := 0 },
  { event := event52883
    frameStart := 0 },
  { event := event52884
    frameStart := 0 },
  { event := event52885
    frameStart := 0 },
  { event := event52886
    frameStart := 0 },
  { event := event52887
    frameStart := 0 },
  { event := event52888
    frameStart := 0 },
  { event := event52889
    frameStart := 0 },
  { event := event52890
    frameStart := 0 },
  { event := event52891
    frameStart := 0 },
  { event := event52892
    frameStart := 0 },
  { event := event52893
    frameStart := 0 },
  { event := event52894
    frameStart := 0 },
  { event := event52895
    frameStart := 0 }
]

def eventLeaf3306 : Array AnnotatedEvent := #[
  { event := event52896
    frameStart := 0 },
  { event := event52897
    frameStart := 0 },
  { event := event52898
    frameStart := 0 },
  { event := event52899
    frameStart := 0 },
  { event := event52900
    frameStart := 52900 },
  { event := event52901
    frameStart := 52900 },
  { event := event52902
    frameStart := 52900 },
  { event := event52903
    frameStart := 52900 },
  { event := event52904
    frameStart := 52900 },
  { event := event52905
    frameStart := 52900 },
  { event := event52906
    frameStart := 52900 },
  { event := event52907
    frameStart := 52900 },
  { event := event52908
    frameStart := 52900 },
  { event := event52909
    frameStart := 52900 },
  { event := event52910
    frameStart := 52900 },
  { event := event52911
    frameStart := 52900 }
]

def eventLeaf3307 : Array AnnotatedEvent := #[
  { event := event52912
    frameStart := 52900 },
  { event := event52913
    frameStart := 52900 },
  { event := event52914
    frameStart := 52900 },
  { event := event52915
    frameStart := 52900 },
  { event := event52916
    frameStart := 52900 },
  { event := event52917
    frameStart := 52900 },
  { event := event52918
    frameStart := 52900 },
  { event := event52919
    frameStart := 52900 },
  { event := event52920
    frameStart := 52900 },
  { event := event52921
    frameStart := 52900 },
  { event := event52922
    frameStart := 52900 },
  { event := event52923
    frameStart := 52900 },
  { event := event52924
    frameStart := 52900 },
  { event := event52925
    frameStart := 52900 },
  { event := event52926
    frameStart := 52900 },
  { event := event52927
    frameStart := 52900 }
]

def eventLeaf3308 : Array AnnotatedEvent := #[
  { event := event52928
    frameStart := 52900 },
  { event := event52929
    frameStart := 52900 },
  { event := event52930
    frameStart := 52900 },
  { event := event52931
    frameStart := 52900 },
  { event := event52932
    frameStart := 52900 },
  { event := event52933
    frameStart := 52900 },
  { event := event52934
    frameStart := 52900 },
  { event := event52935
    frameStart := 52900 },
  { event := event52936
    frameStart := 52900 },
  { event := event52937
    frameStart := 52900 },
  { event := event52938
    frameStart := 52900 },
  { event := event52939
    frameStart := 52900 },
  { event := event52940
    frameStart := 52900 },
  { event := event52941
    frameStart := 52900 },
  { event := event52942
    frameStart := 52900 },
  { event := event52943
    frameStart := 52900 }
]

def eventLeaf3309 : Array AnnotatedEvent := #[
  { event := event52944
    frameStart := 52900 },
  { event := event52945
    frameStart := 52900 },
  { event := event52946
    frameStart := 52900 },
  { event := event52947
    frameStart := 52900 },
  { event := event52948
    frameStart := 52900 },
  { event := event52949
    frameStart := 52900 },
  { event := event52950
    frameStart := 52900 },
  { event := event52951
    frameStart := 52900 },
  { event := event52952
    frameStart := 52900 },
  { event := event52953
    frameStart := 52900 },
  { event := event52954
    frameStart := 52954 },
  { event := event52955
    frameStart := 52954 },
  { event := event52956
    frameStart := 52954 },
  { event := event52957
    frameStart := 52954 },
  { event := event52958
    frameStart := 52954 },
  { event := event52959
    frameStart := 52954 }
]

def eventLeaf3310 : Array AnnotatedEvent := #[
  { event := event52960
    frameStart := 52954 },
  { event := event52961
    frameStart := 52954 },
  { event := event52962
    frameStart := 52954 },
  { event := event52963
    frameStart := 52954 },
  { event := event52964
    frameStart := 52954 },
  { event := event52965
    frameStart := 52954 },
  { event := event52966
    frameStart := 52954 },
  { event := event52967
    frameStart := 52954 },
  { event := event52968
    frameStart := 52954 },
  { event := event52969
    frameStart := 52954 },
  { event := event52970
    frameStart := 52954 },
  { event := event52971
    frameStart := 52954 },
  { event := event52972
    frameStart := 52954 },
  { event := event52973
    frameStart := 52954 },
  { event := event52974
    frameStart := 52954 },
  { event := event52975
    frameStart := 52954 }
]

def eventLeaf3311 : Array AnnotatedEvent := #[
  { event := event52976
    frameStart := 52954 },
  { event := event52977
    frameStart := 52954 },
  { event := event52978
    frameStart := 52954 },
  { event := event52979
    frameStart := 52954 },
  { event := event52980
    frameStart := 52954 },
  { event := event52981
    frameStart := 52954 },
  { event := event52982
    frameStart := 52954 },
  { event := event52983
    frameStart := 52954 },
  { event := event52984
    frameStart := 52954 },
  { event := event52985
    frameStart := 52954 },
  { event := event52986
    frameStart := 52954 },
  { event := event52987
    frameStart := 52954 },
  { event := event52988
    frameStart := 52954 },
  { event := event52989
    frameStart := 52954 },
  { event := event52990
    frameStart := 52954 },
  { event := event52991
    frameStart := 52954 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events206
