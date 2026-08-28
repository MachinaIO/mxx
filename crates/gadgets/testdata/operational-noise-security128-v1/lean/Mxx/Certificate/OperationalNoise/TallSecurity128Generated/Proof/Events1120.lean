import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events1120

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event286720 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event286721 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event286722 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 286721

def event286723 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 286719

def event286724 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 286722 .coefficient) (.value (.predecessor 1 286723 .coefficient)))

def event286725 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event286726 : Event := .predecessor (⟨.program ⟨257⟩, ⟨2376⟩⟩) 0 ⟨392⟩ 286725

def event286727 : Event := .predecessor (⟨.program ⟨257⟩, ⟨2376⟩⟩) 1 ⟨2370⟩ 286717

def event286728 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨2376⟩⟩) (.sum [.predecessor 0 286726 .coefficient, .predecessor 1 286727 .coefficient])

def event286729 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨2376⟩⟩) (.finite 655341)

def event286730 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5487⟩⟩) 0 ⟨2376⟩ 286729

def event286731 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5487⟩⟩) 1 ⟨5426⟩ 286715

def event286732 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5487⟩⟩) (.identity (.predecessor 1 286731 .coefficient))

def event286733 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5487⟩⟩) (.finite 655360)

def event286734 : Event := .predecessor (⟨.program ⟨257⟩, ⟨24698⟩⟩) 0 ⟨5487⟩ 286733

def event286735 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨24698⟩⟩) (.authority (.programFamilyFact))

def exact286736RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨24698⟩⟩], []⟩, (1)⟩]

theorem exact286736RawTermsValid :
    exact286736RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event286736 : Event := .resultExact (⟨.program ⟨257⟩, ⟨24698⟩⟩) exact286736RawTerms (.finite 12) 286735 .exactZero (none)

def event286737 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53363⟩⟩) 0 ⟨5487⟩ 286733

def event286738 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨53363⟩⟩) (.authority (.programFamilyFact))

def exact286739RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨53363⟩⟩], []⟩, (1)⟩]

theorem exact286739RawTermsValid :
    exact286739RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event286739 : Event := .resultExact (⟨.program ⟨257⟩, ⟨53363⟩⟩) exact286739RawTerms (.finite 12) 286738 .exactZero (none)

def event286740 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53364⟩⟩) 0 ⟨53363⟩ 286739

def event286741 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53364⟩⟩) 1 ⟨24698⟩ 286736

def event286742 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨53364⟩⟩) (.product (.predecessor 0 286740 .coefficient) (.predecessor 1 286741 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event286743 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨53364⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨24698⟩⟩, ⟨.program ⟨257⟩, ⟨53363⟩⟩], []⟩) [⟨.result 286739 .coefficient, true, some 1⟩, ⟨.result 286736 .coefficient, true, some 1⟩])

def event286744 : Event := .survivorFold (1) 286743

def exact286745RawTerms : List Term := []

theorem exact286745RawTermsValid :
    exact286745RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event286745 : Event := .resultExact (⟨.program ⟨257⟩, ⟨53364⟩⟩) exact286745RawTerms (.finite 144) 286742 (.finite 144) (some (286743))

def event286746 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53365⟩⟩) 0 ⟨53364⟩ 286745

def event286747 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨53365⟩⟩) (.identity (.predecessor 0 286746 .coefficient))

def event286748 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨53365⟩⟩) (.finite 144)

def event286749 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53820⟩⟩) 0 ⟨53365⟩ 286748

def event286750 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨53820⟩⟩) (.authority (.programFamilyFact))

def exact286751RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨53820⟩⟩], []⟩, (1)⟩]

theorem exact286751RawTermsValid :
    exact286751RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event286751 : Event := .resultExact (⟨.program ⟨257⟩, ⟨53820⟩⟩) exact286751RawTerms (.finite 12) 286750 .exactZero (none)

def event286752 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53821⟩⟩) 0 ⟨53820⟩ 286751

def event286753 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨53821⟩⟩) (.identity (.predecessor 0 286752 .coefficient))

def event286754 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨53821⟩⟩) (.finite 12)

def event286755 : Event := .predecessor (⟨.program ⟨257⟩, ⟨54616⟩⟩) 0 ⟨53821⟩ 286754

def event286756 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨54616⟩⟩) (.authority (.relationPreimageSource ⟨68⟩))

def exact286757RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨54616⟩⟩]⟩, (1)⟩]

theorem exact286757RawTermsValid :
    exact286757RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event286757 : Event := .resultExact (⟨.program ⟨257⟩, ⟨54616⟩⟩) exact286757RawTerms (.finite 5647228698) 286756 .exactZero (none)

def event286758 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35⟩⟩) (.authority (.operator))

def exact286759RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩]⟩, (1)⟩]

theorem exact286759RawTermsValid :
    exact286759RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event286759 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35⟩⟩) exact286759RawTerms .large 286758 .exactZero (none)

def event286760 : Event := .predecessor (⟨.program ⟨257⟩, ⟨54617⟩⟩) 0 ⟨35⟩ 286759

def event286761 : Event := .predecessor (⟨.program ⟨257⟩, ⟨54617⟩⟩) 1 ⟨54616⟩ 286757

def event286762 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨54617⟩⟩) (.product (.predecessor 0 286760 .coefficient) (.predecessor 1 286761 .coefficient) (⟨false, false, none, none, none⟩))

def event286763 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨54617⟩⟩, .operator (⟨286759, 0⟩, ⟨286757, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54616⟩⟩]⟩, (1)⟩)

def exact286764RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54616⟩⟩]⟩, (1)⟩]

theorem exact286764RawTermsValid :
    exact286764RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event286764 : Event := .resultExact (⟨.program ⟨257⟩, ⟨54617⟩⟩) exact286764RawTerms .large 286762 .exactZero (none)

def event286765 : Event := .preFoldPolynomial 286764 [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54616⟩⟩]⟩, (1)⟩] .exactZero none

def exact286766RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54616⟩⟩]⟩, (1)⟩]

def event286766 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨54617⟩⟩) 286765 exact286766RawTerms .large 286762 .exactZero (none)

def event286767 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨55751⟩⟩)

def event286768 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event286769 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event286770 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨2370⟩⟩) (.authority (.operator))

def event286771 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨2370⟩⟩) (.finite 1)

def event286772 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event286773 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event286774 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event286775 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event286776 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 286775

def event286777 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 286773

def event286778 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 286776 .coefficient) (.value (.predecessor 1 286777 .coefficient)))

def event286779 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event286780 : Event := .predecessor (⟨.program ⟨257⟩, ⟨2376⟩⟩) 0 ⟨392⟩ 286779

def event286781 : Event := .predecessor (⟨.program ⟨257⟩, ⟨2376⟩⟩) 1 ⟨2370⟩ 286771

def event286782 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨2376⟩⟩) (.sum [.predecessor 0 286780 .coefficient, .predecessor 1 286781 .coefficient])

def event286783 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨2376⟩⟩) (.finite 655341)

def event286784 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5487⟩⟩) 0 ⟨2376⟩ 286783

def event286785 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5487⟩⟩) 1 ⟨5426⟩ 286769

def event286786 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5487⟩⟩) (.identity (.predecessor 1 286785 .coefficient))

def event286787 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5487⟩⟩) (.finite 655360)

def event286788 : Event := .predecessor (⟨.program ⟨257⟩, ⟨24698⟩⟩) 0 ⟨5487⟩ 286787

def event286789 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨24698⟩⟩) (.authority (.programFamilyFact))

def exact286790RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨24698⟩⟩], []⟩, (1)⟩]

theorem exact286790RawTermsValid :
    exact286790RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event286790 : Event := .resultExact (⟨.program ⟨257⟩, ⟨24698⟩⟩) exact286790RawTerms (.finite 12) 286789 .exactZero (none)

def event286791 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53363⟩⟩) 0 ⟨5487⟩ 286787

def event286792 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨53363⟩⟩) (.authority (.programFamilyFact))

def exact286793RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨53363⟩⟩], []⟩, (1)⟩]

theorem exact286793RawTermsValid :
    exact286793RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event286793 : Event := .resultExact (⟨.program ⟨257⟩, ⟨53363⟩⟩) exact286793RawTerms (.finite 12) 286792 .exactZero (none)

def event286794 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53364⟩⟩) 0 ⟨53363⟩ 286793

def event286795 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53364⟩⟩) 1 ⟨24698⟩ 286790

def event286796 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨53364⟩⟩) (.product (.predecessor 0 286794 .coefficient) (.predecessor 1 286795 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event286797 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨53364⟩⟩, .operator (⟨286793, 0⟩, ⟨286790, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨24698⟩⟩, ⟨.program ⟨257⟩, ⟨53363⟩⟩], []⟩, (1)⟩)

def exact286798RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨24698⟩⟩, ⟨.program ⟨257⟩, ⟨53363⟩⟩], []⟩, (1)⟩]

theorem exact286798RawTermsValid :
    exact286798RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event286798 : Event := .resultExact (⟨.program ⟨257⟩, ⟨53364⟩⟩) exact286798RawTerms (.finite 144) 286796 .exactZero (none)

def event286799 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53365⟩⟩) 0 ⟨53364⟩ 286798

def event286800 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨53365⟩⟩) (.identity (.predecessor 0 286799 .coefficient))

def event286801 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨53365⟩⟩) (.finite 144)

def event286802 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53820⟩⟩) 0 ⟨53365⟩ 286801

def event286803 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨53820⟩⟩) (.authority (.programFamilyFact))

def exact286804RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨53820⟩⟩], []⟩, (1)⟩]

theorem exact286804RawTermsValid :
    exact286804RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event286804 : Event := .resultExact (⟨.program ⟨257⟩, ⟨53820⟩⟩) exact286804RawTerms (.finite 12) 286803 .exactZero (none)

def event286805 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53821⟩⟩) 0 ⟨53820⟩ 286804

def event286806 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨53821⟩⟩) (.identity (.predecessor 0 286805 .coefficient))

def event286807 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨53821⟩⟩) (.finite 12)

def event286808 : Event := .predecessor (⟨.program ⟨257⟩, ⟨55085⟩⟩) 0 ⟨53821⟩ 286807

def event286809 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨55085⟩⟩) (.authority (.programFamilyFact))

def event286810 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨55085⟩⟩) (.finite 3720)

def event286811 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨7177⟩⟩) .missing

def event286812 : Event := .predecessor (⟨.program ⟨257⟩, ⟨55087⟩⟩) 0 ⟨7177⟩ 286811

def event286813 : Event := .predecessor (⟨.program ⟨257⟩, ⟨55087⟩⟩) 1 ⟨55085⟩ 286810

def event286814 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨55087⟩⟩) (.authority (.operator))

def exact286815RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨55087⟩⟩]⟩, (1)⟩]

theorem exact286815RawTermsValid :
    exact286815RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event286815 : Event := .resultExact (⟨.program ⟨257⟩, ⟨55087⟩⟩) exact286815RawTerms .large 286814 .exactZero (none)

def event286816 : Event := .predecessor (⟨.program ⟨257⟩, ⟨55746⟩⟩) 0 ⟨55087⟩ 286815

def event286817 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨55746⟩⟩) (.authority (.operator))

def exact286818RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨55746⟩⟩]⟩, (1)⟩]

theorem exact286818RawTermsValid :
    exact286818RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event286818 : Event := .resultExact (⟨.program ⟨257⟩, ⟨55746⟩⟩) exact286818RawTerms (.finite 8192) 286817 .exactZero (none)

def event286819 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨136⟩⟩) (.authority (.operator))

def event286820 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨136⟩⟩) .exactZero

def event286821 : Event := .predecessor (⟨.program ⟨257⟩, ⟨55322⟩⟩) 0 ⟨53821⟩ 286807

def event286822 : Event := .predecessor (⟨.program ⟨257⟩, ⟨55322⟩⟩) 1 ⟨136⟩ 286820

def event286823 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨55322⟩⟩) (.sum [.predecessor 0 286821 .coefficient, .predecessor 1 286822 .coefficient])

def event286824 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨55322⟩⟩) (.finite 12)

def event286825 : Event := .predecessor (⟨.program ⟨257⟩, ⟨55323⟩⟩) 0 ⟨55322⟩ 286824

def event286826 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨55323⟩⟩) (.identity (.predecessor 0 286825 .coefficient))

def exact286827RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨53820⟩⟩], []⟩, (1)⟩]

theorem exact286827RawTermsValid :
    exact286827RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event286827 : Event := .resultExact (⟨.program ⟨257⟩, ⟨55323⟩⟩) exact286827RawTerms (.finite 12) 286826 .exactZero (none)

def event286828 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6908⟩⟩) (.authority (.factStore))

def exact286829RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact286829RawTermsValid :
    exact286829RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event286829 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6908⟩⟩) exact286829RawTerms .large 286828 .exactZero (none)

def event286830 : Event := .predecessor (⟨.program ⟨257⟩, ⟨55324⟩⟩) 0 ⟨6908⟩ 286829

def event286831 : Event := .predecessor (⟨.program ⟨257⟩, ⟨55324⟩⟩) 1 ⟨55323⟩ 286827

def event286832 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨55324⟩⟩) (.product (.predecessor 0 286830 .coefficient) (.predecessor 1 286831 .coefficient) (⟨false, false, none, none, none⟩))

def event286833 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨55324⟩⟩, .operator (⟨286829, 0⟩, ⟨286827, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨53820⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact286834RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨53820⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact286834RawTermsValid :
    exact286834RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event286834 : Event := .resultExact (⟨.program ⟨257⟩, ⟨55324⟩⟩) exact286834RawTerms .large 286832 .exactZero (none)

def event286835 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7184⟩⟩) 0 ⟨7177⟩ 286811

def event286836 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7184⟩⟩) (.authority (.operator))

def exact286837RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7184⟩⟩]⟩, (1)⟩]

theorem exact286837RawTermsValid :
    exact286837RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event286837 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7184⟩⟩) exact286837RawTerms .large 286836 .exactZero (none)

def event286838 : Event := .predecessor (⟨.program ⟨257⟩, ⟨55325⟩⟩) 0 ⟨7184⟩ 286837

def event286839 : Event := .predecessor (⟨.program ⟨257⟩, ⟨55325⟩⟩) 1 ⟨55324⟩ 286834

def event286840 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨55325⟩⟩) (.sum [.predecessor 0 286838 .coefficient, .predecessor 1 286839 .coefficient])

def exact286841RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7184⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨53820⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact286841RawTermsValid :
    exact286841RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event286841 : Event := .resultExact (⟨.program ⟨257⟩, ⟨55325⟩⟩) exact286841RawTerms .large 286840 .exactZero (none)

def event286842 : Event := .predecessor (⟨.program ⟨257⟩, ⟨55747⟩⟩) 0 ⟨55325⟩ 286841

def event286843 : Event := .predecessor (⟨.program ⟨257⟩, ⟨55747⟩⟩) 1 ⟨55746⟩ 286818

def event286844 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨55747⟩⟩) (.product (.predecessor 0 286842 .coefficient) (.predecessor 1 286843 .coefficient) (⟨false, false, none, none, none⟩))

def event286845 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨55747⟩⟩, .operator (⟨286841, 0⟩, ⟨286818, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7184⟩⟩, ⟨.program ⟨257⟩, ⟨55746⟩⟩]⟩, (1)⟩)

def event286846 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨55747⟩⟩, .operator (⟨286841, 1⟩, ⟨286818, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨53820⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨55746⟩⟩]⟩, (-1)⟩)

def event286847 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨55747⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨53820⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨55746⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨55746⟩⟩) ⟨55087⟩ 286815)

def event286848 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨55747⟩⟩, .relation 286847 0, ⟨[⟨.program ⟨257⟩, ⟨53820⟩⟩], [⟨.program ⟨257⟩, ⟨55087⟩⟩]⟩, (-1)⟩)

def exact286849RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7184⟩⟩, ⟨.program ⟨257⟩, ⟨55746⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨53820⟩⟩], [⟨.program ⟨257⟩, ⟨55087⟩⟩]⟩, (-1)⟩]

theorem exact286849RawTermsValid :
    exact286849RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event286849 : Event := .resultExact (⟨.program ⟨257⟩, ⟨55747⟩⟩) exact286849RawTerms .large 286844 .exactZero (none)

def event286850 : Event := .predecessor (⟨.program ⟨257⟩, ⟨54027⟩⟩) 0 ⟨53821⟩ 286807

def event286851 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨54027⟩⟩) (.authority (.programFamilyFact))

def exact286852RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨54027⟩⟩], []⟩, (1)⟩]

theorem exact286852RawTermsValid :
    exact286852RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event286852 : Event := .resultExact (⟨.program ⟨257⟩, ⟨54027⟩⟩) exact286852RawTerms (.finite 59) 286851 .exactZero (none)

def event286853 : Event := .predecessor (⟨.program ⟨257⟩, ⟨54029⟩⟩) 0 ⟨6908⟩ 286829

def event286854 : Event := .predecessor (⟨.program ⟨257⟩, ⟨54029⟩⟩) 1 ⟨54027⟩ 286852

def event286855 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨54029⟩⟩) (.product (.predecessor 0 286853 .coefficient) (.predecessor 1 286854 .coefficient) (⟨false, true, none, none, some 1⟩))

def event286856 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨54029⟩⟩, .operator (⟨286829, 0⟩, ⟨286852, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨54027⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact286857RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨54027⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact286857RawTermsValid :
    exact286857RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event286857 : Event := .resultExact (⟨.program ⟨257⟩, ⟨54029⟩⟩) exact286857RawTerms .large 286855 .exactZero (none)

def event286858 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7208⟩⟩) 0 ⟨7177⟩ 286811

def event286859 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7208⟩⟩) (.authority (.operator))

def exact286860RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩]

theorem exact286860RawTermsValid :
    exact286860RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event286860 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7208⟩⟩) exact286860RawTerms .large 286859 .exactZero (none)

def event286861 : Event := .predecessor (⟨.program ⟨257⟩, ⟨54030⟩⟩) 0 ⟨7208⟩ 286860

def event286862 : Event := .predecessor (⟨.program ⟨257⟩, ⟨54030⟩⟩) 1 ⟨54029⟩ 286857

def event286863 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨54030⟩⟩) (.sum [.predecessor 0 286861 .coefficient, .predecessor 1 286862 .coefficient])

def exact286864RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨54027⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact286864RawTermsValid :
    exact286864RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event286864 : Event := .resultExact (⟨.program ⟨257⟩, ⟨54030⟩⟩) exact286864RawTerms .large 286863 .exactZero (none)

def event286865 : Event := .predecessor (⟨.program ⟨257⟩, ⟨55751⟩⟩) 0 ⟨54030⟩ 286864

def event286866 : Event := .predecessor (⟨.program ⟨257⟩, ⟨55751⟩⟩) 1 ⟨55747⟩ 286849

def event286867 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨55751⟩⟩) (.sum [.predecessor 0 286865 .coefficient, .predecessor 1 286866 .coefficient])

def exact286868RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7184⟩⟩, ⟨.program ⟨257⟩, ⟨55746⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨53820⟩⟩], [⟨.program ⟨257⟩, ⟨55087⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨54027⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact286868RawTermsValid :
    exact286868RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event286868 : Event := .resultExact (⟨.program ⟨257⟩, ⟨55751⟩⟩) exact286868RawTerms .large 286867 .exactZero (none)

def event286869 : Event := .preFoldPolynomial 286868 [⟨⟨[], [⟨.program ⟨257⟩, ⟨7184⟩⟩, ⟨.program ⟨257⟩, ⟨55746⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨53820⟩⟩], [⟨.program ⟨257⟩, ⟨55087⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨54027⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩] .exactZero none

def exact286870RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7184⟩⟩, ⟨.program ⟨257⟩, ⟨55746⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨53820⟩⟩], [⟨.program ⟨257⟩, ⟨55087⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨54027⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

def event286870 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨55751⟩⟩) 286869 exact286870RawTerms .large 286867 .exactZero (none)

def event286871 : Event := .specializationComputed (⟨.program ⟨257⟩, ⟨53821⟩⟩) ⟨⟨87⟩, ⟨68⟩, ⟨135⟩⟩ ⟨286713, 286871⟩

def event286872 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨54619⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54616⟩⟩]⟩) (1) 0 2 (.universal 286871 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54616⟩⟩]⟩) (none) 286870)

def event286873 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨54619⟩⟩, .relation 286872 1, ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩)

def event286874 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨54619⟩⟩, .relation 286872 0, ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7184⟩⟩, ⟨.program ⟨257⟩, ⟨55746⟩⟩]⟩, (-1)⟩)

def event286875 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨54619⟩⟩, .relation 286872 2, ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨53820⟩⟩], [⟨.program ⟨257⟩, ⟨55087⟩⟩]⟩, (1)⟩)

def event286876 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨54619⟩⟩, .relation 286872 3, ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨54027⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact286877RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7184⟩⟩, ⟨.program ⟨257⟩, ⟨55746⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨53820⟩⟩], [⟨.program ⟨257⟩, ⟨55087⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨54027⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact286877RawTermsValid :
    exact286877RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event286877 : Event := .resultExact (⟨.program ⟨257⟩, ⟨54619⟩⟩) exact286877RawTerms .large 286709 (.finite 202072841853861888) (some (286711))

def event286878 : Event := .predecessor (⟨.program ⟨257⟩, ⟨55749⟩⟩) 0 ⟨54619⟩ 286877

def event286879 : Event := .predecessor (⟨.program ⟨257⟩, ⟨55749⟩⟩) 1 ⟨55748⟩ 286699

def event286880 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨55749⟩⟩) (.sum [.predecessor 0 286878 .coefficient, .predecessor 1 286879 .coefficient])

def event286881 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨55749⟩⟩, .operator (⟨286877, 0⟩, ⟨286699, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7184⟩⟩, ⟨.program ⟨257⟩, ⟨55746⟩⟩]⟩, (1)⟩)

def event286882 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨55749⟩⟩, .operator (⟨286877, 2⟩, ⟨286699, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨53820⟩⟩], [⟨.program ⟨257⟩, ⟨55087⟩⟩]⟩, (-1)⟩)

def event286883 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨55749⟩⟩) (.sum [.result 286877 .summary, .result 286699 .summary])

def exact286884RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨54027⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact286884RawTermsValid :
    exact286884RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event286884 : Event := .resultExact (⟨.program ⟨257⟩, ⟨55749⟩⟩) exact286884RawTerms .large 286880 (.finite 32189789464712143775715074244608) (some (286883))

def event286885 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52105⟩⟩) 0 ⟨50841⟩ 13868

def event286886 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨52105⟩⟩) (.authority (.programFamilyFact))

def event286887 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨52105⟩⟩) (.finite 3720)

def event286888 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52107⟩⟩) 0 ⟨7177⟩ 15500

def event286889 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52107⟩⟩) 1 ⟨52105⟩ 286887

def event286890 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨52107⟩⟩) (.authority (.operator))

def exact286891RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨52107⟩⟩]⟩, (1)⟩]

theorem exact286891RawTermsValid :
    exact286891RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event286891 : Event := .resultExact (⟨.program ⟨257⟩, ⟨52107⟩⟩) exact286891RawTerms .large 286890 .exactZero (none)

def event286892 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52766⟩⟩) 0 ⟨52107⟩ 286891

def event286893 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨52766⟩⟩) (.authority (.operator))

def exact286894RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨52766⟩⟩]⟩, (1)⟩]

theorem exact286894RawTermsValid :
    exact286894RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event286894 : Event := .resultExact (⟨.program ⟨257⟩, ⟨52766⟩⟩) exact286894RawTerms (.finite 8192) 286893 .exactZero (none)

def event286895 : Event := .predecessor (⟨.program ⟨257⟩, ⟨51972⟩⟩) 0 ⟨50385⟩ 13862

def event286896 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨51972⟩⟩) (.authority (.programFamilyFact))

def event286897 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨51972⟩⟩) (.finite 3720)

def event286898 : Event := .predecessor (⟨.program ⟨257⟩, ⟨51973⟩⟩) 0 ⟨7177⟩ 15500

def event286899 : Event := .predecessor (⟨.program ⟨257⟩, ⟨51973⟩⟩) 1 ⟨51972⟩ 286897

def event286900 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨51973⟩⟩) (.authority (.operator))

def exact286901RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨51973⟩⟩]⟩, (1)⟩]

theorem exact286901RawTermsValid :
    exact286901RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event286901 : Event := .resultExact (⟨.program ⟨257⟩, ⟨51973⟩⟩) exact286901RawTerms .large 286900 .exactZero (none)

def event286902 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52453⟩⟩) 0 ⟨51973⟩ 286901

def event286903 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨52453⟩⟩) (.authority (.operator))

def exact286904RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨52453⟩⟩]⟩, (1)⟩]

theorem exact286904RawTermsValid :
    exact286904RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event286904 : Event := .resultExact (⟨.program ⟨257⟩, ⟨52453⟩⟩) exact286904RawTerms (.finite 8192) 286903 .exactZero (none)

def event286905 : Event := .predecessor (⟨.program ⟨257⟩, ⟨24459⟩⟩) 0 ⟨24458⟩ 13851

def event286906 : Event := .predecessor (⟨.program ⟨257⟩, ⟨24459⟩⟩) 1 ⟨6922⟩ 280653

def event286907 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨24459⟩⟩) (.tensor (.predecessor 0 286905 .coefficient) (.predecessor 1 286906 .coefficient) true false)

def event286908 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨24459⟩⟩, .operator (⟨13851, 0⟩, ⟨280653, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨24458⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact286909RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨24458⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact286909RawTermsValid :
    exact286909RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event286909 : Event := .resultExact (⟨.program ⟨257⟩, ⟨24459⟩⟩) exact286909RawTerms .large 286907 .exactZero (none)

def event286910 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7930⟩⟩) 0 ⟨5489⟩ 280523

def event286911 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7930⟩⟩) 1 ⟨7308⟩ 23593

def event286912 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7930⟩⟩) (.product (.predecessor 0 286910 .coefficient) (.predecessor 1 286911 .coefficient) (⟨false, false, none, none, none⟩))

def event286913 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨7930⟩⟩, .operator (⟨280523, 0⟩, ⟨23593, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7308⟩⟩]⟩, (1)⟩)

def exact286914RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7308⟩⟩]⟩, (1)⟩]

theorem exact286914RawTermsValid :
    exact286914RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event286914 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7930⟩⟩) exact286914RawTerms .large 286912 .exactZero (none)

def event286915 : Event := .predecessor (⟨.program ⟨257⟩, ⟨24460⟩⟩) 0 ⟨7930⟩ 286914

def event286916 : Event := .predecessor (⟨.program ⟨257⟩, ⟨24460⟩⟩) 1 ⟨24459⟩ 286909

def event286917 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨24460⟩⟩) (.sum [.predecessor 0 286915 .coefficient, .predecessor 1 286916 .coefficient])

def exact286918RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7308⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨24458⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact286918RawTermsValid :
    exact286918RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event286918 : Event := .resultExact (⟨.program ⟨257⟩, ⟨24460⟩⟩) exact286918RawTerms .large 286917 .exactZero (none)

def event286919 : Event := .predecessor (⟨.program ⟨257⟩, ⟨24461⟩⟩) 0 ⟨24460⟩ 286918

def event286920 : Event := .predecessor (⟨.program ⟨257⟩, ⟨24461⟩⟩) 1 ⟨134⟩ 23585

def event286921 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨24461⟩⟩) (.sum [.predecessor 0 286919 .coefficient, .predecessor 1 286920 .coefficient])

def event286922 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨24461⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨134⟩⟩]⟩) [⟨.result 23585 .coefficient, false, none⟩])

def event286923 : Event := .survivorFold (1) 286922

def exact286924RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7308⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨24458⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact286924RawTermsValid :
    exact286924RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event286924 : Event := .resultExact (⟨.program ⟨257⟩, ⟨24461⟩⟩) exact286924RawTerms .large 286921 (.finite 26) (some (286922))

def event286925 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50386⟩⟩) 0 ⟨24461⟩ 286924

def event286926 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50386⟩⟩) 1 ⟨50383⟩ 13854

def event286927 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50386⟩⟩) (.product (.predecessor 0 286925 .coefficient) (.predecessor 1 286926 .coefficient) (⟨false, true, none, none, some 1⟩))

def event286928 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50386⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨50383⟩⟩], []⟩) [⟨.result 13854 .coefficient, true, some 1⟩])

def event286929 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50386⟩⟩) (.product (.result 286924 .summary) (.transfer 286928) (⟨false, false, none, none, none⟩))

def event286930 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨50386⟩⟩, .operator (⟨286924, 1⟩, ⟨13854, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨24458⟩⟩, ⟨.program ⟨257⟩, ⟨50383⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def event286931 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨50386⟩⟩, .operator (⟨286924, 0⟩, ⟨13854, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨50383⟩⟩], [⟨.program ⟨257⟩, ⟨7308⟩⟩]⟩, (1)⟩)

def exact286932RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨24458⟩⟩, ⟨.program ⟨257⟩, ⟨50383⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨50383⟩⟩], [⟨.program ⟨257⟩, ⟨7308⟩⟩]⟩, (1)⟩]

theorem exact286932RawTermsValid :
    exact286932RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event286932 : Event := .resultExact (⟨.program ⟨257⟩, ⟨50386⟩⟩) exact286932RawTerms .large 286927 (.finite 8519680) (some (286929))

def event286933 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50387⟩⟩) 0 ⟨50383⟩ 13854

def event286934 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50387⟩⟩) 1 ⟨6922⟩ 280653

def event286935 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50387⟩⟩) (.tensor (.predecessor 0 286933 .coefficient) (.predecessor 1 286934 .coefficient) true false)

def event286936 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨50387⟩⟩, .operator (⟨13854, 0⟩, ⟨280653, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨50383⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact286937RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨50383⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact286937RawTermsValid :
    exact286937RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event286937 : Event := .resultExact (⟨.program ⟨257⟩, ⟨50387⟩⟩) exact286937RawTerms .large 286935 .exactZero (none)

def event286938 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7910⟩⟩) 0 ⟨5489⟩ 280523

def event286939 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7910⟩⟩) 1 ⟨7288⟩ 23634

def event286940 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7910⟩⟩) (.product (.predecessor 0 286938 .coefficient) (.predecessor 1 286939 .coefficient) (⟨false, false, none, none, none⟩))

def event286941 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨7910⟩⟩, .operator (⟨280523, 0⟩, ⟨23634, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7288⟩⟩]⟩, (1)⟩)

def exact286942RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7288⟩⟩]⟩, (1)⟩]

theorem exact286942RawTermsValid :
    exact286942RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event286942 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7910⟩⟩) exact286942RawTerms .large 286940 .exactZero (none)

def event286943 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50388⟩⟩) 0 ⟨7910⟩ 286942

def event286944 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50388⟩⟩) 1 ⟨50387⟩ 286937

def event286945 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50388⟩⟩) (.sum [.predecessor 0 286943 .coefficient, .predecessor 1 286944 .coefficient])

def exact286946RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7288⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨50383⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact286946RawTermsValid :
    exact286946RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event286946 : Event := .resultExact (⟨.program ⟨257⟩, ⟨50388⟩⟩) exact286946RawTerms .large 286945 .exactZero (none)

def event286947 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50389⟩⟩) 0 ⟨50388⟩ 286946

def event286948 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50389⟩⟩) 1 ⟨114⟩ 23626

def event286949 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50389⟩⟩) (.sum [.predecessor 0 286947 .coefficient, .predecessor 1 286948 .coefficient])

def event286950 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50389⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨114⟩⟩]⟩) [⟨.result 23626 .coefficient, false, none⟩])

def event286951 : Event := .survivorFold (1) 286950

def exact286952RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7288⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨50383⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact286952RawTermsValid :
    exact286952RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event286952 : Event := .resultExact (⟨.program ⟨257⟩, ⟨50389⟩⟩) exact286952RawTerms .large 286949 (.finite 26) (some (286950))

def event286953 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50390⟩⟩) 0 ⟨50389⟩ 286952

def event286954 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50390⟩⟩) 1 ⟨9581⟩ 23623

def event286955 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50390⟩⟩) (.product (.predecessor 0 286953 .coefficient) (.predecessor 1 286954 .coefficient) (⟨false, false, none, none, none⟩))

def event286956 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50390⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨9580⟩⟩]⟩) [⟨.result 23619 .coefficient, false, none⟩])

def event286957 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50390⟩⟩) (.product (.result 286952 .summary) (.transfer 286956) (⟨false, false, none, none, none⟩))

def event286958 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨50390⟩⟩, .operator (⟨286952, 1⟩, ⟨23623, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨50383⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩]⟩, (-1)⟩)

def event286959 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨50390⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨50383⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨9580⟩⟩) ⟨7308⟩ 23593)

def event286960 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨50390⟩⟩, .relation 286959 0, ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨50383⟩⟩], [⟨.program ⟨257⟩, ⟨7308⟩⟩]⟩, (-1)⟩)

def event286961 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨50390⟩⟩, .operator (⟨286952, 0⟩, ⟨23623, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩]⟩, (1)⟩)

def exact286962RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨50383⟩⟩], [⟨.program ⟨257⟩, ⟨7308⟩⟩]⟩, (-1)⟩]

theorem exact286962RawTermsValid :
    exact286962RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event286962 : Event := .resultExact (⟨.program ⟨257⟩, ⟨50390⟩⟩) exact286962RawTerms .large 286955 (.finite 279172874240) (some (286957))

def event286963 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50391⟩⟩) 0 ⟨50390⟩ 286962

def event286964 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50391⟩⟩) 1 ⟨50386⟩ 286932

def event286965 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50391⟩⟩) (.sum [.predecessor 0 286963 .coefficient, .predecessor 1 286964 .coefficient])

def event286966 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨50391⟩⟩, .operator (⟨286962, 1⟩, ⟨286932, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨50383⟩⟩], [⟨.program ⟨257⟩, ⟨7308⟩⟩]⟩, (1)⟩)

def event286967 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50391⟩⟩) (.sum [.result 286962 .summary, .result 286932 .summary])

def exact286968RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨24458⟩⟩, ⟨.program ⟨257⟩, ⟨50383⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact286968RawTermsValid :
    exact286968RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event286968 : Event := .resultExact (⟨.program ⟨257⟩, ⟨50391⟩⟩) exact286968RawTerms .large 286965 (.finite 279181393920) (some (286967))

def event286969 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52454⟩⟩) 0 ⟨50391⟩ 286968

def event286970 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52454⟩⟩) 1 ⟨52453⟩ 286904

def event286971 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨52454⟩⟩) (.product (.predecessor 0 286969 .coefficient) (.predecessor 1 286970 .coefficient) (⟨false, false, none, none, none⟩))

def event286972 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨52454⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨52453⟩⟩]⟩) [⟨.result 286904 .coefficient, false, none⟩])

def event286973 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨52454⟩⟩) (.product (.result 286968 .summary) (.transfer 286972) (⟨false, false, none, none, none⟩))

def event286974 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨52454⟩⟩, .operator (⟨286968, 1⟩, ⟨286904, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨24458⟩⟩, ⟨.program ⟨257⟩, ⟨50383⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨52453⟩⟩]⟩, (-1)⟩)

def event286975 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨52454⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨24458⟩⟩, ⟨.program ⟨257⟩, ⟨50383⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨52453⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨52453⟩⟩) ⟨51973⟩ 286901)

def eventLeaf17920 : Array AnnotatedEvent := #[
  { event := event286720
    frameStart := 286713 },
  { event := event286721
    frameStart := 286713 },
  { event := event286722
    frameStart := 286713 },
  { event := event286723
    frameStart := 286713 },
  { event := event286724
    frameStart := 286713 },
  { event := event286725
    frameStart := 286713 },
  { event := event286726
    frameStart := 286713 },
  { event := event286727
    frameStart := 286713 },
  { event := event286728
    frameStart := 286713 },
  { event := event286729
    frameStart := 286713 },
  { event := event286730
    frameStart := 286713 },
  { event := event286731
    frameStart := 286713 },
  { event := event286732
    frameStart := 286713 },
  { event := event286733
    frameStart := 286713 },
  { event := event286734
    frameStart := 286713 },
  { event := event286735
    frameStart := 286713 }
]

def eventLeaf17921 : Array AnnotatedEvent := #[
  { event := event286736
    frameStart := 286713 },
  { event := event286737
    frameStart := 286713 },
  { event := event286738
    frameStart := 286713 },
  { event := event286739
    frameStart := 286713 },
  { event := event286740
    frameStart := 286713 },
  { event := event286741
    frameStart := 286713 },
  { event := event286742
    frameStart := 286713 },
  { event := event286743
    frameStart := 286713 },
  { event := event286744
    frameStart := 286713 },
  { event := event286745
    frameStart := 286713 },
  { event := event286746
    frameStart := 286713 },
  { event := event286747
    frameStart := 286713 },
  { event := event286748
    frameStart := 286713 },
  { event := event286749
    frameStart := 286713 },
  { event := event286750
    frameStart := 286713 },
  { event := event286751
    frameStart := 286713 }
]

def eventLeaf17922 : Array AnnotatedEvent := #[
  { event := event286752
    frameStart := 286713 },
  { event := event286753
    frameStart := 286713 },
  { event := event286754
    frameStart := 286713 },
  { event := event286755
    frameStart := 286713 },
  { event := event286756
    frameStart := 286713 },
  { event := event286757
    frameStart := 286713 },
  { event := event286758
    frameStart := 286713 },
  { event := event286759
    frameStart := 286713 },
  { event := event286760
    frameStart := 286713 },
  { event := event286761
    frameStart := 286713 },
  { event := event286762
    frameStart := 286713 },
  { event := event286763
    frameStart := 286713 },
  { event := event286764
    frameStart := 286713 },
  { event := event286765
    frameStart := 286713 },
  { event := event286766
    frameStart := 286713 },
  { event := event286767
    frameStart := 286767 }
]

def eventLeaf17923 : Array AnnotatedEvent := #[
  { event := event286768
    frameStart := 286767 },
  { event := event286769
    frameStart := 286767 },
  { event := event286770
    frameStart := 286767 },
  { event := event286771
    frameStart := 286767 },
  { event := event286772
    frameStart := 286767 },
  { event := event286773
    frameStart := 286767 },
  { event := event286774
    frameStart := 286767 },
  { event := event286775
    frameStart := 286767 },
  { event := event286776
    frameStart := 286767 },
  { event := event286777
    frameStart := 286767 },
  { event := event286778
    frameStart := 286767 },
  { event := event286779
    frameStart := 286767 },
  { event := event286780
    frameStart := 286767 },
  { event := event286781
    frameStart := 286767 },
  { event := event286782
    frameStart := 286767 },
  { event := event286783
    frameStart := 286767 }
]

def eventLeaf17924 : Array AnnotatedEvent := #[
  { event := event286784
    frameStart := 286767 },
  { event := event286785
    frameStart := 286767 },
  { event := event286786
    frameStart := 286767 },
  { event := event286787
    frameStart := 286767 },
  { event := event286788
    frameStart := 286767 },
  { event := event286789
    frameStart := 286767 },
  { event := event286790
    frameStart := 286767 },
  { event := event286791
    frameStart := 286767 },
  { event := event286792
    frameStart := 286767 },
  { event := event286793
    frameStart := 286767 },
  { event := event286794
    frameStart := 286767 },
  { event := event286795
    frameStart := 286767 },
  { event := event286796
    frameStart := 286767 },
  { event := event286797
    frameStart := 286767 },
  { event := event286798
    frameStart := 286767 },
  { event := event286799
    frameStart := 286767 }
]

def eventLeaf17925 : Array AnnotatedEvent := #[
  { event := event286800
    frameStart := 286767 },
  { event := event286801
    frameStart := 286767 },
  { event := event286802
    frameStart := 286767 },
  { event := event286803
    frameStart := 286767 },
  { event := event286804
    frameStart := 286767 },
  { event := event286805
    frameStart := 286767 },
  { event := event286806
    frameStart := 286767 },
  { event := event286807
    frameStart := 286767 },
  { event := event286808
    frameStart := 286767 },
  { event := event286809
    frameStart := 286767 },
  { event := event286810
    frameStart := 286767 },
  { event := event286811
    frameStart := 286767 },
  { event := event286812
    frameStart := 286767 },
  { event := event286813
    frameStart := 286767 },
  { event := event286814
    frameStart := 286767 },
  { event := event286815
    frameStart := 286767 }
]

def eventLeaf17926 : Array AnnotatedEvent := #[
  { event := event286816
    frameStart := 286767 },
  { event := event286817
    frameStart := 286767 },
  { event := event286818
    frameStart := 286767 },
  { event := event286819
    frameStart := 286767 },
  { event := event286820
    frameStart := 286767 },
  { event := event286821
    frameStart := 286767 },
  { event := event286822
    frameStart := 286767 },
  { event := event286823
    frameStart := 286767 },
  { event := event286824
    frameStart := 286767 },
  { event := event286825
    frameStart := 286767 },
  { event := event286826
    frameStart := 286767 },
  { event := event286827
    frameStart := 286767 },
  { event := event286828
    frameStart := 286767 },
  { event := event286829
    frameStart := 286767 },
  { event := event286830
    frameStart := 286767 },
  { event := event286831
    frameStart := 286767 }
]

def eventLeaf17927 : Array AnnotatedEvent := #[
  { event := event286832
    frameStart := 286767 },
  { event := event286833
    frameStart := 286767 },
  { event := event286834
    frameStart := 286767 },
  { event := event286835
    frameStart := 286767 },
  { event := event286836
    frameStart := 286767 },
  { event := event286837
    frameStart := 286767 },
  { event := event286838
    frameStart := 286767 },
  { event := event286839
    frameStart := 286767 },
  { event := event286840
    frameStart := 286767 },
  { event := event286841
    frameStart := 286767 },
  { event := event286842
    frameStart := 286767 },
  { event := event286843
    frameStart := 286767 },
  { event := event286844
    frameStart := 286767 },
  { event := event286845
    frameStart := 286767 },
  { event := event286846
    frameStart := 286767 },
  { event := event286847
    frameStart := 286767 }
]

def eventLeaf17928 : Array AnnotatedEvent := #[
  { event := event286848
    frameStart := 286767 },
  { event := event286849
    frameStart := 286767 },
  { event := event286850
    frameStart := 286767 },
  { event := event286851
    frameStart := 286767 },
  { event := event286852
    frameStart := 286767 },
  { event := event286853
    frameStart := 286767 },
  { event := event286854
    frameStart := 286767 },
  { event := event286855
    frameStart := 286767 },
  { event := event286856
    frameStart := 286767 },
  { event := event286857
    frameStart := 286767 },
  { event := event286858
    frameStart := 286767 },
  { event := event286859
    frameStart := 286767 },
  { event := event286860
    frameStart := 286767 },
  { event := event286861
    frameStart := 286767 },
  { event := event286862
    frameStart := 286767 },
  { event := event286863
    frameStart := 286767 }
]

def eventLeaf17929 : Array AnnotatedEvent := #[
  { event := event286864
    frameStart := 286767 },
  { event := event286865
    frameStart := 286767 },
  { event := event286866
    frameStart := 286767 },
  { event := event286867
    frameStart := 286767 },
  { event := event286868
    frameStart := 286767 },
  { event := event286869
    frameStart := 286767 },
  { event := event286870
    frameStart := 286767 },
  { event := event286871
    frameStart := 0 },
  { event := event286872
    frameStart := 0 },
  { event := event286873
    frameStart := 0 },
  { event := event286874
    frameStart := 0 },
  { event := event286875
    frameStart := 0 },
  { event := event286876
    frameStart := 0 },
  { event := event286877
    frameStart := 0 },
  { event := event286878
    frameStart := 0 },
  { event := event286879
    frameStart := 0 }
]

def eventLeaf17930 : Array AnnotatedEvent := #[
  { event := event286880
    frameStart := 0 },
  { event := event286881
    frameStart := 0 },
  { event := event286882
    frameStart := 0 },
  { event := event286883
    frameStart := 0 },
  { event := event286884
    frameStart := 0 },
  { event := event286885
    frameStart := 0 },
  { event := event286886
    frameStart := 0 },
  { event := event286887
    frameStart := 0 },
  { event := event286888
    frameStart := 0 },
  { event := event286889
    frameStart := 0 },
  { event := event286890
    frameStart := 0 },
  { event := event286891
    frameStart := 0 },
  { event := event286892
    frameStart := 0 },
  { event := event286893
    frameStart := 0 },
  { event := event286894
    frameStart := 0 },
  { event := event286895
    frameStart := 0 }
]

def eventLeaf17931 : Array AnnotatedEvent := #[
  { event := event286896
    frameStart := 0 },
  { event := event286897
    frameStart := 0 },
  { event := event286898
    frameStart := 0 },
  { event := event286899
    frameStart := 0 },
  { event := event286900
    frameStart := 0 },
  { event := event286901
    frameStart := 0 },
  { event := event286902
    frameStart := 0 },
  { event := event286903
    frameStart := 0 },
  { event := event286904
    frameStart := 0 },
  { event := event286905
    frameStart := 0 },
  { event := event286906
    frameStart := 0 },
  { event := event286907
    frameStart := 0 },
  { event := event286908
    frameStart := 0 },
  { event := event286909
    frameStart := 0 },
  { event := event286910
    frameStart := 0 },
  { event := event286911
    frameStart := 0 }
]

def eventLeaf17932 : Array AnnotatedEvent := #[
  { event := event286912
    frameStart := 0 },
  { event := event286913
    frameStart := 0 },
  { event := event286914
    frameStart := 0 },
  { event := event286915
    frameStart := 0 },
  { event := event286916
    frameStart := 0 },
  { event := event286917
    frameStart := 0 },
  { event := event286918
    frameStart := 0 },
  { event := event286919
    frameStart := 0 },
  { event := event286920
    frameStart := 0 },
  { event := event286921
    frameStart := 0 },
  { event := event286922
    frameStart := 0 },
  { event := event286923
    frameStart := 0 },
  { event := event286924
    frameStart := 0 },
  { event := event286925
    frameStart := 0 },
  { event := event286926
    frameStart := 0 },
  { event := event286927
    frameStart := 0 }
]

def eventLeaf17933 : Array AnnotatedEvent := #[
  { event := event286928
    frameStart := 0 },
  { event := event286929
    frameStart := 0 },
  { event := event286930
    frameStart := 0 },
  { event := event286931
    frameStart := 0 },
  { event := event286932
    frameStart := 0 },
  { event := event286933
    frameStart := 0 },
  { event := event286934
    frameStart := 0 },
  { event := event286935
    frameStart := 0 },
  { event := event286936
    frameStart := 0 },
  { event := event286937
    frameStart := 0 },
  { event := event286938
    frameStart := 0 },
  { event := event286939
    frameStart := 0 },
  { event := event286940
    frameStart := 0 },
  { event := event286941
    frameStart := 0 },
  { event := event286942
    frameStart := 0 },
  { event := event286943
    frameStart := 0 }
]

def eventLeaf17934 : Array AnnotatedEvent := #[
  { event := event286944
    frameStart := 0 },
  { event := event286945
    frameStart := 0 },
  { event := event286946
    frameStart := 0 },
  { event := event286947
    frameStart := 0 },
  { event := event286948
    frameStart := 0 },
  { event := event286949
    frameStart := 0 },
  { event := event286950
    frameStart := 0 },
  { event := event286951
    frameStart := 0 },
  { event := event286952
    frameStart := 0 },
  { event := event286953
    frameStart := 0 },
  { event := event286954
    frameStart := 0 },
  { event := event286955
    frameStart := 0 },
  { event := event286956
    frameStart := 0 },
  { event := event286957
    frameStart := 0 },
  { event := event286958
    frameStart := 0 },
  { event := event286959
    frameStart := 0 }
]

def eventLeaf17935 : Array AnnotatedEvent := #[
  { event := event286960
    frameStart := 0 },
  { event := event286961
    frameStart := 0 },
  { event := event286962
    frameStart := 0 },
  { event := event286963
    frameStart := 0 },
  { event := event286964
    frameStart := 0 },
  { event := event286965
    frameStart := 0 },
  { event := event286966
    frameStart := 0 },
  { event := event286967
    frameStart := 0 },
  { event := event286968
    frameStart := 0 },
  { event := event286969
    frameStart := 0 },
  { event := event286970
    frameStart := 0 },
  { event := event286971
    frameStart := 0 },
  { event := event286972
    frameStart := 0 },
  { event := event286973
    frameStart := 0 },
  { event := event286974
    frameStart := 0 },
  { event := event286975
    frameStart := 0 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events1120
