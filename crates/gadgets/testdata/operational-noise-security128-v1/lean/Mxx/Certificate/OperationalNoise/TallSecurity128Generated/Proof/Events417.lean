import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events417

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event106752 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event106753 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event106754 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event106755 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 106754

def event106756 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 106752

def event106757 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 106755 .coefficient) (.value (.predecessor 1 106756 .coefficient)))

def event106758 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event106759 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5756⟩⟩) 0 ⟨392⟩ 106758

def event106760 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5756⟩⟩) 1 ⟨5754⟩ 106750

def event106761 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5756⟩⟩) (.sum [.predecessor 0 106759 .coefficient, .predecessor 1 106760 .coefficient])

def event106762 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5756⟩⟩) (.finite 655353)

def event106763 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5766⟩⟩) 0 ⟨5756⟩ 106762

def event106764 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5766⟩⟩) 1 ⟨5426⟩ 106748

def event106765 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5766⟩⟩) (.identity (.predecessor 1 106764 .coefficient))

def event106766 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5766⟩⟩) (.finite 655360)

def event106767 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39818⟩⟩) 0 ⟨5766⟩ 106766

def event106768 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39818⟩⟩) (.authority (.programFamilyFact))

def exact106769RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨39818⟩⟩], []⟩, (1)⟩]

theorem exact106769RawTermsValid :
    exact106769RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106769 : Event := .resultExact (⟨.program ⟨257⟩, ⟨39818⟩⟩) exact106769RawTerms (.finite 46) 106768 .exactZero (none)

def event106770 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14196⟩⟩) 0 ⟨5766⟩ 106766

def event106771 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14196⟩⟩) (.authority (.programFamilyFact))

def exact106772RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14196⟩⟩], []⟩, (1)⟩]

theorem exact106772RawTermsValid :
    exact106772RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106772 : Event := .resultExact (⟨.program ⟨257⟩, ⟨14196⟩⟩) exact106772RawTerms (.finite 46) 106771 .exactZero (none)

def event106773 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39819⟩⟩) 0 ⟨14196⟩ 106772

def event106774 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39819⟩⟩) 1 ⟨39818⟩ 106769

def event106775 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39819⟩⟩) (.product (.predecessor 0 106773 .coefficient) (.predecessor 1 106774 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event106776 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨39819⟩⟩, .operator (⟨106772, 0⟩, ⟨106769, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨14196⟩⟩, ⟨.program ⟨257⟩, ⟨39818⟩⟩], []⟩, (1)⟩)

def exact106777RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14196⟩⟩, ⟨.program ⟨257⟩, ⟨39818⟩⟩], []⟩, (1)⟩]

theorem exact106777RawTermsValid :
    exact106777RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106777 : Event := .resultExact (⟨.program ⟨257⟩, ⟨39819⟩⟩) exact106777RawTerms (.finite 2116) 106775 .exactZero (none)

def event106778 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39820⟩⟩) 0 ⟨39819⟩ 106777

def event106779 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39820⟩⟩) (.identity (.predecessor 0 106778 .coefficient))

def event106780 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨39820⟩⟩) (.finite 2116)

def event106781 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41114⟩⟩) 0 ⟨39820⟩ 106780

def event106782 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41114⟩⟩) (.authority (.programFamilyFact))

def event106783 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨41114⟩⟩) (.finite 3720)

def event106784 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨7177⟩⟩) .missing

def event106785 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41115⟩⟩) 0 ⟨7177⟩ 106784

def event106786 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41115⟩⟩) 1 ⟨41114⟩ 106783

def event106787 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41115⟩⟩) (.authority (.operator))

def exact106788RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨41115⟩⟩]⟩, (1)⟩]

theorem exact106788RawTermsValid :
    exact106788RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106788 : Event := .resultExact (⟨.program ⟨257⟩, ⟨41115⟩⟩) exact106788RawTerms .large 106787 .exactZero (none)

def event106789 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41630⟩⟩) 0 ⟨41115⟩ 106788

def event106790 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41630⟩⟩) (.authority (.operator))

def exact106791RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨41630⟩⟩]⟩, (1)⟩]

theorem exact106791RawTermsValid :
    exact106791RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106791 : Event := .resultExact (⟨.program ⟨257⟩, ⟨41630⟩⟩) exact106791RawTerms (.finite 8192) 106790 .exactZero (none)

def event106792 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨136⟩⟩) (.authority (.operator))

def event106793 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨136⟩⟩) .exactZero

def event106794 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41390⟩⟩) 0 ⟨39820⟩ 106780

def event106795 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41390⟩⟩) 1 ⟨136⟩ 106793

def event106796 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41390⟩⟩) (.sum [.predecessor 0 106794 .coefficient, .predecessor 1 106795 .coefficient])

def event106797 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨41390⟩⟩) (.finite 2116)

def event106798 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41391⟩⟩) 0 ⟨41390⟩ 106797

def event106799 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41391⟩⟩) (.identity (.predecessor 0 106798 .coefficient))

def exact106800RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14196⟩⟩, ⟨.program ⟨257⟩, ⟨39818⟩⟩], []⟩, (1)⟩]

theorem exact106800RawTermsValid :
    exact106800RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106800 : Event := .resultExact (⟨.program ⟨257⟩, ⟨41391⟩⟩) exact106800RawTerms (.finite 2116) 106799 .exactZero (none)

def event106801 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6908⟩⟩) (.authority (.factStore))

def exact106802RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact106802RawTermsValid :
    exact106802RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106802 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6908⟩⟩) exact106802RawTerms .large 106801 .exactZero (none)

def event106803 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41392⟩⟩) 0 ⟨6908⟩ 106802

def event106804 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41392⟩⟩) 1 ⟨41391⟩ 106800

def event106805 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41392⟩⟩) (.product (.predecessor 0 106803 .coefficient) (.predecessor 1 106804 .coefficient) (⟨false, false, none, none, none⟩))

def event106806 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨41392⟩⟩, .operator (⟨106802, 0⟩, ⟨106800, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨14196⟩⟩, ⟨.program ⟨257⟩, ⟨39818⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact106807RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14196⟩⟩, ⟨.program ⟨257⟩, ⟨39818⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact106807RawTermsValid :
    exact106807RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106807 : Event := .resultExact (⟨.program ⟨257⟩, ⟨41392⟩⟩) exact106807RawTerms .large 106805 .exactZero (none)

def event106808 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨2370⟩⟩) (.authority (.operator))

def event106809 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨2370⟩⟩) (.finite 1)

def event106810 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7178⟩⟩) 0 ⟨7177⟩ 106784

def event106811 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7178⟩⟩) (.authority (.operator))

def exact106812RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7178⟩⟩]⟩, (1)⟩]

theorem exact106812RawTermsValid :
    exact106812RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106812 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7178⟩⟩) exact106812RawTerms .large 106811 .exactZero (none)

def event106813 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7282⟩⟩) 0 ⟨7178⟩ 106812

def event106814 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7282⟩⟩) (.identity (.predecessor 0 106813 .coefficient))

def exact106815RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7282⟩⟩]⟩, (1)⟩]

theorem exact106815RawTermsValid :
    exact106815RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106815 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7282⟩⟩) exact106815RawTerms .large 106814 .exactZero (none)

def event106816 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9556⟩⟩) 0 ⟨7282⟩ 106815

def event106817 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9556⟩⟩) (.authority (.operator))

def exact106818RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨9556⟩⟩]⟩, (1)⟩]

theorem exact106818RawTermsValid :
    exact106818RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106818 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9556⟩⟩) exact106818RawTerms (.finite 8192) 106817 .exactZero (none)

def event106819 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9557⟩⟩) 0 ⟨9556⟩ 106818

def event106820 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9557⟩⟩) 1 ⟨2370⟩ 106809

def event106821 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9557⟩⟩) (.scale (.predecessor 0 106819 .coefficient) (.value (.predecessor 1 106820 .coefficient)))

def exact106822RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨9556⟩⟩]⟩, (1)⟩]

theorem exact106822RawTermsValid :
    exact106822RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106822 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9557⟩⟩) exact106822RawTerms (.finite 8192) 106821 .exactZero (none)

def event106823 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7299⟩⟩) 0 ⟨7178⟩ 106812

def event106824 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7299⟩⟩) (.identity (.predecessor 0 106823 .coefficient))

def exact106825RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7299⟩⟩]⟩, (1)⟩]

theorem exact106825RawTermsValid :
    exact106825RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106825 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7299⟩⟩) exact106825RawTerms .large 106824 .exactZero (none)

def event106826 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9558⟩⟩) 0 ⟨7299⟩ 106825

def event106827 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9558⟩⟩) 1 ⟨9557⟩ 106822

def event106828 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9558⟩⟩) (.product (.predecessor 0 106826 .coefficient) (.predecessor 1 106827 .coefficient) (⟨false, false, none, none, none⟩))

def event106829 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨9558⟩⟩, .operator (⟨106825, 0⟩, ⟨106822, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩]⟩, (1)⟩)

def exact106830RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩]⟩, (1)⟩]

theorem exact106830RawTermsValid :
    exact106830RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106830 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9558⟩⟩) exact106830RawTerms .large 106828 .exactZero (none)

def event106831 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41393⟩⟩) 0 ⟨9558⟩ 106830

def event106832 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41393⟩⟩) 1 ⟨41392⟩ 106807

def event106833 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41393⟩⟩) (.sum [.predecessor 0 106831 .coefficient, .predecessor 1 106832 .coefficient])

def exact106834RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨14196⟩⟩, ⟨.program ⟨257⟩, ⟨39818⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact106834RawTermsValid :
    exact106834RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106834 : Event := .resultExact (⟨.program ⟨257⟩, ⟨41393⟩⟩) exact106834RawTerms .large 106833 .exactZero (none)

def event106835 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41633⟩⟩) 0 ⟨41393⟩ 106834

def event106836 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41633⟩⟩) 1 ⟨41630⟩ 106791

def event106837 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41633⟩⟩) (.product (.predecessor 0 106835 .coefficient) (.predecessor 1 106836 .coefficient) (⟨false, false, none, none, none⟩))

def event106838 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨41633⟩⟩, .operator (⟨106834, 0⟩, ⟨106791, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41630⟩⟩]⟩, (1)⟩)

def event106839 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨41633⟩⟩, .operator (⟨106834, 1⟩, ⟨106791, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨14196⟩⟩, ⟨.program ⟨257⟩, ⟨39818⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨41630⟩⟩]⟩, (-1)⟩)

def event106840 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨41633⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨14196⟩⟩, ⟨.program ⟨257⟩, ⟨39818⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨41630⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨41630⟩⟩) ⟨41115⟩ 106788)

def event106841 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨41633⟩⟩, .relation 106840 0, ⟨[⟨.program ⟨257⟩, ⟨14196⟩⟩, ⟨.program ⟨257⟩, ⟨39818⟩⟩], [⟨.program ⟨257⟩, ⟨41115⟩⟩]⟩, (-1)⟩)

def exact106842RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41630⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨14196⟩⟩, ⟨.program ⟨257⟩, ⟨39818⟩⟩], [⟨.program ⟨257⟩, ⟨41115⟩⟩]⟩, (-1)⟩]

theorem exact106842RawTermsValid :
    exact106842RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106842 : Event := .resultExact (⟨.program ⟨257⟩, ⟨41633⟩⟩) exact106842RawTerms .large 106837 .exactZero (none)

def event106843 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40116⟩⟩) 0 ⟨39820⟩ 106780

def event106844 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨40116⟩⟩) (.authority (.programFamilyFact))

def exact106845RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨40116⟩⟩], []⟩, (1)⟩]

theorem exact106845RawTermsValid :
    exact106845RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106845 : Event := .resultExact (⟨.program ⟨257⟩, ⟨40116⟩⟩) exact106845RawTerms (.finite 46) 106844 .exactZero (none)

def event106846 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40118⟩⟩) 0 ⟨6908⟩ 106802

def event106847 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40118⟩⟩) 1 ⟨40116⟩ 106845

def event106848 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨40118⟩⟩) (.product (.predecessor 0 106846 .coefficient) (.predecessor 1 106847 .coefficient) (⟨false, true, none, none, some 1⟩))

def event106849 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨40118⟩⟩, .operator (⟨106802, 0⟩, ⟨106845, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨40116⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact106850RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨40116⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact106850RawTermsValid :
    exact106850RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106850 : Event := .resultExact (⟨.program ⟨257⟩, ⟨40118⟩⟩) exact106850RawTerms .large 106848 .exactZero (none)

def event106851 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7193⟩⟩) 0 ⟨7177⟩ 106784

def event106852 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7193⟩⟩) (.authority (.operator))

def exact106853RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7193⟩⟩]⟩, (1)⟩]

theorem exact106853RawTermsValid :
    exact106853RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106853 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7193⟩⟩) exact106853RawTerms .large 106852 .exactZero (none)

def event106854 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40119⟩⟩) 0 ⟨7193⟩ 106853

def event106855 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40119⟩⟩) 1 ⟨40118⟩ 106850

def event106856 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨40119⟩⟩) (.sum [.predecessor 0 106854 .coefficient, .predecessor 1 106855 .coefficient])

def exact106857RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7193⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨40116⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact106857RawTermsValid :
    exact106857RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106857 : Event := .resultExact (⟨.program ⟨257⟩, ⟨40119⟩⟩) exact106857RawTerms .large 106856 .exactZero (none)

def event106858 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41634⟩⟩) 0 ⟨40119⟩ 106857

def event106859 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41634⟩⟩) 1 ⟨41633⟩ 106842

def event106860 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41634⟩⟩) (.sum [.predecessor 0 106858 .coefficient, .predecessor 1 106859 .coefficient])

def exact106861RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7193⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41630⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨14196⟩⟩, ⟨.program ⟨257⟩, ⟨39818⟩⟩], [⟨.program ⟨257⟩, ⟨41115⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨40116⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact106861RawTermsValid :
    exact106861RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106861 : Event := .resultExact (⟨.program ⟨257⟩, ⟨41634⟩⟩) exact106861RawTerms .large 106860 .exactZero (none)

def event106862 : Event := .preFoldPolynomial 106861 [⟨⟨[], [⟨.program ⟨257⟩, ⟨7193⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41630⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨14196⟩⟩, ⟨.program ⟨257⟩, ⟨39818⟩⟩], [⟨.program ⟨257⟩, ⟨41115⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨40116⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩] .exactZero none

def exact106863RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7193⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41630⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨14196⟩⟩, ⟨.program ⟨257⟩, ⟨39818⟩⟩], [⟨.program ⟨257⟩, ⟨41115⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨40116⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

def event106863 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨41634⟩⟩) 106862 exact106863RawTerms .large 106860 .exactZero (none)

def event106864 : Event := .specializationComputed (⟨.program ⟨257⟩, ⟨39820⟩⟩) ⟨⟨72⟩, ⟨51⟩, ⟨135⟩⟩ ⟨106698, 106864⟩

def event106865 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨40562⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40559⟩⟩]⟩) (1) 0 2 (.universal 106864 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40559⟩⟩]⟩) (none) 106863)

def event106866 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨40562⟩⟩, .relation 106865 0, ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7193⟩⟩]⟩, (1)⟩)

def event106867 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨40562⟩⟩, .relation 106865 1, ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41630⟩⟩]⟩, (-1)⟩)

def event106868 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨40562⟩⟩, .relation 106865 2, ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨14196⟩⟩, ⟨.program ⟨257⟩, ⟨39818⟩⟩], [⟨.program ⟨257⟩, ⟨41115⟩⟩]⟩, (1)⟩)

def event106869 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨40562⟩⟩, .relation 106865 3, ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨40116⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact106870RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7193⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41630⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨14196⟩⟩, ⟨.program ⟨257⟩, ⟨39818⟩⟩], [⟨.program ⟨257⟩, ⟨41115⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨40116⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact106870RawTermsValid :
    exact106870RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106870 : Event := .resultExact (⟨.program ⟨257⟩, ⟨40562⟩⟩) exact106870RawTerms .large 106694 (.finite 202072841853861888) (some (106696))

def event106871 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41632⟩⟩) 0 ⟨40562⟩ 106870

def event106872 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41632⟩⟩) 1 ⟨41631⟩ 106684

def event106873 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41632⟩⟩) (.sum [.predecessor 0 106871 .coefficient, .predecessor 1 106872 .coefficient])

def event106874 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨41632⟩⟩, .operator (⟨106870, 2⟩, ⟨106684, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨14196⟩⟩, ⟨.program ⟨257⟩, ⟨39818⟩⟩], [⟨.program ⟨257⟩, ⟨41115⟩⟩]⟩, (-1)⟩)

def event106875 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨41632⟩⟩, .operator (⟨106870, 1⟩, ⟨106684, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41630⟩⟩]⟩, (1)⟩)

def event106876 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41632⟩⟩) (.sum [.result 106870 .summary, .result 106684 .summary])

def exact106877RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7193⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨40116⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact106877RawTermsValid :
    exact106877RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106877 : Event := .resultExact (⟨.program ⟨257⟩, ⟨41632⟩⟩) exact106877RawTerms .large 106873 (.finite 2998218789909838430208) (some (106876))

def event106878 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42016⟩⟩) 0 ⟨41632⟩ 106877

def event106879 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42016⟩⟩) 1 ⟨42014⟩ 106600

def event106880 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42016⟩⟩) (.product (.predecessor 0 106878 .coefficient) (.predecessor 1 106879 .coefficient) (⟨false, false, none, none, none⟩))

def event106881 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42016⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨42014⟩⟩]⟩) [⟨.result 106600 .coefficient, false, none⟩])

def event106882 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42016⟩⟩) (.product (.result 106877 .summary) (.transfer 106881) (⟨false, false, none, none, none⟩))

def event106883 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨42016⟩⟩, .operator (⟨106877, 0⟩, ⟨106600, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨42014⟩⟩]⟩, (1)⟩)

def event106884 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨42016⟩⟩, .operator (⟨106877, 1⟩, ⟨106600, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨40116⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨42014⟩⟩]⟩, (-1)⟩)

def event106885 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨42016⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨40116⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨42014⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨42014⟩⟩) ⟨41270⟩ 106597)

def event106886 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨42016⟩⟩, .relation 106885 0, ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨40116⟩⟩], [⟨.program ⟨257⟩, ⟨41270⟩⟩]⟩, (-1)⟩)

def exact106887RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨42014⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨40116⟩⟩], [⟨.program ⟨257⟩, ⟨41270⟩⟩]⟩, (-1)⟩]

theorem exact106887RawTermsValid :
    exact106887RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106887 : Event := .resultExact (⟨.program ⟨257⟩, ⟨42016⟩⟩) exact106887RawTerms .large 106880 (.finite 32193129122288627115968346193920) (some (106882))

def event106888 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40876⟩⟩) 0 ⟨40117⟩ 4668

def event106889 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨40876⟩⟩) (.authority (.relationPreimageSource ⟨87⟩))

def exact106890RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨40876⟩⟩]⟩, (1)⟩]

theorem exact106890RawTermsValid :
    exact106890RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106890 : Event := .resultExact (⟨.program ⟨257⟩, ⟨40876⟩⟩) exact106890RawTerms (.finite 5647228698) 106889 .exactZero (none)

def event106891 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40878⟩⟩) 0 ⟨40876⟩ 106890

def event106892 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40878⟩⟩) 1 ⟨2370⟩ 4

def event106893 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨40878⟩⟩) (.scale (.predecessor 0 106891 .coefficient) (.value (.predecessor 1 106892 .coefficient)))

def exact106894RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨40876⟩⟩]⟩, (1)⟩]

theorem exact106894RawTermsValid :
    exact106894RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106894 : Event := .resultExact (⟨.program ⟨257⟩, ⟨40878⟩⟩) exact106894RawTerms (.finite 5647228698) 106893 .exactZero (none)

def event106895 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40879⟩⟩) 0 ⟨5770⟩ 105245

def event106896 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40879⟩⟩) 1 ⟨40878⟩ 106894

def event106897 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨40879⟩⟩) (.product (.predecessor 0 106895 .coefficient) (.predecessor 1 106896 .coefficient) (⟨false, false, none, none, none⟩))

def event106898 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨40879⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨40876⟩⟩]⟩) [⟨.result 106890 .coefficient, false, none⟩])

def event106899 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨40879⟩⟩) (.product (.result 105245 .summary) (.transfer 106898) (⟨false, false, none, none, none⟩))

def event106900 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨40879⟩⟩, .operator (⟨105245, 0⟩, ⟨106894, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40876⟩⟩]⟩, (1)⟩)

def event106901 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨40877⟩⟩)

def event106902 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event106903 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event106904 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5754⟩⟩) (.authority (.operator))

def event106905 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5754⟩⟩) (.finite 13)

def event106906 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event106907 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event106908 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event106909 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event106910 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 106909

def event106911 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 106907

def event106912 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 106910 .coefficient) (.value (.predecessor 1 106911 .coefficient)))

def event106913 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event106914 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5756⟩⟩) 0 ⟨392⟩ 106913

def event106915 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5756⟩⟩) 1 ⟨5754⟩ 106905

def event106916 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5756⟩⟩) (.sum [.predecessor 0 106914 .coefficient, .predecessor 1 106915 .coefficient])

def event106917 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5756⟩⟩) (.finite 655353)

def event106918 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5766⟩⟩) 0 ⟨5756⟩ 106917

def event106919 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5766⟩⟩) 1 ⟨5426⟩ 106903

def event106920 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5766⟩⟩) (.identity (.predecessor 1 106919 .coefficient))

def event106921 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5766⟩⟩) (.finite 655360)

def event106922 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39818⟩⟩) 0 ⟨5766⟩ 106921

def event106923 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39818⟩⟩) (.authority (.programFamilyFact))

def exact106924RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨39818⟩⟩], []⟩, (1)⟩]

theorem exact106924RawTermsValid :
    exact106924RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106924 : Event := .resultExact (⟨.program ⟨257⟩, ⟨39818⟩⟩) exact106924RawTerms (.finite 46) 106923 .exactZero (none)

def event106925 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14196⟩⟩) 0 ⟨5766⟩ 106921

def event106926 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14196⟩⟩) (.authority (.programFamilyFact))

def exact106927RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14196⟩⟩], []⟩, (1)⟩]

theorem exact106927RawTermsValid :
    exact106927RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106927 : Event := .resultExact (⟨.program ⟨257⟩, ⟨14196⟩⟩) exact106927RawTerms (.finite 46) 106926 .exactZero (none)

def event106928 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39819⟩⟩) 0 ⟨14196⟩ 106927

def event106929 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39819⟩⟩) 1 ⟨39818⟩ 106924

def event106930 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39819⟩⟩) (.product (.predecessor 0 106928 .coefficient) (.predecessor 1 106929 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event106931 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39819⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨14196⟩⟩, ⟨.program ⟨257⟩, ⟨39818⟩⟩], []⟩) [⟨.result 106927 .coefficient, true, some 1⟩, ⟨.result 106924 .coefficient, true, some 1⟩])

def event106932 : Event := .survivorFold (1) 106931

def exact106933RawTerms : List Term := []

theorem exact106933RawTermsValid :
    exact106933RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106933 : Event := .resultExact (⟨.program ⟨257⟩, ⟨39819⟩⟩) exact106933RawTerms (.finite 2116) 106930 (.finite 2116) (some (106931))

def event106934 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39820⟩⟩) 0 ⟨39819⟩ 106933

def event106935 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39820⟩⟩) (.identity (.predecessor 0 106934 .coefficient))

def event106936 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨39820⟩⟩) (.finite 2116)

def event106937 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40116⟩⟩) 0 ⟨39820⟩ 106936

def event106938 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨40116⟩⟩) (.authority (.programFamilyFact))

def exact106939RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨40116⟩⟩], []⟩, (1)⟩]

theorem exact106939RawTermsValid :
    exact106939RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106939 : Event := .resultExact (⟨.program ⟨257⟩, ⟨40116⟩⟩) exact106939RawTerms (.finite 46) 106938 .exactZero (none)

def event106940 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40117⟩⟩) 0 ⟨40116⟩ 106939

def event106941 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨40117⟩⟩) (.identity (.predecessor 0 106940 .coefficient))

def event106942 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨40117⟩⟩) (.finite 46)

def event106943 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40876⟩⟩) 0 ⟨40117⟩ 106942

def event106944 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨40876⟩⟩) (.authority (.relationPreimageSource ⟨87⟩))

def exact106945RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨40876⟩⟩]⟩, (1)⟩]

theorem exact106945RawTermsValid :
    exact106945RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106945 : Event := .resultExact (⟨.program ⟨257⟩, ⟨40876⟩⟩) exact106945RawTerms (.finite 5647228698) 106944 .exactZero (none)

def event106946 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35⟩⟩) (.authority (.operator))

def exact106947RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩]⟩, (1)⟩]

theorem exact106947RawTermsValid :
    exact106947RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106947 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35⟩⟩) exact106947RawTerms .large 106946 .exactZero (none)

def event106948 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40877⟩⟩) 0 ⟨35⟩ 106947

def event106949 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40877⟩⟩) 1 ⟨40876⟩ 106945

def event106950 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨40877⟩⟩) (.product (.predecessor 0 106948 .coefficient) (.predecessor 1 106949 .coefficient) (⟨false, false, none, none, none⟩))

def event106951 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨40877⟩⟩, .operator (⟨106947, 0⟩, ⟨106945, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40876⟩⟩]⟩, (1)⟩)

def exact106952RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40876⟩⟩]⟩, (1)⟩]

theorem exact106952RawTermsValid :
    exact106952RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106952 : Event := .resultExact (⟨.program ⟨257⟩, ⟨40877⟩⟩) exact106952RawTerms .large 106950 .exactZero (none)

def event106953 : Event := .preFoldPolynomial 106952 [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40876⟩⟩]⟩, (1)⟩] .exactZero none

def exact106954RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40876⟩⟩]⟩, (1)⟩]

def event106954 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨40877⟩⟩) 106953 exact106954RawTerms .large 106950 .exactZero (none)

def event106955 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨42018⟩⟩)

def event106956 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event106957 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event106958 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5754⟩⟩) (.authority (.operator))

def event106959 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5754⟩⟩) (.finite 13)

def event106960 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event106961 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event106962 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event106963 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event106964 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 106963

def event106965 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 106961

def event106966 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 106964 .coefficient) (.value (.predecessor 1 106965 .coefficient)))

def event106967 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event106968 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5756⟩⟩) 0 ⟨392⟩ 106967

def event106969 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5756⟩⟩) 1 ⟨5754⟩ 106959

def event106970 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5756⟩⟩) (.sum [.predecessor 0 106968 .coefficient, .predecessor 1 106969 .coefficient])

def event106971 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5756⟩⟩) (.finite 655353)

def event106972 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5766⟩⟩) 0 ⟨5756⟩ 106971

def event106973 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5766⟩⟩) 1 ⟨5426⟩ 106957

def event106974 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5766⟩⟩) (.identity (.predecessor 1 106973 .coefficient))

def event106975 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5766⟩⟩) (.finite 655360)

def event106976 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39818⟩⟩) 0 ⟨5766⟩ 106975

def event106977 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39818⟩⟩) (.authority (.programFamilyFact))

def exact106978RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨39818⟩⟩], []⟩, (1)⟩]

theorem exact106978RawTermsValid :
    exact106978RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106978 : Event := .resultExact (⟨.program ⟨257⟩, ⟨39818⟩⟩) exact106978RawTerms (.finite 46) 106977 .exactZero (none)

def event106979 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14196⟩⟩) 0 ⟨5766⟩ 106975

def event106980 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14196⟩⟩) (.authority (.programFamilyFact))

def exact106981RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14196⟩⟩], []⟩, (1)⟩]

theorem exact106981RawTermsValid :
    exact106981RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106981 : Event := .resultExact (⟨.program ⟨257⟩, ⟨14196⟩⟩) exact106981RawTerms (.finite 46) 106980 .exactZero (none)

def event106982 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39819⟩⟩) 0 ⟨14196⟩ 106981

def event106983 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39819⟩⟩) 1 ⟨39818⟩ 106978

def event106984 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39819⟩⟩) (.product (.predecessor 0 106982 .coefficient) (.predecessor 1 106983 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event106985 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨39819⟩⟩, .operator (⟨106981, 0⟩, ⟨106978, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨14196⟩⟩, ⟨.program ⟨257⟩, ⟨39818⟩⟩], []⟩, (1)⟩)

def exact106986RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14196⟩⟩, ⟨.program ⟨257⟩, ⟨39818⟩⟩], []⟩, (1)⟩]

theorem exact106986RawTermsValid :
    exact106986RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106986 : Event := .resultExact (⟨.program ⟨257⟩, ⟨39819⟩⟩) exact106986RawTerms (.finite 2116) 106984 .exactZero (none)

def event106987 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39820⟩⟩) 0 ⟨39819⟩ 106986

def event106988 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39820⟩⟩) (.identity (.predecessor 0 106987 .coefficient))

def event106989 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨39820⟩⟩) (.finite 2116)

def event106990 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40116⟩⟩) 0 ⟨39820⟩ 106989

def event106991 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨40116⟩⟩) (.authority (.programFamilyFact))

def exact106992RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨40116⟩⟩], []⟩, (1)⟩]

theorem exact106992RawTermsValid :
    exact106992RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106992 : Event := .resultExact (⟨.program ⟨257⟩, ⟨40116⟩⟩) exact106992RawTerms (.finite 46) 106991 .exactZero (none)

def event106993 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40117⟩⟩) 0 ⟨40116⟩ 106992

def event106994 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨40117⟩⟩) (.identity (.predecessor 0 106993 .coefficient))

def event106995 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨40117⟩⟩) (.finite 46)

def event106996 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41268⟩⟩) 0 ⟨40117⟩ 106995

def event106997 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41268⟩⟩) (.authority (.programFamilyFact))

def event106998 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨41268⟩⟩) (.finite 3720)

def event106999 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨7177⟩⟩) .missing

def event107000 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41270⟩⟩) 0 ⟨7177⟩ 106999

def event107001 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41270⟩⟩) 1 ⟨41268⟩ 106998

def event107002 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41270⟩⟩) (.authority (.operator))

def exact107003RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨41270⟩⟩]⟩, (1)⟩]

theorem exact107003RawTermsValid :
    exact107003RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event107003 : Event := .resultExact (⟨.program ⟨257⟩, ⟨41270⟩⟩) exact107003RawTerms .large 107002 .exactZero (none)

def event107004 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42014⟩⟩) 0 ⟨41270⟩ 107003

def event107005 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42014⟩⟩) (.authority (.operator))

def exact107006RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨42014⟩⟩]⟩, (1)⟩]

theorem exact107006RawTermsValid :
    exact107006RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event107006 : Event := .resultExact (⟨.program ⟨257⟩, ⟨42014⟩⟩) exact107006RawTerms (.finite 8192) 107005 .exactZero (none)

def event107007 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨136⟩⟩) (.authority (.operator))

def eventLeaf6672 : Array AnnotatedEvent := #[
  { event := event106752
    frameStart := 106746 },
  { event := event106753
    frameStart := 106746 },
  { event := event106754
    frameStart := 106746 },
  { event := event106755
    frameStart := 106746 },
  { event := event106756
    frameStart := 106746 },
  { event := event106757
    frameStart := 106746 },
  { event := event106758
    frameStart := 106746 },
  { event := event106759
    frameStart := 106746 },
  { event := event106760
    frameStart := 106746 },
  { event := event106761
    frameStart := 106746 },
  { event := event106762
    frameStart := 106746 },
  { event := event106763
    frameStart := 106746 },
  { event := event106764
    frameStart := 106746 },
  { event := event106765
    frameStart := 106746 },
  { event := event106766
    frameStart := 106746 },
  { event := event106767
    frameStart := 106746 }
]

def eventLeaf6673 : Array AnnotatedEvent := #[
  { event := event106768
    frameStart := 106746 },
  { event := event106769
    frameStart := 106746 },
  { event := event106770
    frameStart := 106746 },
  { event := event106771
    frameStart := 106746 },
  { event := event106772
    frameStart := 106746 },
  { event := event106773
    frameStart := 106746 },
  { event := event106774
    frameStart := 106746 },
  { event := event106775
    frameStart := 106746 },
  { event := event106776
    frameStart := 106746 },
  { event := event106777
    frameStart := 106746 },
  { event := event106778
    frameStart := 106746 },
  { event := event106779
    frameStart := 106746 },
  { event := event106780
    frameStart := 106746 },
  { event := event106781
    frameStart := 106746 },
  { event := event106782
    frameStart := 106746 },
  { event := event106783
    frameStart := 106746 }
]

def eventLeaf6674 : Array AnnotatedEvent := #[
  { event := event106784
    frameStart := 106746 },
  { event := event106785
    frameStart := 106746 },
  { event := event106786
    frameStart := 106746 },
  { event := event106787
    frameStart := 106746 },
  { event := event106788
    frameStart := 106746 },
  { event := event106789
    frameStart := 106746 },
  { event := event106790
    frameStart := 106746 },
  { event := event106791
    frameStart := 106746 },
  { event := event106792
    frameStart := 106746 },
  { event := event106793
    frameStart := 106746 },
  { event := event106794
    frameStart := 106746 },
  { event := event106795
    frameStart := 106746 },
  { event := event106796
    frameStart := 106746 },
  { event := event106797
    frameStart := 106746 },
  { event := event106798
    frameStart := 106746 },
  { event := event106799
    frameStart := 106746 }
]

def eventLeaf6675 : Array AnnotatedEvent := #[
  { event := event106800
    frameStart := 106746 },
  { event := event106801
    frameStart := 106746 },
  { event := event106802
    frameStart := 106746 },
  { event := event106803
    frameStart := 106746 },
  { event := event106804
    frameStart := 106746 },
  { event := event106805
    frameStart := 106746 },
  { event := event106806
    frameStart := 106746 },
  { event := event106807
    frameStart := 106746 },
  { event := event106808
    frameStart := 106746 },
  { event := event106809
    frameStart := 106746 },
  { event := event106810
    frameStart := 106746 },
  { event := event106811
    frameStart := 106746 },
  { event := event106812
    frameStart := 106746 },
  { event := event106813
    frameStart := 106746 },
  { event := event106814
    frameStart := 106746 },
  { event := event106815
    frameStart := 106746 }
]

def eventLeaf6676 : Array AnnotatedEvent := #[
  { event := event106816
    frameStart := 106746 },
  { event := event106817
    frameStart := 106746 },
  { event := event106818
    frameStart := 106746 },
  { event := event106819
    frameStart := 106746 },
  { event := event106820
    frameStart := 106746 },
  { event := event106821
    frameStart := 106746 },
  { event := event106822
    frameStart := 106746 },
  { event := event106823
    frameStart := 106746 },
  { event := event106824
    frameStart := 106746 },
  { event := event106825
    frameStart := 106746 },
  { event := event106826
    frameStart := 106746 },
  { event := event106827
    frameStart := 106746 },
  { event := event106828
    frameStart := 106746 },
  { event := event106829
    frameStart := 106746 },
  { event := event106830
    frameStart := 106746 },
  { event := event106831
    frameStart := 106746 }
]

def eventLeaf6677 : Array AnnotatedEvent := #[
  { event := event106832
    frameStart := 106746 },
  { event := event106833
    frameStart := 106746 },
  { event := event106834
    frameStart := 106746 },
  { event := event106835
    frameStart := 106746 },
  { event := event106836
    frameStart := 106746 },
  { event := event106837
    frameStart := 106746 },
  { event := event106838
    frameStart := 106746 },
  { event := event106839
    frameStart := 106746 },
  { event := event106840
    frameStart := 106746 },
  { event := event106841
    frameStart := 106746 },
  { event := event106842
    frameStart := 106746 },
  { event := event106843
    frameStart := 106746 },
  { event := event106844
    frameStart := 106746 },
  { event := event106845
    frameStart := 106746 },
  { event := event106846
    frameStart := 106746 },
  { event := event106847
    frameStart := 106746 }
]

def eventLeaf6678 : Array AnnotatedEvent := #[
  { event := event106848
    frameStart := 106746 },
  { event := event106849
    frameStart := 106746 },
  { event := event106850
    frameStart := 106746 },
  { event := event106851
    frameStart := 106746 },
  { event := event106852
    frameStart := 106746 },
  { event := event106853
    frameStart := 106746 },
  { event := event106854
    frameStart := 106746 },
  { event := event106855
    frameStart := 106746 },
  { event := event106856
    frameStart := 106746 },
  { event := event106857
    frameStart := 106746 },
  { event := event106858
    frameStart := 106746 },
  { event := event106859
    frameStart := 106746 },
  { event := event106860
    frameStart := 106746 },
  { event := event106861
    frameStart := 106746 },
  { event := event106862
    frameStart := 106746 },
  { event := event106863
    frameStart := 106746 }
]

def eventLeaf6679 : Array AnnotatedEvent := #[
  { event := event106864
    frameStart := 0 },
  { event := event106865
    frameStart := 0 },
  { event := event106866
    frameStart := 0 },
  { event := event106867
    frameStart := 0 },
  { event := event106868
    frameStart := 0 },
  { event := event106869
    frameStart := 0 },
  { event := event106870
    frameStart := 0 },
  { event := event106871
    frameStart := 0 },
  { event := event106872
    frameStart := 0 },
  { event := event106873
    frameStart := 0 },
  { event := event106874
    frameStart := 0 },
  { event := event106875
    frameStart := 0 },
  { event := event106876
    frameStart := 0 },
  { event := event106877
    frameStart := 0 },
  { event := event106878
    frameStart := 0 },
  { event := event106879
    frameStart := 0 }
]

def eventLeaf6680 : Array AnnotatedEvent := #[
  { event := event106880
    frameStart := 0 },
  { event := event106881
    frameStart := 0 },
  { event := event106882
    frameStart := 0 },
  { event := event106883
    frameStart := 0 },
  { event := event106884
    frameStart := 0 },
  { event := event106885
    frameStart := 0 },
  { event := event106886
    frameStart := 0 },
  { event := event106887
    frameStart := 0 },
  { event := event106888
    frameStart := 0 },
  { event := event106889
    frameStart := 0 },
  { event := event106890
    frameStart := 0 },
  { event := event106891
    frameStart := 0 },
  { event := event106892
    frameStart := 0 },
  { event := event106893
    frameStart := 0 },
  { event := event106894
    frameStart := 0 },
  { event := event106895
    frameStart := 0 }
]

def eventLeaf6681 : Array AnnotatedEvent := #[
  { event := event106896
    frameStart := 0 },
  { event := event106897
    frameStart := 0 },
  { event := event106898
    frameStart := 0 },
  { event := event106899
    frameStart := 0 },
  { event := event106900
    frameStart := 0 },
  { event := event106901
    frameStart := 106901 },
  { event := event106902
    frameStart := 106901 },
  { event := event106903
    frameStart := 106901 },
  { event := event106904
    frameStart := 106901 },
  { event := event106905
    frameStart := 106901 },
  { event := event106906
    frameStart := 106901 },
  { event := event106907
    frameStart := 106901 },
  { event := event106908
    frameStart := 106901 },
  { event := event106909
    frameStart := 106901 },
  { event := event106910
    frameStart := 106901 },
  { event := event106911
    frameStart := 106901 }
]

def eventLeaf6682 : Array AnnotatedEvent := #[
  { event := event106912
    frameStart := 106901 },
  { event := event106913
    frameStart := 106901 },
  { event := event106914
    frameStart := 106901 },
  { event := event106915
    frameStart := 106901 },
  { event := event106916
    frameStart := 106901 },
  { event := event106917
    frameStart := 106901 },
  { event := event106918
    frameStart := 106901 },
  { event := event106919
    frameStart := 106901 },
  { event := event106920
    frameStart := 106901 },
  { event := event106921
    frameStart := 106901 },
  { event := event106922
    frameStart := 106901 },
  { event := event106923
    frameStart := 106901 },
  { event := event106924
    frameStart := 106901 },
  { event := event106925
    frameStart := 106901 },
  { event := event106926
    frameStart := 106901 },
  { event := event106927
    frameStart := 106901 }
]

def eventLeaf6683 : Array AnnotatedEvent := #[
  { event := event106928
    frameStart := 106901 },
  { event := event106929
    frameStart := 106901 },
  { event := event106930
    frameStart := 106901 },
  { event := event106931
    frameStart := 106901 },
  { event := event106932
    frameStart := 106901 },
  { event := event106933
    frameStart := 106901 },
  { event := event106934
    frameStart := 106901 },
  { event := event106935
    frameStart := 106901 },
  { event := event106936
    frameStart := 106901 },
  { event := event106937
    frameStart := 106901 },
  { event := event106938
    frameStart := 106901 },
  { event := event106939
    frameStart := 106901 },
  { event := event106940
    frameStart := 106901 },
  { event := event106941
    frameStart := 106901 },
  { event := event106942
    frameStart := 106901 },
  { event := event106943
    frameStart := 106901 }
]

def eventLeaf6684 : Array AnnotatedEvent := #[
  { event := event106944
    frameStart := 106901 },
  { event := event106945
    frameStart := 106901 },
  { event := event106946
    frameStart := 106901 },
  { event := event106947
    frameStart := 106901 },
  { event := event106948
    frameStart := 106901 },
  { event := event106949
    frameStart := 106901 },
  { event := event106950
    frameStart := 106901 },
  { event := event106951
    frameStart := 106901 },
  { event := event106952
    frameStart := 106901 },
  { event := event106953
    frameStart := 106901 },
  { event := event106954
    frameStart := 106901 },
  { event := event106955
    frameStart := 106955 },
  { event := event106956
    frameStart := 106955 },
  { event := event106957
    frameStart := 106955 },
  { event := event106958
    frameStart := 106955 },
  { event := event106959
    frameStart := 106955 }
]

def eventLeaf6685 : Array AnnotatedEvent := #[
  { event := event106960
    frameStart := 106955 },
  { event := event106961
    frameStart := 106955 },
  { event := event106962
    frameStart := 106955 },
  { event := event106963
    frameStart := 106955 },
  { event := event106964
    frameStart := 106955 },
  { event := event106965
    frameStart := 106955 },
  { event := event106966
    frameStart := 106955 },
  { event := event106967
    frameStart := 106955 },
  { event := event106968
    frameStart := 106955 },
  { event := event106969
    frameStart := 106955 },
  { event := event106970
    frameStart := 106955 },
  { event := event106971
    frameStart := 106955 },
  { event := event106972
    frameStart := 106955 },
  { event := event106973
    frameStart := 106955 },
  { event := event106974
    frameStart := 106955 },
  { event := event106975
    frameStart := 106955 }
]

def eventLeaf6686 : Array AnnotatedEvent := #[
  { event := event106976
    frameStart := 106955 },
  { event := event106977
    frameStart := 106955 },
  { event := event106978
    frameStart := 106955 },
  { event := event106979
    frameStart := 106955 },
  { event := event106980
    frameStart := 106955 },
  { event := event106981
    frameStart := 106955 },
  { event := event106982
    frameStart := 106955 },
  { event := event106983
    frameStart := 106955 },
  { event := event106984
    frameStart := 106955 },
  { event := event106985
    frameStart := 106955 },
  { event := event106986
    frameStart := 106955 },
  { event := event106987
    frameStart := 106955 },
  { event := event106988
    frameStart := 106955 },
  { event := event106989
    frameStart := 106955 },
  { event := event106990
    frameStart := 106955 },
  { event := event106991
    frameStart := 106955 }
]

def eventLeaf6687 : Array AnnotatedEvent := #[
  { event := event106992
    frameStart := 106955 },
  { event := event106993
    frameStart := 106955 },
  { event := event106994
    frameStart := 106955 },
  { event := event106995
    frameStart := 106955 },
  { event := event106996
    frameStart := 106955 },
  { event := event106997
    frameStart := 106955 },
  { event := event106998
    frameStart := 106955 },
  { event := event106999
    frameStart := 106955 },
  { event := event107000
    frameStart := 106955 },
  { event := event107001
    frameStart := 106955 },
  { event := event107002
    frameStart := 106955 },
  { event := event107003
    frameStart := 106955 },
  { event := event107004
    frameStart := 106955 },
  { event := event107005
    frameStart := 106955 },
  { event := event107006
    frameStart := 106955 },
  { event := event107007
    frameStart := 106955 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events417
