import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events558

open Mxx.Certificate.OperationalNoise
open CertificateABI

def exact142848RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15732⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact142848RawTermsValid :
    exact142848RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event142848 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15734⟩⟩) exact142848RawTerms .large 142846 .exactZero (none)

def event142849 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7179⟩⟩) 0 ⟨7177⟩ 142782

def event142850 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7179⟩⟩) (.authority (.operator))

def exact142851RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩]⟩, (1)⟩]

theorem exact142851RawTermsValid :
    exact142851RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event142851 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7179⟩⟩) exact142851RawTerms .large 142850 .exactZero (none)

def event142852 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15735⟩⟩) 0 ⟨7179⟩ 142851

def event142853 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15735⟩⟩) 1 ⟨15734⟩ 142848

def event142854 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15735⟩⟩) (.sum [.predecessor 0 142852 .coefficient, .predecessor 1 142853 .coefficient])

def exact142855RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15732⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact142855RawTermsValid :
    exact142855RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event142855 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15735⟩⟩) exact142855RawTerms .large 142854 .exactZero (none)

def event142856 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17286⟩⟩) 0 ⟨15735⟩ 142855

def event142857 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17286⟩⟩) 1 ⟨17285⟩ 142840

def event142858 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17286⟩⟩) (.sum [.predecessor 0 142856 .coefficient, .predecessor 1 142857 .coefficient])

def exact142859RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩, ⟨.program ⟨257⟩, ⟨17282⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨12276⟩⟩, ⟨.program ⟨257⟩, ⟨15306⟩⟩], [⟨.program ⟨257⟩, ⟨16807⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15732⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact142859RawTermsValid :
    exact142859RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event142859 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17286⟩⟩) exact142859RawTerms .large 142858 .exactZero (none)

def event142860 : Event := .preFoldPolynomial 142859 [⟨⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩, ⟨.program ⟨257⟩, ⟨17282⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨12276⟩⟩, ⟨.program ⟨257⟩, ⟨15306⟩⟩], [⟨.program ⟨257⟩, ⟨16807⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15732⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩] .exactZero none

def exact142861RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩, ⟨.program ⟨257⟩, ⟨17282⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨12276⟩⟩, ⟨.program ⟨257⟩, ⟨15306⟩⟩], [⟨.program ⟨257⟩, ⟨16807⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15732⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

def event142861 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨17286⟩⟩) 142860 exact142861RawTerms .large 142858 .exactZero (none)

def event142862 : Event := .specializationComputed (⟨.program ⟨257⟩, ⟨15308⟩⟩) ⟨⟨58⟩, ⟨36⟩, ⟨135⟩⟩ ⟨142696, 142862⟩

def event142863 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨16222⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16219⟩⟩]⟩) (1) 0 2 (.universal 142862 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16219⟩⟩]⟩) (none) 142861)

def event142864 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨16222⟩⟩, .relation 142863 0, ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7179⟩⟩]⟩, (1)⟩)

def event142865 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨16222⟩⟩, .relation 142863 1, ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩, ⟨.program ⟨257⟩, ⟨17282⟩⟩]⟩, (-1)⟩)

def event142866 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨16222⟩⟩, .relation 142863 2, ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨12276⟩⟩, ⟨.program ⟨257⟩, ⟨15306⟩⟩], [⟨.program ⟨257⟩, ⟨16807⟩⟩]⟩, (1)⟩)

def event142867 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨16222⟩⟩, .relation 142863 3, ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨15732⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact142868RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7179⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩, ⟨.program ⟨257⟩, ⟨17282⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨12276⟩⟩, ⟨.program ⟨257⟩, ⟨15306⟩⟩], [⟨.program ⟨257⟩, ⟨16807⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨15732⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact142868RawTermsValid :
    exact142868RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event142868 : Event := .resultExact (⟨.program ⟨257⟩, ⟨16222⟩⟩) exact142868RawTerms .large 142692 (.finite 202072841853861888) (some (142694))

def event142869 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17284⟩⟩) 0 ⟨16222⟩ 142868

def event142870 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17284⟩⟩) 1 ⟨17283⟩ 142682

def event142871 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17284⟩⟩) (.sum [.predecessor 0 142869 .coefficient, .predecessor 1 142870 .coefficient])

def event142872 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17284⟩⟩, .operator (⟨142868, 2⟩, ⟨142682, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨12276⟩⟩, ⟨.program ⟨257⟩, ⟨15306⟩⟩], [⟨.program ⟨257⟩, ⟨16807⟩⟩]⟩, (-1)⟩)

def event142873 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17284⟩⟩, .operator (⟨142868, 1⟩, ⟨142682, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩, ⟨.program ⟨257⟩, ⟨17282⟩⟩]⟩, (1)⟩)

def event142874 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17284⟩⟩) (.sum [.result 142868 .summary, .result 142682 .summary])

def exact142875RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7179⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨15732⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact142875RawTermsValid :
    exact142875RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event142875 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17284⟩⟩) exact142875RawTerms .large 142871 (.finite 2997816280693142192128) (some (142874))

def event142876 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17567⟩⟩) 0 ⟨17284⟩ 142875

def event142877 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17567⟩⟩) 1 ⟨17565⟩ 142598

def event142878 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17567⟩⟩) (.product (.predecessor 0 142876 .coefficient) (.predecessor 1 142877 .coefficient) (⟨false, false, none, none, none⟩))

def event142879 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17567⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨17565⟩⟩]⟩) [⟨.result 142598 .coefficient, false, none⟩])

def event142880 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17567⟩⟩) (.product (.result 142875 .summary) (.transfer 142879) (⟨false, false, none, none, none⟩))

def event142881 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17567⟩⟩, .operator (⟨142875, 0⟩, ⟨142598, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17565⟩⟩]⟩, (1)⟩)

def event142882 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17567⟩⟩, .operator (⟨142875, 1⟩, ⟨142598, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨15732⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17565⟩⟩]⟩, (-1)⟩)

def event142883 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨17567⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨15732⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17565⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨17565⟩⟩) ⟨16938⟩ 142595)

def event142884 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17567⟩⟩, .relation 142883 0, ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨15732⟩⟩], [⟨.program ⟨257⟩, ⟨16938⟩⟩]⟩, (-1)⟩)

def exact142885RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17565⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨15732⟩⟩], [⟨.program ⟨257⟩, ⟨16938⟩⟩]⟩, (-1)⟩]

theorem exact142885RawTermsValid :
    exact142885RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event142885 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17567⟩⟩) exact142885RawTerms .large 142878 (.finite 32188807212483504816668771614720) (some (142880))

def event142886 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16456⟩⟩) 0 ⟨15733⟩ 6486

def event142887 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨16456⟩⟩) (.authority (.relationPreimageSource ⟨57⟩))

def exact142888RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨16456⟩⟩]⟩, (1)⟩]

theorem exact142888RawTermsValid :
    exact142888RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event142888 : Event := .resultExact (⟨.program ⟨257⟩, ⟨16456⟩⟩) exact142888RawTerms (.finite 5647228698) 142887 .exactZero (none)

def event142889 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16458⟩⟩) 0 ⟨16456⟩ 142888

def event142890 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16458⟩⟩) 1 ⟨2370⟩ 4

def event142891 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨16458⟩⟩) (.scale (.predecessor 0 142889 .coefficient) (.value (.predecessor 1 142890 .coefficient)))

def exact142892RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨16456⟩⟩]⟩, (1)⟩]

theorem exact142892RawTermsValid :
    exact142892RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event142892 : Event := .resultExact (⟨.program ⟨257⟩, ⟨16458⟩⟩) exact142892RawTerms (.finite 5647228698) 142891 .exactZero (none)

def event142893 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16459⟩⟩) 0 ⟨5473⟩ 134495

def event142894 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16459⟩⟩) 1 ⟨16458⟩ 142892

def event142895 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨16459⟩⟩) (.product (.predecessor 0 142893 .coefficient) (.predecessor 1 142894 .coefficient) (⟨false, false, none, none, none⟩))

def event142896 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨16459⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨16456⟩⟩]⟩) [⟨.result 142888 .coefficient, false, none⟩])

def event142897 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨16459⟩⟩) (.product (.result 134495 .summary) (.transfer 142896) (⟨false, false, none, none, none⟩))

def event142898 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨16459⟩⟩, .operator (⟨134495, 0⟩, ⟨142892, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16456⟩⟩]⟩, (1)⟩)

def event142899 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨16457⟩⟩)

def event142900 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event142901 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event142902 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨455⟩⟩) (.authority (.operator))

def event142903 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨455⟩⟩) (.finite 11)

def event142904 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event142905 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event142906 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event142907 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event142908 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 142907

def event142909 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 142905

def event142910 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 142908 .coefficient) (.value (.predecessor 1 142909 .coefficient)))

def event142911 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event142912 : Event := .predecessor (⟨.program ⟨257⟩, ⟨457⟩⟩) 0 ⟨392⟩ 142911

def event142913 : Event := .predecessor (⟨.program ⟨257⟩, ⟨457⟩⟩) 1 ⟨455⟩ 142903

def event142914 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨457⟩⟩) (.sum [.predecessor 0 142912 .coefficient, .predecessor 1 142913 .coefficient])

def event142915 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨457⟩⟩) (.finite 655351)

def event142916 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5469⟩⟩) 0 ⟨457⟩ 142915

def event142917 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5469⟩⟩) 1 ⟨5426⟩ 142901

def event142918 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5469⟩⟩) (.identity (.predecessor 1 142917 .coefficient))

def event142919 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5469⟩⟩) (.finite 655360)

def event142920 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15306⟩⟩) 0 ⟨5469⟩ 142919

def event142921 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15306⟩⟩) (.authority (.programFamilyFact))

def exact142922RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15306⟩⟩], []⟩, (1)⟩]

theorem exact142922RawTermsValid :
    exact142922RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event142922 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15306⟩⟩) exact142922RawTerms (.finite 2) 142921 .exactZero (none)

def event142923 : Event := .predecessor (⟨.program ⟨257⟩, ⟨12276⟩⟩) 0 ⟨5469⟩ 142919

def event142924 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨12276⟩⟩) (.authority (.programFamilyFact))

def exact142925RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨12276⟩⟩], []⟩, (1)⟩]

theorem exact142925RawTermsValid :
    exact142925RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event142925 : Event := .resultExact (⟨.program ⟨257⟩, ⟨12276⟩⟩) exact142925RawTerms (.finite 2) 142924 .exactZero (none)

def event142926 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15307⟩⟩) 0 ⟨12276⟩ 142925

def event142927 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15307⟩⟩) 1 ⟨15306⟩ 142922

def event142928 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15307⟩⟩) (.product (.predecessor 0 142926 .coefficient) (.predecessor 1 142927 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event142929 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15307⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨12276⟩⟩, ⟨.program ⟨257⟩, ⟨15306⟩⟩], []⟩) [⟨.result 142925 .coefficient, true, some 1⟩, ⟨.result 142922 .coefficient, true, some 1⟩])

def event142930 : Event := .survivorFold (1) 142929

def exact142931RawTerms : List Term := []

theorem exact142931RawTermsValid :
    exact142931RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event142931 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15307⟩⟩) exact142931RawTerms (.finite 4) 142928 (.finite 4) (some (142929))

def event142932 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15308⟩⟩) 0 ⟨15307⟩ 142931

def event142933 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15308⟩⟩) (.identity (.predecessor 0 142932 .coefficient))

def event142934 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨15308⟩⟩) (.finite 4)

def event142935 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15732⟩⟩) 0 ⟨15308⟩ 142934

def event142936 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15732⟩⟩) (.authority (.programFamilyFact))

def exact142937RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15732⟩⟩], []⟩, (1)⟩]

theorem exact142937RawTermsValid :
    exact142937RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event142937 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15732⟩⟩) exact142937RawTerms (.finite 2) 142936 .exactZero (none)

def event142938 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15733⟩⟩) 0 ⟨15732⟩ 142937

def event142939 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15733⟩⟩) (.identity (.predecessor 0 142938 .coefficient))

def event142940 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨15733⟩⟩) (.finite 2)

def event142941 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16456⟩⟩) 0 ⟨15733⟩ 142940

def event142942 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨16456⟩⟩) (.authority (.relationPreimageSource ⟨57⟩))

def exact142943RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨16456⟩⟩]⟩, (1)⟩]

theorem exact142943RawTermsValid :
    exact142943RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event142943 : Event := .resultExact (⟨.program ⟨257⟩, ⟨16456⟩⟩) exact142943RawTerms (.finite 5647228698) 142942 .exactZero (none)

def event142944 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35⟩⟩) (.authority (.operator))

def exact142945RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩]⟩, (1)⟩]

theorem exact142945RawTermsValid :
    exact142945RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event142945 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35⟩⟩) exact142945RawTerms .large 142944 .exactZero (none)

def event142946 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16457⟩⟩) 0 ⟨35⟩ 142945

def event142947 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16457⟩⟩) 1 ⟨16456⟩ 142943

def event142948 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨16457⟩⟩) (.product (.predecessor 0 142946 .coefficient) (.predecessor 1 142947 .coefficient) (⟨false, false, none, none, none⟩))

def event142949 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨16457⟩⟩, .operator (⟨142945, 0⟩, ⟨142943, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16456⟩⟩]⟩, (1)⟩)

def exact142950RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16456⟩⟩]⟩, (1)⟩]

theorem exact142950RawTermsValid :
    exact142950RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event142950 : Event := .resultExact (⟨.program ⟨257⟩, ⟨16457⟩⟩) exact142950RawTerms .large 142948 .exactZero (none)

def event142951 : Event := .preFoldPolynomial 142950 [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16456⟩⟩]⟩, (1)⟩] .exactZero none

def exact142952RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16456⟩⟩]⟩, (1)⟩]

def event142952 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨16457⟩⟩) 142951 exact142952RawTerms .large 142948 .exactZero (none)

def event142953 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨17569⟩⟩)

def event142954 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event142955 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event142956 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨455⟩⟩) (.authority (.operator))

def event142957 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨455⟩⟩) (.finite 11)

def event142958 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event142959 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event142960 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event142961 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event142962 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 142961

def event142963 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 142959

def event142964 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 142962 .coefficient) (.value (.predecessor 1 142963 .coefficient)))

def event142965 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event142966 : Event := .predecessor (⟨.program ⟨257⟩, ⟨457⟩⟩) 0 ⟨392⟩ 142965

def event142967 : Event := .predecessor (⟨.program ⟨257⟩, ⟨457⟩⟩) 1 ⟨455⟩ 142957

def event142968 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨457⟩⟩) (.sum [.predecessor 0 142966 .coefficient, .predecessor 1 142967 .coefficient])

def event142969 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨457⟩⟩) (.finite 655351)

def event142970 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5469⟩⟩) 0 ⟨457⟩ 142969

def event142971 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5469⟩⟩) 1 ⟨5426⟩ 142955

def event142972 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5469⟩⟩) (.identity (.predecessor 1 142971 .coefficient))

def event142973 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5469⟩⟩) (.finite 655360)

def event142974 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15306⟩⟩) 0 ⟨5469⟩ 142973

def event142975 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15306⟩⟩) (.authority (.programFamilyFact))

def exact142976RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15306⟩⟩], []⟩, (1)⟩]

theorem exact142976RawTermsValid :
    exact142976RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event142976 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15306⟩⟩) exact142976RawTerms (.finite 2) 142975 .exactZero (none)

def event142977 : Event := .predecessor (⟨.program ⟨257⟩, ⟨12276⟩⟩) 0 ⟨5469⟩ 142973

def event142978 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨12276⟩⟩) (.authority (.programFamilyFact))

def exact142979RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨12276⟩⟩], []⟩, (1)⟩]

theorem exact142979RawTermsValid :
    exact142979RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event142979 : Event := .resultExact (⟨.program ⟨257⟩, ⟨12276⟩⟩) exact142979RawTerms (.finite 2) 142978 .exactZero (none)

def event142980 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15307⟩⟩) 0 ⟨12276⟩ 142979

def event142981 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15307⟩⟩) 1 ⟨15306⟩ 142976

def event142982 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15307⟩⟩) (.product (.predecessor 0 142980 .coefficient) (.predecessor 1 142981 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event142983 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨15307⟩⟩, .operator (⟨142979, 0⟩, ⟨142976, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨12276⟩⟩, ⟨.program ⟨257⟩, ⟨15306⟩⟩], []⟩, (1)⟩)

def exact142984RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨12276⟩⟩, ⟨.program ⟨257⟩, ⟨15306⟩⟩], []⟩, (1)⟩]

theorem exact142984RawTermsValid :
    exact142984RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event142984 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15307⟩⟩) exact142984RawTerms (.finite 4) 142982 .exactZero (none)

def event142985 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15308⟩⟩) 0 ⟨15307⟩ 142984

def event142986 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15308⟩⟩) (.identity (.predecessor 0 142985 .coefficient))

def event142987 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨15308⟩⟩) (.finite 4)

def event142988 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15732⟩⟩) 0 ⟨15308⟩ 142987

def event142989 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15732⟩⟩) (.authority (.programFamilyFact))

def exact142990RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15732⟩⟩], []⟩, (1)⟩]

theorem exact142990RawTermsValid :
    exact142990RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event142990 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15732⟩⟩) exact142990RawTerms (.finite 2) 142989 .exactZero (none)

def event142991 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15733⟩⟩) 0 ⟨15732⟩ 142990

def event142992 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15733⟩⟩) (.identity (.predecessor 0 142991 .coefficient))

def event142993 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨15733⟩⟩) (.finite 2)

def event142994 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16936⟩⟩) 0 ⟨15733⟩ 142993

def event142995 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨16936⟩⟩) (.authority (.programFamilyFact))

def event142996 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨16936⟩⟩) (.finite 3720)

def event142997 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨7177⟩⟩) .missing

def event142998 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16938⟩⟩) 0 ⟨7177⟩ 142997

def event142999 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16938⟩⟩) 1 ⟨16936⟩ 142996

def event143000 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨16938⟩⟩) (.authority (.operator))

def exact143001RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨16938⟩⟩]⟩, (1)⟩]

theorem exact143001RawTermsValid :
    exact143001RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event143001 : Event := .resultExact (⟨.program ⟨257⟩, ⟨16938⟩⟩) exact143001RawTerms .large 143000 .exactZero (none)

def event143002 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17565⟩⟩) 0 ⟨16938⟩ 143001

def event143003 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17565⟩⟩) (.authority (.operator))

def exact143004RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨17565⟩⟩]⟩, (1)⟩]

theorem exact143004RawTermsValid :
    exact143004RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event143004 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17565⟩⟩) exact143004RawTerms (.finite 8192) 143003 .exactZero (none)

def event143005 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨136⟩⟩) (.authority (.operator))

def event143006 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨136⟩⟩) .exactZero

def event143007 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17178⟩⟩) 0 ⟨15733⟩ 142993

def event143008 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17178⟩⟩) 1 ⟨136⟩ 143006

def event143009 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17178⟩⟩) (.sum [.predecessor 0 143007 .coefficient, .predecessor 1 143008 .coefficient])

def event143010 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨17178⟩⟩) (.finite 2)

def event143011 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17179⟩⟩) 0 ⟨17178⟩ 143010

def event143012 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17179⟩⟩) (.identity (.predecessor 0 143011 .coefficient))

def exact143013RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15732⟩⟩], []⟩, (1)⟩]

theorem exact143013RawTermsValid :
    exact143013RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event143013 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17179⟩⟩) exact143013RawTerms (.finite 2) 143012 .exactZero (none)

def event143014 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6908⟩⟩) (.authority (.factStore))

def exact143015RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact143015RawTermsValid :
    exact143015RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event143015 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6908⟩⟩) exact143015RawTerms .large 143014 .exactZero (none)

def event143016 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17180⟩⟩) 0 ⟨6908⟩ 143015

def event143017 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17180⟩⟩) 1 ⟨17179⟩ 143013

def event143018 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17180⟩⟩) (.product (.predecessor 0 143016 .coefficient) (.predecessor 1 143017 .coefficient) (⟨false, false, none, none, none⟩))

def event143019 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17180⟩⟩, .operator (⟨143015, 0⟩, ⟨143013, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨15732⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact143020RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15732⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact143020RawTermsValid :
    exact143020RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event143020 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17180⟩⟩) exact143020RawTerms .large 143018 .exactZero (none)

def event143021 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7179⟩⟩) 0 ⟨7177⟩ 142997

def event143022 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7179⟩⟩) (.authority (.operator))

def exact143023RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩]⟩, (1)⟩]

theorem exact143023RawTermsValid :
    exact143023RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event143023 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7179⟩⟩) exact143023RawTerms .large 143022 .exactZero (none)

def event143024 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17181⟩⟩) 0 ⟨7179⟩ 143023

def event143025 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17181⟩⟩) 1 ⟨17180⟩ 143020

def event143026 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17181⟩⟩) (.sum [.predecessor 0 143024 .coefficient, .predecessor 1 143025 .coefficient])

def exact143027RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15732⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact143027RawTermsValid :
    exact143027RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event143027 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17181⟩⟩) exact143027RawTerms .large 143026 .exactZero (none)

def event143028 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17566⟩⟩) 0 ⟨17181⟩ 143027

def event143029 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17566⟩⟩) 1 ⟨17565⟩ 143004

def event143030 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17566⟩⟩) (.product (.predecessor 0 143028 .coefficient) (.predecessor 1 143029 .coefficient) (⟨false, false, none, none, none⟩))

def event143031 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17566⟩⟩, .operator (⟨143027, 0⟩, ⟨143004, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17565⟩⟩]⟩, (1)⟩)

def event143032 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17566⟩⟩, .operator (⟨143027, 1⟩, ⟨143004, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨15732⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17565⟩⟩]⟩, (-1)⟩)

def event143033 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨17566⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨15732⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17565⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨17565⟩⟩) ⟨16938⟩ 143001)

def event143034 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17566⟩⟩, .relation 143033 0, ⟨[⟨.program ⟨257⟩, ⟨15732⟩⟩], [⟨.program ⟨257⟩, ⟨16938⟩⟩]⟩, (-1)⟩)

def exact143035RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17565⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15732⟩⟩], [⟨.program ⟨257⟩, ⟨16938⟩⟩]⟩, (-1)⟩]

theorem exact143035RawTermsValid :
    exact143035RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event143035 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17566⟩⟩) exact143035RawTerms .large 143030 .exactZero (none)

def event143036 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15923⟩⟩) 0 ⟨15733⟩ 142993

def event143037 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15923⟩⟩) (.authority (.programFamilyFact))

def exact143038RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15923⟩⟩], []⟩, (1)⟩]

theorem exact143038RawTermsValid :
    exact143038RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event143038 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15923⟩⟩) exact143038RawTerms (.finite 43) 143037 .exactZero (none)

def event143039 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15924⟩⟩) 0 ⟨6908⟩ 143015

def event143040 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15924⟩⟩) 1 ⟨15923⟩ 143038

def event143041 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15924⟩⟩) (.product (.predecessor 0 143039 .coefficient) (.predecessor 1 143040 .coefficient) (⟨false, true, none, none, some 1⟩))

def event143042 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨15924⟩⟩, .operator (⟨143015, 0⟩, ⟨143038, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨15923⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact143043RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15923⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact143043RawTermsValid :
    exact143043RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event143043 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15924⟩⟩) exact143043RawTerms .large 143041 .exactZero (none)

def event143044 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7198⟩⟩) 0 ⟨7177⟩ 142997

def event143045 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7198⟩⟩) (.authority (.operator))

def exact143046RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩]

theorem exact143046RawTermsValid :
    exact143046RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event143046 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7198⟩⟩) exact143046RawTerms .large 143045 .exactZero (none)

def event143047 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15925⟩⟩) 0 ⟨7198⟩ 143046

def event143048 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15925⟩⟩) 1 ⟨15924⟩ 143043

def event143049 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15925⟩⟩) (.sum [.predecessor 0 143047 .coefficient, .predecessor 1 143048 .coefficient])

def exact143050RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15923⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact143050RawTermsValid :
    exact143050RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event143050 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15925⟩⟩) exact143050RawTerms .large 143049 .exactZero (none)

def event143051 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17569⟩⟩) 0 ⟨15925⟩ 143050

def event143052 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17569⟩⟩) 1 ⟨17566⟩ 143035

def event143053 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17569⟩⟩) (.sum [.predecessor 0 143051 .coefficient, .predecessor 1 143052 .coefficient])

def exact143054RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17565⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15732⟩⟩], [⟨.program ⟨257⟩, ⟨16938⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15923⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact143054RawTermsValid :
    exact143054RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event143054 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17569⟩⟩) exact143054RawTerms .large 143053 .exactZero (none)

def event143055 : Event := .preFoldPolynomial 143054 [⟨⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17565⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15732⟩⟩], [⟨.program ⟨257⟩, ⟨16938⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15923⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩] .exactZero none

def exact143056RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17565⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15732⟩⟩], [⟨.program ⟨257⟩, ⟨16938⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15923⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

def event143056 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨17569⟩⟩) 143055 exact143056RawTerms .large 143053 .exactZero (none)

def event143057 : Event := .specializationComputed (⟨.program ⟨257⟩, ⟨15733⟩⟩) ⟨⟨77⟩, ⟨57⟩, ⟨135⟩⟩ ⟨142899, 143057⟩

def event143058 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨16459⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16456⟩⟩]⟩) (1) 0 2 (.universal 143057 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16456⟩⟩]⟩) (none) 143056)

def event143059 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨16459⟩⟩, .relation 143058 1, ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩)

def event143060 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨16459⟩⟩, .relation 143058 0, ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17565⟩⟩]⟩, (-1)⟩)

def event143061 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨16459⟩⟩, .relation 143058 2, ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨15732⟩⟩], [⟨.program ⟨257⟩, ⟨16938⟩⟩]⟩, (1)⟩)

def event143062 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨16459⟩⟩, .relation 143058 3, ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨15923⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact143063RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17565⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨15732⟩⟩], [⟨.program ⟨257⟩, ⟨16938⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨15923⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact143063RawTermsValid :
    exact143063RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event143063 : Event := .resultExact (⟨.program ⟨257⟩, ⟨16459⟩⟩) exact143063RawTerms .large 142895 (.finite 202072841853861888) (some (142897))

def event143064 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17568⟩⟩) 0 ⟨16459⟩ 143063

def event143065 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17568⟩⟩) 1 ⟨17567⟩ 142885

def event143066 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17568⟩⟩) (.sum [.predecessor 0 143064 .coefficient, .predecessor 1 143065 .coefficient])

def event143067 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17568⟩⟩, .operator (⟨143063, 0⟩, ⟨142885, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17565⟩⟩]⟩, (1)⟩)

def event143068 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17568⟩⟩, .operator (⟨143063, 2⟩, ⟨142885, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨15732⟩⟩], [⟨.program ⟨257⟩, ⟨16938⟩⟩]⟩, (-1)⟩)

def event143069 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17568⟩⟩) (.sum [.result 143063 .summary, .result 142885 .summary])

def exact143070RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨15923⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact143070RawTermsValid :
    exact143070RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event143070 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17568⟩⟩) exact143070RawTerms .large 143066 (.finite 32188807212483706889510625476608) (some (143069))

def event143071 : Event := .predecessor (⟨.program ⟨257⟩, ⟨20439⟩⟩) 0 ⟨17568⟩ 143070

def event143072 : Event := .predecessor (⟨.program ⟨257⟩, ⟨20439⟩⟩) 1 ⟨20438⟩ 142588

def event143073 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨20439⟩⟩) (.sum [.predecessor 0 143071 .coefficient, .predecessor 1 143072 .coefficient])

def event143074 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨20439⟩⟩) (.sum [.result 143070 .summary, .result 142588 .summary])

def exact143075RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨15923⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨18733⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact143075RawTermsValid :
    exact143075RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event143075 : Event := .resultExact (⟨.program ⟨257⟩, ⟨20439⟩⟩) exact143075RawTerms .large 143073 (.finite 64377712650190257467641695830016) (some (143074))

def event143076 : Event := .predecessor (⟨.program ⟨257⟩, ⟨23659⟩⟩) 0 ⟨20439⟩ 143075

def event143077 : Event := .predecessor (⟨.program ⟨257⟩, ⟨23659⟩⟩) 1 ⟨23658⟩ 142106

def event143078 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨23659⟩⟩) (.sum [.predecessor 0 143076 .coefficient, .predecessor 1 143077 .coefficient])

def event143079 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨23659⟩⟩) (.sum [.result 143075 .summary, .result 142106 .summary])

def exact143080RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨15923⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨18733⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨21953⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact143080RawTermsValid :
    exact143080RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event143080 : Event := .resultExact (⟨.program ⟨257⟩, ⟨23659⟩⟩) exact143080RawTerms .large 143078 (.finite 96566716313119651734393211060224) (some (143079))

def event143081 : Event := .predecessor (⟨.program ⟨257⟩, ⟨33679⟩⟩) 0 ⟨23659⟩ 143080

def event143082 : Event := .predecessor (⟨.program ⟨257⟩, ⟨33679⟩⟩) 1 ⟨33678⟩ 141624

def event143083 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨33679⟩⟩) (.sum [.predecessor 0 143081 .coefficient, .predecessor 1 143082 .coefficient])

def event143084 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨33679⟩⟩) (.sum [.result 143080 .summary, .result 141624 .summary])

def exact143085RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨15923⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨18733⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨21953⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨31973⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact143085RawTermsValid :
    exact143085RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event143085 : Event := .resultExact (⟨.program ⟨257⟩, ⟨33679⟩⟩) exact143085RawTerms .large 143083 (.finite 128755916426494733378385616044032) (some (143084))

def event143086 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52739⟩⟩) 0 ⟨33679⟩ 143085

def event143087 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52739⟩⟩) 1 ⟨52738⟩ 141142

def event143088 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨52739⟩⟩) (.sum [.predecessor 0 143086 .coefficient, .predecessor 1 143087 .coefficient])

def event143089 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨52739⟩⟩) (.sum [.result 143085 .summary, .result 141142 .summary])

def exact143090RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨15923⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨18733⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨21953⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨31973⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨51028⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact143090RawTermsValid :
    exact143090RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event143090 : Event := .resultExact (⟨.program ⟨257⟩, ⟨52739⟩⟩) exact143090RawTerms .large 143088 (.finite 160945509440761189776859800535040) (some (143089))

def event143091 : Event := .predecessor (⟨.program ⟨257⟩, ⟨55719⟩⟩) 0 ⟨52739⟩ 143090

def event143092 : Event := .predecessor (⟨.program ⟨257⟩, ⟨55719⟩⟩) 1 ⟨55718⟩ 140660

def event143093 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨55719⟩⟩) (.sum [.predecessor 0 143091 .coefficient, .predecessor 1 143092 .coefficient])

def event143094 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨55719⟩⟩) (.sum [.result 143090 .summary, .result 140660 .summary])

def exact143095RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨15923⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨18733⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨21953⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨31973⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨51028⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨54008⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact143095RawTermsValid :
    exact143095RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event143095 : Event := .resultExact (⟨.program ⟨257⟩, ⟨55719⟩⟩) exact143095RawTerms .large 143093 (.finite 193135298905473333552574874779648) (some (143094))

def event143096 : Event := .predecessor (⟨.program ⟨257⟩, ⟨58699⟩⟩) 0 ⟨55719⟩ 143095

def event143097 : Event := .predecessor (⟨.program ⟨257⟩, ⟨58699⟩⟩) 1 ⟨58698⟩ 140178

def event143098 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨58699⟩⟩) (.sum [.predecessor 0 143096 .coefficient, .predecessor 1 143097 .coefficient])

def event143099 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨58699⟩⟩) (.sum [.result 143095 .summary, .result 140178 .summary])

def exact143100RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨15923⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨18733⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨21953⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨31973⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨51028⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨54008⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨56988⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact143100RawTermsValid :
    exact143100RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event143100 : Event := .resultExact (⟨.program ⟨257⟩, ⟨58699⟩⟩) exact143100RawTerms .large 143098 (.finite 225325481271076852082771728531456) (some (143099))

def event143101 : Event := .predecessor (⟨.program ⟨257⟩, ⟨61679⟩⟩) 0 ⟨58699⟩ 143100

def event143102 : Event := .predecessor (⟨.program ⟨257⟩, ⟨61679⟩⟩) 1 ⟨61678⟩ 139696

def event143103 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨61679⟩⟩) (.sum [.predecessor 0 143101 .coefficient, .predecessor 1 143102 .coefficient])

def eventLeaf8928 : Array AnnotatedEvent := #[
  { event := event142848
    frameStart := 142744 },
  { event := event142849
    frameStart := 142744 },
  { event := event142850
    frameStart := 142744 },
  { event := event142851
    frameStart := 142744 },
  { event := event142852
    frameStart := 142744 },
  { event := event142853
    frameStart := 142744 },
  { event := event142854
    frameStart := 142744 },
  { event := event142855
    frameStart := 142744 },
  { event := event142856
    frameStart := 142744 },
  { event := event142857
    frameStart := 142744 },
  { event := event142858
    frameStart := 142744 },
  { event := event142859
    frameStart := 142744 },
  { event := event142860
    frameStart := 142744 },
  { event := event142861
    frameStart := 142744 },
  { event := event142862
    frameStart := 0 },
  { event := event142863
    frameStart := 0 }
]

def eventLeaf8929 : Array AnnotatedEvent := #[
  { event := event142864
    frameStart := 0 },
  { event := event142865
    frameStart := 0 },
  { event := event142866
    frameStart := 0 },
  { event := event142867
    frameStart := 0 },
  { event := event142868
    frameStart := 0 },
  { event := event142869
    frameStart := 0 },
  { event := event142870
    frameStart := 0 },
  { event := event142871
    frameStart := 0 },
  { event := event142872
    frameStart := 0 },
  { event := event142873
    frameStart := 0 },
  { event := event142874
    frameStart := 0 },
  { event := event142875
    frameStart := 0 },
  { event := event142876
    frameStart := 0 },
  { event := event142877
    frameStart := 0 },
  { event := event142878
    frameStart := 0 },
  { event := event142879
    frameStart := 0 }
]

def eventLeaf8930 : Array AnnotatedEvent := #[
  { event := event142880
    frameStart := 0 },
  { event := event142881
    frameStart := 0 },
  { event := event142882
    frameStart := 0 },
  { event := event142883
    frameStart := 0 },
  { event := event142884
    frameStart := 0 },
  { event := event142885
    frameStart := 0 },
  { event := event142886
    frameStart := 0 },
  { event := event142887
    frameStart := 0 },
  { event := event142888
    frameStart := 0 },
  { event := event142889
    frameStart := 0 },
  { event := event142890
    frameStart := 0 },
  { event := event142891
    frameStart := 0 },
  { event := event142892
    frameStart := 0 },
  { event := event142893
    frameStart := 0 },
  { event := event142894
    frameStart := 0 },
  { event := event142895
    frameStart := 0 }
]

def eventLeaf8931 : Array AnnotatedEvent := #[
  { event := event142896
    frameStart := 0 },
  { event := event142897
    frameStart := 0 },
  { event := event142898
    frameStart := 0 },
  { event := event142899
    frameStart := 142899 },
  { event := event142900
    frameStart := 142899 },
  { event := event142901
    frameStart := 142899 },
  { event := event142902
    frameStart := 142899 },
  { event := event142903
    frameStart := 142899 },
  { event := event142904
    frameStart := 142899 },
  { event := event142905
    frameStart := 142899 },
  { event := event142906
    frameStart := 142899 },
  { event := event142907
    frameStart := 142899 },
  { event := event142908
    frameStart := 142899 },
  { event := event142909
    frameStart := 142899 },
  { event := event142910
    frameStart := 142899 },
  { event := event142911
    frameStart := 142899 }
]

def eventLeaf8932 : Array AnnotatedEvent := #[
  { event := event142912
    frameStart := 142899 },
  { event := event142913
    frameStart := 142899 },
  { event := event142914
    frameStart := 142899 },
  { event := event142915
    frameStart := 142899 },
  { event := event142916
    frameStart := 142899 },
  { event := event142917
    frameStart := 142899 },
  { event := event142918
    frameStart := 142899 },
  { event := event142919
    frameStart := 142899 },
  { event := event142920
    frameStart := 142899 },
  { event := event142921
    frameStart := 142899 },
  { event := event142922
    frameStart := 142899 },
  { event := event142923
    frameStart := 142899 },
  { event := event142924
    frameStart := 142899 },
  { event := event142925
    frameStart := 142899 },
  { event := event142926
    frameStart := 142899 },
  { event := event142927
    frameStart := 142899 }
]

def eventLeaf8933 : Array AnnotatedEvent := #[
  { event := event142928
    frameStart := 142899 },
  { event := event142929
    frameStart := 142899 },
  { event := event142930
    frameStart := 142899 },
  { event := event142931
    frameStart := 142899 },
  { event := event142932
    frameStart := 142899 },
  { event := event142933
    frameStart := 142899 },
  { event := event142934
    frameStart := 142899 },
  { event := event142935
    frameStart := 142899 },
  { event := event142936
    frameStart := 142899 },
  { event := event142937
    frameStart := 142899 },
  { event := event142938
    frameStart := 142899 },
  { event := event142939
    frameStart := 142899 },
  { event := event142940
    frameStart := 142899 },
  { event := event142941
    frameStart := 142899 },
  { event := event142942
    frameStart := 142899 },
  { event := event142943
    frameStart := 142899 }
]

def eventLeaf8934 : Array AnnotatedEvent := #[
  { event := event142944
    frameStart := 142899 },
  { event := event142945
    frameStart := 142899 },
  { event := event142946
    frameStart := 142899 },
  { event := event142947
    frameStart := 142899 },
  { event := event142948
    frameStart := 142899 },
  { event := event142949
    frameStart := 142899 },
  { event := event142950
    frameStart := 142899 },
  { event := event142951
    frameStart := 142899 },
  { event := event142952
    frameStart := 142899 },
  { event := event142953
    frameStart := 142953 },
  { event := event142954
    frameStart := 142953 },
  { event := event142955
    frameStart := 142953 },
  { event := event142956
    frameStart := 142953 },
  { event := event142957
    frameStart := 142953 },
  { event := event142958
    frameStart := 142953 },
  { event := event142959
    frameStart := 142953 }
]

def eventLeaf8935 : Array AnnotatedEvent := #[
  { event := event142960
    frameStart := 142953 },
  { event := event142961
    frameStart := 142953 },
  { event := event142962
    frameStart := 142953 },
  { event := event142963
    frameStart := 142953 },
  { event := event142964
    frameStart := 142953 },
  { event := event142965
    frameStart := 142953 },
  { event := event142966
    frameStart := 142953 },
  { event := event142967
    frameStart := 142953 },
  { event := event142968
    frameStart := 142953 },
  { event := event142969
    frameStart := 142953 },
  { event := event142970
    frameStart := 142953 },
  { event := event142971
    frameStart := 142953 },
  { event := event142972
    frameStart := 142953 },
  { event := event142973
    frameStart := 142953 },
  { event := event142974
    frameStart := 142953 },
  { event := event142975
    frameStart := 142953 }
]

def eventLeaf8936 : Array AnnotatedEvent := #[
  { event := event142976
    frameStart := 142953 },
  { event := event142977
    frameStart := 142953 },
  { event := event142978
    frameStart := 142953 },
  { event := event142979
    frameStart := 142953 },
  { event := event142980
    frameStart := 142953 },
  { event := event142981
    frameStart := 142953 },
  { event := event142982
    frameStart := 142953 },
  { event := event142983
    frameStart := 142953 },
  { event := event142984
    frameStart := 142953 },
  { event := event142985
    frameStart := 142953 },
  { event := event142986
    frameStart := 142953 },
  { event := event142987
    frameStart := 142953 },
  { event := event142988
    frameStart := 142953 },
  { event := event142989
    frameStart := 142953 },
  { event := event142990
    frameStart := 142953 },
  { event := event142991
    frameStart := 142953 }
]

def eventLeaf8937 : Array AnnotatedEvent := #[
  { event := event142992
    frameStart := 142953 },
  { event := event142993
    frameStart := 142953 },
  { event := event142994
    frameStart := 142953 },
  { event := event142995
    frameStart := 142953 },
  { event := event142996
    frameStart := 142953 },
  { event := event142997
    frameStart := 142953 },
  { event := event142998
    frameStart := 142953 },
  { event := event142999
    frameStart := 142953 },
  { event := event143000
    frameStart := 142953 },
  { event := event143001
    frameStart := 142953 },
  { event := event143002
    frameStart := 142953 },
  { event := event143003
    frameStart := 142953 },
  { event := event143004
    frameStart := 142953 },
  { event := event143005
    frameStart := 142953 },
  { event := event143006
    frameStart := 142953 },
  { event := event143007
    frameStart := 142953 }
]

def eventLeaf8938 : Array AnnotatedEvent := #[
  { event := event143008
    frameStart := 142953 },
  { event := event143009
    frameStart := 142953 },
  { event := event143010
    frameStart := 142953 },
  { event := event143011
    frameStart := 142953 },
  { event := event143012
    frameStart := 142953 },
  { event := event143013
    frameStart := 142953 },
  { event := event143014
    frameStart := 142953 },
  { event := event143015
    frameStart := 142953 },
  { event := event143016
    frameStart := 142953 },
  { event := event143017
    frameStart := 142953 },
  { event := event143018
    frameStart := 142953 },
  { event := event143019
    frameStart := 142953 },
  { event := event143020
    frameStart := 142953 },
  { event := event143021
    frameStart := 142953 },
  { event := event143022
    frameStart := 142953 },
  { event := event143023
    frameStart := 142953 }
]

def eventLeaf8939 : Array AnnotatedEvent := #[
  { event := event143024
    frameStart := 142953 },
  { event := event143025
    frameStart := 142953 },
  { event := event143026
    frameStart := 142953 },
  { event := event143027
    frameStart := 142953 },
  { event := event143028
    frameStart := 142953 },
  { event := event143029
    frameStart := 142953 },
  { event := event143030
    frameStart := 142953 },
  { event := event143031
    frameStart := 142953 },
  { event := event143032
    frameStart := 142953 },
  { event := event143033
    frameStart := 142953 },
  { event := event143034
    frameStart := 142953 },
  { event := event143035
    frameStart := 142953 },
  { event := event143036
    frameStart := 142953 },
  { event := event143037
    frameStart := 142953 },
  { event := event143038
    frameStart := 142953 },
  { event := event143039
    frameStart := 142953 }
]

def eventLeaf8940 : Array AnnotatedEvent := #[
  { event := event143040
    frameStart := 142953 },
  { event := event143041
    frameStart := 142953 },
  { event := event143042
    frameStart := 142953 },
  { event := event143043
    frameStart := 142953 },
  { event := event143044
    frameStart := 142953 },
  { event := event143045
    frameStart := 142953 },
  { event := event143046
    frameStart := 142953 },
  { event := event143047
    frameStart := 142953 },
  { event := event143048
    frameStart := 142953 },
  { event := event143049
    frameStart := 142953 },
  { event := event143050
    frameStart := 142953 },
  { event := event143051
    frameStart := 142953 },
  { event := event143052
    frameStart := 142953 },
  { event := event143053
    frameStart := 142953 },
  { event := event143054
    frameStart := 142953 },
  { event := event143055
    frameStart := 142953 }
]

def eventLeaf8941 : Array AnnotatedEvent := #[
  { event := event143056
    frameStart := 142953 },
  { event := event143057
    frameStart := 0 },
  { event := event143058
    frameStart := 0 },
  { event := event143059
    frameStart := 0 },
  { event := event143060
    frameStart := 0 },
  { event := event143061
    frameStart := 0 },
  { event := event143062
    frameStart := 0 },
  { event := event143063
    frameStart := 0 },
  { event := event143064
    frameStart := 0 },
  { event := event143065
    frameStart := 0 },
  { event := event143066
    frameStart := 0 },
  { event := event143067
    frameStart := 0 },
  { event := event143068
    frameStart := 0 },
  { event := event143069
    frameStart := 0 },
  { event := event143070
    frameStart := 0 },
  { event := event143071
    frameStart := 0 }
]

def eventLeaf8942 : Array AnnotatedEvent := #[
  { event := event143072
    frameStart := 0 },
  { event := event143073
    frameStart := 0 },
  { event := event143074
    frameStart := 0 },
  { event := event143075
    frameStart := 0 },
  { event := event143076
    frameStart := 0 },
  { event := event143077
    frameStart := 0 },
  { event := event143078
    frameStart := 0 },
  { event := event143079
    frameStart := 0 },
  { event := event143080
    frameStart := 0 },
  { event := event143081
    frameStart := 0 },
  { event := event143082
    frameStart := 0 },
  { event := event143083
    frameStart := 0 },
  { event := event143084
    frameStart := 0 },
  { event := event143085
    frameStart := 0 },
  { event := event143086
    frameStart := 0 },
  { event := event143087
    frameStart := 0 }
]

def eventLeaf8943 : Array AnnotatedEvent := #[
  { event := event143088
    frameStart := 0 },
  { event := event143089
    frameStart := 0 },
  { event := event143090
    frameStart := 0 },
  { event := event143091
    frameStart := 0 },
  { event := event143092
    frameStart := 0 },
  { event := event143093
    frameStart := 0 },
  { event := event143094
    frameStart := 0 },
  { event := event143095
    frameStart := 0 },
  { event := event143096
    frameStart := 0 },
  { event := event143097
    frameStart := 0 },
  { event := event143098
    frameStart := 0 },
  { event := event143099
    frameStart := 0 },
  { event := event143100
    frameStart := 0 },
  { event := event143101
    frameStart := 0 },
  { event := event143102
    frameStart := 0 },
  { event := event143103
    frameStart := 0 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events558
