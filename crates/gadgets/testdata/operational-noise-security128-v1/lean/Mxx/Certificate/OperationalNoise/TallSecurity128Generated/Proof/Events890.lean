import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events890

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event227840 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨56479⟩⟩) (.product (.predecessor 0 227838 .coefficient) (.predecessor 1 227839 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event227841 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨56479⟩⟩, .operator (⟨227837, 0⟩, ⟨227834, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨24998⟩⟩, ⟨.program ⟨257⟩, ⟨56478⟩⟩], []⟩, (1)⟩)

def exact227842RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨24998⟩⟩, ⟨.program ⟨257⟩, ⟨56478⟩⟩], []⟩, (1)⟩]

theorem exact227842RawTermsValid :
    exact227842RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event227842 : Event := .resultExact (⟨.program ⟨257⟩, ⟨56479⟩⟩) exact227842RawTerms (.finite 256) 227840 .exactZero (none)

def event227843 : Event := .predecessor (⟨.program ⟨257⟩, ⟨56480⟩⟩) 0 ⟨56479⟩ 227842

def event227844 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨56480⟩⟩) (.identity (.predecessor 0 227843 .coefficient))

def event227845 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨56480⟩⟩) (.finite 256)

def event227846 : Event := .predecessor (⟨.program ⟨257⟩, ⟨56840⟩⟩) 0 ⟨56480⟩ 227845

def event227847 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨56840⟩⟩) (.authority (.programFamilyFact))

def exact227848RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨56840⟩⟩], []⟩, (1)⟩]

theorem exact227848RawTermsValid :
    exact227848RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event227848 : Event := .resultExact (⟨.program ⟨257⟩, ⟨56840⟩⟩) exact227848RawTerms (.finite 16) 227847 .exactZero (none)

def event227849 : Event := .predecessor (⟨.program ⟨257⟩, ⟨56841⟩⟩) 0 ⟨56840⟩ 227848

def event227850 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨56841⟩⟩) (.identity (.predecessor 0 227849 .coefficient))

def event227851 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨56841⟩⟩) (.finite 16)

def event227852 : Event := .predecessor (⟨.program ⟨257⟩, ⟨58110⟩⟩) 0 ⟨56841⟩ 227851

def event227853 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨58110⟩⟩) (.authority (.programFamilyFact))

def event227854 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨58110⟩⟩) (.finite 3720)

def event227855 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨7177⟩⟩) .missing

def event227856 : Event := .predecessor (⟨.program ⟨257⟩, ⟨58112⟩⟩) 0 ⟨7177⟩ 227855

def event227857 : Event := .predecessor (⟨.program ⟨257⟩, ⟨58112⟩⟩) 1 ⟨58110⟩ 227854

def event227858 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨58112⟩⟩) (.authority (.operator))

def exact227859RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨58112⟩⟩]⟩, (1)⟩]

theorem exact227859RawTermsValid :
    exact227859RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event227859 : Event := .resultExact (⟨.program ⟨257⟩, ⟨58112⟩⟩) exact227859RawTerms .large 227858 .exactZero (none)

def event227860 : Event := .predecessor (⟨.program ⟨257⟩, ⟨58881⟩⟩) 0 ⟨58112⟩ 227859

def event227861 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨58881⟩⟩) (.authority (.operator))

def exact227862RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨58881⟩⟩]⟩, (1)⟩]

theorem exact227862RawTermsValid :
    exact227862RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event227862 : Event := .resultExact (⟨.program ⟨257⟩, ⟨58881⟩⟩) exact227862RawTerms (.finite 8192) 227861 .exactZero (none)

def event227863 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨136⟩⟩) (.authority (.operator))

def event227864 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨136⟩⟩) .exactZero

def event227865 : Event := .predecessor (⟨.program ⟨257⟩, ⟨58322⟩⟩) 0 ⟨56841⟩ 227851

def event227866 : Event := .predecessor (⟨.program ⟨257⟩, ⟨58322⟩⟩) 1 ⟨136⟩ 227864

def event227867 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨58322⟩⟩) (.sum [.predecessor 0 227865 .coefficient, .predecessor 1 227866 .coefficient])

def event227868 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨58322⟩⟩) (.finite 16)

def event227869 : Event := .predecessor (⟨.program ⟨257⟩, ⟨58323⟩⟩) 0 ⟨58322⟩ 227868

def event227870 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨58323⟩⟩) (.identity (.predecessor 0 227869 .coefficient))

def exact227871RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨56840⟩⟩], []⟩, (1)⟩]

theorem exact227871RawTermsValid :
    exact227871RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event227871 : Event := .resultExact (⟨.program ⟨257⟩, ⟨58323⟩⟩) exact227871RawTerms (.finite 16) 227870 .exactZero (none)

def event227872 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6908⟩⟩) (.authority (.factStore))

def exact227873RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact227873RawTermsValid :
    exact227873RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event227873 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6908⟩⟩) exact227873RawTerms .large 227872 .exactZero (none)

def event227874 : Event := .predecessor (⟨.program ⟨257⟩, ⟨58324⟩⟩) 0 ⟨6908⟩ 227873

def event227875 : Event := .predecessor (⟨.program ⟨257⟩, ⟨58324⟩⟩) 1 ⟨58323⟩ 227871

def event227876 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨58324⟩⟩) (.product (.predecessor 0 227874 .coefficient) (.predecessor 1 227875 .coefficient) (⟨false, false, none, none, none⟩))

def event227877 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨58324⟩⟩, .operator (⟨227873, 0⟩, ⟨227871, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨56840⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact227878RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨56840⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact227878RawTermsValid :
    exact227878RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event227878 : Event := .resultExact (⟨.program ⟨257⟩, ⟨58324⟩⟩) exact227878RawTerms .large 227876 .exactZero (none)

def event227879 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7185⟩⟩) 0 ⟨7177⟩ 227855

def event227880 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7185⟩⟩) (.authority (.operator))

def exact227881RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7185⟩⟩]⟩, (1)⟩]

theorem exact227881RawTermsValid :
    exact227881RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event227881 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7185⟩⟩) exact227881RawTerms .large 227880 .exactZero (none)

def event227882 : Event := .predecessor (⟨.program ⟨257⟩, ⟨58325⟩⟩) 0 ⟨7185⟩ 227881

def event227883 : Event := .predecessor (⟨.program ⟨257⟩, ⟨58325⟩⟩) 1 ⟨58324⟩ 227878

def event227884 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨58325⟩⟩) (.sum [.predecessor 0 227882 .coefficient, .predecessor 1 227883 .coefficient])

def exact227885RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7185⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨56840⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact227885RawTermsValid :
    exact227885RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event227885 : Event := .resultExact (⟨.program ⟨257⟩, ⟨58325⟩⟩) exact227885RawTerms .large 227884 .exactZero (none)

def event227886 : Event := .predecessor (⟨.program ⟨257⟩, ⟨58882⟩⟩) 0 ⟨58325⟩ 227885

def event227887 : Event := .predecessor (⟨.program ⟨257⟩, ⟨58882⟩⟩) 1 ⟨58881⟩ 227862

def event227888 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨58882⟩⟩) (.product (.predecessor 0 227886 .coefficient) (.predecessor 1 227887 .coefficient) (⟨false, false, none, none, none⟩))

def event227889 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨58882⟩⟩, .operator (⟨227885, 0⟩, ⟨227862, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7185⟩⟩, ⟨.program ⟨257⟩, ⟨58881⟩⟩]⟩, (1)⟩)

def event227890 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨58882⟩⟩, .operator (⟨227885, 1⟩, ⟨227862, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨56840⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨58881⟩⟩]⟩, (-1)⟩)

def event227891 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨58882⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨56840⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨58881⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨58881⟩⟩) ⟨58112⟩ 227859)

def event227892 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨58882⟩⟩, .relation 227891 0, ⟨[⟨.program ⟨257⟩, ⟨56840⟩⟩], [⟨.program ⟨257⟩, ⟨58112⟩⟩]⟩, (-1)⟩)

def exact227893RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7185⟩⟩, ⟨.program ⟨257⟩, ⟨58881⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨56840⟩⟩], [⟨.program ⟨257⟩, ⟨58112⟩⟩]⟩, (-1)⟩]

theorem exact227893RawTermsValid :
    exact227893RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event227893 : Event := .resultExact (⟨.program ⟨257⟩, ⟨58882⟩⟩) exact227893RawTerms .large 227888 .exactZero (none)

def event227894 : Event := .predecessor (⟨.program ⟨257⟩, ⟨57102⟩⟩) 0 ⟨56841⟩ 227851

def event227895 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨57102⟩⟩) (.authority (.programFamilyFact))

def exact227896RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨57102⟩⟩], []⟩, (1)⟩]

theorem exact227896RawTermsValid :
    exact227896RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event227896 : Event := .resultExact (⟨.program ⟨257⟩, ⟨57102⟩⟩) exact227896RawTerms (.finite 60) 227895 .exactZero (none)

def event227897 : Event := .predecessor (⟨.program ⟨257⟩, ⟨57104⟩⟩) 0 ⟨6908⟩ 227873

def event227898 : Event := .predecessor (⟨.program ⟨257⟩, ⟨57104⟩⟩) 1 ⟨57102⟩ 227896

def event227899 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨57104⟩⟩) (.product (.predecessor 0 227897 .coefficient) (.predecessor 1 227898 .coefficient) (⟨false, true, none, none, some 1⟩))

def event227900 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨57104⟩⟩, .operator (⟨227873, 0⟩, ⟨227896, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨57102⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact227901RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨57102⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact227901RawTermsValid :
    exact227901RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event227901 : Event := .resultExact (⟨.program ⟨257⟩, ⟨57104⟩⟩) exact227901RawTerms .large 227899 .exactZero (none)

def event227902 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7210⟩⟩) 0 ⟨7177⟩ 227855

def event227903 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7210⟩⟩) (.authority (.operator))

def exact227904RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩]

theorem exact227904RawTermsValid :
    exact227904RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event227904 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7210⟩⟩) exact227904RawTerms .large 227903 .exactZero (none)

def event227905 : Event := .predecessor (⟨.program ⟨257⟩, ⟨57105⟩⟩) 0 ⟨7210⟩ 227904

def event227906 : Event := .predecessor (⟨.program ⟨257⟩, ⟨57105⟩⟩) 1 ⟨57104⟩ 227901

def event227907 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨57105⟩⟩) (.sum [.predecessor 0 227905 .coefficient, .predecessor 1 227906 .coefficient])

def exact227908RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨57102⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact227908RawTermsValid :
    exact227908RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event227908 : Event := .resultExact (⟨.program ⟨257⟩, ⟨57105⟩⟩) exact227908RawTerms .large 227907 .exactZero (none)

def event227909 : Event := .predecessor (⟨.program ⟨257⟩, ⟨58886⟩⟩) 0 ⟨57105⟩ 227908

def event227910 : Event := .predecessor (⟨.program ⟨257⟩, ⟨58886⟩⟩) 1 ⟨58882⟩ 227893

def event227911 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨58886⟩⟩) (.sum [.predecessor 0 227909 .coefficient, .predecessor 1 227910 .coefficient])

def exact227912RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7185⟩⟩, ⟨.program ⟨257⟩, ⟨58881⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨56840⟩⟩], [⟨.program ⟨257⟩, ⟨58112⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨57102⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact227912RawTermsValid :
    exact227912RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event227912 : Event := .resultExact (⟨.program ⟨257⟩, ⟨58886⟩⟩) exact227912RawTerms .large 227911 .exactZero (none)

def event227913 : Event := .preFoldPolynomial 227912 [⟨⟨[], [⟨.program ⟨257⟩, ⟨7185⟩⟩, ⟨.program ⟨257⟩, ⟨58881⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨56840⟩⟩], [⟨.program ⟨257⟩, ⟨58112⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨57102⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩] .exactZero none

def exact227914RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7185⟩⟩, ⟨.program ⟨257⟩, ⟨58881⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨56840⟩⟩], [⟨.program ⟨257⟩, ⟨58112⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨57102⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

def event227914 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨58886⟩⟩) 227913 exact227914RawTerms .large 227911 .exactZero (none)

def event227915 : Event := .specializationComputed (⟨.program ⟨257⟩, ⟨56841⟩⟩) ⟨⟨89⟩, ⟨70⟩, ⟨135⟩⟩ ⟨227757, 227915⟩

def event227916 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨57699⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57696⟩⟩]⟩) (1) 0 2 (.universal 227915 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57696⟩⟩]⟩) (none) 227914)

def event227917 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨57699⟩⟩, .relation 227916 1, ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩)

def event227918 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨57699⟩⟩, .relation 227916 0, ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7185⟩⟩, ⟨.program ⟨257⟩, ⟨58881⟩⟩]⟩, (-1)⟩)

def event227919 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨57699⟩⟩, .relation 227916 2, ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨56840⟩⟩], [⟨.program ⟨257⟩, ⟨58112⟩⟩]⟩, (1)⟩)

def event227920 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨57699⟩⟩, .relation 227916 3, ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨57102⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact227921RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7185⟩⟩, ⟨.program ⟨257⟩, ⟨58881⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨56840⟩⟩], [⟨.program ⟨257⟩, ⟨58112⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨57102⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact227921RawTermsValid :
    exact227921RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event227921 : Event := .resultExact (⟨.program ⟨257⟩, ⟨57699⟩⟩) exact227921RawTerms .large 227753 (.finite 202072841853861888) (some (227755))

def event227922 : Event := .predecessor (⟨.program ⟨257⟩, ⟨58884⟩⟩) 0 ⟨57699⟩ 227921

def event227923 : Event := .predecessor (⟨.program ⟨257⟩, ⟨58884⟩⟩) 1 ⟨58883⟩ 227743

def event227924 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨58884⟩⟩) (.sum [.predecessor 0 227922 .coefficient, .predecessor 1 227923 .coefficient])

def event227925 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨58884⟩⟩, .operator (⟨227921, 0⟩, ⟨227743, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7185⟩⟩, ⟨.program ⟨257⟩, ⟨58881⟩⟩]⟩, (1)⟩)

def event227926 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨58884⟩⟩, .operator (⟨227921, 2⟩, ⟨227743, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨56840⟩⟩], [⟨.program ⟨257⟩, ⟨58112⟩⟩]⟩, (-1)⟩)

def event227927 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨58884⟩⟩) (.sum [.result 227921 .summary, .result 227743 .summary])

def exact227928RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨57102⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact227928RawTermsValid :
    exact227928RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event227928 : Event := .resultExact (⟨.program ⟨257⟩, ⟨58884⟩⟩) exact227928RawTerms .large 227924 (.finite 32190182365603518530196853751808) (some (227927))

def event227929 : Event := .predecessor (⟨.program ⟨257⟩, ⟨55130⟩⟩) 0 ⟨53861⟩ 10859

def event227930 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨55130⟩⟩) (.authority (.programFamilyFact))

def event227931 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨55130⟩⟩) (.finite 3720)

def event227932 : Event := .predecessor (⟨.program ⟨257⟩, ⟨55132⟩⟩) 0 ⟨7177⟩ 15500

def event227933 : Event := .predecessor (⟨.program ⟨257⟩, ⟨55132⟩⟩) 1 ⟨55130⟩ 227931

def event227934 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨55132⟩⟩) (.authority (.operator))

def exact227935RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨55132⟩⟩]⟩, (1)⟩]

theorem exact227935RawTermsValid :
    exact227935RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event227935 : Event := .resultExact (⟨.program ⟨257⟩, ⟨55132⟩⟩) exact227935RawTerms .large 227934 .exactZero (none)

def event227936 : Event := .predecessor (⟨.program ⟨257⟩, ⟨55901⟩⟩) 0 ⟨55132⟩ 227935

def event227937 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨55901⟩⟩) (.authority (.operator))

def exact227938RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨55901⟩⟩]⟩, (1)⟩]

theorem exact227938RawTermsValid :
    exact227938RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event227938 : Event := .resultExact (⟨.program ⟨257⟩, ⟨55901⟩⟩) exact227938RawTerms (.finite 8192) 227937 .exactZero (none)

def event227939 : Event := .predecessor (⟨.program ⟨257⟩, ⟨54982⟩⟩) 0 ⟨53500⟩ 10853

def event227940 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨54982⟩⟩) (.authority (.programFamilyFact))

def event227941 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨54982⟩⟩) (.finite 3720)

def event227942 : Event := .predecessor (⟨.program ⟨257⟩, ⟨54983⟩⟩) 0 ⟨7177⟩ 15500

def event227943 : Event := .predecessor (⟨.program ⟨257⟩, ⟨54983⟩⟩) 1 ⟨54982⟩ 227941

def event227944 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨54983⟩⟩) (.authority (.operator))

def exact227945RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨54983⟩⟩]⟩, (1)⟩]

theorem exact227945RawTermsValid :
    exact227945RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event227945 : Event := .resultExact (⟨.program ⟨257⟩, ⟨54983⟩⟩) exact227945RawTerms .large 227944 .exactZero (none)

def event227946 : Event := .predecessor (⟨.program ⟨257⟩, ⟨55488⟩⟩) 0 ⟨54983⟩ 227945

def event227947 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨55488⟩⟩) (.authority (.operator))

def exact227948RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨55488⟩⟩]⟩, (1)⟩]

theorem exact227948RawTermsValid :
    exact227948RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event227948 : Event := .resultExact (⟨.program ⟨257⟩, ⟨55488⟩⟩) exact227948RawTerms (.finite 8192) 227947 .exactZero (none)

def event227949 : Event := .predecessor (⟨.program ⟨257⟩, ⟨24759⟩⟩) 0 ⟨24758⟩ 10842

def event227950 : Event := .predecessor (⟨.program ⟨257⟩, ⟨24759⟩⟩) 1 ⟨6937⟩ 222153

def event227951 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨24759⟩⟩) (.tensor (.predecessor 0 227949 .coefficient) (.predecessor 1 227950 .coefficient) true false)

def event227952 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨24759⟩⟩, .operator (⟨10842, 0⟩, ⟨222153, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨24758⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact227953RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨24758⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact227953RawTermsValid :
    exact227953RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event227953 : Event := .resultExact (⟨.program ⟨257⟩, ⟨24759⟩⟩) exact227953RawTerms .large 227951 .exactZero (none)

def event227954 : Event := .predecessor (⟨.program ⟨257⟩, ⟨8464⟩⟩) 0 ⟨5579⟩ 222023

def event227955 : Event := .predecessor (⟨.program ⟨257⟩, ⟨8464⟩⟩) 1 ⟨7272⟩ 23092

def event227956 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨8464⟩⟩) (.product (.predecessor 0 227954 .coefficient) (.predecessor 1 227955 .coefficient) (⟨false, false, none, none, none⟩))

def event227957 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨8464⟩⟩, .operator (⟨222023, 0⟩, ⟨23092, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7272⟩⟩]⟩, (1)⟩)

def exact227958RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7272⟩⟩]⟩, (1)⟩]

theorem exact227958RawTermsValid :
    exact227958RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event227958 : Event := .resultExact (⟨.program ⟨257⟩, ⟨8464⟩⟩) exact227958RawTerms .large 227956 .exactZero (none)

def event227959 : Event := .predecessor (⟨.program ⟨257⟩, ⟨24760⟩⟩) 0 ⟨8464⟩ 227958

def event227960 : Event := .predecessor (⟨.program ⟨257⟩, ⟨24760⟩⟩) 1 ⟨24759⟩ 227953

def event227961 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨24760⟩⟩) (.sum [.predecessor 0 227959 .coefficient, .predecessor 1 227960 .coefficient])

def exact227962RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7272⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨24758⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact227962RawTermsValid :
    exact227962RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event227962 : Event := .resultExact (⟨.program ⟨257⟩, ⟨24760⟩⟩) exact227962RawTerms .large 227961 .exactZero (none)

def event227963 : Event := .predecessor (⟨.program ⟨257⟩, ⟨24761⟩⟩) 0 ⟨24760⟩ 227962

def event227964 : Event := .predecessor (⟨.program ⟨257⟩, ⟨24761⟩⟩) 1 ⟨98⟩ 23084

def event227965 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨24761⟩⟩) (.sum [.predecessor 0 227963 .coefficient, .predecessor 1 227964 .coefficient])

def event227966 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨24761⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨98⟩⟩]⟩) [⟨.result 23084 .coefficient, false, none⟩])

def event227967 : Event := .survivorFold (1) 227966

def exact227968RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7272⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨24758⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact227968RawTermsValid :
    exact227968RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event227968 : Event := .resultExact (⟨.program ⟨257⟩, ⟨24761⟩⟩) exact227968RawTerms .large 227965 (.finite 26) (some (227966))

def event227969 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53501⟩⟩) 0 ⟨24761⟩ 227968

def event227970 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53501⟩⟩) 1 ⟨53498⟩ 10845

def event227971 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨53501⟩⟩) (.product (.predecessor 0 227969 .coefficient) (.predecessor 1 227970 .coefficient) (⟨false, true, none, none, some 1⟩))

def event227972 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨53501⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨53498⟩⟩], []⟩) [⟨.result 10845 .coefficient, true, some 1⟩])

def event227973 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨53501⟩⟩) (.product (.result 227968 .summary) (.transfer 227972) (⟨false, false, none, none, none⟩))

def event227974 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨53501⟩⟩, .operator (⟨227968, 1⟩, ⟨10845, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨24758⟩⟩, ⟨.program ⟨257⟩, ⟨53498⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def event227975 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨53501⟩⟩, .operator (⟨227968, 0⟩, ⟨10845, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨53498⟩⟩], [⟨.program ⟨257⟩, ⟨7272⟩⟩]⟩, (1)⟩)

def exact227976RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨24758⟩⟩, ⟨.program ⟨257⟩, ⟨53498⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨53498⟩⟩], [⟨.program ⟨257⟩, ⟨7272⟩⟩]⟩, (1)⟩]

theorem exact227976RawTermsValid :
    exact227976RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event227976 : Event := .resultExact (⟨.program ⟨257⟩, ⟨53501⟩⟩) exact227976RawTerms .large 227971 (.finite 10223616) (some (227973))

def event227977 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53502⟩⟩) 0 ⟨53498⟩ 10845

def event227978 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53502⟩⟩) 1 ⟨6937⟩ 222153

def event227979 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨53502⟩⟩) (.tensor (.predecessor 0 227977 .coefficient) (.predecessor 1 227978 .coefficient) true false)

def event227980 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨53502⟩⟩, .operator (⟨10845, 0⟩, ⟨222153, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨53498⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact227981RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨53498⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact227981RawTermsValid :
    exact227981RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event227981 : Event := .resultExact (⟨.program ⟨257⟩, ⟨53502⟩⟩) exact227981RawTerms .large 227979 .exactZero (none)

def event227982 : Event := .predecessor (⟨.program ⟨257⟩, ⟨8481⟩⟩) 0 ⟨5579⟩ 222023

def event227983 : Event := .predecessor (⟨.program ⟨257⟩, ⟨8481⟩⟩) 1 ⟨7289⟩ 23133

def event227984 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨8481⟩⟩) (.product (.predecessor 0 227982 .coefficient) (.predecessor 1 227983 .coefficient) (⟨false, false, none, none, none⟩))

def event227985 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨8481⟩⟩, .operator (⟨222023, 0⟩, ⟨23133, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7289⟩⟩]⟩, (1)⟩)

def exact227986RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7289⟩⟩]⟩, (1)⟩]

theorem exact227986RawTermsValid :
    exact227986RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event227986 : Event := .resultExact (⟨.program ⟨257⟩, ⟨8481⟩⟩) exact227986RawTerms .large 227984 .exactZero (none)

def event227987 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53503⟩⟩) 0 ⟨8481⟩ 227986

def event227988 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53503⟩⟩) 1 ⟨53502⟩ 227981

def event227989 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨53503⟩⟩) (.sum [.predecessor 0 227987 .coefficient, .predecessor 1 227988 .coefficient])

def exact227990RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7289⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨53498⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact227990RawTermsValid :
    exact227990RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event227990 : Event := .resultExact (⟨.program ⟨257⟩, ⟨53503⟩⟩) exact227990RawTerms .large 227989 .exactZero (none)

def event227991 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53504⟩⟩) 0 ⟨53503⟩ 227990

def event227992 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53504⟩⟩) 1 ⟨115⟩ 23125

def event227993 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨53504⟩⟩) (.sum [.predecessor 0 227991 .coefficient, .predecessor 1 227992 .coefficient])

def event227994 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨53504⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨115⟩⟩]⟩) [⟨.result 23125 .coefficient, false, none⟩])

def event227995 : Event := .survivorFold (1) 227994

def exact227996RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7289⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨53498⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact227996RawTermsValid :
    exact227996RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event227996 : Event := .resultExact (⟨.program ⟨257⟩, ⟨53504⟩⟩) exact227996RawTerms .large 227993 (.finite 26) (some (227994))

def event227997 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53505⟩⟩) 0 ⟨53504⟩ 227996

def event227998 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53505⟩⟩) 1 ⟨9530⟩ 23122

def event227999 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨53505⟩⟩) (.product (.predecessor 0 227997 .coefficient) (.predecessor 1 227998 .coefficient) (⟨false, false, none, none, none⟩))

def event228000 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨53505⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨9529⟩⟩]⟩) [⟨.result 23118 .coefficient, false, none⟩])

def event228001 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨53505⟩⟩) (.product (.result 227996 .summary) (.transfer 228000) (⟨false, false, none, none, none⟩))

def event228002 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨53505⟩⟩, .operator (⟨227996, 1⟩, ⟨23122, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨53498⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩]⟩, (-1)⟩)

def event228003 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨53505⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨53498⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨9529⟩⟩) ⟨7272⟩ 23092)

def event228004 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨53505⟩⟩, .relation 228003 0, ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨53498⟩⟩], [⟨.program ⟨257⟩, ⟨7272⟩⟩]⟩, (-1)⟩)

def event228005 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨53505⟩⟩, .operator (⟨227996, 0⟩, ⟨23122, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩]⟩, (1)⟩)

def exact228006RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨53498⟩⟩], [⟨.program ⟨257⟩, ⟨7272⟩⟩]⟩, (-1)⟩]

theorem exact228006RawTermsValid :
    exact228006RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event228006 : Event := .resultExact (⟨.program ⟨257⟩, ⟨53505⟩⟩) exact228006RawTerms .large 227999 (.finite 279172874240) (some (228001))

def event228007 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53506⟩⟩) 0 ⟨53505⟩ 228006

def event228008 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53506⟩⟩) 1 ⟨53501⟩ 227976

def event228009 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨53506⟩⟩) (.sum [.predecessor 0 228007 .coefficient, .predecessor 1 228008 .coefficient])

def event228010 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨53506⟩⟩, .operator (⟨228006, 1⟩, ⟨227976, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨53498⟩⟩], [⟨.program ⟨257⟩, ⟨7272⟩⟩]⟩, (1)⟩)

def event228011 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨53506⟩⟩) (.sum [.result 228006 .summary, .result 227976 .summary])

def exact228012RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨24758⟩⟩, ⟨.program ⟨257⟩, ⟨53498⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact228012RawTermsValid :
    exact228012RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event228012 : Event := .resultExact (⟨.program ⟨257⟩, ⟨53506⟩⟩) exact228012RawTerms .large 228009 (.finite 279183097856) (some (228011))

def event228013 : Event := .predecessor (⟨.program ⟨257⟩, ⟨55489⟩⟩) 0 ⟨53506⟩ 228012

def event228014 : Event := .predecessor (⟨.program ⟨257⟩, ⟨55489⟩⟩) 1 ⟨55488⟩ 227948

def event228015 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨55489⟩⟩) (.product (.predecessor 0 228013 .coefficient) (.predecessor 1 228014 .coefficient) (⟨false, false, none, none, none⟩))

def event228016 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨55489⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨55488⟩⟩]⟩) [⟨.result 227948 .coefficient, false, none⟩])

def event228017 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨55489⟩⟩) (.product (.result 228012 .summary) (.transfer 228016) (⟨false, false, none, none, none⟩))

def event228018 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨55489⟩⟩, .operator (⟨228012, 1⟩, ⟨227948, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨24758⟩⟩, ⟨.program ⟨257⟩, ⟨53498⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨55488⟩⟩]⟩, (-1)⟩)

def event228019 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨55489⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨24758⟩⟩, ⟨.program ⟨257⟩, ⟨53498⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨55488⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨55488⟩⟩) ⟨54983⟩ 227945)

def event228020 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨55489⟩⟩, .relation 228019 0, ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨24758⟩⟩, ⟨.program ⟨257⟩, ⟨53498⟩⟩], [⟨.program ⟨257⟩, ⟨54983⟩⟩]⟩, (-1)⟩)

def event228021 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨55489⟩⟩, .operator (⟨228012, 0⟩, ⟨227948, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55488⟩⟩]⟩, (1)⟩)

def exact228022RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55488⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨24758⟩⟩, ⟨.program ⟨257⟩, ⟨53498⟩⟩], [⟨.program ⟨257⟩, ⟨54983⟩⟩]⟩, (-1)⟩]

theorem exact228022RawTermsValid :
    exact228022RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event228022 : Event := .resultExact (⟨.program ⟨257⟩, ⟨55489⟩⟩) exact228022RawTerms .large 228015 (.finite 2997705687218719293440) (some (228017))

def event228023 : Event := .predecessor (⟨.program ⟨257⟩, ⟨54419⟩⟩) 0 ⟨53500⟩ 10853

def event228024 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨54419⟩⟩) (.authority (.relationPreimageSource ⟨41⟩))

def exact228025RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨54419⟩⟩]⟩, (1)⟩]

theorem exact228025RawTermsValid :
    exact228025RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event228025 : Event := .resultExact (⟨.program ⟨257⟩, ⟨54419⟩⟩) exact228025RawTerms (.finite 5647228698) 228024 .exactZero (none)

def event228026 : Event := .predecessor (⟨.program ⟨257⟩, ⟨54421⟩⟩) 0 ⟨54419⟩ 228025

def event228027 : Event := .predecessor (⟨.program ⟨257⟩, ⟨54421⟩⟩) 1 ⟨2370⟩ 4

def event228028 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨54421⟩⟩) (.scale (.predecessor 0 228026 .coefficient) (.value (.predecessor 1 228027 .coefficient)))

def exact228029RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨54419⟩⟩]⟩, (1)⟩]

theorem exact228029RawTermsValid :
    exact228029RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event228029 : Event := .resultExact (⟨.program ⟨257⟩, ⟨54421⟩⟩) exact228029RawTerms (.finite 5647228698) 228028 .exactZero (none)

def event228030 : Event := .predecessor (⟨.program ⟨257⟩, ⟨54422⟩⟩) 0 ⟨5581⟩ 222245

def event228031 : Event := .predecessor (⟨.program ⟨257⟩, ⟨54422⟩⟩) 1 ⟨54421⟩ 228029

def event228032 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨54422⟩⟩) (.product (.predecessor 0 228030 .coefficient) (.predecessor 1 228031 .coefficient) (⟨false, false, none, none, none⟩))

def event228033 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨54422⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨54419⟩⟩]⟩) [⟨.result 228025 .coefficient, false, none⟩])

def event228034 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨54422⟩⟩) (.product (.result 222245 .summary) (.transfer 228033) (⟨false, false, none, none, none⟩))

def event228035 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨54422⟩⟩, .operator (⟨222245, 0⟩, ⟨228029, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54419⟩⟩]⟩, (1)⟩)

def event228036 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨54420⟩⟩)

def event228037 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event228038 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event228039 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨4990⟩⟩) (.authority (.operator))

def event228040 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨4990⟩⟩) (.finite 5)

def event228041 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event228042 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event228043 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event228044 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event228045 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 228044

def event228046 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 228042

def event228047 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 228045 .coefficient) (.value (.predecessor 1 228046 .coefficient)))

def event228048 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event228049 : Event := .predecessor (⟨.program ⟨257⟩, ⟨4992⟩⟩) 0 ⟨392⟩ 228048

def event228050 : Event := .predecessor (⟨.program ⟨257⟩, ⟨4992⟩⟩) 1 ⟨4990⟩ 228040

def event228051 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨4992⟩⟩) (.sum [.predecessor 0 228049 .coefficient, .predecessor 1 228050 .coefficient])

def event228052 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨4992⟩⟩) (.finite 655345)

def event228053 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5577⟩⟩) 0 ⟨4992⟩ 228052

def event228054 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5577⟩⟩) 1 ⟨5426⟩ 228038

def event228055 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5577⟩⟩) (.identity (.predecessor 1 228054 .coefficient))

def event228056 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5577⟩⟩) (.finite 655360)

def event228057 : Event := .predecessor (⟨.program ⟨257⟩, ⟨24758⟩⟩) 0 ⟨5577⟩ 228056

def event228058 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨24758⟩⟩) (.authority (.programFamilyFact))

def exact228059RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨24758⟩⟩], []⟩, (1)⟩]

theorem exact228059RawTermsValid :
    exact228059RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event228059 : Event := .resultExact (⟨.program ⟨257⟩, ⟨24758⟩⟩) exact228059RawTerms (.finite 12) 228058 .exactZero (none)

def event228060 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53498⟩⟩) 0 ⟨5577⟩ 228056

def event228061 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨53498⟩⟩) (.authority (.programFamilyFact))

def exact228062RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨53498⟩⟩], []⟩, (1)⟩]

theorem exact228062RawTermsValid :
    exact228062RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event228062 : Event := .resultExact (⟨.program ⟨257⟩, ⟨53498⟩⟩) exact228062RawTerms (.finite 12) 228061 .exactZero (none)

def event228063 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53499⟩⟩) 0 ⟨53498⟩ 228062

def event228064 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53499⟩⟩) 1 ⟨24758⟩ 228059

def event228065 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨53499⟩⟩) (.product (.predecessor 0 228063 .coefficient) (.predecessor 1 228064 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event228066 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨53499⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨24758⟩⟩, ⟨.program ⟨257⟩, ⟨53498⟩⟩], []⟩) [⟨.result 228062 .coefficient, true, some 1⟩, ⟨.result 228059 .coefficient, true, some 1⟩])

def event228067 : Event := .survivorFold (1) 228066

def exact228068RawTerms : List Term := []

theorem exact228068RawTermsValid :
    exact228068RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event228068 : Event := .resultExact (⟨.program ⟨257⟩, ⟨53499⟩⟩) exact228068RawTerms (.finite 144) 228065 (.finite 144) (some (228066))

def event228069 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53500⟩⟩) 0 ⟨53499⟩ 228068

def event228070 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨53500⟩⟩) (.identity (.predecessor 0 228069 .coefficient))

def event228071 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨53500⟩⟩) (.finite 144)

def event228072 : Event := .predecessor (⟨.program ⟨257⟩, ⟨54419⟩⟩) 0 ⟨53500⟩ 228071

def event228073 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨54419⟩⟩) (.authority (.relationPreimageSource ⟨41⟩))

def exact228074RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨54419⟩⟩]⟩, (1)⟩]

theorem exact228074RawTermsValid :
    exact228074RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event228074 : Event := .resultExact (⟨.program ⟨257⟩, ⟨54419⟩⟩) exact228074RawTerms (.finite 5647228698) 228073 .exactZero (none)

def event228075 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35⟩⟩) (.authority (.operator))

def exact228076RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩]⟩, (1)⟩]

theorem exact228076RawTermsValid :
    exact228076RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event228076 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35⟩⟩) exact228076RawTerms .large 228075 .exactZero (none)

def event228077 : Event := .predecessor (⟨.program ⟨257⟩, ⟨54420⟩⟩) 0 ⟨35⟩ 228076

def event228078 : Event := .predecessor (⟨.program ⟨257⟩, ⟨54420⟩⟩) 1 ⟨54419⟩ 228074

def event228079 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨54420⟩⟩) (.product (.predecessor 0 228077 .coefficient) (.predecessor 1 228078 .coefficient) (⟨false, false, none, none, none⟩))

def event228080 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨54420⟩⟩, .operator (⟨228076, 0⟩, ⟨228074, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54419⟩⟩]⟩, (1)⟩)

def exact228081RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54419⟩⟩]⟩, (1)⟩]

theorem exact228081RawTermsValid :
    exact228081RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event228081 : Event := .resultExact (⟨.program ⟨257⟩, ⟨54420⟩⟩) exact228081RawTerms .large 228079 .exactZero (none)

def event228082 : Event := .preFoldPolynomial 228081 [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54419⟩⟩]⟩, (1)⟩] .exactZero none

def exact228083RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54419⟩⟩]⟩, (1)⟩]

def event228083 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨54420⟩⟩) 228082 exact228083RawTerms .large 228079 .exactZero (none)

def event228084 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨55492⟩⟩)

def event228085 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event228086 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event228087 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨4990⟩⟩) (.authority (.operator))

def event228088 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨4990⟩⟩) (.finite 5)

def event228089 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event228090 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event228091 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event228092 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event228093 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 228092

def event228094 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 228090

def event228095 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 228093 .coefficient) (.value (.predecessor 1 228094 .coefficient)))

def eventLeaf14240 : Array AnnotatedEvent := #[
  { event := event227840
    frameStart := 227811 },
  { event := event227841
    frameStart := 227811 },
  { event := event227842
    frameStart := 227811 },
  { event := event227843
    frameStart := 227811 },
  { event := event227844
    frameStart := 227811 },
  { event := event227845
    frameStart := 227811 },
  { event := event227846
    frameStart := 227811 },
  { event := event227847
    frameStart := 227811 },
  { event := event227848
    frameStart := 227811 },
  { event := event227849
    frameStart := 227811 },
  { event := event227850
    frameStart := 227811 },
  { event := event227851
    frameStart := 227811 },
  { event := event227852
    frameStart := 227811 },
  { event := event227853
    frameStart := 227811 },
  { event := event227854
    frameStart := 227811 },
  { event := event227855
    frameStart := 227811 }
]

def eventLeaf14241 : Array AnnotatedEvent := #[
  { event := event227856
    frameStart := 227811 },
  { event := event227857
    frameStart := 227811 },
  { event := event227858
    frameStart := 227811 },
  { event := event227859
    frameStart := 227811 },
  { event := event227860
    frameStart := 227811 },
  { event := event227861
    frameStart := 227811 },
  { event := event227862
    frameStart := 227811 },
  { event := event227863
    frameStart := 227811 },
  { event := event227864
    frameStart := 227811 },
  { event := event227865
    frameStart := 227811 },
  { event := event227866
    frameStart := 227811 },
  { event := event227867
    frameStart := 227811 },
  { event := event227868
    frameStart := 227811 },
  { event := event227869
    frameStart := 227811 },
  { event := event227870
    frameStart := 227811 },
  { event := event227871
    frameStart := 227811 }
]

def eventLeaf14242 : Array AnnotatedEvent := #[
  { event := event227872
    frameStart := 227811 },
  { event := event227873
    frameStart := 227811 },
  { event := event227874
    frameStart := 227811 },
  { event := event227875
    frameStart := 227811 },
  { event := event227876
    frameStart := 227811 },
  { event := event227877
    frameStart := 227811 },
  { event := event227878
    frameStart := 227811 },
  { event := event227879
    frameStart := 227811 },
  { event := event227880
    frameStart := 227811 },
  { event := event227881
    frameStart := 227811 },
  { event := event227882
    frameStart := 227811 },
  { event := event227883
    frameStart := 227811 },
  { event := event227884
    frameStart := 227811 },
  { event := event227885
    frameStart := 227811 },
  { event := event227886
    frameStart := 227811 },
  { event := event227887
    frameStart := 227811 }
]

def eventLeaf14243 : Array AnnotatedEvent := #[
  { event := event227888
    frameStart := 227811 },
  { event := event227889
    frameStart := 227811 },
  { event := event227890
    frameStart := 227811 },
  { event := event227891
    frameStart := 227811 },
  { event := event227892
    frameStart := 227811 },
  { event := event227893
    frameStart := 227811 },
  { event := event227894
    frameStart := 227811 },
  { event := event227895
    frameStart := 227811 },
  { event := event227896
    frameStart := 227811 },
  { event := event227897
    frameStart := 227811 },
  { event := event227898
    frameStart := 227811 },
  { event := event227899
    frameStart := 227811 },
  { event := event227900
    frameStart := 227811 },
  { event := event227901
    frameStart := 227811 },
  { event := event227902
    frameStart := 227811 },
  { event := event227903
    frameStart := 227811 }
]

def eventLeaf14244 : Array AnnotatedEvent := #[
  { event := event227904
    frameStart := 227811 },
  { event := event227905
    frameStart := 227811 },
  { event := event227906
    frameStart := 227811 },
  { event := event227907
    frameStart := 227811 },
  { event := event227908
    frameStart := 227811 },
  { event := event227909
    frameStart := 227811 },
  { event := event227910
    frameStart := 227811 },
  { event := event227911
    frameStart := 227811 },
  { event := event227912
    frameStart := 227811 },
  { event := event227913
    frameStart := 227811 },
  { event := event227914
    frameStart := 227811 },
  { event := event227915
    frameStart := 0 },
  { event := event227916
    frameStart := 0 },
  { event := event227917
    frameStart := 0 },
  { event := event227918
    frameStart := 0 },
  { event := event227919
    frameStart := 0 }
]

def eventLeaf14245 : Array AnnotatedEvent := #[
  { event := event227920
    frameStart := 0 },
  { event := event227921
    frameStart := 0 },
  { event := event227922
    frameStart := 0 },
  { event := event227923
    frameStart := 0 },
  { event := event227924
    frameStart := 0 },
  { event := event227925
    frameStart := 0 },
  { event := event227926
    frameStart := 0 },
  { event := event227927
    frameStart := 0 },
  { event := event227928
    frameStart := 0 },
  { event := event227929
    frameStart := 0 },
  { event := event227930
    frameStart := 0 },
  { event := event227931
    frameStart := 0 },
  { event := event227932
    frameStart := 0 },
  { event := event227933
    frameStart := 0 },
  { event := event227934
    frameStart := 0 },
  { event := event227935
    frameStart := 0 }
]

def eventLeaf14246 : Array AnnotatedEvent := #[
  { event := event227936
    frameStart := 0 },
  { event := event227937
    frameStart := 0 },
  { event := event227938
    frameStart := 0 },
  { event := event227939
    frameStart := 0 },
  { event := event227940
    frameStart := 0 },
  { event := event227941
    frameStart := 0 },
  { event := event227942
    frameStart := 0 },
  { event := event227943
    frameStart := 0 },
  { event := event227944
    frameStart := 0 },
  { event := event227945
    frameStart := 0 },
  { event := event227946
    frameStart := 0 },
  { event := event227947
    frameStart := 0 },
  { event := event227948
    frameStart := 0 },
  { event := event227949
    frameStart := 0 },
  { event := event227950
    frameStart := 0 },
  { event := event227951
    frameStart := 0 }
]

def eventLeaf14247 : Array AnnotatedEvent := #[
  { event := event227952
    frameStart := 0 },
  { event := event227953
    frameStart := 0 },
  { event := event227954
    frameStart := 0 },
  { event := event227955
    frameStart := 0 },
  { event := event227956
    frameStart := 0 },
  { event := event227957
    frameStart := 0 },
  { event := event227958
    frameStart := 0 },
  { event := event227959
    frameStart := 0 },
  { event := event227960
    frameStart := 0 },
  { event := event227961
    frameStart := 0 },
  { event := event227962
    frameStart := 0 },
  { event := event227963
    frameStart := 0 },
  { event := event227964
    frameStart := 0 },
  { event := event227965
    frameStart := 0 },
  { event := event227966
    frameStart := 0 },
  { event := event227967
    frameStart := 0 }
]

def eventLeaf14248 : Array AnnotatedEvent := #[
  { event := event227968
    frameStart := 0 },
  { event := event227969
    frameStart := 0 },
  { event := event227970
    frameStart := 0 },
  { event := event227971
    frameStart := 0 },
  { event := event227972
    frameStart := 0 },
  { event := event227973
    frameStart := 0 },
  { event := event227974
    frameStart := 0 },
  { event := event227975
    frameStart := 0 },
  { event := event227976
    frameStart := 0 },
  { event := event227977
    frameStart := 0 },
  { event := event227978
    frameStart := 0 },
  { event := event227979
    frameStart := 0 },
  { event := event227980
    frameStart := 0 },
  { event := event227981
    frameStart := 0 },
  { event := event227982
    frameStart := 0 },
  { event := event227983
    frameStart := 0 }
]

def eventLeaf14249 : Array AnnotatedEvent := #[
  { event := event227984
    frameStart := 0 },
  { event := event227985
    frameStart := 0 },
  { event := event227986
    frameStart := 0 },
  { event := event227987
    frameStart := 0 },
  { event := event227988
    frameStart := 0 },
  { event := event227989
    frameStart := 0 },
  { event := event227990
    frameStart := 0 },
  { event := event227991
    frameStart := 0 },
  { event := event227992
    frameStart := 0 },
  { event := event227993
    frameStart := 0 },
  { event := event227994
    frameStart := 0 },
  { event := event227995
    frameStart := 0 },
  { event := event227996
    frameStart := 0 },
  { event := event227997
    frameStart := 0 },
  { event := event227998
    frameStart := 0 },
  { event := event227999
    frameStart := 0 }
]

def eventLeaf14250 : Array AnnotatedEvent := #[
  { event := event228000
    frameStart := 0 },
  { event := event228001
    frameStart := 0 },
  { event := event228002
    frameStart := 0 },
  { event := event228003
    frameStart := 0 },
  { event := event228004
    frameStart := 0 },
  { event := event228005
    frameStart := 0 },
  { event := event228006
    frameStart := 0 },
  { event := event228007
    frameStart := 0 },
  { event := event228008
    frameStart := 0 },
  { event := event228009
    frameStart := 0 },
  { event := event228010
    frameStart := 0 },
  { event := event228011
    frameStart := 0 },
  { event := event228012
    frameStart := 0 },
  { event := event228013
    frameStart := 0 },
  { event := event228014
    frameStart := 0 },
  { event := event228015
    frameStart := 0 }
]

def eventLeaf14251 : Array AnnotatedEvent := #[
  { event := event228016
    frameStart := 0 },
  { event := event228017
    frameStart := 0 },
  { event := event228018
    frameStart := 0 },
  { event := event228019
    frameStart := 0 },
  { event := event228020
    frameStart := 0 },
  { event := event228021
    frameStart := 0 },
  { event := event228022
    frameStart := 0 },
  { event := event228023
    frameStart := 0 },
  { event := event228024
    frameStart := 0 },
  { event := event228025
    frameStart := 0 },
  { event := event228026
    frameStart := 0 },
  { event := event228027
    frameStart := 0 },
  { event := event228028
    frameStart := 0 },
  { event := event228029
    frameStart := 0 },
  { event := event228030
    frameStart := 0 },
  { event := event228031
    frameStart := 0 }
]

def eventLeaf14252 : Array AnnotatedEvent := #[
  { event := event228032
    frameStart := 0 },
  { event := event228033
    frameStart := 0 },
  { event := event228034
    frameStart := 0 },
  { event := event228035
    frameStart := 0 },
  { event := event228036
    frameStart := 228036 },
  { event := event228037
    frameStart := 228036 },
  { event := event228038
    frameStart := 228036 },
  { event := event228039
    frameStart := 228036 },
  { event := event228040
    frameStart := 228036 },
  { event := event228041
    frameStart := 228036 },
  { event := event228042
    frameStart := 228036 },
  { event := event228043
    frameStart := 228036 },
  { event := event228044
    frameStart := 228036 },
  { event := event228045
    frameStart := 228036 },
  { event := event228046
    frameStart := 228036 },
  { event := event228047
    frameStart := 228036 }
]

def eventLeaf14253 : Array AnnotatedEvent := #[
  { event := event228048
    frameStart := 228036 },
  { event := event228049
    frameStart := 228036 },
  { event := event228050
    frameStart := 228036 },
  { event := event228051
    frameStart := 228036 },
  { event := event228052
    frameStart := 228036 },
  { event := event228053
    frameStart := 228036 },
  { event := event228054
    frameStart := 228036 },
  { event := event228055
    frameStart := 228036 },
  { event := event228056
    frameStart := 228036 },
  { event := event228057
    frameStart := 228036 },
  { event := event228058
    frameStart := 228036 },
  { event := event228059
    frameStart := 228036 },
  { event := event228060
    frameStart := 228036 },
  { event := event228061
    frameStart := 228036 },
  { event := event228062
    frameStart := 228036 },
  { event := event228063
    frameStart := 228036 }
]

def eventLeaf14254 : Array AnnotatedEvent := #[
  { event := event228064
    frameStart := 228036 },
  { event := event228065
    frameStart := 228036 },
  { event := event228066
    frameStart := 228036 },
  { event := event228067
    frameStart := 228036 },
  { event := event228068
    frameStart := 228036 },
  { event := event228069
    frameStart := 228036 },
  { event := event228070
    frameStart := 228036 },
  { event := event228071
    frameStart := 228036 },
  { event := event228072
    frameStart := 228036 },
  { event := event228073
    frameStart := 228036 },
  { event := event228074
    frameStart := 228036 },
  { event := event228075
    frameStart := 228036 },
  { event := event228076
    frameStart := 228036 },
  { event := event228077
    frameStart := 228036 },
  { event := event228078
    frameStart := 228036 },
  { event := event228079
    frameStart := 228036 }
]

def eventLeaf14255 : Array AnnotatedEvent := #[
  { event := event228080
    frameStart := 228036 },
  { event := event228081
    frameStart := 228036 },
  { event := event228082
    frameStart := 228036 },
  { event := event228083
    frameStart := 228036 },
  { event := event228084
    frameStart := 228084 },
  { event := event228085
    frameStart := 228084 },
  { event := event228086
    frameStart := 228084 },
  { event := event228087
    frameStart := 228084 },
  { event := event228088
    frameStart := 228084 },
  { event := event228089
    frameStart := 228084 },
  { event := event228090
    frameStart := 228084 },
  { event := event228091
    frameStart := 228084 },
  { event := event228092
    frameStart := 228084 },
  { event := event228093
    frameStart := 228084 },
  { event := event228094
    frameStart := 228084 },
  { event := event228095
    frameStart := 228084 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events890
