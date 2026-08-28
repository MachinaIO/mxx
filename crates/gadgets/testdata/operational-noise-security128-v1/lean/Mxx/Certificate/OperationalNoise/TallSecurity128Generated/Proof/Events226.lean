import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events226

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event57856 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37307⟩⟩) 0 ⟨14001⟩ 57855

def event57857 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37307⟩⟩) 1 ⟨37306⟩ 57852

def event57858 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37307⟩⟩) (.product (.predecessor 0 57856 .coefficient) (.predecessor 1 57857 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event57859 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37307⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨14001⟩⟩, ⟨.program ⟨257⟩, ⟨37306⟩⟩], []⟩) [⟨.result 57855 .coefficient, true, some 1⟩, ⟨.result 57852 .coefficient, true, some 1⟩])

def event57860 : Event := .survivorFold (1) 57859

def exact57861RawTerms : List Term := []

theorem exact57861RawTermsValid :
    exact57861RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event57861 : Event := .resultExact (⟨.program ⟨257⟩, ⟨37307⟩⟩) exact57861RawTerms (.finite 1764) 57858 (.finite 1764) (some (57859))

def event57862 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37308⟩⟩) 0 ⟨37307⟩ 57861

def event57863 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37308⟩⟩) (.identity (.predecessor 0 57862 .coefficient))

def event57864 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨37308⟩⟩) (.finite 1764)

def event57865 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37492⟩⟩) 0 ⟨37308⟩ 57864

def event57866 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37492⟩⟩) (.authority (.programFamilyFact))

def exact57867RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨37492⟩⟩], []⟩, (1)⟩]

theorem exact57867RawTermsValid :
    exact57867RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event57867 : Event := .resultExact (⟨.program ⟨257⟩, ⟨37492⟩⟩) exact57867RawTerms (.finite 42) 57866 .exactZero (none)

def event57868 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37493⟩⟩) 0 ⟨37492⟩ 57867

def event57869 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37493⟩⟩) (.identity (.predecessor 0 57868 .coefficient))

def event57870 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨37493⟩⟩) (.finite 42)

def event57871 : Event := .predecessor (⟨.program ⟨257⟩, ⟨38332⟩⟩) 0 ⟨37493⟩ 57870

def event57872 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨38332⟩⟩) (.authority (.relationPreimageSource ⟨84⟩))

def exact57873RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨38332⟩⟩]⟩, (1)⟩]

theorem exact57873RawTermsValid :
    exact57873RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event57873 : Event := .resultExact (⟨.program ⟨257⟩, ⟨38332⟩⟩) exact57873RawTerms (.finite 5647228698) 57872 .exactZero (none)

def event57874 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35⟩⟩) (.authority (.operator))

def exact57875RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩]⟩, (1)⟩]

theorem exact57875RawTermsValid :
    exact57875RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event57875 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35⟩⟩) exact57875RawTerms .large 57874 .exactZero (none)

def event57876 : Event := .predecessor (⟨.program ⟨257⟩, ⟨38333⟩⟩) 0 ⟨35⟩ 57875

def event57877 : Event := .predecessor (⟨.program ⟨257⟩, ⟨38333⟩⟩) 1 ⟨38332⟩ 57873

def event57878 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨38333⟩⟩) (.product (.predecessor 0 57876 .coefficient) (.predecessor 1 57877 .coefficient) (⟨false, false, none, none, none⟩))

def event57879 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨38333⟩⟩, .operator (⟨57875, 0⟩, ⟨57873, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨38332⟩⟩]⟩, (1)⟩)

def exact57880RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨38332⟩⟩]⟩, (1)⟩]

theorem exact57880RawTermsValid :
    exact57880RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event57880 : Event := .resultExact (⟨.program ⟨257⟩, ⟨38333⟩⟩) exact57880RawTerms .large 57878 .exactZero (none)

def event57881 : Event := .preFoldPolynomial 57880 [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨38332⟩⟩]⟩, (1)⟩] .exactZero none

def exact57882RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨38332⟩⟩]⟩, (1)⟩]

def event57882 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨38333⟩⟩) 57881 exact57882RawTerms .large 57878 .exactZero (none)

def event57883 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨39508⟩⟩)

def event57884 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event57885 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event57886 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨11115⟩⟩) (.authority (.operator))

def event57887 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨11115⟩⟩) (.finite 17)

def event57888 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event57889 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event57890 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event57891 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event57892 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 57891

def event57893 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 57889

def event57894 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 57892 .coefficient) (.value (.predecessor 1 57893 .coefficient)))

def event57895 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event57896 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11117⟩⟩) 0 ⟨392⟩ 57895

def event57897 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11117⟩⟩) 1 ⟨11115⟩ 57887

def event57898 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨11117⟩⟩) (.sum [.predecessor 0 57896 .coefficient, .predecessor 1 57897 .coefficient])

def event57899 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨11117⟩⟩) (.finite 655357)

def event57900 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11173⟩⟩) 0 ⟨11117⟩ 57899

def event57901 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11173⟩⟩) 1 ⟨5426⟩ 57885

def event57902 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨11173⟩⟩) (.identity (.predecessor 1 57901 .coefficient))

def event57903 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨11173⟩⟩) (.finite 655360)

def event57904 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37306⟩⟩) 0 ⟨11173⟩ 57903

def event57905 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37306⟩⟩) (.authority (.programFamilyFact))

def exact57906RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨37306⟩⟩], []⟩, (1)⟩]

theorem exact57906RawTermsValid :
    exact57906RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event57906 : Event := .resultExact (⟨.program ⟨257⟩, ⟨37306⟩⟩) exact57906RawTerms (.finite 42) 57905 .exactZero (none)

def event57907 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14001⟩⟩) 0 ⟨11173⟩ 57903

def event57908 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14001⟩⟩) (.authority (.programFamilyFact))

def exact57909RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14001⟩⟩], []⟩, (1)⟩]

theorem exact57909RawTermsValid :
    exact57909RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event57909 : Event := .resultExact (⟨.program ⟨257⟩, ⟨14001⟩⟩) exact57909RawTerms (.finite 42) 57908 .exactZero (none)

def event57910 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37307⟩⟩) 0 ⟨14001⟩ 57909

def event57911 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37307⟩⟩) 1 ⟨37306⟩ 57906

def event57912 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37307⟩⟩) (.product (.predecessor 0 57910 .coefficient) (.predecessor 1 57911 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event57913 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨37307⟩⟩, .operator (⟨57909, 0⟩, ⟨57906, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨14001⟩⟩, ⟨.program ⟨257⟩, ⟨37306⟩⟩], []⟩, (1)⟩)

def exact57914RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14001⟩⟩, ⟨.program ⟨257⟩, ⟨37306⟩⟩], []⟩, (1)⟩]

theorem exact57914RawTermsValid :
    exact57914RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event57914 : Event := .resultExact (⟨.program ⟨257⟩, ⟨37307⟩⟩) exact57914RawTerms (.finite 1764) 57912 .exactZero (none)

def event57915 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37308⟩⟩) 0 ⟨37307⟩ 57914

def event57916 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37308⟩⟩) (.identity (.predecessor 0 57915 .coefficient))

def event57917 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨37308⟩⟩) (.finite 1764)

def event57918 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37492⟩⟩) 0 ⟨37308⟩ 57917

def event57919 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37492⟩⟩) (.authority (.programFamilyFact))

def exact57920RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨37492⟩⟩], []⟩, (1)⟩]

theorem exact57920RawTermsValid :
    exact57920RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event57920 : Event := .resultExact (⟨.program ⟨257⟩, ⟨37492⟩⟩) exact57920RawTerms (.finite 42) 57919 .exactZero (none)

def event57921 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37493⟩⟩) 0 ⟨37492⟩ 57920

def event57922 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37493⟩⟩) (.identity (.predecessor 0 57921 .coefficient))

def event57923 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨37493⟩⟩) (.finite 42)

def event57924 : Event := .predecessor (⟨.program ⟨257⟩, ⟨38651⟩⟩) 0 ⟨37493⟩ 57923

def event57925 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨38651⟩⟩) (.authority (.programFamilyFact))

def event57926 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨38651⟩⟩) (.finite 3720)

def event57927 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨7177⟩⟩) .missing

def event57928 : Event := .predecessor (⟨.program ⟨257⟩, ⟨38652⟩⟩) 0 ⟨7177⟩ 57927

def event57929 : Event := .predecessor (⟨.program ⟨257⟩, ⟨38652⟩⟩) 1 ⟨38651⟩ 57926

def event57930 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨38652⟩⟩) (.authority (.operator))

def exact57931RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨38652⟩⟩]⟩, (1)⟩]

theorem exact57931RawTermsValid :
    exact57931RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event57931 : Event := .resultExact (⟨.program ⟨257⟩, ⟨38652⟩⟩) exact57931RawTerms .large 57930 .exactZero (none)

def event57932 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39503⟩⟩) 0 ⟨38652⟩ 57931

def event57933 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39503⟩⟩) (.authority (.operator))

def exact57934RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨39503⟩⟩]⟩, (1)⟩]

theorem exact57934RawTermsValid :
    exact57934RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event57934 : Event := .resultExact (⟨.program ⟨257⟩, ⟨39503⟩⟩) exact57934RawTerms (.finite 8192) 57933 .exactZero (none)

def event57935 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨136⟩⟩) (.authority (.operator))

def event57936 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨136⟩⟩) .exactZero

def event57937 : Event := .predecessor (⟨.program ⟨257⟩, ⟨38818⟩⟩) 0 ⟨37493⟩ 57923

def event57938 : Event := .predecessor (⟨.program ⟨257⟩, ⟨38818⟩⟩) 1 ⟨136⟩ 57936

def event57939 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨38818⟩⟩) (.sum [.predecessor 0 57937 .coefficient, .predecessor 1 57938 .coefficient])

def event57940 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨38818⟩⟩) (.finite 42)

def event57941 : Event := .predecessor (⟨.program ⟨257⟩, ⟨38819⟩⟩) 0 ⟨38818⟩ 57940

def event57942 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨38819⟩⟩) (.identity (.predecessor 0 57941 .coefficient))

def exact57943RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨37492⟩⟩], []⟩, (1)⟩]

theorem exact57943RawTermsValid :
    exact57943RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event57943 : Event := .resultExact (⟨.program ⟨257⟩, ⟨38819⟩⟩) exact57943RawTerms (.finite 42) 57942 .exactZero (none)

def event57944 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6908⟩⟩) (.authority (.factStore))

def exact57945RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact57945RawTermsValid :
    exact57945RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event57945 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6908⟩⟩) exact57945RawTerms .large 57944 .exactZero (none)

def event57946 : Event := .predecessor (⟨.program ⟨257⟩, ⟨38820⟩⟩) 0 ⟨6908⟩ 57945

def event57947 : Event := .predecessor (⟨.program ⟨257⟩, ⟨38820⟩⟩) 1 ⟨38819⟩ 57943

def event57948 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨38820⟩⟩) (.product (.predecessor 0 57946 .coefficient) (.predecessor 1 57947 .coefficient) (⟨false, false, none, none, none⟩))

def event57949 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨38820⟩⟩, .operator (⟨57945, 0⟩, ⟨57943, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨37492⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact57950RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨37492⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact57950RawTermsValid :
    exact57950RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event57950 : Event := .resultExact (⟨.program ⟨257⟩, ⟨38820⟩⟩) exact57950RawTerms .large 57948 .exactZero (none)

def event57951 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7192⟩⟩) 0 ⟨7177⟩ 57927

def event57952 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7192⟩⟩) (.authority (.operator))

def exact57953RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7192⟩⟩]⟩, (1)⟩]

theorem exact57953RawTermsValid :
    exact57953RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event57953 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7192⟩⟩) exact57953RawTerms .large 57952 .exactZero (none)

def event57954 : Event := .predecessor (⟨.program ⟨257⟩, ⟨38821⟩⟩) 0 ⟨7192⟩ 57953

def event57955 : Event := .predecessor (⟨.program ⟨257⟩, ⟨38821⟩⟩) 1 ⟨38820⟩ 57950

def event57956 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨38821⟩⟩) (.sum [.predecessor 0 57954 .coefficient, .predecessor 1 57955 .coefficient])

def exact57957RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7192⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨37492⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact57957RawTermsValid :
    exact57957RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event57957 : Event := .resultExact (⟨.program ⟨257⟩, ⟨38821⟩⟩) exact57957RawTerms .large 57956 .exactZero (none)

def event57958 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39504⟩⟩) 0 ⟨38821⟩ 57957

def event57959 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39504⟩⟩) 1 ⟨39503⟩ 57934

def event57960 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39504⟩⟩) (.product (.predecessor 0 57958 .coefficient) (.predecessor 1 57959 .coefficient) (⟨false, false, none, none, none⟩))

def event57961 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨39504⟩⟩, .operator (⟨57957, 0⟩, ⟨57934, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7192⟩⟩, ⟨.program ⟨257⟩, ⟨39503⟩⟩]⟩, (1)⟩)

def event57962 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨39504⟩⟩, .operator (⟨57957, 1⟩, ⟨57934, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨37492⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨39503⟩⟩]⟩, (-1)⟩)

def event57963 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨39504⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨37492⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨39503⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨39503⟩⟩) ⟨38652⟩ 57931)

def event57964 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨39504⟩⟩, .relation 57963 0, ⟨[⟨.program ⟨257⟩, ⟨37492⟩⟩], [⟨.program ⟨257⟩, ⟨38652⟩⟩]⟩, (-1)⟩)

def exact57965RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7192⟩⟩, ⟨.program ⟨257⟩, ⟨39503⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨37492⟩⟩], [⟨.program ⟨257⟩, ⟨38652⟩⟩]⟩, (-1)⟩]

theorem exact57965RawTermsValid :
    exact57965RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event57965 : Event := .resultExact (⟨.program ⟨257⟩, ⟨39504⟩⟩) exact57965RawTerms .large 57960 .exactZero (none)

def event57966 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37743⟩⟩) 0 ⟨37493⟩ 57923

def event57967 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37743⟩⟩) (.authority (.programFamilyFact))

def exact57968RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨37743⟩⟩], []⟩, (1)⟩]

theorem exact57968RawTermsValid :
    exact57968RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event57968 : Event := .resultExact (⟨.program ⟨257⟩, ⟨37743⟩⟩) exact57968RawTerms (.finite 42) 57967 .exactZero (none)

def event57969 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37745⟩⟩) 0 ⟨6908⟩ 57945

def event57970 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37745⟩⟩) 1 ⟨37743⟩ 57968

def event57971 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37745⟩⟩) (.product (.predecessor 0 57969 .coefficient) (.predecessor 1 57970 .coefficient) (⟨false, true, none, none, some 1⟩))

def event57972 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨37745⟩⟩, .operator (⟨57945, 0⟩, ⟨57968, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨37743⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact57973RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨37743⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact57973RawTermsValid :
    exact57973RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event57973 : Event := .resultExact (⟨.program ⟨257⟩, ⟨37745⟩⟩) exact57973RawTerms .large 57971 .exactZero (none)

def event57974 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7223⟩⟩) 0 ⟨7177⟩ 57927

def event57975 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7223⟩⟩) (.authority (.operator))

def exact57976RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7223⟩⟩]⟩, (1)⟩]

theorem exact57976RawTermsValid :
    exact57976RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event57976 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7223⟩⟩) exact57976RawTerms .large 57975 .exactZero (none)

def event57977 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37746⟩⟩) 0 ⟨7223⟩ 57976

def event57978 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37746⟩⟩) 1 ⟨37745⟩ 57973

def event57979 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37746⟩⟩) (.sum [.predecessor 0 57977 .coefficient, .predecessor 1 57978 .coefficient])

def exact57980RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7223⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨37743⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact57980RawTermsValid :
    exact57980RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event57980 : Event := .resultExact (⟨.program ⟨257⟩, ⟨37746⟩⟩) exact57980RawTerms .large 57979 .exactZero (none)

def event57981 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39508⟩⟩) 0 ⟨37746⟩ 57980

def event57982 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39508⟩⟩) 1 ⟨39504⟩ 57965

def event57983 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39508⟩⟩) (.sum [.predecessor 0 57981 .coefficient, .predecessor 1 57982 .coefficient])

def exact57984RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7192⟩⟩, ⟨.program ⟨257⟩, ⟨39503⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7223⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨37492⟩⟩], [⟨.program ⟨257⟩, ⟨38652⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨37743⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact57984RawTermsValid :
    exact57984RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event57984 : Event := .resultExact (⟨.program ⟨257⟩, ⟨39508⟩⟩) exact57984RawTerms .large 57983 .exactZero (none)

def event57985 : Event := .preFoldPolynomial 57984 [⟨⟨[], [⟨.program ⟨257⟩, ⟨7192⟩⟩, ⟨.program ⟨257⟩, ⟨39503⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7223⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨37492⟩⟩], [⟨.program ⟨257⟩, ⟨38652⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨37743⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩] .exactZero none

def exact57986RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7192⟩⟩, ⟨.program ⟨257⟩, ⟨39503⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7223⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨37492⟩⟩], [⟨.program ⟨257⟩, ⟨38652⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨37743⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

def event57986 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨39508⟩⟩) 57985 exact57986RawTerms .large 57983 .exactZero (none)

def event57987 : Event := .specializationComputed (⟨.program ⟨257⟩, ⟨37493⟩⟩) ⟨⟨102⟩, ⟨84⟩, ⟨135⟩⟩ ⟨57829, 57987⟩

def event57988 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨38335⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨38332⟩⟩]⟩) (1) 0 2 (.universal 57987 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨38332⟩⟩]⟩) (none) 57986)

def event57989 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨38335⟩⟩, .relation 57988 1, ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7223⟩⟩]⟩, (1)⟩)

def event57990 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨38335⟩⟩, .relation 57988 0, ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7192⟩⟩, ⟨.program ⟨257⟩, ⟨39503⟩⟩]⟩, (-1)⟩)

def event57991 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨38335⟩⟩, .relation 57988 2, ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨37492⟩⟩], [⟨.program ⟨257⟩, ⟨38652⟩⟩]⟩, (1)⟩)

def event57992 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨38335⟩⟩, .relation 57988 3, ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨37743⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact57993RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7192⟩⟩, ⟨.program ⟨257⟩, ⟨39503⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7223⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨37492⟩⟩], [⟨.program ⟨257⟩, ⟨38652⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨37743⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact57993RawTermsValid :
    exact57993RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event57993 : Event := .resultExact (⟨.program ⟨257⟩, ⟨38335⟩⟩) exact57993RawTerms .large 57825 (.finite 202072841853861888) (some (57827))

def event57994 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39506⟩⟩) 0 ⟨38335⟩ 57993

def event57995 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39506⟩⟩) 1 ⟨39505⟩ 57815

def event57996 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39506⟩⟩) (.sum [.predecessor 0 57994 .coefficient, .predecessor 1 57995 .coefficient])

def event57997 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨39506⟩⟩, .operator (⟨57993, 0⟩, ⟨57815, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7192⟩⟩, ⟨.program ⟨257⟩, ⟨39503⟩⟩]⟩, (1)⟩)

def event57998 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨39506⟩⟩, .operator (⟨57993, 2⟩, ⟨57815, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨37492⟩⟩], [⟨.program ⟨257⟩, ⟨38652⟩⟩]⟩, (-1)⟩)

def event57999 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39506⟩⟩) (.sum [.result 57993 .summary, .result 57815 .summary])

def exact58000RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7223⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨37743⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact58000RawTermsValid :
    exact58000RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event58000 : Event := .resultExact (⟨.program ⟨257⟩, ⟨39506⟩⟩) exact58000RawTerms .large 57996 (.finite 32192736221397454434328420548608) (some (57999))

def event58001 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39507⟩⟩) 0 ⟨39506⟩ 58000

def event58002 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39507⟩⟩) 1 ⟨7162⟩ 15622

def event58003 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39507⟩⟩) (.product (.predecessor 0 58001 .coefficient) (.predecessor 1 58002 .coefficient) (⟨false, false, none, none, none⟩))

def event58004 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39507⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨7161⟩⟩]⟩) [⟨.result 15618 .coefficient, false, none⟩])

def event58005 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39507⟩⟩) (.product (.result 58000 .summary) (.transfer 58004) (⟨false, false, none, none, none⟩))

def event58006 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨39507⟩⟩, .operator (⟨58000, 0⟩, ⟨15622, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7223⟩⟩, ⟨.program ⟨257⟩, ⟨7161⟩⟩]⟩, (1)⟩)

def event58007 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨39507⟩⟩, .operator (⟨58000, 1⟩, ⟨15622, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨37743⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7161⟩⟩]⟩, (-1)⟩)

def event58008 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨39507⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨37743⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7161⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨7161⟩⟩) ⟨7046⟩ 15615)

def event58009 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨39507⟩⟩, .relation 58008 0, ⟨[⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨37743⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact58010RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨37743⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7223⟩⟩, ⟨.program ⟨257⟩, ⟨7161⟩⟩]⟩, (1)⟩]

theorem exact58010RawTermsValid :
    exact58010RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event58010 : Event := .resultExact (⟨.program ⟨257⟩, ⟨39507⟩⟩) exact58010RawTerms .large 58003 (.finite 345666873099141705532726864949014345809920) (some (58005))

def event58011 : Event := .predecessor (⟨.program ⟨257⟩, ⟨35972⟩⟩) 0 ⟨7177⟩ 15500

def event58012 : Event := .predecessor (⟨.program ⟨257⟩, ⟨35972⟩⟩) 1 ⟨35971⟩ 49057

def event58013 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35972⟩⟩) (.authority (.operator))

def exact58014RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35972⟩⟩]⟩, (1)⟩]

theorem exact58014RawTermsValid :
    exact58014RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event58014 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35972⟩⟩) exact58014RawTerms .large 58013 .exactZero (none)

def event58015 : Event := .predecessor (⟨.program ⟨257⟩, ⟨36823⟩⟩) 0 ⟨35972⟩ 58014

def event58016 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨36823⟩⟩) (.authority (.operator))

def exact58017RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨36823⟩⟩]⟩, (1)⟩]

theorem exact58017RawTermsValid :
    exact58017RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event58017 : Event := .resultExact (⟨.program ⟨257⟩, ⟨36823⟩⟩) exact58017RawTerms (.finite 8192) 58016 .exactZero (none)

def event58018 : Event := .predecessor (⟨.program ⟨257⟩, ⟨36825⟩⟩) 0 ⟨36349⟩ 49341

def event58019 : Event := .predecessor (⟨.program ⟨257⟩, ⟨36825⟩⟩) 1 ⟨36823⟩ 58017

def event58020 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨36825⟩⟩) (.product (.predecessor 0 58018 .coefficient) (.predecessor 1 58019 .coefficient) (⟨false, false, none, none, none⟩))

def event58021 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨36825⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨36823⟩⟩]⟩) [⟨.result 58017 .coefficient, false, none⟩])

def event58022 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨36825⟩⟩) (.product (.result 49341 .summary) (.transfer 58021) (⟨false, false, none, none, none⟩))

def event58023 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨36825⟩⟩, .operator (⟨49341, 0⟩, ⟨58017, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7191⟩⟩, ⟨.program ⟨257⟩, ⟨36823⟩⟩]⟩, (1)⟩)

def event58024 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨36825⟩⟩, .operator (⟨49341, 1⟩, ⟨58017, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨34812⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨36823⟩⟩]⟩, (-1)⟩)

def event58025 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨36825⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨34812⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨36823⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨36823⟩⟩) ⟨35972⟩ 58014)

def event58026 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨36825⟩⟩, .relation 58025 0, ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨34812⟩⟩], [⟨.program ⟨257⟩, ⟨35972⟩⟩]⟩, (-1)⟩)

def exact58027RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7191⟩⟩, ⟨.program ⟨257⟩, ⟨36823⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨34812⟩⟩], [⟨.program ⟨257⟩, ⟨35972⟩⟩]⟩, (-1)⟩]

theorem exact58027RawTermsValid :
    exact58027RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event58027 : Event := .resultExact (⟨.program ⟨257⟩, ⟨36825⟩⟩) exact58027RawTerms .large 58020 (.finite 32192539770951564984245676933120) (some (58022))

def event58028 : Event := .predecessor (⟨.program ⟨257⟩, ⟨35652⟩⟩) 0 ⟨34813⟩ 1722

def event58029 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35652⟩⟩) (.authority (.relationPreimageSource ⟨82⟩))

def exact58030RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35652⟩⟩]⟩, (1)⟩]

theorem exact58030RawTermsValid :
    exact58030RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event58030 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35652⟩⟩) exact58030RawTerms (.finite 5647228698) 58029 .exactZero (none)

def event58031 : Event := .predecessor (⟨.program ⟨257⟩, ⟨35654⟩⟩) 0 ⟨35652⟩ 58030

def event58032 : Event := .predecessor (⟨.program ⟨257⟩, ⟨35654⟩⟩) 1 ⟨2370⟩ 4

def event58033 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35654⟩⟩) (.scale (.predecessor 0 58031 .coefficient) (.value (.predecessor 1 58032 .coefficient)))

def exact58034RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35652⟩⟩]⟩, (1)⟩]

theorem exact58034RawTermsValid :
    exact58034RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event58034 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35654⟩⟩) exact58034RawTerms (.finite 5647228698) 58033 .exactZero (none)

def event58035 : Event := .predecessor (⟨.program ⟨257⟩, ⟨35655⟩⟩) 0 ⟨11216⟩ 46745

def event58036 : Event := .predecessor (⟨.program ⟨257⟩, ⟨35655⟩⟩) 1 ⟨35654⟩ 58034

def event58037 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35655⟩⟩) (.product (.predecessor 0 58035 .coefficient) (.predecessor 1 58036 .coefficient) (⟨false, false, none, none, none⟩))

def event58038 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35655⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨35652⟩⟩]⟩) [⟨.result 58030 .coefficient, false, none⟩])

def event58039 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35655⟩⟩) (.product (.result 46745 .summary) (.transfer 58038) (⟨false, false, none, none, none⟩))

def event58040 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨35655⟩⟩, .operator (⟨46745, 0⟩, ⟨58034, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35652⟩⟩]⟩, (1)⟩)

def event58041 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨35653⟩⟩)

def event58042 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event58043 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event58044 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨11115⟩⟩) (.authority (.operator))

def event58045 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨11115⟩⟩) (.finite 17)

def event58046 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event58047 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event58048 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event58049 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event58050 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 58049

def event58051 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 58047

def event58052 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 58050 .coefficient) (.value (.predecessor 1 58051 .coefficient)))

def event58053 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event58054 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11117⟩⟩) 0 ⟨392⟩ 58053

def event58055 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11117⟩⟩) 1 ⟨11115⟩ 58045

def event58056 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨11117⟩⟩) (.sum [.predecessor 0 58054 .coefficient, .predecessor 1 58055 .coefficient])

def event58057 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨11117⟩⟩) (.finite 655357)

def event58058 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11173⟩⟩) 0 ⟨11117⟩ 58057

def event58059 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11173⟩⟩) 1 ⟨5426⟩ 58043

def event58060 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨11173⟩⟩) (.identity (.predecessor 1 58059 .coefficient))

def event58061 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨11173⟩⟩) (.finite 655360)

def event58062 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34626⟩⟩) 0 ⟨11173⟩ 58061

def event58063 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34626⟩⟩) (.authority (.programFamilyFact))

def exact58064RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨34626⟩⟩], []⟩, (1)⟩]

theorem exact58064RawTermsValid :
    exact58064RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event58064 : Event := .resultExact (⟨.program ⟨257⟩, ⟨34626⟩⟩) exact58064RawTerms (.finite 40) 58063 .exactZero (none)

def event58065 : Event := .predecessor (⟨.program ⟨257⟩, ⟨13701⟩⟩) 0 ⟨11173⟩ 58061

def event58066 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨13701⟩⟩) (.authority (.programFamilyFact))

def exact58067RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨13701⟩⟩], []⟩, (1)⟩]

theorem exact58067RawTermsValid :
    exact58067RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event58067 : Event := .resultExact (⟨.program ⟨257⟩, ⟨13701⟩⟩) exact58067RawTerms (.finite 40) 58066 .exactZero (none)

def event58068 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34627⟩⟩) 0 ⟨13701⟩ 58067

def event58069 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34627⟩⟩) 1 ⟨34626⟩ 58064

def event58070 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34627⟩⟩) (.product (.predecessor 0 58068 .coefficient) (.predecessor 1 58069 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event58071 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34627⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨13701⟩⟩, ⟨.program ⟨257⟩, ⟨34626⟩⟩], []⟩) [⟨.result 58067 .coefficient, true, some 1⟩, ⟨.result 58064 .coefficient, true, some 1⟩])

def event58072 : Event := .survivorFold (1) 58071

def exact58073RawTerms : List Term := []

theorem exact58073RawTermsValid :
    exact58073RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event58073 : Event := .resultExact (⟨.program ⟨257⟩, ⟨34627⟩⟩) exact58073RawTerms (.finite 1600) 58070 (.finite 1600) (some (58071))

def event58074 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34628⟩⟩) 0 ⟨34627⟩ 58073

def event58075 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34628⟩⟩) (.identity (.predecessor 0 58074 .coefficient))

def event58076 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨34628⟩⟩) (.finite 1600)

def event58077 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34812⟩⟩) 0 ⟨34628⟩ 58076

def event58078 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34812⟩⟩) (.authority (.programFamilyFact))

def exact58079RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨34812⟩⟩], []⟩, (1)⟩]

theorem exact58079RawTermsValid :
    exact58079RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event58079 : Event := .resultExact (⟨.program ⟨257⟩, ⟨34812⟩⟩) exact58079RawTerms (.finite 40) 58078 .exactZero (none)

def event58080 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34813⟩⟩) 0 ⟨34812⟩ 58079

def event58081 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34813⟩⟩) (.identity (.predecessor 0 58080 .coefficient))

def event58082 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨34813⟩⟩) (.finite 40)

def event58083 : Event := .predecessor (⟨.program ⟨257⟩, ⟨35652⟩⟩) 0 ⟨34813⟩ 58082

def event58084 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35652⟩⟩) (.authority (.relationPreimageSource ⟨82⟩))

def exact58085RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35652⟩⟩]⟩, (1)⟩]

theorem exact58085RawTermsValid :
    exact58085RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event58085 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35652⟩⟩) exact58085RawTerms (.finite 5647228698) 58084 .exactZero (none)

def event58086 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35⟩⟩) (.authority (.operator))

def exact58087RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩]⟩, (1)⟩]

theorem exact58087RawTermsValid :
    exact58087RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event58087 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35⟩⟩) exact58087RawTerms .large 58086 .exactZero (none)

def event58088 : Event := .predecessor (⟨.program ⟨257⟩, ⟨35653⟩⟩) 0 ⟨35⟩ 58087

def event58089 : Event := .predecessor (⟨.program ⟨257⟩, ⟨35653⟩⟩) 1 ⟨35652⟩ 58085

def event58090 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35653⟩⟩) (.product (.predecessor 0 58088 .coefficient) (.predecessor 1 58089 .coefficient) (⟨false, false, none, none, none⟩))

def event58091 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨35653⟩⟩, .operator (⟨58087, 0⟩, ⟨58085, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35652⟩⟩]⟩, (1)⟩)

def exact58092RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35652⟩⟩]⟩, (1)⟩]

theorem exact58092RawTermsValid :
    exact58092RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event58092 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35653⟩⟩) exact58092RawTerms .large 58090 .exactZero (none)

def event58093 : Event := .preFoldPolynomial 58092 [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35652⟩⟩]⟩, (1)⟩] .exactZero none

def exact58094RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35652⟩⟩]⟩, (1)⟩]

def event58094 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨35653⟩⟩) 58093 exact58094RawTerms .large 58090 .exactZero (none)

def event58095 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨36828⟩⟩)

def event58096 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event58097 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event58098 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨11115⟩⟩) (.authority (.operator))

def event58099 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨11115⟩⟩) (.finite 17)

def event58100 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event58101 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event58102 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event58103 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event58104 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 58103

def event58105 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 58101

def event58106 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 58104 .coefficient) (.value (.predecessor 1 58105 .coefficient)))

def event58107 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event58108 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11117⟩⟩) 0 ⟨392⟩ 58107

def event58109 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11117⟩⟩) 1 ⟨11115⟩ 58099

def event58110 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨11117⟩⟩) (.sum [.predecessor 0 58108 .coefficient, .predecessor 1 58109 .coefficient])

def event58111 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨11117⟩⟩) (.finite 655357)

def eventLeaf3616 : Array AnnotatedEvent := #[
  { event := event57856
    frameStart := 57829 },
  { event := event57857
    frameStart := 57829 },
  { event := event57858
    frameStart := 57829 },
  { event := event57859
    frameStart := 57829 },
  { event := event57860
    frameStart := 57829 },
  { event := event57861
    frameStart := 57829 },
  { event := event57862
    frameStart := 57829 },
  { event := event57863
    frameStart := 57829 },
  { event := event57864
    frameStart := 57829 },
  { event := event57865
    frameStart := 57829 },
  { event := event57866
    frameStart := 57829 },
  { event := event57867
    frameStart := 57829 },
  { event := event57868
    frameStart := 57829 },
  { event := event57869
    frameStart := 57829 },
  { event := event57870
    frameStart := 57829 },
  { event := event57871
    frameStart := 57829 }
]

def eventLeaf3617 : Array AnnotatedEvent := #[
  { event := event57872
    frameStart := 57829 },
  { event := event57873
    frameStart := 57829 },
  { event := event57874
    frameStart := 57829 },
  { event := event57875
    frameStart := 57829 },
  { event := event57876
    frameStart := 57829 },
  { event := event57877
    frameStart := 57829 },
  { event := event57878
    frameStart := 57829 },
  { event := event57879
    frameStart := 57829 },
  { event := event57880
    frameStart := 57829 },
  { event := event57881
    frameStart := 57829 },
  { event := event57882
    frameStart := 57829 },
  { event := event57883
    frameStart := 57883 },
  { event := event57884
    frameStart := 57883 },
  { event := event57885
    frameStart := 57883 },
  { event := event57886
    frameStart := 57883 },
  { event := event57887
    frameStart := 57883 }
]

def eventLeaf3618 : Array AnnotatedEvent := #[
  { event := event57888
    frameStart := 57883 },
  { event := event57889
    frameStart := 57883 },
  { event := event57890
    frameStart := 57883 },
  { event := event57891
    frameStart := 57883 },
  { event := event57892
    frameStart := 57883 },
  { event := event57893
    frameStart := 57883 },
  { event := event57894
    frameStart := 57883 },
  { event := event57895
    frameStart := 57883 },
  { event := event57896
    frameStart := 57883 },
  { event := event57897
    frameStart := 57883 },
  { event := event57898
    frameStart := 57883 },
  { event := event57899
    frameStart := 57883 },
  { event := event57900
    frameStart := 57883 },
  { event := event57901
    frameStart := 57883 },
  { event := event57902
    frameStart := 57883 },
  { event := event57903
    frameStart := 57883 }
]

def eventLeaf3619 : Array AnnotatedEvent := #[
  { event := event57904
    frameStart := 57883 },
  { event := event57905
    frameStart := 57883 },
  { event := event57906
    frameStart := 57883 },
  { event := event57907
    frameStart := 57883 },
  { event := event57908
    frameStart := 57883 },
  { event := event57909
    frameStart := 57883 },
  { event := event57910
    frameStart := 57883 },
  { event := event57911
    frameStart := 57883 },
  { event := event57912
    frameStart := 57883 },
  { event := event57913
    frameStart := 57883 },
  { event := event57914
    frameStart := 57883 },
  { event := event57915
    frameStart := 57883 },
  { event := event57916
    frameStart := 57883 },
  { event := event57917
    frameStart := 57883 },
  { event := event57918
    frameStart := 57883 },
  { event := event57919
    frameStart := 57883 }
]

def eventLeaf3620 : Array AnnotatedEvent := #[
  { event := event57920
    frameStart := 57883 },
  { event := event57921
    frameStart := 57883 },
  { event := event57922
    frameStart := 57883 },
  { event := event57923
    frameStart := 57883 },
  { event := event57924
    frameStart := 57883 },
  { event := event57925
    frameStart := 57883 },
  { event := event57926
    frameStart := 57883 },
  { event := event57927
    frameStart := 57883 },
  { event := event57928
    frameStart := 57883 },
  { event := event57929
    frameStart := 57883 },
  { event := event57930
    frameStart := 57883 },
  { event := event57931
    frameStart := 57883 },
  { event := event57932
    frameStart := 57883 },
  { event := event57933
    frameStart := 57883 },
  { event := event57934
    frameStart := 57883 },
  { event := event57935
    frameStart := 57883 }
]

def eventLeaf3621 : Array AnnotatedEvent := #[
  { event := event57936
    frameStart := 57883 },
  { event := event57937
    frameStart := 57883 },
  { event := event57938
    frameStart := 57883 },
  { event := event57939
    frameStart := 57883 },
  { event := event57940
    frameStart := 57883 },
  { event := event57941
    frameStart := 57883 },
  { event := event57942
    frameStart := 57883 },
  { event := event57943
    frameStart := 57883 },
  { event := event57944
    frameStart := 57883 },
  { event := event57945
    frameStart := 57883 },
  { event := event57946
    frameStart := 57883 },
  { event := event57947
    frameStart := 57883 },
  { event := event57948
    frameStart := 57883 },
  { event := event57949
    frameStart := 57883 },
  { event := event57950
    frameStart := 57883 },
  { event := event57951
    frameStart := 57883 }
]

def eventLeaf3622 : Array AnnotatedEvent := #[
  { event := event57952
    frameStart := 57883 },
  { event := event57953
    frameStart := 57883 },
  { event := event57954
    frameStart := 57883 },
  { event := event57955
    frameStart := 57883 },
  { event := event57956
    frameStart := 57883 },
  { event := event57957
    frameStart := 57883 },
  { event := event57958
    frameStart := 57883 },
  { event := event57959
    frameStart := 57883 },
  { event := event57960
    frameStart := 57883 },
  { event := event57961
    frameStart := 57883 },
  { event := event57962
    frameStart := 57883 },
  { event := event57963
    frameStart := 57883 },
  { event := event57964
    frameStart := 57883 },
  { event := event57965
    frameStart := 57883 },
  { event := event57966
    frameStart := 57883 },
  { event := event57967
    frameStart := 57883 }
]

def eventLeaf3623 : Array AnnotatedEvent := #[
  { event := event57968
    frameStart := 57883 },
  { event := event57969
    frameStart := 57883 },
  { event := event57970
    frameStart := 57883 },
  { event := event57971
    frameStart := 57883 },
  { event := event57972
    frameStart := 57883 },
  { event := event57973
    frameStart := 57883 },
  { event := event57974
    frameStart := 57883 },
  { event := event57975
    frameStart := 57883 },
  { event := event57976
    frameStart := 57883 },
  { event := event57977
    frameStart := 57883 },
  { event := event57978
    frameStart := 57883 },
  { event := event57979
    frameStart := 57883 },
  { event := event57980
    frameStart := 57883 },
  { event := event57981
    frameStart := 57883 },
  { event := event57982
    frameStart := 57883 },
  { event := event57983
    frameStart := 57883 }
]

def eventLeaf3624 : Array AnnotatedEvent := #[
  { event := event57984
    frameStart := 57883 },
  { event := event57985
    frameStart := 57883 },
  { event := event57986
    frameStart := 57883 },
  { event := event57987
    frameStart := 0 },
  { event := event57988
    frameStart := 0 },
  { event := event57989
    frameStart := 0 },
  { event := event57990
    frameStart := 0 },
  { event := event57991
    frameStart := 0 },
  { event := event57992
    frameStart := 0 },
  { event := event57993
    frameStart := 0 },
  { event := event57994
    frameStart := 0 },
  { event := event57995
    frameStart := 0 },
  { event := event57996
    frameStart := 0 },
  { event := event57997
    frameStart := 0 },
  { event := event57998
    frameStart := 0 },
  { event := event57999
    frameStart := 0 }
]

def eventLeaf3625 : Array AnnotatedEvent := #[
  { event := event58000
    frameStart := 0 },
  { event := event58001
    frameStart := 0 },
  { event := event58002
    frameStart := 0 },
  { event := event58003
    frameStart := 0 },
  { event := event58004
    frameStart := 0 },
  { event := event58005
    frameStart := 0 },
  { event := event58006
    frameStart := 0 },
  { event := event58007
    frameStart := 0 },
  { event := event58008
    frameStart := 0 },
  { event := event58009
    frameStart := 0 },
  { event := event58010
    frameStart := 0 },
  { event := event58011
    frameStart := 0 },
  { event := event58012
    frameStart := 0 },
  { event := event58013
    frameStart := 0 },
  { event := event58014
    frameStart := 0 },
  { event := event58015
    frameStart := 0 }
]

def eventLeaf3626 : Array AnnotatedEvent := #[
  { event := event58016
    frameStart := 0 },
  { event := event58017
    frameStart := 0 },
  { event := event58018
    frameStart := 0 },
  { event := event58019
    frameStart := 0 },
  { event := event58020
    frameStart := 0 },
  { event := event58021
    frameStart := 0 },
  { event := event58022
    frameStart := 0 },
  { event := event58023
    frameStart := 0 },
  { event := event58024
    frameStart := 0 },
  { event := event58025
    frameStart := 0 },
  { event := event58026
    frameStart := 0 },
  { event := event58027
    frameStart := 0 },
  { event := event58028
    frameStart := 0 },
  { event := event58029
    frameStart := 0 },
  { event := event58030
    frameStart := 0 },
  { event := event58031
    frameStart := 0 }
]

def eventLeaf3627 : Array AnnotatedEvent := #[
  { event := event58032
    frameStart := 0 },
  { event := event58033
    frameStart := 0 },
  { event := event58034
    frameStart := 0 },
  { event := event58035
    frameStart := 0 },
  { event := event58036
    frameStart := 0 },
  { event := event58037
    frameStart := 0 },
  { event := event58038
    frameStart := 0 },
  { event := event58039
    frameStart := 0 },
  { event := event58040
    frameStart := 0 },
  { event := event58041
    frameStart := 58041 },
  { event := event58042
    frameStart := 58041 },
  { event := event58043
    frameStart := 58041 },
  { event := event58044
    frameStart := 58041 },
  { event := event58045
    frameStart := 58041 },
  { event := event58046
    frameStart := 58041 },
  { event := event58047
    frameStart := 58041 }
]

def eventLeaf3628 : Array AnnotatedEvent := #[
  { event := event58048
    frameStart := 58041 },
  { event := event58049
    frameStart := 58041 },
  { event := event58050
    frameStart := 58041 },
  { event := event58051
    frameStart := 58041 },
  { event := event58052
    frameStart := 58041 },
  { event := event58053
    frameStart := 58041 },
  { event := event58054
    frameStart := 58041 },
  { event := event58055
    frameStart := 58041 },
  { event := event58056
    frameStart := 58041 },
  { event := event58057
    frameStart := 58041 },
  { event := event58058
    frameStart := 58041 },
  { event := event58059
    frameStart := 58041 },
  { event := event58060
    frameStart := 58041 },
  { event := event58061
    frameStart := 58041 },
  { event := event58062
    frameStart := 58041 },
  { event := event58063
    frameStart := 58041 }
]

def eventLeaf3629 : Array AnnotatedEvent := #[
  { event := event58064
    frameStart := 58041 },
  { event := event58065
    frameStart := 58041 },
  { event := event58066
    frameStart := 58041 },
  { event := event58067
    frameStart := 58041 },
  { event := event58068
    frameStart := 58041 },
  { event := event58069
    frameStart := 58041 },
  { event := event58070
    frameStart := 58041 },
  { event := event58071
    frameStart := 58041 },
  { event := event58072
    frameStart := 58041 },
  { event := event58073
    frameStart := 58041 },
  { event := event58074
    frameStart := 58041 },
  { event := event58075
    frameStart := 58041 },
  { event := event58076
    frameStart := 58041 },
  { event := event58077
    frameStart := 58041 },
  { event := event58078
    frameStart := 58041 },
  { event := event58079
    frameStart := 58041 }
]

def eventLeaf3630 : Array AnnotatedEvent := #[
  { event := event58080
    frameStart := 58041 },
  { event := event58081
    frameStart := 58041 },
  { event := event58082
    frameStart := 58041 },
  { event := event58083
    frameStart := 58041 },
  { event := event58084
    frameStart := 58041 },
  { event := event58085
    frameStart := 58041 },
  { event := event58086
    frameStart := 58041 },
  { event := event58087
    frameStart := 58041 },
  { event := event58088
    frameStart := 58041 },
  { event := event58089
    frameStart := 58041 },
  { event := event58090
    frameStart := 58041 },
  { event := event58091
    frameStart := 58041 },
  { event := event58092
    frameStart := 58041 },
  { event := event58093
    frameStart := 58041 },
  { event := event58094
    frameStart := 58041 },
  { event := event58095
    frameStart := 58095 }
]

def eventLeaf3631 : Array AnnotatedEvent := #[
  { event := event58096
    frameStart := 58095 },
  { event := event58097
    frameStart := 58095 },
  { event := event58098
    frameStart := 58095 },
  { event := event58099
    frameStart := 58095 },
  { event := event58100
    frameStart := 58095 },
  { event := event58101
    frameStart := 58095 },
  { event := event58102
    frameStart := 58095 },
  { event := event58103
    frameStart := 58095 },
  { event := event58104
    frameStart := 58095 },
  { event := event58105
    frameStart := 58095 },
  { event := event58106
    frameStart := 58095 },
  { event := event58107
    frameStart := 58095 },
  { event := event58108
    frameStart := 58095 },
  { event := event58109
    frameStart := 58095 },
  { event := event58110
    frameStart := 58095 },
  { event := event58111
    frameStart := 58095 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events226
