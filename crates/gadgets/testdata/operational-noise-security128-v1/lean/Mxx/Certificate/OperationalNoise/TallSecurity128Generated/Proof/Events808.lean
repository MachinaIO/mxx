import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events808

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event206848 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5895⟩⟩) 0 ⟨392⟩ 206847

def event206849 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5895⟩⟩) 1 ⟨5893⟩ 206839

def event206850 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5895⟩⟩) (.sum [.predecessor 0 206848 .coefficient, .predecessor 1 206849 .coefficient])

def event206851 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5895⟩⟩) (.finite 655347)

def event206852 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5905⟩⟩) 0 ⟨5895⟩ 206851

def event206853 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5905⟩⟩) 1 ⟨5426⟩ 206837

def event206854 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5905⟩⟩) (.identity (.predecessor 1 206853 .coefficient))

def event206855 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5905⟩⟩) (.finite 655360)

def event206856 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15522⟩⟩) 0 ⟨5905⟩ 206855

def event206857 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15522⟩⟩) (.authority (.programFamilyFact))

def exact206858RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15522⟩⟩], []⟩, (1)⟩]

theorem exact206858RawTermsValid :
    exact206858RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event206858 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15522⟩⟩) exact206858RawTerms (.finite 2) 206857 .exactZero (none)

def event206859 : Event := .predecessor (⟨.program ⟨257⟩, ⟨12411⟩⟩) 0 ⟨5905⟩ 206855

def event206860 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨12411⟩⟩) (.authority (.programFamilyFact))

def exact206861RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨12411⟩⟩], []⟩, (1)⟩]

theorem exact206861RawTermsValid :
    exact206861RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event206861 : Event := .resultExact (⟨.program ⟨257⟩, ⟨12411⟩⟩) exact206861RawTerms (.finite 2) 206860 .exactZero (none)

def event206862 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15523⟩⟩) 0 ⟨12411⟩ 206861

def event206863 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15523⟩⟩) 1 ⟨15522⟩ 206858

def event206864 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15523⟩⟩) (.product (.predecessor 0 206862 .coefficient) (.predecessor 1 206863 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event206865 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15523⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨12411⟩⟩, ⟨.program ⟨257⟩, ⟨15522⟩⟩], []⟩) [⟨.result 206861 .coefficient, true, some 1⟩, ⟨.result 206858 .coefficient, true, some 1⟩])

def event206866 : Event := .survivorFold (1) 206865

def exact206867RawTerms : List Term := []

theorem exact206867RawTermsValid :
    exact206867RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event206867 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15523⟩⟩) exact206867RawTerms (.finite 4) 206864 (.finite 4) (some (206865))

def event206868 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15524⟩⟩) 0 ⟨15523⟩ 206867

def event206869 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15524⟩⟩) (.identity (.predecessor 0 206868 .coefficient))

def event206870 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨15524⟩⟩) (.finite 4)

def event206871 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15804⟩⟩) 0 ⟨15524⟩ 206870

def event206872 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15804⟩⟩) (.authority (.programFamilyFact))

def exact206873RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15804⟩⟩], []⟩, (1)⟩]

theorem exact206873RawTermsValid :
    exact206873RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event206873 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15804⟩⟩) exact206873RawTerms (.finite 2) 206872 .exactZero (none)

def event206874 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15805⟩⟩) 0 ⟨15804⟩ 206873

def event206875 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15805⟩⟩) (.identity (.predecessor 0 206874 .coefficient))

def event206876 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨15805⟩⟩) (.finite 2)

def event206877 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16632⟩⟩) 0 ⟨15805⟩ 206876

def event206878 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨16632⟩⟩) (.authority (.relationPreimageSource ⟨56⟩))

def exact206879RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨16632⟩⟩]⟩, (1)⟩]

theorem exact206879RawTermsValid :
    exact206879RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event206879 : Event := .resultExact (⟨.program ⟨257⟩, ⟨16632⟩⟩) exact206879RawTerms (.finite 5647228698) 206878 .exactZero (none)

def event206880 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35⟩⟩) (.authority (.operator))

def exact206881RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩]⟩, (1)⟩]

theorem exact206881RawTermsValid :
    exact206881RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event206881 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35⟩⟩) exact206881RawTerms .large 206880 .exactZero (none)

def event206882 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16633⟩⟩) 0 ⟨35⟩ 206881

def event206883 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16633⟩⟩) 1 ⟨16632⟩ 206879

def event206884 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨16633⟩⟩) (.product (.predecessor 0 206882 .coefficient) (.predecessor 1 206883 .coefficient) (⟨false, false, none, none, none⟩))

def event206885 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨16633⟩⟩, .operator (⟨206881, 0⟩, ⟨206879, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16632⟩⟩]⟩, (1)⟩)

def exact206886RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16632⟩⟩]⟩, (1)⟩]

theorem exact206886RawTermsValid :
    exact206886RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event206886 : Event := .resultExact (⟨.program ⟨257⟩, ⟨16633⟩⟩) exact206886RawTerms .large 206884 .exactZero (none)

def event206887 : Event := .preFoldPolynomial 206886 [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16632⟩⟩]⟩, (1)⟩] .exactZero none

def exact206888RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16632⟩⟩]⟩, (1)⟩]

def event206888 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨16633⟩⟩) 206887 exact206888RawTerms .large 206884 .exactZero (none)

def event206889 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨17816⟩⟩)

def event206890 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event206891 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event206892 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5893⟩⟩) (.authority (.operator))

def event206893 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5893⟩⟩) (.finite 7)

def event206894 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event206895 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event206896 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event206897 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event206898 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 206897

def event206899 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 206895

def event206900 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 206898 .coefficient) (.value (.predecessor 1 206899 .coefficient)))

def event206901 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event206902 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5895⟩⟩) 0 ⟨392⟩ 206901

def event206903 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5895⟩⟩) 1 ⟨5893⟩ 206893

def event206904 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5895⟩⟩) (.sum [.predecessor 0 206902 .coefficient, .predecessor 1 206903 .coefficient])

def event206905 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5895⟩⟩) (.finite 655347)

def event206906 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5905⟩⟩) 0 ⟨5895⟩ 206905

def event206907 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5905⟩⟩) 1 ⟨5426⟩ 206891

def event206908 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5905⟩⟩) (.identity (.predecessor 1 206907 .coefficient))

def event206909 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5905⟩⟩) (.finite 655360)

def event206910 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15522⟩⟩) 0 ⟨5905⟩ 206909

def event206911 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15522⟩⟩) (.authority (.programFamilyFact))

def exact206912RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15522⟩⟩], []⟩, (1)⟩]

theorem exact206912RawTermsValid :
    exact206912RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event206912 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15522⟩⟩) exact206912RawTerms (.finite 2) 206911 .exactZero (none)

def event206913 : Event := .predecessor (⟨.program ⟨257⟩, ⟨12411⟩⟩) 0 ⟨5905⟩ 206909

def event206914 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨12411⟩⟩) (.authority (.programFamilyFact))

def exact206915RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨12411⟩⟩], []⟩, (1)⟩]

theorem exact206915RawTermsValid :
    exact206915RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event206915 : Event := .resultExact (⟨.program ⟨257⟩, ⟨12411⟩⟩) exact206915RawTerms (.finite 2) 206914 .exactZero (none)

def event206916 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15523⟩⟩) 0 ⟨12411⟩ 206915

def event206917 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15523⟩⟩) 1 ⟨15522⟩ 206912

def event206918 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15523⟩⟩) (.product (.predecessor 0 206916 .coefficient) (.predecessor 1 206917 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event206919 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨15523⟩⟩, .operator (⟨206915, 0⟩, ⟨206912, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨12411⟩⟩, ⟨.program ⟨257⟩, ⟨15522⟩⟩], []⟩, (1)⟩)

def exact206920RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨12411⟩⟩, ⟨.program ⟨257⟩, ⟨15522⟩⟩], []⟩, (1)⟩]

theorem exact206920RawTermsValid :
    exact206920RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event206920 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15523⟩⟩) exact206920RawTerms (.finite 4) 206918 .exactZero (none)

def event206921 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15524⟩⟩) 0 ⟨15523⟩ 206920

def event206922 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15524⟩⟩) (.identity (.predecessor 0 206921 .coefficient))

def event206923 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨15524⟩⟩) (.finite 4)

def event206924 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15804⟩⟩) 0 ⟨15524⟩ 206923

def event206925 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15804⟩⟩) (.authority (.programFamilyFact))

def exact206926RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15804⟩⟩], []⟩, (1)⟩]

theorem exact206926RawTermsValid :
    exact206926RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event206926 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15804⟩⟩) exact206926RawTerms (.finite 2) 206925 .exactZero (none)

def event206927 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15805⟩⟩) 0 ⟨15804⟩ 206926

def event206928 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15805⟩⟩) (.identity (.predecessor 0 206927 .coefficient))

def event206929 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨15805⟩⟩) (.finite 2)

def event206930 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17017⟩⟩) 0 ⟨15805⟩ 206929

def event206931 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17017⟩⟩) (.authority (.programFamilyFact))

def event206932 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨17017⟩⟩) (.finite 3720)

def event206933 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨7177⟩⟩) .missing

def event206934 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17018⟩⟩) 0 ⟨7177⟩ 206933

def event206935 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17018⟩⟩) 1 ⟨17017⟩ 206932

def event206936 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17018⟩⟩) (.authority (.operator))

def exact206937RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨17018⟩⟩]⟩, (1)⟩]

theorem exact206937RawTermsValid :
    exact206937RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event206937 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17018⟩⟩) exact206937RawTerms .large 206936 .exactZero (none)

def event206938 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17810⟩⟩) 0 ⟨17018⟩ 206937

def event206939 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17810⟩⟩) (.authority (.operator))

def exact206940RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨17810⟩⟩]⟩, (1)⟩]

theorem exact206940RawTermsValid :
    exact206940RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event206940 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17810⟩⟩) exact206940RawTerms (.finite 8192) 206939 .exactZero (none)

def event206941 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨136⟩⟩) (.authority (.operator))

def event206942 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨136⟩⟩) .exactZero

def event206943 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17214⟩⟩) 0 ⟨15805⟩ 206929

def event206944 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17214⟩⟩) 1 ⟨136⟩ 206942

def event206945 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17214⟩⟩) (.sum [.predecessor 0 206943 .coefficient, .predecessor 1 206944 .coefficient])

def event206946 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨17214⟩⟩) (.finite 2)

def event206947 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17215⟩⟩) 0 ⟨17214⟩ 206946

def event206948 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17215⟩⟩) (.identity (.predecessor 0 206947 .coefficient))

def exact206949RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15804⟩⟩], []⟩, (1)⟩]

theorem exact206949RawTermsValid :
    exact206949RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event206949 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17215⟩⟩) exact206949RawTerms (.finite 2) 206948 .exactZero (none)

def event206950 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6908⟩⟩) (.authority (.factStore))

def exact206951RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact206951RawTermsValid :
    exact206951RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event206951 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6908⟩⟩) exact206951RawTerms .large 206950 .exactZero (none)

def event206952 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17216⟩⟩) 0 ⟨6908⟩ 206951

def event206953 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17216⟩⟩) 1 ⟨17215⟩ 206949

def event206954 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17216⟩⟩) (.product (.predecessor 0 206952 .coefficient) (.predecessor 1 206953 .coefficient) (⟨false, false, none, none, none⟩))

def event206955 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17216⟩⟩, .operator (⟨206951, 0⟩, ⟨206949, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨15804⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact206956RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15804⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact206956RawTermsValid :
    exact206956RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event206956 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17216⟩⟩) exact206956RawTerms .large 206954 .exactZero (none)

def event206957 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7179⟩⟩) 0 ⟨7177⟩ 206933

def event206958 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7179⟩⟩) (.authority (.operator))

def exact206959RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩]⟩, (1)⟩]

theorem exact206959RawTermsValid :
    exact206959RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event206959 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7179⟩⟩) exact206959RawTerms .large 206958 .exactZero (none)

def event206960 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17217⟩⟩) 0 ⟨7179⟩ 206959

def event206961 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17217⟩⟩) 1 ⟨17216⟩ 206956

def event206962 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17217⟩⟩) (.sum [.predecessor 0 206960 .coefficient, .predecessor 1 206961 .coefficient])

def exact206963RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15804⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact206963RawTermsValid :
    exact206963RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event206963 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17217⟩⟩) exact206963RawTerms .large 206962 .exactZero (none)

def event206964 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17811⟩⟩) 0 ⟨17217⟩ 206963

def event206965 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17811⟩⟩) 1 ⟨17810⟩ 206940

def event206966 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17811⟩⟩) (.product (.predecessor 0 206964 .coefficient) (.predecessor 1 206965 .coefficient) (⟨false, false, none, none, none⟩))

def event206967 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17811⟩⟩, .operator (⟨206963, 0⟩, ⟨206940, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17810⟩⟩]⟩, (1)⟩)

def event206968 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17811⟩⟩, .operator (⟨206963, 1⟩, ⟨206940, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨15804⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17810⟩⟩]⟩, (-1)⟩)

def event206969 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨17811⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨15804⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17810⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨17810⟩⟩) ⟨17018⟩ 206937)

def event206970 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17811⟩⟩, .relation 206969 0, ⟨[⟨.program ⟨257⟩, ⟨15804⟩⟩], [⟨.program ⟨257⟩, ⟨17018⟩⟩]⟩, (-1)⟩)

def exact206971RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17810⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15804⟩⟩], [⟨.program ⟨257⟩, ⟨17018⟩⟩]⟩, (-1)⟩]

theorem exact206971RawTermsValid :
    exact206971RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event206971 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17811⟩⟩) exact206971RawTerms .large 206966 .exactZero (none)

def event206972 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16062⟩⟩) 0 ⟨15805⟩ 206929

def event206973 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨16062⟩⟩) (.authority (.programFamilyFact))

def exact206974RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨16062⟩⟩], []⟩, (1)⟩]

theorem exact206974RawTermsValid :
    exact206974RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event206974 : Event := .resultExact (⟨.program ⟨257⟩, ⟨16062⟩⟩) exact206974RawTerms (.finite 2) 206973 .exactZero (none)

def event206975 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16065⟩⟩) 0 ⟨6908⟩ 206951

def event206976 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16065⟩⟩) 1 ⟨16062⟩ 206974

def event206977 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨16065⟩⟩) (.product (.predecessor 0 206975 .coefficient) (.predecessor 1 206976 .coefficient) (⟨false, true, none, none, some 1⟩))

def event206978 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨16065⟩⟩, .operator (⟨206951, 0⟩, ⟨206974, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨16062⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact206979RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨16062⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact206979RawTermsValid :
    exact206979RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event206979 : Event := .resultExact (⟨.program ⟨257⟩, ⟨16065⟩⟩) exact206979RawTerms .large 206977 .exactZero (none)

def event206980 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7197⟩⟩) 0 ⟨7177⟩ 206933

def event206981 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7197⟩⟩) (.authority (.operator))

def exact206982RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7197⟩⟩]⟩, (1)⟩]

theorem exact206982RawTermsValid :
    exact206982RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event206982 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7197⟩⟩) exact206982RawTerms .large 206981 .exactZero (none)

def event206983 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16066⟩⟩) 0 ⟨7197⟩ 206982

def event206984 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16066⟩⟩) 1 ⟨16065⟩ 206979

def event206985 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨16066⟩⟩) (.sum [.predecessor 0 206983 .coefficient, .predecessor 1 206984 .coefficient])

def exact206986RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7197⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨16062⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact206986RawTermsValid :
    exact206986RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event206986 : Event := .resultExact (⟨.program ⟨257⟩, ⟨16066⟩⟩) exact206986RawTerms .large 206985 .exactZero (none)

def event206987 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17816⟩⟩) 0 ⟨16066⟩ 206986

def event206988 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17816⟩⟩) 1 ⟨17811⟩ 206971

def event206989 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17816⟩⟩) (.sum [.predecessor 0 206987 .coefficient, .predecessor 1 206988 .coefficient])

def exact206990RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17810⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7197⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15804⟩⟩], [⟨.program ⟨257⟩, ⟨17018⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨16062⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact206990RawTermsValid :
    exact206990RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event206990 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17816⟩⟩) exact206990RawTerms .large 206989 .exactZero (none)

def event206991 : Event := .preFoldPolynomial 206990 [⟨⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17810⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7197⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15804⟩⟩], [⟨.program ⟨257⟩, ⟨17018⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨16062⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩] .exactZero none

def exact206992RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17810⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7197⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15804⟩⟩], [⟨.program ⟨257⟩, ⟨17018⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨16062⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

def event206992 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨17816⟩⟩) 206991 exact206992RawTerms .large 206989 .exactZero (none)

def event206993 : Event := .specializationComputed (⟨.program ⟨257⟩, ⟨15805⟩⟩) ⟨⟨76⟩, ⟨56⟩, ⟨135⟩⟩ ⟨206835, 206993⟩

def event206994 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨16635⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16632⟩⟩]⟩) (1) 0 2 (.universal 206993 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16632⟩⟩]⟩) (none) 206992)

def event206995 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨16635⟩⟩, .relation 206994 1, ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩]⟩, (1)⟩)

def event206996 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨16635⟩⟩, .relation 206994 0, ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17810⟩⟩]⟩, (-1)⟩)

def event206997 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨16635⟩⟩, .relation 206994 2, ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨15804⟩⟩], [⟨.program ⟨257⟩, ⟨17018⟩⟩]⟩, (1)⟩)

def event206998 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨16635⟩⟩, .relation 206994 3, ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨16062⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact206999RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17810⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨15804⟩⟩], [⟨.program ⟨257⟩, ⟨17018⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨16062⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact206999RawTermsValid :
    exact206999RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event206999 : Event := .resultExact (⟨.program ⟨257⟩, ⟨16635⟩⟩) exact206999RawTerms .large 206831 (.finite 202072841853861888) (some (206833))

def event207000 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17813⟩⟩) 0 ⟨16635⟩ 206999

def event207001 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17813⟩⟩) 1 ⟨17812⟩ 206821

def event207002 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17813⟩⟩) (.sum [.predecessor 0 207000 .coefficient, .predecessor 1 207001 .coefficient])

def event207003 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17813⟩⟩, .operator (⟨206999, 0⟩, ⟨206821, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17810⟩⟩]⟩, (1)⟩)

def event207004 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17813⟩⟩, .operator (⟨206999, 2⟩, ⟨206821, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨15804⟩⟩], [⟨.program ⟨257⟩, ⟨17018⟩⟩]⟩, (-1)⟩)

def event207005 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17813⟩⟩) (.sum [.result 206999 .summary, .result 206821 .summary])

def exact207006RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨16062⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact207006RawTermsValid :
    exact207006RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event207006 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17813⟩⟩) exact207006RawTerms .large 207002 (.finite 32188807212483706889510625476608) (some (207005))

def event207007 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17814⟩⟩) 0 ⟨17813⟩ 207006

def event207008 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17814⟩⟩) 1 ⟨7172⟩ 15882

def event207009 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17814⟩⟩) (.product (.predecessor 0 207007 .coefficient) (.predecessor 1 207008 .coefficient) (⟨false, false, none, none, none⟩))

def event207010 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17814⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩) [⟨.result 15878 .coefficient, false, none⟩])

def event207011 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17814⟩⟩) (.product (.result 207006 .summary) (.transfer 207010) (⟨false, false, none, none, none⟩))

def event207012 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17814⟩⟩, .operator (⟨207006, 0⟩, ⟨15882, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩)

def event207013 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17814⟩⟩, .operator (⟨207006, 1⟩, ⟨15882, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨16062⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (-1)⟩)

def event207014 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨17814⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨16062⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨7171⟩⟩) ⟨7051⟩ 15875)

def event207015 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17814⟩⟩, .relation 207014 0, ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16062⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact207016RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16062⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact207016RawTermsValid :
    exact207016RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event207016 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17814⟩⟩) exact207016RawTerms .large 207009 (.finite 345624685687166110058245054666339432529920) (some (207011))

def event207017 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7091⟩⟩) 0 ⟨6727⟩ 723

def event207018 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7091⟩⟩) 1 ⟨6998⟩ 192903

def event207019 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7091⟩⟩) (.tensor (.predecessor 0 207017 .coefficient) (.predecessor 1 207018 .coefficient) true false)

def event207020 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨7091⟩⟩, .operator (⟨723, 0⟩, ⟨192903, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6727⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact207021RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6727⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact207021RawTermsValid :
    exact207021RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event207021 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7091⟩⟩) exact207021RawTerms .large 207019 .exactZero (none)

def event207022 : Event := .predecessor (⟨.program ⟨257⟩, ⟨8826⟩⟩) 0 ⟨5907⟩ 192773

def event207023 : Event := .predecessor (⟨.program ⟨257⟩, ⟨8826⟩⟩) 1 ⟨7292⟩ 15896

def event207024 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨8826⟩⟩) (.product (.predecessor 0 207022 .coefficient) (.predecessor 1 207023 .coefficient) (⟨false, false, none, none, none⟩))

def event207025 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨8826⟩⟩, .operator (⟨192773, 0⟩, ⟨15896, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7292⟩⟩]⟩, (1)⟩)

def exact207026RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7292⟩⟩]⟩, (1)⟩]

theorem exact207026RawTermsValid :
    exact207026RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event207026 : Event := .resultExact (⟨.program ⟨257⟩, ⟨8826⟩⟩) exact207026RawTerms .large 207024 .exactZero (none)

def event207027 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9413⟩⟩) 0 ⟨8826⟩ 207026

def event207028 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9413⟩⟩) 1 ⟨7091⟩ 207021

def event207029 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9413⟩⟩) (.sum [.predecessor 0 207027 .coefficient, .predecessor 1 207028 .coefficient])

def exact207030RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7292⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6727⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact207030RawTermsValid :
    exact207030RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event207030 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9413⟩⟩) exact207030RawTerms .large 207029 .exactZero (none)

def event207031 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9414⟩⟩) 0 ⟨9413⟩ 207030

def event207032 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9414⟩⟩) 1 ⟨118⟩ 31516

def event207033 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9414⟩⟩) (.sum [.predecessor 0 207031 .coefficient, .predecessor 1 207032 .coefficient])

def event207034 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9414⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨118⟩⟩]⟩) [⟨.result 31516 .coefficient, false, none⟩])

def event207035 : Event := .survivorFold (1) 207034

def exact207036RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7292⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6727⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact207036RawTermsValid :
    exact207036RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event207036 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9414⟩⟩) exact207036RawTerms .large 207033 (.finite 26) (some (207034))

def event207037 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9483⟩⟩) 0 ⟨9414⟩ 207036

def event207038 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9483⟩⟩) 1 ⟨9414⟩ 207036

def event207039 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9483⟩⟩) (.sum [.predecessor 0 207037 .coefficient, .predecessor 1 207038 .coefficient])

def event207040 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨9483⟩⟩, .operator (⟨207036, 1⟩, ⟨207036, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6727⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def event207041 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨9483⟩⟩, .operator (⟨207036, 0⟩, ⟨207036, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7292⟩⟩]⟩, (-1)⟩)

def event207042 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9483⟩⟩) (.sum [.result 207036 .summary, .result 207036 .summary])

def exact207043RawTerms : List Term := []

theorem exact207043RawTermsValid :
    exact207043RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event207043 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9483⟩⟩) exact207043RawTerms .large 207039 (.finite 52) (some (207042))

def event207044 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17815⟩⟩) 0 ⟨9483⟩ 207043

def event207045 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17815⟩⟩) 1 ⟨17814⟩ 207016

def event207046 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17815⟩⟩) (.sum [.predecessor 0 207044 .coefficient, .predecessor 1 207045 .coefficient])

def event207047 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17815⟩⟩) (.sum [.result 207043 .summary, .result 207016 .summary])

def exact207048RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16062⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact207048RawTermsValid :
    exact207048RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event207048 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17815⟩⟩) exact207048RawTerms .large 207046 (.finite 345624685687166110058245054666339432529972) (some (207047))

def event207049 : Event := .predecessor (⟨.program ⟨257⟩, ⟨20712⟩⟩) 0 ⟨17815⟩ 207048

def event207050 : Event := .predecessor (⟨.program ⟨257⟩, ⟨20712⟩⟩) 1 ⟨20711⟩ 206804

def event207051 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨20712⟩⟩) (.sum [.predecessor 0 207049 .coefficient, .predecessor 1 207050 .coefficient])

def event207052 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨20712⟩⟩) (.sum [.result 207048 .summary, .result 206804 .summary])

def exact207053RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7199⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18899⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16062⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact207053RawTermsValid :
    exact207053RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event207053 : Event := .resultExact (⟨.program ⟨257⟩, ⟨20712⟩⟩) exact207053RawTerms .large 207051 (.finite 691250426059631610003352154589745737891892) (some (207052))

def event207054 : Event := .predecessor (⟨.program ⟨257⟩, ⟨23932⟩⟩) 0 ⟨20712⟩ 207053

def event207055 : Event := .predecessor (⟨.program ⟨257⟩, ⟨23932⟩⟩) 1 ⟨23931⟩ 206592

def event207056 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨23932⟩⟩) (.sum [.predecessor 0 207054 .coefficient, .predecessor 1 207055 .coefficient])

def event207057 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨23932⟩⟩) (.sum [.result 207053 .summary, .result 206592 .summary])

def exact207058RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7199⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22119⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18899⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16062⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact207058RawTermsValid :
    exact207058RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event207058 : Event := .resultExact (⟨.program ⟨257⟩, ⟨23932⟩⟩) exact207058RawTerms .large 207056 (.finite 1036877221117396499835321299770218916085812) (some (207057))

def event207059 : Event := .predecessor (⟨.program ⟨257⟩, ⟨33952⟩⟩) 0 ⟨23932⟩ 207058

def event207060 : Event := .predecessor (⟨.program ⟨257⟩, ⟨33952⟩⟩) 1 ⟨33951⟩ 206380

def event207061 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨33952⟩⟩) (.sum [.predecessor 0 207059 .coefficient, .predecessor 1 207060 .coefficient])

def event207062 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨33952⟩⟩) (.sum [.result 207058 .summary, .result 206380 .summary])

def exact207063RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7199⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7203⟩⟩, ⟨.program ⟨257⟩, ⟨7145⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32139⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22119⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18899⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16062⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact207063RawTermsValid :
    exact207063RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event207063 : Event := .resultExact (⟨.program ⟨257⟩, ⟨33952⟩⟩) exact207063RawTerms .large 207061 (.finite 1382506125545760169441014535464825839943732) (some (207062))

def event207064 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53012⟩⟩) 0 ⟨33952⟩ 207063

def event207065 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53012⟩⟩) 1 ⟨53011⟩ 206168

def event207066 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨53012⟩⟩) (.sum [.predecessor 0 207064 .coefficient, .predecessor 1 207065 .coefficient])

def event207067 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨53012⟩⟩) (.sum [.result 207063 .summary, .result 206168 .summary])

def exact207068RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7199⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7203⟩⟩, ⟨.program ⟨257⟩, ⟨7145⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7205⟩⟩, ⟨.program ⟨257⟩, ⟨7131⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51203⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32139⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22119⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18899⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16062⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact207068RawTermsValid :
    exact207068RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event207068 : Event := .resultExact (⟨.program ⟨257⟩, ⟨53012⟩⟩) exact207068RawTerms .large 207066 (.finite 1728139248715321398594155952187700255129652) (some (207067))

def event207069 : Event := .predecessor (⟨.program ⟨257⟩, ⟨55992⟩⟩) 0 ⟨53012⟩ 207068

def event207070 : Event := .predecessor (⟨.program ⟨257⟩, ⟨55992⟩⟩) 1 ⟨55991⟩ 205956

def event207071 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨55992⟩⟩) (.sum [.predecessor 0 207069 .coefficient, .predecessor 1 207070 .coefficient])

def event207072 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨55992⟩⟩) (.sum [.result 207068 .summary, .result 205956 .summary])

def exact207073RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7199⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7203⟩⟩, ⟨.program ⟨257⟩, ⟨7145⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7205⟩⟩, ⟨.program ⟨257⟩, ⟨7131⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7207⟩⟩, ⟨.program ⟨257⟩, ⟨7125⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54183⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51203⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32139⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22119⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18899⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16062⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact207073RawTermsValid :
    exact207073RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event207073 : Event := .resultExact (⟨.program ⟨257⟩, ⟨55992⟩⟩) exact207073RawTerms .large 207071 (.finite 2073774481255481407521021459424708415979572) (some (207072))

def event207074 : Event := .predecessor (⟨.program ⟨257⟩, ⟨58972⟩⟩) 0 ⟨55992⟩ 207073

def event207075 : Event := .predecessor (⟨.program ⟨257⟩, ⟨58972⟩⟩) 1 ⟨58971⟩ 205744

def event207076 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨58972⟩⟩) (.sum [.predecessor 0 207074 .coefficient, .predecessor 1 207075 .coefficient])

def event207077 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨58972⟩⟩) (.sum [.result 207073 .summary, .result 205744 .summary])

def exact207078RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7199⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7203⟩⟩, ⟨.program ⟨257⟩, ⟨7145⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7205⟩⟩, ⟨.program ⟨257⟩, ⟨7131⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7207⟩⟩, ⟨.program ⟨257⟩, ⟨7125⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7209⟩⟩, ⟨.program ⟨257⟩, ⟨7107⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57163⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54183⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51203⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32139⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22119⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18899⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16062⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact207078RawTermsValid :
    exact207078RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event207078 : Event := .resultExact (⟨.program ⟨257⟩, ⟨58972⟩⟩) exact207078RawTerms .large 207076 (.finite 2419413932536838975995335147689984068157492) (some (207077))

def event207079 : Event := .predecessor (⟨.program ⟨257⟩, ⟨61952⟩⟩) 0 ⟨58972⟩ 207078

def event207080 : Event := .predecessor (⟨.program ⟨257⟩, ⟨61952⟩⟩) 1 ⟨61951⟩ 205532

def event207081 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨61952⟩⟩) (.sum [.predecessor 0 207079 .coefficient, .predecessor 1 207080 .coefficient])

def event207082 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨61952⟩⟩) (.sum [.result 207078 .summary, .result 205532 .summary])

def exact207083RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7199⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7203⟩⟩, ⟨.program ⟨257⟩, ⟨7145⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7205⟩⟩, ⟨.program ⟨257⟩, ⟨7131⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7207⟩⟩, ⟨.program ⟨257⟩, ⟨7125⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7209⟩⟩, ⟨.program ⟨257⟩, ⟨7107⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7211⟩⟩, ⟨.program ⟨257⟩, ⟨7103⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60143⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57163⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54183⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51203⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32139⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22119⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18899⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16062⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact207083RawTermsValid :
    exact207083RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event207083 : Event := .resultExact (⟨.program ⟨257⟩, ⟨61952⟩⟩) exact207083RawTerms .large 207081 (.finite 2765055493188795324243372926469393465999412) (some (207082))

def event207084 : Event := .predecessor (⟨.program ⟨257⟩, ⟨64932⟩⟩) 0 ⟨61952⟩ 207083

def event207085 : Event := .predecessor (⟨.program ⟨257⟩, ⟨64932⟩⟩) 1 ⟨64931⟩ 205320

def event207086 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨64932⟩⟩) (.sum [.predecessor 0 207084 .coefficient, .predecessor 1 207085 .coefficient])

def event207087 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨64932⟩⟩) (.sum [.result 207083 .summary, .result 205320 .summary])

def exact207088RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7199⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7203⟩⟩, ⟨.program ⟨257⟩, ⟨7145⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7205⟩⟩, ⟨.program ⟨257⟩, ⟨7131⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7207⟩⟩, ⟨.program ⟨257⟩, ⟨7125⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7209⟩⟩, ⟨.program ⟨257⟩, ⟨7107⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7211⟩⟩, ⟨.program ⟨257⟩, ⟨7103⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7213⟩⟩, ⟨.program ⟨257⟩, ⟨7099⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨63123⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60143⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57163⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54183⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51203⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32139⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22119⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18899⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16062⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact207088RawTermsValid :
    exact207088RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event207088 : Event := .resultExact (⟨.program ⟨257⟩, ⟨64932⟩⟩) exact207088RawTerms .large 207086 (.finite 3110701272581949232038858886277070355169332) (some (207087))

def event207089 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70325⟩⟩) 0 ⟨64932⟩ 207088

def event207090 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70325⟩⟩) 1 ⟨70324⟩ 205108

def event207091 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70325⟩⟩) (.sum [.predecessor 0 207089 .coefficient, .predecessor 1 207090 .coefficient])

def event207092 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70325⟩⟩) (.sum [.result 207088 .summary, .result 205108 .summary])

def exact207093RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7199⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7203⟩⟩, ⟨.program ⟨257⟩, ⟨7145⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7205⟩⟩, ⟨.program ⟨257⟩, ⟨7131⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7207⟩⟩, ⟨.program ⟨257⟩, ⟨7125⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7209⟩⟩, ⟨.program ⟨257⟩, ⟨7107⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7211⟩⟩, ⟨.program ⟨257⟩, ⟨7103⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7213⟩⟩, ⟨.program ⟨257⟩, ⟨7099⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7215⟩⟩, ⟨.program ⟨257⟩, ⟨7173⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨63123⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60143⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57163⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54183⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51203⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32139⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22119⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18899⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16062⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66728⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact207093RawTermsValid :
    exact207093RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event207093 : Event := .resultExact (⟨.program ⟨257⟩, ⟨70325⟩⟩) exact207093RawTerms .large 207091 (.finite 3456353380086899479155517117627148481331252) (some (207092))

def event207094 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70326⟩⟩) 0 ⟨70325⟩ 207093

def event207095 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70326⟩⟩) 1 ⟨28337⟩ 204896

def event207096 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70326⟩⟩) (.sum [.predecessor 0 207094 .coefficient, .predecessor 1 207095 .coefficient])

def event207097 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70326⟩⟩) (.sum [.result 207093 .summary, .result 204896 .summary])

def exact207098RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7199⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7203⟩⟩, ⟨.program ⟨257⟩, ⟨7145⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7205⟩⟩, ⟨.program ⟨257⟩, ⟨7131⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7207⟩⟩, ⟨.program ⟨257⟩, ⟨7125⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7209⟩⟩, ⟨.program ⟨257⟩, ⟨7107⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7211⟩⟩, ⟨.program ⟨257⟩, ⟨7103⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7213⟩⟩, ⟨.program ⟨257⟩, ⟨7099⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7215⟩⟩, ⟨.program ⟨257⟩, ⟨7173⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7217⟩⟩, ⟨.program ⟨257⟩, ⟨7169⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨63123⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60143⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57163⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54183⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51203⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32139⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22119⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18899⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26648⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16062⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66728⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact207098RawTermsValid :
    exact207098RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event207098 : Event := .resultExact (⟨.program ⟨257⟩, ⟨70326⟩⟩) exact207098RawTerms .large 207096 (.finite 3802007596962448506045899439491360353157172) (some (207097))

def event207099 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70327⟩⟩) 0 ⟨70326⟩ 207098

def event207100 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70327⟩⟩) 1 ⟨31017⟩ 204684

def event207101 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70327⟩⟩) (.sum [.predecessor 0 207099 .coefficient, .predecessor 1 207100 .coefficient])

def event207102 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70327⟩⟩) (.sum [.result 207098 .summary, .result 204684 .summary])

def exact207103RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7199⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7203⟩⟩, ⟨.program ⟨257⟩, ⟨7145⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7205⟩⟩, ⟨.program ⟨257⟩, ⟨7131⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7207⟩⟩, ⟨.program ⟨257⟩, ⟨7125⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7209⟩⟩, ⟨.program ⟨257⟩, ⟨7107⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7211⟩⟩, ⟨.program ⟨257⟩, ⟨7103⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7213⟩⟩, ⟨.program ⟨257⟩, ⟨7099⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7215⟩⟩, ⟨.program ⟨257⟩, ⟨7173⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7217⟩⟩, ⟨.program ⟨257⟩, ⟨7169⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7219⟩⟩, ⟨.program ⟨257⟩, ⟨7167⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨63123⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60143⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57163⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54183⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51203⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32139⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22119⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18899⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29328⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26648⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16062⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66728⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact207103RawTermsValid :
    exact207103RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event207103 : Event := .resultExact (⟨.program ⟨257⟩, ⟨70327⟩⟩) exact207103RawTerms .large 207101 (.finite 4147668141949793872257454032897973461975092) (some (207102))

def eventLeaf12928 : Array AnnotatedEvent := #[
  { event := event206848
    frameStart := 206835 },
  { event := event206849
    frameStart := 206835 },
  { event := event206850
    frameStart := 206835 },
  { event := event206851
    frameStart := 206835 },
  { event := event206852
    frameStart := 206835 },
  { event := event206853
    frameStart := 206835 },
  { event := event206854
    frameStart := 206835 },
  { event := event206855
    frameStart := 206835 },
  { event := event206856
    frameStart := 206835 },
  { event := event206857
    frameStart := 206835 },
  { event := event206858
    frameStart := 206835 },
  { event := event206859
    frameStart := 206835 },
  { event := event206860
    frameStart := 206835 },
  { event := event206861
    frameStart := 206835 },
  { event := event206862
    frameStart := 206835 },
  { event := event206863
    frameStart := 206835 }
]

def eventLeaf12929 : Array AnnotatedEvent := #[
  { event := event206864
    frameStart := 206835 },
  { event := event206865
    frameStart := 206835 },
  { event := event206866
    frameStart := 206835 },
  { event := event206867
    frameStart := 206835 },
  { event := event206868
    frameStart := 206835 },
  { event := event206869
    frameStart := 206835 },
  { event := event206870
    frameStart := 206835 },
  { event := event206871
    frameStart := 206835 },
  { event := event206872
    frameStart := 206835 },
  { event := event206873
    frameStart := 206835 },
  { event := event206874
    frameStart := 206835 },
  { event := event206875
    frameStart := 206835 },
  { event := event206876
    frameStart := 206835 },
  { event := event206877
    frameStart := 206835 },
  { event := event206878
    frameStart := 206835 },
  { event := event206879
    frameStart := 206835 }
]

def eventLeaf12930 : Array AnnotatedEvent := #[
  { event := event206880
    frameStart := 206835 },
  { event := event206881
    frameStart := 206835 },
  { event := event206882
    frameStart := 206835 },
  { event := event206883
    frameStart := 206835 },
  { event := event206884
    frameStart := 206835 },
  { event := event206885
    frameStart := 206835 },
  { event := event206886
    frameStart := 206835 },
  { event := event206887
    frameStart := 206835 },
  { event := event206888
    frameStart := 206835 },
  { event := event206889
    frameStart := 206889 },
  { event := event206890
    frameStart := 206889 },
  { event := event206891
    frameStart := 206889 },
  { event := event206892
    frameStart := 206889 },
  { event := event206893
    frameStart := 206889 },
  { event := event206894
    frameStart := 206889 },
  { event := event206895
    frameStart := 206889 }
]

def eventLeaf12931 : Array AnnotatedEvent := #[
  { event := event206896
    frameStart := 206889 },
  { event := event206897
    frameStart := 206889 },
  { event := event206898
    frameStart := 206889 },
  { event := event206899
    frameStart := 206889 },
  { event := event206900
    frameStart := 206889 },
  { event := event206901
    frameStart := 206889 },
  { event := event206902
    frameStart := 206889 },
  { event := event206903
    frameStart := 206889 },
  { event := event206904
    frameStart := 206889 },
  { event := event206905
    frameStart := 206889 },
  { event := event206906
    frameStart := 206889 },
  { event := event206907
    frameStart := 206889 },
  { event := event206908
    frameStart := 206889 },
  { event := event206909
    frameStart := 206889 },
  { event := event206910
    frameStart := 206889 },
  { event := event206911
    frameStart := 206889 }
]

def eventLeaf12932 : Array AnnotatedEvent := #[
  { event := event206912
    frameStart := 206889 },
  { event := event206913
    frameStart := 206889 },
  { event := event206914
    frameStart := 206889 },
  { event := event206915
    frameStart := 206889 },
  { event := event206916
    frameStart := 206889 },
  { event := event206917
    frameStart := 206889 },
  { event := event206918
    frameStart := 206889 },
  { event := event206919
    frameStart := 206889 },
  { event := event206920
    frameStart := 206889 },
  { event := event206921
    frameStart := 206889 },
  { event := event206922
    frameStart := 206889 },
  { event := event206923
    frameStart := 206889 },
  { event := event206924
    frameStart := 206889 },
  { event := event206925
    frameStart := 206889 },
  { event := event206926
    frameStart := 206889 },
  { event := event206927
    frameStart := 206889 }
]

def eventLeaf12933 : Array AnnotatedEvent := #[
  { event := event206928
    frameStart := 206889 },
  { event := event206929
    frameStart := 206889 },
  { event := event206930
    frameStart := 206889 },
  { event := event206931
    frameStart := 206889 },
  { event := event206932
    frameStart := 206889 },
  { event := event206933
    frameStart := 206889 },
  { event := event206934
    frameStart := 206889 },
  { event := event206935
    frameStart := 206889 },
  { event := event206936
    frameStart := 206889 },
  { event := event206937
    frameStart := 206889 },
  { event := event206938
    frameStart := 206889 },
  { event := event206939
    frameStart := 206889 },
  { event := event206940
    frameStart := 206889 },
  { event := event206941
    frameStart := 206889 },
  { event := event206942
    frameStart := 206889 },
  { event := event206943
    frameStart := 206889 }
]

def eventLeaf12934 : Array AnnotatedEvent := #[
  { event := event206944
    frameStart := 206889 },
  { event := event206945
    frameStart := 206889 },
  { event := event206946
    frameStart := 206889 },
  { event := event206947
    frameStart := 206889 },
  { event := event206948
    frameStart := 206889 },
  { event := event206949
    frameStart := 206889 },
  { event := event206950
    frameStart := 206889 },
  { event := event206951
    frameStart := 206889 },
  { event := event206952
    frameStart := 206889 },
  { event := event206953
    frameStart := 206889 },
  { event := event206954
    frameStart := 206889 },
  { event := event206955
    frameStart := 206889 },
  { event := event206956
    frameStart := 206889 },
  { event := event206957
    frameStart := 206889 },
  { event := event206958
    frameStart := 206889 },
  { event := event206959
    frameStart := 206889 }
]

def eventLeaf12935 : Array AnnotatedEvent := #[
  { event := event206960
    frameStart := 206889 },
  { event := event206961
    frameStart := 206889 },
  { event := event206962
    frameStart := 206889 },
  { event := event206963
    frameStart := 206889 },
  { event := event206964
    frameStart := 206889 },
  { event := event206965
    frameStart := 206889 },
  { event := event206966
    frameStart := 206889 },
  { event := event206967
    frameStart := 206889 },
  { event := event206968
    frameStart := 206889 },
  { event := event206969
    frameStart := 206889 },
  { event := event206970
    frameStart := 206889 },
  { event := event206971
    frameStart := 206889 },
  { event := event206972
    frameStart := 206889 },
  { event := event206973
    frameStart := 206889 },
  { event := event206974
    frameStart := 206889 },
  { event := event206975
    frameStart := 206889 }
]

def eventLeaf12936 : Array AnnotatedEvent := #[
  { event := event206976
    frameStart := 206889 },
  { event := event206977
    frameStart := 206889 },
  { event := event206978
    frameStart := 206889 },
  { event := event206979
    frameStart := 206889 },
  { event := event206980
    frameStart := 206889 },
  { event := event206981
    frameStart := 206889 },
  { event := event206982
    frameStart := 206889 },
  { event := event206983
    frameStart := 206889 },
  { event := event206984
    frameStart := 206889 },
  { event := event206985
    frameStart := 206889 },
  { event := event206986
    frameStart := 206889 },
  { event := event206987
    frameStart := 206889 },
  { event := event206988
    frameStart := 206889 },
  { event := event206989
    frameStart := 206889 },
  { event := event206990
    frameStart := 206889 },
  { event := event206991
    frameStart := 206889 }
]

def eventLeaf12937 : Array AnnotatedEvent := #[
  { event := event206992
    frameStart := 206889 },
  { event := event206993
    frameStart := 0 },
  { event := event206994
    frameStart := 0 },
  { event := event206995
    frameStart := 0 },
  { event := event206996
    frameStart := 0 },
  { event := event206997
    frameStart := 0 },
  { event := event206998
    frameStart := 0 },
  { event := event206999
    frameStart := 0 },
  { event := event207000
    frameStart := 0 },
  { event := event207001
    frameStart := 0 },
  { event := event207002
    frameStart := 0 },
  { event := event207003
    frameStart := 0 },
  { event := event207004
    frameStart := 0 },
  { event := event207005
    frameStart := 0 },
  { event := event207006
    frameStart := 0 },
  { event := event207007
    frameStart := 0 }
]

def eventLeaf12938 : Array AnnotatedEvent := #[
  { event := event207008
    frameStart := 0 },
  { event := event207009
    frameStart := 0 },
  { event := event207010
    frameStart := 0 },
  { event := event207011
    frameStart := 0 },
  { event := event207012
    frameStart := 0 },
  { event := event207013
    frameStart := 0 },
  { event := event207014
    frameStart := 0 },
  { event := event207015
    frameStart := 0 },
  { event := event207016
    frameStart := 0 },
  { event := event207017
    frameStart := 0 },
  { event := event207018
    frameStart := 0 },
  { event := event207019
    frameStart := 0 },
  { event := event207020
    frameStart := 0 },
  { event := event207021
    frameStart := 0 },
  { event := event207022
    frameStart := 0 },
  { event := event207023
    frameStart := 0 }
]

def eventLeaf12939 : Array AnnotatedEvent := #[
  { event := event207024
    frameStart := 0 },
  { event := event207025
    frameStart := 0 },
  { event := event207026
    frameStart := 0 },
  { event := event207027
    frameStart := 0 },
  { event := event207028
    frameStart := 0 },
  { event := event207029
    frameStart := 0 },
  { event := event207030
    frameStart := 0 },
  { event := event207031
    frameStart := 0 },
  { event := event207032
    frameStart := 0 },
  { event := event207033
    frameStart := 0 },
  { event := event207034
    frameStart := 0 },
  { event := event207035
    frameStart := 0 },
  { event := event207036
    frameStart := 0 },
  { event := event207037
    frameStart := 0 },
  { event := event207038
    frameStart := 0 },
  { event := event207039
    frameStart := 0 }
]

def eventLeaf12940 : Array AnnotatedEvent := #[
  { event := event207040
    frameStart := 0 },
  { event := event207041
    frameStart := 0 },
  { event := event207042
    frameStart := 0 },
  { event := event207043
    frameStart := 0 },
  { event := event207044
    frameStart := 0 },
  { event := event207045
    frameStart := 0 },
  { event := event207046
    frameStart := 0 },
  { event := event207047
    frameStart := 0 },
  { event := event207048
    frameStart := 0 },
  { event := event207049
    frameStart := 0 },
  { event := event207050
    frameStart := 0 },
  { event := event207051
    frameStart := 0 },
  { event := event207052
    frameStart := 0 },
  { event := event207053
    frameStart := 0 },
  { event := event207054
    frameStart := 0 },
  { event := event207055
    frameStart := 0 }
]

def eventLeaf12941 : Array AnnotatedEvent := #[
  { event := event207056
    frameStart := 0 },
  { event := event207057
    frameStart := 0 },
  { event := event207058
    frameStart := 0 },
  { event := event207059
    frameStart := 0 },
  { event := event207060
    frameStart := 0 },
  { event := event207061
    frameStart := 0 },
  { event := event207062
    frameStart := 0 },
  { event := event207063
    frameStart := 0 },
  { event := event207064
    frameStart := 0 },
  { event := event207065
    frameStart := 0 },
  { event := event207066
    frameStart := 0 },
  { event := event207067
    frameStart := 0 },
  { event := event207068
    frameStart := 0 },
  { event := event207069
    frameStart := 0 },
  { event := event207070
    frameStart := 0 },
  { event := event207071
    frameStart := 0 }
]

def eventLeaf12942 : Array AnnotatedEvent := #[
  { event := event207072
    frameStart := 0 },
  { event := event207073
    frameStart := 0 },
  { event := event207074
    frameStart := 0 },
  { event := event207075
    frameStart := 0 },
  { event := event207076
    frameStart := 0 },
  { event := event207077
    frameStart := 0 },
  { event := event207078
    frameStart := 0 },
  { event := event207079
    frameStart := 0 },
  { event := event207080
    frameStart := 0 },
  { event := event207081
    frameStart := 0 },
  { event := event207082
    frameStart := 0 },
  { event := event207083
    frameStart := 0 },
  { event := event207084
    frameStart := 0 },
  { event := event207085
    frameStart := 0 },
  { event := event207086
    frameStart := 0 },
  { event := event207087
    frameStart := 0 }
]

def eventLeaf12943 : Array AnnotatedEvent := #[
  { event := event207088
    frameStart := 0 },
  { event := event207089
    frameStart := 0 },
  { event := event207090
    frameStart := 0 },
  { event := event207091
    frameStart := 0 },
  { event := event207092
    frameStart := 0 },
  { event := event207093
    frameStart := 0 },
  { event := event207094
    frameStart := 0 },
  { event := event207095
    frameStart := 0 },
  { event := event207096
    frameStart := 0 },
  { event := event207097
    frameStart := 0 },
  { event := event207098
    frameStart := 0 },
  { event := event207099
    frameStart := 0 },
  { event := event207100
    frameStart := 0 },
  { event := event207101
    frameStart := 0 },
  { event := event207102
    frameStart := 0 },
  { event := event207103
    frameStart := 0 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events808
