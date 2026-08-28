import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events398

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event101888 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35945⟩⟩) (.authority (.operator))

def exact101889RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35945⟩⟩]⟩, (1)⟩]

theorem exact101889RawTermsValid :
    exact101889RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event101889 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35945⟩⟩) exact101889RawTerms .large 101888 .exactZero (none)

def event101890 : Event := .predecessor (⟨.program ⟨257⟩, ⟨36748⟩⟩) 0 ⟨35945⟩ 101889

def event101891 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨36748⟩⟩) (.authority (.operator))

def exact101892RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨36748⟩⟩]⟩, (1)⟩]

theorem exact101892RawTermsValid :
    exact101892RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event101892 : Event := .resultExact (⟨.program ⟨257⟩, ⟨36748⟩⟩) exact101892RawTerms (.finite 8192) 101891 .exactZero (none)

def event101893 : Event := .predecessor (⟨.program ⟨257⟩, ⟨36750⟩⟩) 0 ⟨36316⟩ 93216

def event101894 : Event := .predecessor (⟨.program ⟨257⟩, ⟨36750⟩⟩) 1 ⟨36748⟩ 101892

def event101895 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨36750⟩⟩) (.product (.predecessor 0 101893 .coefficient) (.predecessor 1 101894 .coefficient) (⟨false, false, none, none, none⟩))

def event101896 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨36750⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨36748⟩⟩]⟩) [⟨.result 101892 .coefficient, false, none⟩])

def event101897 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨36750⟩⟩) (.product (.result 93216 .summary) (.transfer 101896) (⟨false, false, none, none, none⟩))

def event101898 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨36750⟩⟩, .operator (⟨93216, 0⟩, ⟨101892, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7191⟩⟩, ⟨.program ⟨257⟩, ⟨36748⟩⟩]⟩, (1)⟩)

def event101899 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨36750⟩⟩, .operator (⟨93216, 1⟩, ⟨101892, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨34788⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨36748⟩⟩]⟩, (-1)⟩)

def event101900 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨36750⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨34788⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨36748⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨36748⟩⟩) ⟨35945⟩ 101889)

def event101901 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨36750⟩⟩, .relation 101900 0, ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨34788⟩⟩], [⟨.program ⟨257⟩, ⟨35945⟩⟩]⟩, (-1)⟩)

def exact101902RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7191⟩⟩, ⟨.program ⟨257⟩, ⟨36748⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨34788⟩⟩], [⟨.program ⟨257⟩, ⟨35945⟩⟩]⟩, (-1)⟩]

theorem exact101902RawTermsValid :
    exact101902RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event101902 : Event := .resultExact (⟨.program ⟨257⟩, ⟨36750⟩⟩) exact101902RawTerms .large 101895 (.finite 32192539770951564984245676933120) (some (101897))

def event101903 : Event := .predecessor (⟨.program ⟨257⟩, ⟨35592⟩⟩) 0 ⟨34789⟩ 3966

def event101904 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35592⟩⟩) (.authority (.relationPreimageSource ⟨82⟩))

def exact101905RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35592⟩⟩]⟩, (1)⟩]

theorem exact101905RawTermsValid :
    exact101905RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event101905 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35592⟩⟩) exact101905RawTerms (.finite 5647228698) 101904 .exactZero (none)

def event101906 : Event := .predecessor (⟨.program ⟨257⟩, ⟨35594⟩⟩) 0 ⟨35592⟩ 101905

def event101907 : Event := .predecessor (⟨.program ⟨257⟩, ⟨35594⟩⟩) 1 ⟨2370⟩ 4

def event101908 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35594⟩⟩) (.scale (.predecessor 0 101906 .coefficient) (.value (.predecessor 1 101907 .coefficient)))

def exact101909RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35592⟩⟩]⟩, (1)⟩]

theorem exact101909RawTermsValid :
    exact101909RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event101909 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35594⟩⟩) exact101909RawTerms (.finite 5647228698) 101908 .exactZero (none)

def event101910 : Event := .predecessor (⟨.program ⟨257⟩, ⟨35595⟩⟩) 0 ⟨9944⟩ 90620

def event101911 : Event := .predecessor (⟨.program ⟨257⟩, ⟨35595⟩⟩) 1 ⟨35594⟩ 101909

def event101912 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35595⟩⟩) (.product (.predecessor 0 101910 .coefficient) (.predecessor 1 101911 .coefficient) (⟨false, false, none, none, none⟩))

def event101913 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35595⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨35592⟩⟩]⟩) [⟨.result 101905 .coefficient, false, none⟩])

def event101914 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35595⟩⟩) (.product (.result 90620 .summary) (.transfer 101913) (⟨false, false, none, none, none⟩))

def event101915 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨35595⟩⟩, .operator (⟨90620, 0⟩, ⟨101909, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35592⟩⟩]⟩, (1)⟩)

def event101916 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨35593⟩⟩)

def event101917 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event101918 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event101919 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9843⟩⟩) (.authority (.operator))

def event101920 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨9843⟩⟩) (.finite 14)

def event101921 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event101922 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event101923 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event101924 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event101925 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 101924

def event101926 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 101922

def event101927 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 101925 .coefficient) (.value (.predecessor 1 101926 .coefficient)))

def event101928 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event101929 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9845⟩⟩) 0 ⟨392⟩ 101928

def event101930 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9845⟩⟩) 1 ⟨9843⟩ 101920

def event101931 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9845⟩⟩) (.sum [.predecessor 0 101929 .coefficient, .predecessor 1 101930 .coefficient])

def event101932 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨9845⟩⟩) (.finite 655354)

def event101933 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9901⟩⟩) 0 ⟨9845⟩ 101932

def event101934 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9901⟩⟩) 1 ⟨5426⟩ 101918

def event101935 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9901⟩⟩) (.identity (.predecessor 1 101934 .coefficient))

def event101936 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨9901⟩⟩) (.finite 655360)

def event101937 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34554⟩⟩) 0 ⟨9901⟩ 101936

def event101938 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34554⟩⟩) (.authority (.programFamilyFact))

def exact101939RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨34554⟩⟩], []⟩, (1)⟩]

theorem exact101939RawTermsValid :
    exact101939RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event101939 : Event := .resultExact (⟨.program ⟨257⟩, ⟨34554⟩⟩) exact101939RawTerms (.finite 40) 101938 .exactZero (none)

def event101940 : Event := .predecessor (⟨.program ⟨257⟩, ⟨13656⟩⟩) 0 ⟨9901⟩ 101936

def event101941 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨13656⟩⟩) (.authority (.programFamilyFact))

def exact101942RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨13656⟩⟩], []⟩, (1)⟩]

theorem exact101942RawTermsValid :
    exact101942RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event101942 : Event := .resultExact (⟨.program ⟨257⟩, ⟨13656⟩⟩) exact101942RawTerms (.finite 40) 101941 .exactZero (none)

def event101943 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34555⟩⟩) 0 ⟨13656⟩ 101942

def event101944 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34555⟩⟩) 1 ⟨34554⟩ 101939

def event101945 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34555⟩⟩) (.product (.predecessor 0 101943 .coefficient) (.predecessor 1 101944 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event101946 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34555⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨13656⟩⟩, ⟨.program ⟨257⟩, ⟨34554⟩⟩], []⟩) [⟨.result 101942 .coefficient, true, some 1⟩, ⟨.result 101939 .coefficient, true, some 1⟩])

def event101947 : Event := .survivorFold (1) 101946

def exact101948RawTerms : List Term := []

theorem exact101948RawTermsValid :
    exact101948RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event101948 : Event := .resultExact (⟨.program ⟨257⟩, ⟨34555⟩⟩) exact101948RawTerms (.finite 1600) 101945 (.finite 1600) (some (101946))

def event101949 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34556⟩⟩) 0 ⟨34555⟩ 101948

def event101950 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34556⟩⟩) (.identity (.predecessor 0 101949 .coefficient))

def event101951 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨34556⟩⟩) (.finite 1600)

def event101952 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34788⟩⟩) 0 ⟨34556⟩ 101951

def event101953 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34788⟩⟩) (.authority (.programFamilyFact))

def exact101954RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨34788⟩⟩], []⟩, (1)⟩]

theorem exact101954RawTermsValid :
    exact101954RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event101954 : Event := .resultExact (⟨.program ⟨257⟩, ⟨34788⟩⟩) exact101954RawTerms (.finite 40) 101953 .exactZero (none)

def event101955 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34789⟩⟩) 0 ⟨34788⟩ 101954

def event101956 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34789⟩⟩) (.identity (.predecessor 0 101955 .coefficient))

def event101957 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨34789⟩⟩) (.finite 40)

def event101958 : Event := .predecessor (⟨.program ⟨257⟩, ⟨35592⟩⟩) 0 ⟨34789⟩ 101957

def event101959 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35592⟩⟩) (.authority (.relationPreimageSource ⟨82⟩))

def exact101960RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35592⟩⟩]⟩, (1)⟩]

theorem exact101960RawTermsValid :
    exact101960RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event101960 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35592⟩⟩) exact101960RawTerms (.finite 5647228698) 101959 .exactZero (none)

def event101961 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35⟩⟩) (.authority (.operator))

def exact101962RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩]⟩, (1)⟩]

theorem exact101962RawTermsValid :
    exact101962RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event101962 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35⟩⟩) exact101962RawTerms .large 101961 .exactZero (none)

def event101963 : Event := .predecessor (⟨.program ⟨257⟩, ⟨35593⟩⟩) 0 ⟨35⟩ 101962

def event101964 : Event := .predecessor (⟨.program ⟨257⟩, ⟨35593⟩⟩) 1 ⟨35592⟩ 101960

def event101965 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35593⟩⟩) (.product (.predecessor 0 101963 .coefficient) (.predecessor 1 101964 .coefficient) (⟨false, false, none, none, none⟩))

def event101966 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨35593⟩⟩, .operator (⟨101962, 0⟩, ⟨101960, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35592⟩⟩]⟩, (1)⟩)

def exact101967RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35592⟩⟩]⟩, (1)⟩]

theorem exact101967RawTermsValid :
    exact101967RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event101967 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35593⟩⟩) exact101967RawTerms .large 101965 .exactZero (none)

def event101968 : Event := .preFoldPolynomial 101967 [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35592⟩⟩]⟩, (1)⟩] .exactZero none

def exact101969RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35592⟩⟩]⟩, (1)⟩]

def event101969 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨35593⟩⟩) 101968 exact101969RawTerms .large 101965 .exactZero (none)

def event101970 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨36753⟩⟩)

def event101971 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event101972 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event101973 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9843⟩⟩) (.authority (.operator))

def event101974 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨9843⟩⟩) (.finite 14)

def event101975 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event101976 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event101977 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event101978 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event101979 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 101978

def event101980 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 101976

def event101981 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 101979 .coefficient) (.value (.predecessor 1 101980 .coefficient)))

def event101982 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event101983 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9845⟩⟩) 0 ⟨392⟩ 101982

def event101984 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9845⟩⟩) 1 ⟨9843⟩ 101974

def event101985 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9845⟩⟩) (.sum [.predecessor 0 101983 .coefficient, .predecessor 1 101984 .coefficient])

def event101986 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨9845⟩⟩) (.finite 655354)

def event101987 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9901⟩⟩) 0 ⟨9845⟩ 101986

def event101988 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9901⟩⟩) 1 ⟨5426⟩ 101972

def event101989 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9901⟩⟩) (.identity (.predecessor 1 101988 .coefficient))

def event101990 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨9901⟩⟩) (.finite 655360)

def event101991 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34554⟩⟩) 0 ⟨9901⟩ 101990

def event101992 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34554⟩⟩) (.authority (.programFamilyFact))

def exact101993RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨34554⟩⟩], []⟩, (1)⟩]

theorem exact101993RawTermsValid :
    exact101993RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event101993 : Event := .resultExact (⟨.program ⟨257⟩, ⟨34554⟩⟩) exact101993RawTerms (.finite 40) 101992 .exactZero (none)

def event101994 : Event := .predecessor (⟨.program ⟨257⟩, ⟨13656⟩⟩) 0 ⟨9901⟩ 101990

def event101995 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨13656⟩⟩) (.authority (.programFamilyFact))

def exact101996RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨13656⟩⟩], []⟩, (1)⟩]

theorem exact101996RawTermsValid :
    exact101996RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event101996 : Event := .resultExact (⟨.program ⟨257⟩, ⟨13656⟩⟩) exact101996RawTerms (.finite 40) 101995 .exactZero (none)

def event101997 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34555⟩⟩) 0 ⟨13656⟩ 101996

def event101998 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34555⟩⟩) 1 ⟨34554⟩ 101993

def event101999 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34555⟩⟩) (.product (.predecessor 0 101997 .coefficient) (.predecessor 1 101998 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event102000 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨34555⟩⟩, .operator (⟨101996, 0⟩, ⟨101993, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨13656⟩⟩, ⟨.program ⟨257⟩, ⟨34554⟩⟩], []⟩, (1)⟩)

def exact102001RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨13656⟩⟩, ⟨.program ⟨257⟩, ⟨34554⟩⟩], []⟩, (1)⟩]

theorem exact102001RawTermsValid :
    exact102001RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event102001 : Event := .resultExact (⟨.program ⟨257⟩, ⟨34555⟩⟩) exact102001RawTerms (.finite 1600) 101999 .exactZero (none)

def event102002 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34556⟩⟩) 0 ⟨34555⟩ 102001

def event102003 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34556⟩⟩) (.identity (.predecessor 0 102002 .coefficient))

def event102004 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨34556⟩⟩) (.finite 1600)

def event102005 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34788⟩⟩) 0 ⟨34556⟩ 102004

def event102006 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34788⟩⟩) (.authority (.programFamilyFact))

def exact102007RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨34788⟩⟩], []⟩, (1)⟩]

theorem exact102007RawTermsValid :
    exact102007RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event102007 : Event := .resultExact (⟨.program ⟨257⟩, ⟨34788⟩⟩) exact102007RawTerms (.finite 40) 102006 .exactZero (none)

def event102008 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34789⟩⟩) 0 ⟨34788⟩ 102007

def event102009 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34789⟩⟩) (.identity (.predecessor 0 102008 .coefficient))

def event102010 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨34789⟩⟩) (.finite 40)

def event102011 : Event := .predecessor (⟨.program ⟨257⟩, ⟨35944⟩⟩) 0 ⟨34789⟩ 102010

def event102012 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35944⟩⟩) (.authority (.programFamilyFact))

def event102013 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨35944⟩⟩) (.finite 3720)

def event102014 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨7177⟩⟩) .missing

def event102015 : Event := .predecessor (⟨.program ⟨257⟩, ⟨35945⟩⟩) 0 ⟨7177⟩ 102014

def event102016 : Event := .predecessor (⟨.program ⟨257⟩, ⟨35945⟩⟩) 1 ⟨35944⟩ 102013

def event102017 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35945⟩⟩) (.authority (.operator))

def exact102018RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35945⟩⟩]⟩, (1)⟩]

theorem exact102018RawTermsValid :
    exact102018RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event102018 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35945⟩⟩) exact102018RawTerms .large 102017 .exactZero (none)

def event102019 : Event := .predecessor (⟨.program ⟨257⟩, ⟨36748⟩⟩) 0 ⟨35945⟩ 102018

def event102020 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨36748⟩⟩) (.authority (.operator))

def exact102021RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨36748⟩⟩]⟩, (1)⟩]

theorem exact102021RawTermsValid :
    exact102021RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event102021 : Event := .resultExact (⟨.program ⟨257⟩, ⟨36748⟩⟩) exact102021RawTerms (.finite 8192) 102020 .exactZero (none)

def event102022 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨136⟩⟩) (.authority (.operator))

def event102023 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨136⟩⟩) .exactZero

def event102024 : Event := .predecessor (⟨.program ⟨257⟩, ⟨36126⟩⟩) 0 ⟨34789⟩ 102010

def event102025 : Event := .predecessor (⟨.program ⟨257⟩, ⟨36126⟩⟩) 1 ⟨136⟩ 102023

def event102026 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨36126⟩⟩) (.sum [.predecessor 0 102024 .coefficient, .predecessor 1 102025 .coefficient])

def event102027 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨36126⟩⟩) (.finite 40)

def event102028 : Event := .predecessor (⟨.program ⟨257⟩, ⟨36127⟩⟩) 0 ⟨36126⟩ 102027

def event102029 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨36127⟩⟩) (.identity (.predecessor 0 102028 .coefficient))

def exact102030RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨34788⟩⟩], []⟩, (1)⟩]

theorem exact102030RawTermsValid :
    exact102030RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event102030 : Event := .resultExact (⟨.program ⟨257⟩, ⟨36127⟩⟩) exact102030RawTerms (.finite 40) 102029 .exactZero (none)

def event102031 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6908⟩⟩) (.authority (.factStore))

def exact102032RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact102032RawTermsValid :
    exact102032RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event102032 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6908⟩⟩) exact102032RawTerms .large 102031 .exactZero (none)

def event102033 : Event := .predecessor (⟨.program ⟨257⟩, ⟨36128⟩⟩) 0 ⟨6908⟩ 102032

def event102034 : Event := .predecessor (⟨.program ⟨257⟩, ⟨36128⟩⟩) 1 ⟨36127⟩ 102030

def event102035 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨36128⟩⟩) (.product (.predecessor 0 102033 .coefficient) (.predecessor 1 102034 .coefficient) (⟨false, false, none, none, none⟩))

def event102036 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨36128⟩⟩, .operator (⟨102032, 0⟩, ⟨102030, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨34788⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact102037RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨34788⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact102037RawTermsValid :
    exact102037RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event102037 : Event := .resultExact (⟨.program ⟨257⟩, ⟨36128⟩⟩) exact102037RawTerms .large 102035 .exactZero (none)

def event102038 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7191⟩⟩) 0 ⟨7177⟩ 102014

def event102039 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7191⟩⟩) (.authority (.operator))

def exact102040RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7191⟩⟩]⟩, (1)⟩]

theorem exact102040RawTermsValid :
    exact102040RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event102040 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7191⟩⟩) exact102040RawTerms .large 102039 .exactZero (none)

def event102041 : Event := .predecessor (⟨.program ⟨257⟩, ⟨36129⟩⟩) 0 ⟨7191⟩ 102040

def event102042 : Event := .predecessor (⟨.program ⟨257⟩, ⟨36129⟩⟩) 1 ⟨36128⟩ 102037

def event102043 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨36129⟩⟩) (.sum [.predecessor 0 102041 .coefficient, .predecessor 1 102042 .coefficient])

def exact102044RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7191⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨34788⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact102044RawTermsValid :
    exact102044RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event102044 : Event := .resultExact (⟨.program ⟨257⟩, ⟨36129⟩⟩) exact102044RawTerms .large 102043 .exactZero (none)

def event102045 : Event := .predecessor (⟨.program ⟨257⟩, ⟨36749⟩⟩) 0 ⟨36129⟩ 102044

def event102046 : Event := .predecessor (⟨.program ⟨257⟩, ⟨36749⟩⟩) 1 ⟨36748⟩ 102021

def event102047 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨36749⟩⟩) (.product (.predecessor 0 102045 .coefficient) (.predecessor 1 102046 .coefficient) (⟨false, false, none, none, none⟩))

def event102048 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨36749⟩⟩, .operator (⟨102044, 0⟩, ⟨102021, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7191⟩⟩, ⟨.program ⟨257⟩, ⟨36748⟩⟩]⟩, (1)⟩)

def event102049 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨36749⟩⟩, .operator (⟨102044, 1⟩, ⟨102021, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨34788⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨36748⟩⟩]⟩, (-1)⟩)

def event102050 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨36749⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨34788⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨36748⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨36748⟩⟩) ⟨35945⟩ 102018)

def event102051 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨36749⟩⟩, .relation 102050 0, ⟨[⟨.program ⟨257⟩, ⟨34788⟩⟩], [⟨.program ⟨257⟩, ⟨35945⟩⟩]⟩, (-1)⟩)

def exact102052RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7191⟩⟩, ⟨.program ⟨257⟩, ⟨36748⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨34788⟩⟩], [⟨.program ⟨257⟩, ⟨35945⟩⟩]⟩, (-1)⟩]

theorem exact102052RawTermsValid :
    exact102052RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event102052 : Event := .resultExact (⟨.program ⟨257⟩, ⟨36749⟩⟩) exact102052RawTerms .large 102047 .exactZero (none)

def event102053 : Event := .predecessor (⟨.program ⟨257⟩, ⟨35024⟩⟩) 0 ⟨34789⟩ 102010

def event102054 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35024⟩⟩) (.authority (.programFamilyFact))

def exact102055RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨35024⟩⟩], []⟩, (1)⟩]

theorem exact102055RawTermsValid :
    exact102055RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event102055 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35024⟩⟩) exact102055RawTerms (.finite 40) 102054 .exactZero (none)

def event102056 : Event := .predecessor (⟨.program ⟨257⟩, ⟨35026⟩⟩) 0 ⟨6908⟩ 102032

def event102057 : Event := .predecessor (⟨.program ⟨257⟩, ⟨35026⟩⟩) 1 ⟨35024⟩ 102055

def event102058 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35026⟩⟩) (.product (.predecessor 0 102056 .coefficient) (.predecessor 1 102057 .coefficient) (⟨false, true, none, none, some 1⟩))

def event102059 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨35026⟩⟩, .operator (⟨102032, 0⟩, ⟨102055, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨35024⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact102060RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨35024⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact102060RawTermsValid :
    exact102060RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event102060 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35026⟩⟩) exact102060RawTerms .large 102058 .exactZero (none)

def event102061 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7221⟩⟩) 0 ⟨7177⟩ 102014

def event102062 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7221⟩⟩) (.authority (.operator))

def exact102063RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7221⟩⟩]⟩, (1)⟩]

theorem exact102063RawTermsValid :
    exact102063RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event102063 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7221⟩⟩) exact102063RawTerms .large 102062 .exactZero (none)

def event102064 : Event := .predecessor (⟨.program ⟨257⟩, ⟨35027⟩⟩) 0 ⟨7221⟩ 102063

def event102065 : Event := .predecessor (⟨.program ⟨257⟩, ⟨35027⟩⟩) 1 ⟨35026⟩ 102060

def event102066 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35027⟩⟩) (.sum [.predecessor 0 102064 .coefficient, .predecessor 1 102065 .coefficient])

def exact102067RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7221⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨35024⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact102067RawTermsValid :
    exact102067RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event102067 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35027⟩⟩) exact102067RawTerms .large 102066 .exactZero (none)

def event102068 : Event := .predecessor (⟨.program ⟨257⟩, ⟨36753⟩⟩) 0 ⟨35027⟩ 102067

def event102069 : Event := .predecessor (⟨.program ⟨257⟩, ⟨36753⟩⟩) 1 ⟨36749⟩ 102052

def event102070 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨36753⟩⟩) (.sum [.predecessor 0 102068 .coefficient, .predecessor 1 102069 .coefficient])

def exact102071RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7191⟩⟩, ⟨.program ⟨257⟩, ⟨36748⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7221⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨34788⟩⟩], [⟨.program ⟨257⟩, ⟨35945⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨35024⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact102071RawTermsValid :
    exact102071RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event102071 : Event := .resultExact (⟨.program ⟨257⟩, ⟨36753⟩⟩) exact102071RawTerms .large 102070 .exactZero (none)

def event102072 : Event := .preFoldPolynomial 102071 [⟨⟨[], [⟨.program ⟨257⟩, ⟨7191⟩⟩, ⟨.program ⟨257⟩, ⟨36748⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7221⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨34788⟩⟩], [⟨.program ⟨257⟩, ⟨35945⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨35024⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩] .exactZero none

def exact102073RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7191⟩⟩, ⟨.program ⟨257⟩, ⟨36748⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7221⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨34788⟩⟩], [⟨.program ⟨257⟩, ⟨35945⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨35024⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

def event102073 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨36753⟩⟩) 102072 exact102073RawTerms .large 102070 .exactZero (none)

def event102074 : Event := .specializationComputed (⟨.program ⟨257⟩, ⟨34789⟩⟩) ⟨⟨100⟩, ⟨82⟩, ⟨135⟩⟩ ⟨101916, 102074⟩

def event102075 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨35595⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35592⟩⟩]⟩) (1) 0 2 (.universal 102074 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35592⟩⟩]⟩) (none) 102073)

def event102076 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨35595⟩⟩, .relation 102075 1, ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7221⟩⟩]⟩, (1)⟩)

def event102077 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨35595⟩⟩, .relation 102075 0, ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7191⟩⟩, ⟨.program ⟨257⟩, ⟨36748⟩⟩]⟩, (-1)⟩)

def event102078 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨35595⟩⟩, .relation 102075 2, ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨34788⟩⟩], [⟨.program ⟨257⟩, ⟨35945⟩⟩]⟩, (1)⟩)

def event102079 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨35595⟩⟩, .relation 102075 3, ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨35024⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact102080RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7191⟩⟩, ⟨.program ⟨257⟩, ⟨36748⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7221⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨34788⟩⟩], [⟨.program ⟨257⟩, ⟨35945⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨35024⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact102080RawTermsValid :
    exact102080RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event102080 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35595⟩⟩) exact102080RawTerms .large 101912 (.finite 202072841853861888) (some (101914))

def event102081 : Event := .predecessor (⟨.program ⟨257⟩, ⟨36751⟩⟩) 0 ⟨35595⟩ 102080

def event102082 : Event := .predecessor (⟨.program ⟨257⟩, ⟨36751⟩⟩) 1 ⟨36750⟩ 101902

def event102083 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨36751⟩⟩) (.sum [.predecessor 0 102081 .coefficient, .predecessor 1 102082 .coefficient])

def event102084 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨36751⟩⟩, .operator (⟨102080, 0⟩, ⟨101902, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7191⟩⟩, ⟨.program ⟨257⟩, ⟨36748⟩⟩]⟩, (1)⟩)

def event102085 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨36751⟩⟩, .operator (⟨102080, 2⟩, ⟨101902, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨34788⟩⟩], [⟨.program ⟨257⟩, ⟨35945⟩⟩]⟩, (-1)⟩)

def event102086 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨36751⟩⟩) (.sum [.result 102080 .summary, .result 101902 .summary])

def exact102087RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7221⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨35024⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact102087RawTermsValid :
    exact102087RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event102087 : Event := .resultExact (⟨.program ⟨257⟩, ⟨36751⟩⟩) exact102087RawTerms .large 102083 (.finite 32192539770951767057087530795008) (some (102086))

def event102088 : Event := .predecessor (⟨.program ⟨257⟩, ⟨36752⟩⟩) 0 ⟨36751⟩ 102087

def event102089 : Event := .predecessor (⟨.program ⟨257⟩, ⟨36752⟩⟩) 1 ⟨7164⟩ 15642

def event102090 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨36752⟩⟩) (.product (.predecessor 0 102088 .coefficient) (.predecessor 1 102089 .coefficient) (⟨false, false, none, none, none⟩))

def event102091 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨36752⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨7163⟩⟩]⟩) [⟨.result 15638 .coefficient, false, none⟩])

def event102092 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨36752⟩⟩) (.product (.result 102087 .summary) (.transfer 102091) (⟨false, false, none, none, none⟩))

def event102093 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨36752⟩⟩, .operator (⟨102087, 0⟩, ⟨15642, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7221⟩⟩, ⟨.program ⟨257⟩, ⟨7163⟩⟩]⟩, (1)⟩)

def event102094 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨36752⟩⟩, .operator (⟨102087, 1⟩, ⟨15642, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨35024⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7163⟩⟩]⟩, (-1)⟩)

def event102095 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨36752⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨35024⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7163⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨7163⟩⟩) ⟨7047⟩ 15635)

def event102096 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨36752⟩⟩, .relation 102095 0, ⟨[⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨35024⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact102097RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨35024⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7221⟩⟩, ⟨.program ⟨257⟩, ⟨7163⟩⟩]⟩, (1)⟩]

theorem exact102097RawTermsValid :
    exact102097RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event102097 : Event := .resultExact (⟨.program ⟨257⟩, ⟨36752⟩⟩) exact102097RawTerms .large 102090 (.finite 345664763728542925759002774434880600145920) (some (102092))

def event102098 : Event := .predecessor (⟨.program ⟨257⟩, ⟨30285⟩⟩) 0 ⟨7177⟩ 15500

def event102099 : Event := .predecessor (⟨.program ⟨257⟩, ⟨30285⟩⟩) 1 ⟨30284⟩ 93414

def event102100 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨30285⟩⟩) (.authority (.operator))

def exact102101RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨30285⟩⟩]⟩, (1)⟩]

theorem exact102101RawTermsValid :
    exact102101RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event102101 : Event := .resultExact (⟨.program ⟨257⟩, ⟨30285⟩⟩) exact102101RawTerms .large 102100 .exactZero (none)

def event102102 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31088⟩⟩) 0 ⟨30285⟩ 102101

def event102103 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31088⟩⟩) (.authority (.operator))

def exact102104RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨31088⟩⟩]⟩, (1)⟩]

theorem exact102104RawTermsValid :
    exact102104RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event102104 : Event := .resultExact (⟨.program ⟨257⟩, ⟨31088⟩⟩) exact102104RawTerms (.finite 8192) 102103 .exactZero (none)

def event102105 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31090⟩⟩) 0 ⟨30656⟩ 93698

def event102106 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31090⟩⟩) 1 ⟨31088⟩ 102104

def event102107 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31090⟩⟩) (.product (.predecessor 0 102105 .coefficient) (.predecessor 1 102106 .coefficient) (⟨false, false, none, none, none⟩))

def event102108 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31090⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨31088⟩⟩]⟩) [⟨.result 102104 .coefficient, false, none⟩])

def event102109 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31090⟩⟩) (.product (.result 93698 .summary) (.transfer 102108) (⟨false, false, none, none, none⟩))

def event102110 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨31090⟩⟩, .operator (⟨93698, 0⟩, ⟨102104, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨31088⟩⟩]⟩, (1)⟩)

def event102111 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨31090⟩⟩, .operator (⟨93698, 1⟩, ⟨102104, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨29128⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨31088⟩⟩]⟩, (-1)⟩)

def event102112 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨31090⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨29128⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨31088⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨31088⟩⟩) ⟨30285⟩ 102101)

def event102113 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨31090⟩⟩, .relation 102112 0, ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨29128⟩⟩], [⟨.program ⟨257⟩, ⟨30285⟩⟩]⟩, (-1)⟩)

def exact102114RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨31088⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨29128⟩⟩], [⟨.program ⟨257⟩, ⟨30285⟩⟩]⟩, (-1)⟩]

theorem exact102114RawTermsValid :
    exact102114RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event102114 : Event := .resultExact (⟨.program ⟨257⟩, ⟨31090⟩⟩) exact102114RawTerms .large 102107 (.finite 32192146870060190229763897425920) (some (102109))

def event102115 : Event := .predecessor (⟨.program ⟨257⟩, ⟨29932⟩⟩) 0 ⟨29129⟩ 3989

def event102116 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨29932⟩⟩) (.authority (.relationPreimageSource ⟨80⟩))

def exact102117RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨29932⟩⟩]⟩, (1)⟩]

theorem exact102117RawTermsValid :
    exact102117RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event102117 : Event := .resultExact (⟨.program ⟨257⟩, ⟨29932⟩⟩) exact102117RawTerms (.finite 5647228698) 102116 .exactZero (none)

def event102118 : Event := .predecessor (⟨.program ⟨257⟩, ⟨29934⟩⟩) 0 ⟨29932⟩ 102117

def event102119 : Event := .predecessor (⟨.program ⟨257⟩, ⟨29934⟩⟩) 1 ⟨2370⟩ 4

def event102120 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨29934⟩⟩) (.scale (.predecessor 0 102118 .coefficient) (.value (.predecessor 1 102119 .coefficient)))

def exact102121RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨29932⟩⟩]⟩, (1)⟩]

theorem exact102121RawTermsValid :
    exact102121RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event102121 : Event := .resultExact (⟨.program ⟨257⟩, ⟨29934⟩⟩) exact102121RawTerms (.finite 5647228698) 102120 .exactZero (none)

def event102122 : Event := .predecessor (⟨.program ⟨257⟩, ⟨29935⟩⟩) 0 ⟨9944⟩ 90620

def event102123 : Event := .predecessor (⟨.program ⟨257⟩, ⟨29935⟩⟩) 1 ⟨29934⟩ 102121

def event102124 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨29935⟩⟩) (.product (.predecessor 0 102122 .coefficient) (.predecessor 1 102123 .coefficient) (⟨false, false, none, none, none⟩))

def event102125 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨29935⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨29932⟩⟩]⟩) [⟨.result 102117 .coefficient, false, none⟩])

def event102126 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨29935⟩⟩) (.product (.result 90620 .summary) (.transfer 102125) (⟨false, false, none, none, none⟩))

def event102127 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨29935⟩⟩, .operator (⟨90620, 0⟩, ⟨102121, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29932⟩⟩]⟩, (1)⟩)

def event102128 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨29933⟩⟩)

def event102129 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event102130 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event102131 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9843⟩⟩) (.authority (.operator))

def event102132 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨9843⟩⟩) (.finite 14)

def event102133 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event102134 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event102135 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event102136 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event102137 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 102136

def event102138 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 102134

def event102139 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 102137 .coefficient) (.value (.predecessor 1 102138 .coefficient)))

def event102140 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event102141 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9845⟩⟩) 0 ⟨392⟩ 102140

def event102142 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9845⟩⟩) 1 ⟨9843⟩ 102132

def event102143 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9845⟩⟩) (.sum [.predecessor 0 102141 .coefficient, .predecessor 1 102142 .coefficient])

def eventLeaf6368 : Array AnnotatedEvent := #[
  { event := event101888
    frameStart := 0 },
  { event := event101889
    frameStart := 0 },
  { event := event101890
    frameStart := 0 },
  { event := event101891
    frameStart := 0 },
  { event := event101892
    frameStart := 0 },
  { event := event101893
    frameStart := 0 },
  { event := event101894
    frameStart := 0 },
  { event := event101895
    frameStart := 0 },
  { event := event101896
    frameStart := 0 },
  { event := event101897
    frameStart := 0 },
  { event := event101898
    frameStart := 0 },
  { event := event101899
    frameStart := 0 },
  { event := event101900
    frameStart := 0 },
  { event := event101901
    frameStart := 0 },
  { event := event101902
    frameStart := 0 },
  { event := event101903
    frameStart := 0 }
]

def eventLeaf6369 : Array AnnotatedEvent := #[
  { event := event101904
    frameStart := 0 },
  { event := event101905
    frameStart := 0 },
  { event := event101906
    frameStart := 0 },
  { event := event101907
    frameStart := 0 },
  { event := event101908
    frameStart := 0 },
  { event := event101909
    frameStart := 0 },
  { event := event101910
    frameStart := 0 },
  { event := event101911
    frameStart := 0 },
  { event := event101912
    frameStart := 0 },
  { event := event101913
    frameStart := 0 },
  { event := event101914
    frameStart := 0 },
  { event := event101915
    frameStart := 0 },
  { event := event101916
    frameStart := 101916 },
  { event := event101917
    frameStart := 101916 },
  { event := event101918
    frameStart := 101916 },
  { event := event101919
    frameStart := 101916 }
]

def eventLeaf6370 : Array AnnotatedEvent := #[
  { event := event101920
    frameStart := 101916 },
  { event := event101921
    frameStart := 101916 },
  { event := event101922
    frameStart := 101916 },
  { event := event101923
    frameStart := 101916 },
  { event := event101924
    frameStart := 101916 },
  { event := event101925
    frameStart := 101916 },
  { event := event101926
    frameStart := 101916 },
  { event := event101927
    frameStart := 101916 },
  { event := event101928
    frameStart := 101916 },
  { event := event101929
    frameStart := 101916 },
  { event := event101930
    frameStart := 101916 },
  { event := event101931
    frameStart := 101916 },
  { event := event101932
    frameStart := 101916 },
  { event := event101933
    frameStart := 101916 },
  { event := event101934
    frameStart := 101916 },
  { event := event101935
    frameStart := 101916 }
]

def eventLeaf6371 : Array AnnotatedEvent := #[
  { event := event101936
    frameStart := 101916 },
  { event := event101937
    frameStart := 101916 },
  { event := event101938
    frameStart := 101916 },
  { event := event101939
    frameStart := 101916 },
  { event := event101940
    frameStart := 101916 },
  { event := event101941
    frameStart := 101916 },
  { event := event101942
    frameStart := 101916 },
  { event := event101943
    frameStart := 101916 },
  { event := event101944
    frameStart := 101916 },
  { event := event101945
    frameStart := 101916 },
  { event := event101946
    frameStart := 101916 },
  { event := event101947
    frameStart := 101916 },
  { event := event101948
    frameStart := 101916 },
  { event := event101949
    frameStart := 101916 },
  { event := event101950
    frameStart := 101916 },
  { event := event101951
    frameStart := 101916 }
]

def eventLeaf6372 : Array AnnotatedEvent := #[
  { event := event101952
    frameStart := 101916 },
  { event := event101953
    frameStart := 101916 },
  { event := event101954
    frameStart := 101916 },
  { event := event101955
    frameStart := 101916 },
  { event := event101956
    frameStart := 101916 },
  { event := event101957
    frameStart := 101916 },
  { event := event101958
    frameStart := 101916 },
  { event := event101959
    frameStart := 101916 },
  { event := event101960
    frameStart := 101916 },
  { event := event101961
    frameStart := 101916 },
  { event := event101962
    frameStart := 101916 },
  { event := event101963
    frameStart := 101916 },
  { event := event101964
    frameStart := 101916 },
  { event := event101965
    frameStart := 101916 },
  { event := event101966
    frameStart := 101916 },
  { event := event101967
    frameStart := 101916 }
]

def eventLeaf6373 : Array AnnotatedEvent := #[
  { event := event101968
    frameStart := 101916 },
  { event := event101969
    frameStart := 101916 },
  { event := event101970
    frameStart := 101970 },
  { event := event101971
    frameStart := 101970 },
  { event := event101972
    frameStart := 101970 },
  { event := event101973
    frameStart := 101970 },
  { event := event101974
    frameStart := 101970 },
  { event := event101975
    frameStart := 101970 },
  { event := event101976
    frameStart := 101970 },
  { event := event101977
    frameStart := 101970 },
  { event := event101978
    frameStart := 101970 },
  { event := event101979
    frameStart := 101970 },
  { event := event101980
    frameStart := 101970 },
  { event := event101981
    frameStart := 101970 },
  { event := event101982
    frameStart := 101970 },
  { event := event101983
    frameStart := 101970 }
]

def eventLeaf6374 : Array AnnotatedEvent := #[
  { event := event101984
    frameStart := 101970 },
  { event := event101985
    frameStart := 101970 },
  { event := event101986
    frameStart := 101970 },
  { event := event101987
    frameStart := 101970 },
  { event := event101988
    frameStart := 101970 },
  { event := event101989
    frameStart := 101970 },
  { event := event101990
    frameStart := 101970 },
  { event := event101991
    frameStart := 101970 },
  { event := event101992
    frameStart := 101970 },
  { event := event101993
    frameStart := 101970 },
  { event := event101994
    frameStart := 101970 },
  { event := event101995
    frameStart := 101970 },
  { event := event101996
    frameStart := 101970 },
  { event := event101997
    frameStart := 101970 },
  { event := event101998
    frameStart := 101970 },
  { event := event101999
    frameStart := 101970 }
]

def eventLeaf6375 : Array AnnotatedEvent := #[
  { event := event102000
    frameStart := 101970 },
  { event := event102001
    frameStart := 101970 },
  { event := event102002
    frameStart := 101970 },
  { event := event102003
    frameStart := 101970 },
  { event := event102004
    frameStart := 101970 },
  { event := event102005
    frameStart := 101970 },
  { event := event102006
    frameStart := 101970 },
  { event := event102007
    frameStart := 101970 },
  { event := event102008
    frameStart := 101970 },
  { event := event102009
    frameStart := 101970 },
  { event := event102010
    frameStart := 101970 },
  { event := event102011
    frameStart := 101970 },
  { event := event102012
    frameStart := 101970 },
  { event := event102013
    frameStart := 101970 },
  { event := event102014
    frameStart := 101970 },
  { event := event102015
    frameStart := 101970 }
]

def eventLeaf6376 : Array AnnotatedEvent := #[
  { event := event102016
    frameStart := 101970 },
  { event := event102017
    frameStart := 101970 },
  { event := event102018
    frameStart := 101970 },
  { event := event102019
    frameStart := 101970 },
  { event := event102020
    frameStart := 101970 },
  { event := event102021
    frameStart := 101970 },
  { event := event102022
    frameStart := 101970 },
  { event := event102023
    frameStart := 101970 },
  { event := event102024
    frameStart := 101970 },
  { event := event102025
    frameStart := 101970 },
  { event := event102026
    frameStart := 101970 },
  { event := event102027
    frameStart := 101970 },
  { event := event102028
    frameStart := 101970 },
  { event := event102029
    frameStart := 101970 },
  { event := event102030
    frameStart := 101970 },
  { event := event102031
    frameStart := 101970 }
]

def eventLeaf6377 : Array AnnotatedEvent := #[
  { event := event102032
    frameStart := 101970 },
  { event := event102033
    frameStart := 101970 },
  { event := event102034
    frameStart := 101970 },
  { event := event102035
    frameStart := 101970 },
  { event := event102036
    frameStart := 101970 },
  { event := event102037
    frameStart := 101970 },
  { event := event102038
    frameStart := 101970 },
  { event := event102039
    frameStart := 101970 },
  { event := event102040
    frameStart := 101970 },
  { event := event102041
    frameStart := 101970 },
  { event := event102042
    frameStart := 101970 },
  { event := event102043
    frameStart := 101970 },
  { event := event102044
    frameStart := 101970 },
  { event := event102045
    frameStart := 101970 },
  { event := event102046
    frameStart := 101970 },
  { event := event102047
    frameStart := 101970 }
]

def eventLeaf6378 : Array AnnotatedEvent := #[
  { event := event102048
    frameStart := 101970 },
  { event := event102049
    frameStart := 101970 },
  { event := event102050
    frameStart := 101970 },
  { event := event102051
    frameStart := 101970 },
  { event := event102052
    frameStart := 101970 },
  { event := event102053
    frameStart := 101970 },
  { event := event102054
    frameStart := 101970 },
  { event := event102055
    frameStart := 101970 },
  { event := event102056
    frameStart := 101970 },
  { event := event102057
    frameStart := 101970 },
  { event := event102058
    frameStart := 101970 },
  { event := event102059
    frameStart := 101970 },
  { event := event102060
    frameStart := 101970 },
  { event := event102061
    frameStart := 101970 },
  { event := event102062
    frameStart := 101970 },
  { event := event102063
    frameStart := 101970 }
]

def eventLeaf6379 : Array AnnotatedEvent := #[
  { event := event102064
    frameStart := 101970 },
  { event := event102065
    frameStart := 101970 },
  { event := event102066
    frameStart := 101970 },
  { event := event102067
    frameStart := 101970 },
  { event := event102068
    frameStart := 101970 },
  { event := event102069
    frameStart := 101970 },
  { event := event102070
    frameStart := 101970 },
  { event := event102071
    frameStart := 101970 },
  { event := event102072
    frameStart := 101970 },
  { event := event102073
    frameStart := 101970 },
  { event := event102074
    frameStart := 0 },
  { event := event102075
    frameStart := 0 },
  { event := event102076
    frameStart := 0 },
  { event := event102077
    frameStart := 0 },
  { event := event102078
    frameStart := 0 },
  { event := event102079
    frameStart := 0 }
]

def eventLeaf6380 : Array AnnotatedEvent := #[
  { event := event102080
    frameStart := 0 },
  { event := event102081
    frameStart := 0 },
  { event := event102082
    frameStart := 0 },
  { event := event102083
    frameStart := 0 },
  { event := event102084
    frameStart := 0 },
  { event := event102085
    frameStart := 0 },
  { event := event102086
    frameStart := 0 },
  { event := event102087
    frameStart := 0 },
  { event := event102088
    frameStart := 0 },
  { event := event102089
    frameStart := 0 },
  { event := event102090
    frameStart := 0 },
  { event := event102091
    frameStart := 0 },
  { event := event102092
    frameStart := 0 },
  { event := event102093
    frameStart := 0 },
  { event := event102094
    frameStart := 0 },
  { event := event102095
    frameStart := 0 }
]

def eventLeaf6381 : Array AnnotatedEvent := #[
  { event := event102096
    frameStart := 0 },
  { event := event102097
    frameStart := 0 },
  { event := event102098
    frameStart := 0 },
  { event := event102099
    frameStart := 0 },
  { event := event102100
    frameStart := 0 },
  { event := event102101
    frameStart := 0 },
  { event := event102102
    frameStart := 0 },
  { event := event102103
    frameStart := 0 },
  { event := event102104
    frameStart := 0 },
  { event := event102105
    frameStart := 0 },
  { event := event102106
    frameStart := 0 },
  { event := event102107
    frameStart := 0 },
  { event := event102108
    frameStart := 0 },
  { event := event102109
    frameStart := 0 },
  { event := event102110
    frameStart := 0 },
  { event := event102111
    frameStart := 0 }
]

def eventLeaf6382 : Array AnnotatedEvent := #[
  { event := event102112
    frameStart := 0 },
  { event := event102113
    frameStart := 0 },
  { event := event102114
    frameStart := 0 },
  { event := event102115
    frameStart := 0 },
  { event := event102116
    frameStart := 0 },
  { event := event102117
    frameStart := 0 },
  { event := event102118
    frameStart := 0 },
  { event := event102119
    frameStart := 0 },
  { event := event102120
    frameStart := 0 },
  { event := event102121
    frameStart := 0 },
  { event := event102122
    frameStart := 0 },
  { event := event102123
    frameStart := 0 },
  { event := event102124
    frameStart := 0 },
  { event := event102125
    frameStart := 0 },
  { event := event102126
    frameStart := 0 },
  { event := event102127
    frameStart := 0 }
]

def eventLeaf6383 : Array AnnotatedEvent := #[
  { event := event102128
    frameStart := 102128 },
  { event := event102129
    frameStart := 102128 },
  { event := event102130
    frameStart := 102128 },
  { event := event102131
    frameStart := 102128 },
  { event := event102132
    frameStart := 102128 },
  { event := event102133
    frameStart := 102128 },
  { event := event102134
    frameStart := 102128 },
  { event := event102135
    frameStart := 102128 },
  { event := event102136
    frameStart := 102128 },
  { event := event102137
    frameStart := 102128 },
  { event := event102138
    frameStart := 102128 },
  { event := event102139
    frameStart := 102128 },
  { event := event102140
    frameStart := 102128 },
  { event := event102141
    frameStart := 102128 },
  { event := event102142
    frameStart := 102128 },
  { event := event102143
    frameStart := 102128 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events398
