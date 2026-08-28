import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events148

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event37888 : Event := .predecessor (⟨.program ⟨257⟩, ⟨55599⟩⟩) 0 ⟨53776⟩ 37887

def event37889 : Event := .predecessor (⟨.program ⟨257⟩, ⟨55599⟩⟩) 1 ⟨55598⟩ 37823

def event37890 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨55599⟩⟩) (.product (.predecessor 0 37888 .coefficient) (.predecessor 1 37889 .coefficient) (⟨false, false, none, none, none⟩))

def event37891 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨55599⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨55598⟩⟩]⟩) [⟨.result 37823 .coefficient, false, none⟩])

def event37892 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨55599⟩⟩) (.product (.result 37887 .summary) (.transfer 37891) (⟨false, false, none, none, none⟩))

def event37893 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨55599⟩⟩, .operator (⟨37887, 1⟩, ⟨37823, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨24878⟩⟩, ⟨.program ⟨257⟩, ⟨53768⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨55598⟩⟩]⟩, (-1)⟩)

def event37894 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨55599⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨24878⟩⟩, ⟨.program ⟨257⟩, ⟨53768⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨55598⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨55598⟩⟩) ⟨55043⟩ 37820)

def event37895 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨55599⟩⟩, .relation 37894 0, ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨24878⟩⟩, ⟨.program ⟨257⟩, ⟨53768⟩⟩], [⟨.program ⟨257⟩, ⟨55043⟩⟩]⟩, (-1)⟩)

def event37896 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨55599⟩⟩, .operator (⟨37887, 0⟩, ⟨37823, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55598⟩⟩]⟩, (1)⟩)

def exact37897RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55598⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨24878⟩⟩, ⟨.program ⟨257⟩, ⟨53768⟩⟩], [⟨.program ⟨257⟩, ⟨55043⟩⟩]⟩, (-1)⟩]

theorem exact37897RawTermsValid :
    exact37897RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event37897 : Event := .resultExact (⟨.program ⟨257⟩, ⟨55599⟩⟩) exact37897RawTerms .large 37890 (.finite 2997705687218719293440) (some (37892))

def event37898 : Event := .predecessor (⟨.program ⟨257⟩, ⟨54519⟩⟩) 0 ⟨53770⟩ 1129

def event37899 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨54519⟩⟩) (.authority (.relationPreimageSource ⟨41⟩))

def exact37900RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨54519⟩⟩]⟩, (1)⟩]

theorem exact37900RawTermsValid :
    exact37900RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event37900 : Event := .resultExact (⟨.program ⟨257⟩, ⟨54519⟩⟩) exact37900RawTerms (.finite 5647228698) 37899 .exactZero (none)

def event37901 : Event := .predecessor (⟨.program ⟨257⟩, ⟨54521⟩⟩) 0 ⟨54519⟩ 37900

def event37902 : Event := .predecessor (⟨.program ⟨257⟩, ⟨54521⟩⟩) 1 ⟨2370⟩ 4

def event37903 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨54521⟩⟩) (.scale (.predecessor 0 37901 .coefficient) (.value (.predecessor 1 37902 .coefficient)))

def exact37904RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨54519⟩⟩]⟩, (1)⟩]

theorem exact37904RawTermsValid :
    exact37904RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event37904 : Event := .resultExact (⟨.program ⟨257⟩, ⟨54521⟩⟩) exact37904RawTerms (.finite 5647228698) 37903 .exactZero (none)

def event37905 : Event := .predecessor (⟨.program ⟨257⟩, ⟨54522⟩⟩) 0 ⟨11643⟩ 32120

def event37906 : Event := .predecessor (⟨.program ⟨257⟩, ⟨54522⟩⟩) 1 ⟨54521⟩ 37904

def event37907 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨54522⟩⟩) (.product (.predecessor 0 37905 .coefficient) (.predecessor 1 37906 .coefficient) (⟨false, false, none, none, none⟩))

def event37908 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨54522⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨54519⟩⟩]⟩) [⟨.result 37900 .coefficient, false, none⟩])

def event37909 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨54522⟩⟩) (.product (.result 32120 .summary) (.transfer 37908) (⟨false, false, none, none, none⟩))

def event37910 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨54522⟩⟩, .operator (⟨32120, 0⟩, ⟨37904, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54519⟩⟩]⟩, (1)⟩)

def event37911 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨54520⟩⟩)

def event37912 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event37913 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event37914 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨11541⟩⟩) (.authority (.operator))

def event37915 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨11541⟩⟩) (.finite 18)

def event37916 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event37917 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event37918 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event37919 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event37920 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 37919

def event37921 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 37917

def event37922 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 37920 .coefficient) (.value (.predecessor 1 37921 .coefficient)))

def event37923 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event37924 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11543⟩⟩) 0 ⟨392⟩ 37923

def event37925 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11543⟩⟩) 1 ⟨11541⟩ 37915

def event37926 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨11543⟩⟩) (.sum [.predecessor 0 37924 .coefficient, .predecessor 1 37925 .coefficient])

def event37927 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨11543⟩⟩) (.finite 655358)

def event37928 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11600⟩⟩) 0 ⟨11543⟩ 37927

def event37929 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11600⟩⟩) 1 ⟨5426⟩ 37913

def event37930 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨11600⟩⟩) (.identity (.predecessor 1 37929 .coefficient))

def event37931 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨11600⟩⟩) (.finite 655360)

def event37932 : Event := .predecessor (⟨.program ⟨257⟩, ⟨24878⟩⟩) 0 ⟨11600⟩ 37931

def event37933 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨24878⟩⟩) (.authority (.programFamilyFact))

def exact37934RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨24878⟩⟩], []⟩, (1)⟩]

theorem exact37934RawTermsValid :
    exact37934RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event37934 : Event := .resultExact (⟨.program ⟨257⟩, ⟨24878⟩⟩) exact37934RawTerms (.finite 12) 37933 .exactZero (none)

def event37935 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53768⟩⟩) 0 ⟨11600⟩ 37931

def event37936 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨53768⟩⟩) (.authority (.programFamilyFact))

def exact37937RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨53768⟩⟩], []⟩, (1)⟩]

theorem exact37937RawTermsValid :
    exact37937RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event37937 : Event := .resultExact (⟨.program ⟨257⟩, ⟨53768⟩⟩) exact37937RawTerms (.finite 12) 37936 .exactZero (none)

def event37938 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53769⟩⟩) 0 ⟨53768⟩ 37937

def event37939 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53769⟩⟩) 1 ⟨24878⟩ 37934

def event37940 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨53769⟩⟩) (.product (.predecessor 0 37938 .coefficient) (.predecessor 1 37939 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event37941 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨53769⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨24878⟩⟩, ⟨.program ⟨257⟩, ⟨53768⟩⟩], []⟩) [⟨.result 37937 .coefficient, true, some 1⟩, ⟨.result 37934 .coefficient, true, some 1⟩])

def event37942 : Event := .survivorFold (1) 37941

def exact37943RawTerms : List Term := []

theorem exact37943RawTermsValid :
    exact37943RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event37943 : Event := .resultExact (⟨.program ⟨257⟩, ⟨53769⟩⟩) exact37943RawTerms (.finite 144) 37940 (.finite 144) (some (37941))

def event37944 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53770⟩⟩) 0 ⟨53769⟩ 37943

def event37945 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨53770⟩⟩) (.identity (.predecessor 0 37944 .coefficient))

def event37946 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨53770⟩⟩) (.finite 144)

def event37947 : Event := .predecessor (⟨.program ⟨257⟩, ⟨54519⟩⟩) 0 ⟨53770⟩ 37946

def event37948 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨54519⟩⟩) (.authority (.relationPreimageSource ⟨41⟩))

def exact37949RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨54519⟩⟩]⟩, (1)⟩]

theorem exact37949RawTermsValid :
    exact37949RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event37949 : Event := .resultExact (⟨.program ⟨257⟩, ⟨54519⟩⟩) exact37949RawTerms (.finite 5647228698) 37948 .exactZero (none)

def event37950 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35⟩⟩) (.authority (.operator))

def exact37951RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩]⟩, (1)⟩]

theorem exact37951RawTermsValid :
    exact37951RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event37951 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35⟩⟩) exact37951RawTerms .large 37950 .exactZero (none)

def event37952 : Event := .predecessor (⟨.program ⟨257⟩, ⟨54520⟩⟩) 0 ⟨35⟩ 37951

def event37953 : Event := .predecessor (⟨.program ⟨257⟩, ⟨54520⟩⟩) 1 ⟨54519⟩ 37949

def event37954 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨54520⟩⟩) (.product (.predecessor 0 37952 .coefficient) (.predecessor 1 37953 .coefficient) (⟨false, false, none, none, none⟩))

def event37955 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨54520⟩⟩, .operator (⟨37951, 0⟩, ⟨37949, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54519⟩⟩]⟩, (1)⟩)

def exact37956RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54519⟩⟩]⟩, (1)⟩]

theorem exact37956RawTermsValid :
    exact37956RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event37956 : Event := .resultExact (⟨.program ⟨257⟩, ⟨54520⟩⟩) exact37956RawTerms .large 37954 .exactZero (none)

def event37957 : Event := .preFoldPolynomial 37956 [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54519⟩⟩]⟩, (1)⟩] .exactZero none

def exact37958RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54519⟩⟩]⟩, (1)⟩]

def event37958 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨54520⟩⟩) 37957 exact37958RawTerms .large 37954 .exactZero (none)

def event37959 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨55602⟩⟩)

def event37960 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event37961 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event37962 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨11541⟩⟩) (.authority (.operator))

def event37963 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨11541⟩⟩) (.finite 18)

def event37964 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event37965 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event37966 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event37967 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event37968 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 37967

def event37969 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 37965

def event37970 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 37968 .coefficient) (.value (.predecessor 1 37969 .coefficient)))

def event37971 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event37972 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11543⟩⟩) 0 ⟨392⟩ 37971

def event37973 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11543⟩⟩) 1 ⟨11541⟩ 37963

def event37974 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨11543⟩⟩) (.sum [.predecessor 0 37972 .coefficient, .predecessor 1 37973 .coefficient])

def event37975 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨11543⟩⟩) (.finite 655358)

def event37976 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11600⟩⟩) 0 ⟨11543⟩ 37975

def event37977 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11600⟩⟩) 1 ⟨5426⟩ 37961

def event37978 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨11600⟩⟩) (.identity (.predecessor 1 37977 .coefficient))

def event37979 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨11600⟩⟩) (.finite 655360)

def event37980 : Event := .predecessor (⟨.program ⟨257⟩, ⟨24878⟩⟩) 0 ⟨11600⟩ 37979

def event37981 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨24878⟩⟩) (.authority (.programFamilyFact))

def exact37982RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨24878⟩⟩], []⟩, (1)⟩]

theorem exact37982RawTermsValid :
    exact37982RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event37982 : Event := .resultExact (⟨.program ⟨257⟩, ⟨24878⟩⟩) exact37982RawTerms (.finite 12) 37981 .exactZero (none)

def event37983 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53768⟩⟩) 0 ⟨11600⟩ 37979

def event37984 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨53768⟩⟩) (.authority (.programFamilyFact))

def exact37985RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨53768⟩⟩], []⟩, (1)⟩]

theorem exact37985RawTermsValid :
    exact37985RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event37985 : Event := .resultExact (⟨.program ⟨257⟩, ⟨53768⟩⟩) exact37985RawTerms (.finite 12) 37984 .exactZero (none)

def event37986 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53769⟩⟩) 0 ⟨53768⟩ 37985

def event37987 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53769⟩⟩) 1 ⟨24878⟩ 37982

def event37988 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨53769⟩⟩) (.product (.predecessor 0 37986 .coefficient) (.predecessor 1 37987 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event37989 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨53769⟩⟩, .operator (⟨37985, 0⟩, ⟨37982, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨24878⟩⟩, ⟨.program ⟨257⟩, ⟨53768⟩⟩], []⟩, (1)⟩)

def exact37990RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨24878⟩⟩, ⟨.program ⟨257⟩, ⟨53768⟩⟩], []⟩, (1)⟩]

theorem exact37990RawTermsValid :
    exact37990RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event37990 : Event := .resultExact (⟨.program ⟨257⟩, ⟨53769⟩⟩) exact37990RawTerms (.finite 144) 37988 .exactZero (none)

def event37991 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53770⟩⟩) 0 ⟨53769⟩ 37990

def event37992 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨53770⟩⟩) (.identity (.predecessor 0 37991 .coefficient))

def event37993 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨53770⟩⟩) (.finite 144)

def event37994 : Event := .predecessor (⟨.program ⟨257⟩, ⟨55042⟩⟩) 0 ⟨53770⟩ 37993

def event37995 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨55042⟩⟩) (.authority (.programFamilyFact))

def event37996 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨55042⟩⟩) (.finite 3720)

def event37997 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨7177⟩⟩) .missing

def event37998 : Event := .predecessor (⟨.program ⟨257⟩, ⟨55043⟩⟩) 0 ⟨7177⟩ 37997

def event37999 : Event := .predecessor (⟨.program ⟨257⟩, ⟨55043⟩⟩) 1 ⟨55042⟩ 37996

def event38000 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨55043⟩⟩) (.authority (.operator))

def exact38001RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨55043⟩⟩]⟩, (1)⟩]

theorem exact38001RawTermsValid :
    exact38001RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event38001 : Event := .resultExact (⟨.program ⟨257⟩, ⟨55043⟩⟩) exact38001RawTerms .large 38000 .exactZero (none)

def event38002 : Event := .predecessor (⟨.program ⟨257⟩, ⟨55598⟩⟩) 0 ⟨55043⟩ 38001

def event38003 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨55598⟩⟩) (.authority (.operator))

def exact38004RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨55598⟩⟩]⟩, (1)⟩]

theorem exact38004RawTermsValid :
    exact38004RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event38004 : Event := .resultExact (⟨.program ⟨257⟩, ⟨55598⟩⟩) exact38004RawTerms (.finite 8192) 38003 .exactZero (none)

def event38005 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨136⟩⟩) (.authority (.operator))

def event38006 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨136⟩⟩) .exactZero

def event38007 : Event := .predecessor (⟨.program ⟨257⟩, ⟨55302⟩⟩) 0 ⟨53770⟩ 37993

def event38008 : Event := .predecessor (⟨.program ⟨257⟩, ⟨55302⟩⟩) 1 ⟨136⟩ 38006

def event38009 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨55302⟩⟩) (.sum [.predecessor 0 38007 .coefficient, .predecessor 1 38008 .coefficient])

def event38010 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨55302⟩⟩) (.finite 144)

def event38011 : Event := .predecessor (⟨.program ⟨257⟩, ⟨55303⟩⟩) 0 ⟨55302⟩ 38010

def event38012 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨55303⟩⟩) (.identity (.predecessor 0 38011 .coefficient))

def exact38013RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨24878⟩⟩, ⟨.program ⟨257⟩, ⟨53768⟩⟩], []⟩, (1)⟩]

theorem exact38013RawTermsValid :
    exact38013RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event38013 : Event := .resultExact (⟨.program ⟨257⟩, ⟨55303⟩⟩) exact38013RawTerms (.finite 144) 38012 .exactZero (none)

def event38014 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6908⟩⟩) (.authority (.factStore))

def exact38015RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact38015RawTermsValid :
    exact38015RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event38015 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6908⟩⟩) exact38015RawTerms .large 38014 .exactZero (none)

def event38016 : Event := .predecessor (⟨.program ⟨257⟩, ⟨55304⟩⟩) 0 ⟨6908⟩ 38015

def event38017 : Event := .predecessor (⟨.program ⟨257⟩, ⟨55304⟩⟩) 1 ⟨55303⟩ 38013

def event38018 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨55304⟩⟩) (.product (.predecessor 0 38016 .coefficient) (.predecessor 1 38017 .coefficient) (⟨false, false, none, none, none⟩))

def event38019 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨55304⟩⟩, .operator (⟨38015, 0⟩, ⟨38013, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨24878⟩⟩, ⟨.program ⟨257⟩, ⟨53768⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact38020RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨24878⟩⟩, ⟨.program ⟨257⟩, ⟨53768⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact38020RawTermsValid :
    exact38020RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event38020 : Event := .resultExact (⟨.program ⟨257⟩, ⟨55304⟩⟩) exact38020RawTerms .large 38018 .exactZero (none)

def event38021 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨2370⟩⟩) (.authority (.operator))

def event38022 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨2370⟩⟩) (.finite 1)

def event38023 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7178⟩⟩) 0 ⟨7177⟩ 37997

def event38024 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7178⟩⟩) (.authority (.operator))

def exact38025RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7178⟩⟩]⟩, (1)⟩]

theorem exact38025RawTermsValid :
    exact38025RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event38025 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7178⟩⟩) exact38025RawTerms .large 38024 .exactZero (none)

def event38026 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7272⟩⟩) 0 ⟨7178⟩ 38025

def event38027 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7272⟩⟩) (.identity (.predecessor 0 38026 .coefficient))

def exact38028RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7272⟩⟩]⟩, (1)⟩]

theorem exact38028RawTermsValid :
    exact38028RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event38028 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7272⟩⟩) exact38028RawTerms .large 38027 .exactZero (none)

def event38029 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9529⟩⟩) 0 ⟨7272⟩ 38028

def event38030 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9529⟩⟩) (.authority (.operator))

def exact38031RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨9529⟩⟩]⟩, (1)⟩]

theorem exact38031RawTermsValid :
    exact38031RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event38031 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9529⟩⟩) exact38031RawTerms (.finite 8192) 38030 .exactZero (none)

def event38032 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9530⟩⟩) 0 ⟨9529⟩ 38031

def event38033 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9530⟩⟩) 1 ⟨2370⟩ 38022

def event38034 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9530⟩⟩) (.scale (.predecessor 0 38032 .coefficient) (.value (.predecessor 1 38033 .coefficient)))

def exact38035RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨9529⟩⟩]⟩, (1)⟩]

theorem exact38035RawTermsValid :
    exact38035RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event38035 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9530⟩⟩) exact38035RawTerms (.finite 8192) 38034 .exactZero (none)

def event38036 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7289⟩⟩) 0 ⟨7178⟩ 38025

def event38037 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7289⟩⟩) (.identity (.predecessor 0 38036 .coefficient))

def exact38038RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7289⟩⟩]⟩, (1)⟩]

theorem exact38038RawTermsValid :
    exact38038RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event38038 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7289⟩⟩) exact38038RawTerms .large 38037 .exactZero (none)

def event38039 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9531⟩⟩) 0 ⟨7289⟩ 38038

def event38040 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9531⟩⟩) 1 ⟨9530⟩ 38035

def event38041 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9531⟩⟩) (.product (.predecessor 0 38039 .coefficient) (.predecessor 1 38040 .coefficient) (⟨false, false, none, none, none⟩))

def event38042 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨9531⟩⟩, .operator (⟨38038, 0⟩, ⟨38035, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩]⟩, (1)⟩)

def exact38043RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩]⟩, (1)⟩]

theorem exact38043RawTermsValid :
    exact38043RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event38043 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9531⟩⟩) exact38043RawTerms .large 38041 .exactZero (none)

def event38044 : Event := .predecessor (⟨.program ⟨257⟩, ⟨55305⟩⟩) 0 ⟨9531⟩ 38043

def event38045 : Event := .predecessor (⟨.program ⟨257⟩, ⟨55305⟩⟩) 1 ⟨55304⟩ 38020

def event38046 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨55305⟩⟩) (.sum [.predecessor 0 38044 .coefficient, .predecessor 1 38045 .coefficient])

def exact38047RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨24878⟩⟩, ⟨.program ⟨257⟩, ⟨53768⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact38047RawTermsValid :
    exact38047RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event38047 : Event := .resultExact (⟨.program ⟨257⟩, ⟨55305⟩⟩) exact38047RawTerms .large 38046 .exactZero (none)

def event38048 : Event := .predecessor (⟨.program ⟨257⟩, ⟨55601⟩⟩) 0 ⟨55305⟩ 38047

def event38049 : Event := .predecessor (⟨.program ⟨257⟩, ⟨55601⟩⟩) 1 ⟨55598⟩ 38004

def event38050 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨55601⟩⟩) (.product (.predecessor 0 38048 .coefficient) (.predecessor 1 38049 .coefficient) (⟨false, false, none, none, none⟩))

def event38051 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨55601⟩⟩, .operator (⟨38047, 0⟩, ⟨38004, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55598⟩⟩]⟩, (1)⟩)

def event38052 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨55601⟩⟩, .operator (⟨38047, 1⟩, ⟨38004, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨24878⟩⟩, ⟨.program ⟨257⟩, ⟨53768⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨55598⟩⟩]⟩, (-1)⟩)

def event38053 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨55601⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨24878⟩⟩, ⟨.program ⟨257⟩, ⟨53768⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨55598⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨55598⟩⟩) ⟨55043⟩ 38001)

def event38054 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨55601⟩⟩, .relation 38053 0, ⟨[⟨.program ⟨257⟩, ⟨24878⟩⟩, ⟨.program ⟨257⟩, ⟨53768⟩⟩], [⟨.program ⟨257⟩, ⟨55043⟩⟩]⟩, (-1)⟩)

def exact38055RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55598⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨24878⟩⟩, ⟨.program ⟨257⟩, ⟨53768⟩⟩], [⟨.program ⟨257⟩, ⟨55043⟩⟩]⟩, (-1)⟩]

theorem exact38055RawTermsValid :
    exact38055RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event38055 : Event := .resultExact (⟨.program ⟨257⟩, ⟨55601⟩⟩) exact38055RawTerms .large 38050 .exactZero (none)

def event38056 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53940⟩⟩) 0 ⟨53770⟩ 37993

def event38057 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨53940⟩⟩) (.authority (.programFamilyFact))

def exact38058RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨53940⟩⟩], []⟩, (1)⟩]

theorem exact38058RawTermsValid :
    exact38058RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event38058 : Event := .resultExact (⟨.program ⟨257⟩, ⟨53940⟩⟩) exact38058RawTerms (.finite 12) 38057 .exactZero (none)

def event38059 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53942⟩⟩) 0 ⟨6908⟩ 38015

def event38060 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53942⟩⟩) 1 ⟨53940⟩ 38058

def event38061 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨53942⟩⟩) (.product (.predecessor 0 38059 .coefficient) (.predecessor 1 38060 .coefficient) (⟨false, true, none, none, some 1⟩))

def event38062 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨53942⟩⟩, .operator (⟨38015, 0⟩, ⟨38058, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨53940⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact38063RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨53940⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact38063RawTermsValid :
    exact38063RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event38063 : Event := .resultExact (⟨.program ⟨257⟩, ⟨53942⟩⟩) exact38063RawTerms .large 38061 .exactZero (none)

def event38064 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7184⟩⟩) 0 ⟨7177⟩ 37997

def event38065 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7184⟩⟩) (.authority (.operator))

def exact38066RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7184⟩⟩]⟩, (1)⟩]

theorem exact38066RawTermsValid :
    exact38066RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event38066 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7184⟩⟩) exact38066RawTerms .large 38065 .exactZero (none)

def event38067 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53943⟩⟩) 0 ⟨7184⟩ 38066

def event38068 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53943⟩⟩) 1 ⟨53942⟩ 38063

def event38069 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨53943⟩⟩) (.sum [.predecessor 0 38067 .coefficient, .predecessor 1 38068 .coefficient])

def exact38070RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7184⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨53940⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact38070RawTermsValid :
    exact38070RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event38070 : Event := .resultExact (⟨.program ⟨257⟩, ⟨53943⟩⟩) exact38070RawTerms .large 38069 .exactZero (none)

def event38071 : Event := .predecessor (⟨.program ⟨257⟩, ⟨55602⟩⟩) 0 ⟨53943⟩ 38070

def event38072 : Event := .predecessor (⟨.program ⟨257⟩, ⟨55602⟩⟩) 1 ⟨55601⟩ 38055

def event38073 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨55602⟩⟩) (.sum [.predecessor 0 38071 .coefficient, .predecessor 1 38072 .coefficient])

def exact38074RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7184⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55598⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨24878⟩⟩, ⟨.program ⟨257⟩, ⟨53768⟩⟩], [⟨.program ⟨257⟩, ⟨55043⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨53940⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact38074RawTermsValid :
    exact38074RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event38074 : Event := .resultExact (⟨.program ⟨257⟩, ⟨55602⟩⟩) exact38074RawTerms .large 38073 .exactZero (none)

def event38075 : Event := .preFoldPolynomial 38074 [⟨⟨[], [⟨.program ⟨257⟩, ⟨7184⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55598⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨24878⟩⟩, ⟨.program ⟨257⟩, ⟨53768⟩⟩], [⟨.program ⟨257⟩, ⟨55043⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨53940⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩] .exactZero none

def exact38076RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7184⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55598⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨24878⟩⟩, ⟨.program ⟨257⟩, ⟨53768⟩⟩], [⟨.program ⟨257⟩, ⟨55043⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨53940⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

def event38076 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨55602⟩⟩) 38075 exact38076RawTerms .large 38073 .exactZero (none)

def event38077 : Event := .specializationComputed (⟨.program ⟨257⟩, ⟨53770⟩⟩) ⟨⟨63⟩, ⟨41⟩, ⟨135⟩⟩ ⟨37911, 38077⟩

def event38078 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨54522⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54519⟩⟩]⟩) (1) 0 2 (.universal 38077 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54519⟩⟩]⟩) (none) 38076)

def event38079 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨54522⟩⟩, .relation 38078 0, ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7184⟩⟩]⟩, (1)⟩)

def event38080 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨54522⟩⟩, .relation 38078 1, ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55598⟩⟩]⟩, (-1)⟩)

def event38081 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨54522⟩⟩, .relation 38078 2, ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨24878⟩⟩, ⟨.program ⟨257⟩, ⟨53768⟩⟩], [⟨.program ⟨257⟩, ⟨55043⟩⟩]⟩, (1)⟩)

def event38082 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨54522⟩⟩, .relation 38078 3, ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨53940⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact38083RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7184⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55598⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨24878⟩⟩, ⟨.program ⟨257⟩, ⟨53768⟩⟩], [⟨.program ⟨257⟩, ⟨55043⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨53940⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact38083RawTermsValid :
    exact38083RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event38083 : Event := .resultExact (⟨.program ⟨257⟩, ⟨54522⟩⟩) exact38083RawTerms .large 37907 (.finite 202072841853861888) (some (37909))

def event38084 : Event := .predecessor (⟨.program ⟨257⟩, ⟨55600⟩⟩) 0 ⟨54522⟩ 38083

def event38085 : Event := .predecessor (⟨.program ⟨257⟩, ⟨55600⟩⟩) 1 ⟨55599⟩ 37897

def event38086 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨55600⟩⟩) (.sum [.predecessor 0 38084 .coefficient, .predecessor 1 38085 .coefficient])

def event38087 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨55600⟩⟩, .operator (⟨38083, 2⟩, ⟨37897, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨24878⟩⟩, ⟨.program ⟨257⟩, ⟨53768⟩⟩], [⟨.program ⟨257⟩, ⟨55043⟩⟩]⟩, (-1)⟩)

def event38088 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨55600⟩⟩, .operator (⟨38083, 1⟩, ⟨37897, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55598⟩⟩]⟩, (1)⟩)

def event38089 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨55600⟩⟩) (.sum [.result 38083 .summary, .result 37897 .summary])

def exact38090RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7184⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨53940⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact38090RawTermsValid :
    exact38090RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event38090 : Event := .resultExact (⟨.program ⟨257⟩, ⟨55600⟩⟩) exact38090RawTerms .large 38086 (.finite 2997907760060573155328) (some (38089))

def event38091 : Event := .predecessor (⟨.program ⟨257⟩, ⟨56213⟩⟩) 0 ⟨55600⟩ 38090

def event38092 : Event := .predecessor (⟨.program ⟨257⟩, ⟨56213⟩⟩) 1 ⟨56211⟩ 37813

def event38093 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨56213⟩⟩) (.product (.predecessor 0 38091 .coefficient) (.predecessor 1 38092 .coefficient) (⟨false, false, none, none, none⟩))

def event38094 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨56213⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨56211⟩⟩]⟩) [⟨.result 37813 .coefficient, false, none⟩])

def event38095 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨56213⟩⟩) (.product (.result 38090 .summary) (.transfer 38094) (⟨false, false, none, none, none⟩))

def event38096 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨56213⟩⟩, .operator (⟨38090, 0⟩, ⟨37813, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7184⟩⟩, ⟨.program ⟨257⟩, ⟨56211⟩⟩]⟩, (1)⟩)

def event38097 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨56213⟩⟩, .operator (⟨38090, 1⟩, ⟨37813, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨53940⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨56211⟩⟩]⟩, (-1)⟩)

def event38098 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨56213⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨53940⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨56211⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨56211⟩⟩) ⟨55222⟩ 37810)

def event38099 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨56213⟩⟩, .relation 38098 0, ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨53940⟩⟩], [⟨.program ⟨257⟩, ⟨55222⟩⟩]⟩, (-1)⟩)

def exact38100RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7184⟩⟩, ⟨.program ⟨257⟩, ⟨56211⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨53940⟩⟩], [⟨.program ⟨257⟩, ⟨55222⟩⟩]⟩, (-1)⟩]

theorem exact38100RawTermsValid :
    exact38100RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event38100 : Event := .resultExact (⟨.program ⟨257⟩, ⟨56213⟩⟩) exact38100RawTerms .large 38093 (.finite 32189789464711941702873220382720) (some (38095))

def event38101 : Event := .predecessor (⟨.program ⟨257⟩, ⟨54916⟩⟩) 0 ⟨53941⟩ 1135

def event38102 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨54916⟩⟩) (.authority (.relationPreimageSource ⟨68⟩))

def exact38103RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨54916⟩⟩]⟩, (1)⟩]

theorem exact38103RawTermsValid :
    exact38103RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event38103 : Event := .resultExact (⟨.program ⟨257⟩, ⟨54916⟩⟩) exact38103RawTerms (.finite 5647228698) 38102 .exactZero (none)

def event38104 : Event := .predecessor (⟨.program ⟨257⟩, ⟨54918⟩⟩) 0 ⟨54916⟩ 38103

def event38105 : Event := .predecessor (⟨.program ⟨257⟩, ⟨54918⟩⟩) 1 ⟨2370⟩ 4

def event38106 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨54918⟩⟩) (.scale (.predecessor 0 38104 .coefficient) (.value (.predecessor 1 38105 .coefficient)))

def exact38107RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨54916⟩⟩]⟩, (1)⟩]

theorem exact38107RawTermsValid :
    exact38107RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event38107 : Event := .resultExact (⟨.program ⟨257⟩, ⟨54918⟩⟩) exact38107RawTerms (.finite 5647228698) 38106 .exactZero (none)

def event38108 : Event := .predecessor (⟨.program ⟨257⟩, ⟨54919⟩⟩) 0 ⟨11643⟩ 32120

def event38109 : Event := .predecessor (⟨.program ⟨257⟩, ⟨54919⟩⟩) 1 ⟨54918⟩ 38107

def event38110 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨54919⟩⟩) (.product (.predecessor 0 38108 .coefficient) (.predecessor 1 38109 .coefficient) (⟨false, false, none, none, none⟩))

def event38111 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨54919⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨54916⟩⟩]⟩) [⟨.result 38103 .coefficient, false, none⟩])

def event38112 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨54919⟩⟩) (.product (.result 32120 .summary) (.transfer 38111) (⟨false, false, none, none, none⟩))

def event38113 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨54919⟩⟩, .operator (⟨32120, 0⟩, ⟨38107, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54916⟩⟩]⟩, (1)⟩)

def event38114 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨54917⟩⟩)

def event38115 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event38116 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event38117 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨11541⟩⟩) (.authority (.operator))

def event38118 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨11541⟩⟩) (.finite 18)

def event38119 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event38120 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event38121 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event38122 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event38123 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 38122

def event38124 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 38120

def event38125 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 38123 .coefficient) (.value (.predecessor 1 38124 .coefficient)))

def event38126 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event38127 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11543⟩⟩) 0 ⟨392⟩ 38126

def event38128 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11543⟩⟩) 1 ⟨11541⟩ 38118

def event38129 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨11543⟩⟩) (.sum [.predecessor 0 38127 .coefficient, .predecessor 1 38128 .coefficient])

def event38130 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨11543⟩⟩) (.finite 655358)

def event38131 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11600⟩⟩) 0 ⟨11543⟩ 38130

def event38132 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11600⟩⟩) 1 ⟨5426⟩ 38116

def event38133 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨11600⟩⟩) (.identity (.predecessor 1 38132 .coefficient))

def event38134 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨11600⟩⟩) (.finite 655360)

def event38135 : Event := .predecessor (⟨.program ⟨257⟩, ⟨24878⟩⟩) 0 ⟨11600⟩ 38134

def event38136 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨24878⟩⟩) (.authority (.programFamilyFact))

def exact38137RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨24878⟩⟩], []⟩, (1)⟩]

theorem exact38137RawTermsValid :
    exact38137RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event38137 : Event := .resultExact (⟨.program ⟨257⟩, ⟨24878⟩⟩) exact38137RawTerms (.finite 12) 38136 .exactZero (none)

def event38138 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53768⟩⟩) 0 ⟨11600⟩ 38134

def event38139 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨53768⟩⟩) (.authority (.programFamilyFact))

def exact38140RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨53768⟩⟩], []⟩, (1)⟩]

theorem exact38140RawTermsValid :
    exact38140RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event38140 : Event := .resultExact (⟨.program ⟨257⟩, ⟨53768⟩⟩) exact38140RawTerms (.finite 12) 38139 .exactZero (none)

def event38141 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53769⟩⟩) 0 ⟨53768⟩ 38140

def event38142 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53769⟩⟩) 1 ⟨24878⟩ 38137

def event38143 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨53769⟩⟩) (.product (.predecessor 0 38141 .coefficient) (.predecessor 1 38142 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def eventLeaf2368 : Array AnnotatedEvent := #[
  { event := event37888
    frameStart := 0 },
  { event := event37889
    frameStart := 0 },
  { event := event37890
    frameStart := 0 },
  { event := event37891
    frameStart := 0 },
  { event := event37892
    frameStart := 0 },
  { event := event37893
    frameStart := 0 },
  { event := event37894
    frameStart := 0 },
  { event := event37895
    frameStart := 0 },
  { event := event37896
    frameStart := 0 },
  { event := event37897
    frameStart := 0 },
  { event := event37898
    frameStart := 0 },
  { event := event37899
    frameStart := 0 },
  { event := event37900
    frameStart := 0 },
  { event := event37901
    frameStart := 0 },
  { event := event37902
    frameStart := 0 },
  { event := event37903
    frameStart := 0 }
]

def eventLeaf2369 : Array AnnotatedEvent := #[
  { event := event37904
    frameStart := 0 },
  { event := event37905
    frameStart := 0 },
  { event := event37906
    frameStart := 0 },
  { event := event37907
    frameStart := 0 },
  { event := event37908
    frameStart := 0 },
  { event := event37909
    frameStart := 0 },
  { event := event37910
    frameStart := 0 },
  { event := event37911
    frameStart := 37911 },
  { event := event37912
    frameStart := 37911 },
  { event := event37913
    frameStart := 37911 },
  { event := event37914
    frameStart := 37911 },
  { event := event37915
    frameStart := 37911 },
  { event := event37916
    frameStart := 37911 },
  { event := event37917
    frameStart := 37911 },
  { event := event37918
    frameStart := 37911 },
  { event := event37919
    frameStart := 37911 }
]

def eventLeaf2370 : Array AnnotatedEvent := #[
  { event := event37920
    frameStart := 37911 },
  { event := event37921
    frameStart := 37911 },
  { event := event37922
    frameStart := 37911 },
  { event := event37923
    frameStart := 37911 },
  { event := event37924
    frameStart := 37911 },
  { event := event37925
    frameStart := 37911 },
  { event := event37926
    frameStart := 37911 },
  { event := event37927
    frameStart := 37911 },
  { event := event37928
    frameStart := 37911 },
  { event := event37929
    frameStart := 37911 },
  { event := event37930
    frameStart := 37911 },
  { event := event37931
    frameStart := 37911 },
  { event := event37932
    frameStart := 37911 },
  { event := event37933
    frameStart := 37911 },
  { event := event37934
    frameStart := 37911 },
  { event := event37935
    frameStart := 37911 }
]

def eventLeaf2371 : Array AnnotatedEvent := #[
  { event := event37936
    frameStart := 37911 },
  { event := event37937
    frameStart := 37911 },
  { event := event37938
    frameStart := 37911 },
  { event := event37939
    frameStart := 37911 },
  { event := event37940
    frameStart := 37911 },
  { event := event37941
    frameStart := 37911 },
  { event := event37942
    frameStart := 37911 },
  { event := event37943
    frameStart := 37911 },
  { event := event37944
    frameStart := 37911 },
  { event := event37945
    frameStart := 37911 },
  { event := event37946
    frameStart := 37911 },
  { event := event37947
    frameStart := 37911 },
  { event := event37948
    frameStart := 37911 },
  { event := event37949
    frameStart := 37911 },
  { event := event37950
    frameStart := 37911 },
  { event := event37951
    frameStart := 37911 }
]

def eventLeaf2372 : Array AnnotatedEvent := #[
  { event := event37952
    frameStart := 37911 },
  { event := event37953
    frameStart := 37911 },
  { event := event37954
    frameStart := 37911 },
  { event := event37955
    frameStart := 37911 },
  { event := event37956
    frameStart := 37911 },
  { event := event37957
    frameStart := 37911 },
  { event := event37958
    frameStart := 37911 },
  { event := event37959
    frameStart := 37959 },
  { event := event37960
    frameStart := 37959 },
  { event := event37961
    frameStart := 37959 },
  { event := event37962
    frameStart := 37959 },
  { event := event37963
    frameStart := 37959 },
  { event := event37964
    frameStart := 37959 },
  { event := event37965
    frameStart := 37959 },
  { event := event37966
    frameStart := 37959 },
  { event := event37967
    frameStart := 37959 }
]

def eventLeaf2373 : Array AnnotatedEvent := #[
  { event := event37968
    frameStart := 37959 },
  { event := event37969
    frameStart := 37959 },
  { event := event37970
    frameStart := 37959 },
  { event := event37971
    frameStart := 37959 },
  { event := event37972
    frameStart := 37959 },
  { event := event37973
    frameStart := 37959 },
  { event := event37974
    frameStart := 37959 },
  { event := event37975
    frameStart := 37959 },
  { event := event37976
    frameStart := 37959 },
  { event := event37977
    frameStart := 37959 },
  { event := event37978
    frameStart := 37959 },
  { event := event37979
    frameStart := 37959 },
  { event := event37980
    frameStart := 37959 },
  { event := event37981
    frameStart := 37959 },
  { event := event37982
    frameStart := 37959 },
  { event := event37983
    frameStart := 37959 }
]

def eventLeaf2374 : Array AnnotatedEvent := #[
  { event := event37984
    frameStart := 37959 },
  { event := event37985
    frameStart := 37959 },
  { event := event37986
    frameStart := 37959 },
  { event := event37987
    frameStart := 37959 },
  { event := event37988
    frameStart := 37959 },
  { event := event37989
    frameStart := 37959 },
  { event := event37990
    frameStart := 37959 },
  { event := event37991
    frameStart := 37959 },
  { event := event37992
    frameStart := 37959 },
  { event := event37993
    frameStart := 37959 },
  { event := event37994
    frameStart := 37959 },
  { event := event37995
    frameStart := 37959 },
  { event := event37996
    frameStart := 37959 },
  { event := event37997
    frameStart := 37959 },
  { event := event37998
    frameStart := 37959 },
  { event := event37999
    frameStart := 37959 }
]

def eventLeaf2375 : Array AnnotatedEvent := #[
  { event := event38000
    frameStart := 37959 },
  { event := event38001
    frameStart := 37959 },
  { event := event38002
    frameStart := 37959 },
  { event := event38003
    frameStart := 37959 },
  { event := event38004
    frameStart := 37959 },
  { event := event38005
    frameStart := 37959 },
  { event := event38006
    frameStart := 37959 },
  { event := event38007
    frameStart := 37959 },
  { event := event38008
    frameStart := 37959 },
  { event := event38009
    frameStart := 37959 },
  { event := event38010
    frameStart := 37959 },
  { event := event38011
    frameStart := 37959 },
  { event := event38012
    frameStart := 37959 },
  { event := event38013
    frameStart := 37959 },
  { event := event38014
    frameStart := 37959 },
  { event := event38015
    frameStart := 37959 }
]

def eventLeaf2376 : Array AnnotatedEvent := #[
  { event := event38016
    frameStart := 37959 },
  { event := event38017
    frameStart := 37959 },
  { event := event38018
    frameStart := 37959 },
  { event := event38019
    frameStart := 37959 },
  { event := event38020
    frameStart := 37959 },
  { event := event38021
    frameStart := 37959 },
  { event := event38022
    frameStart := 37959 },
  { event := event38023
    frameStart := 37959 },
  { event := event38024
    frameStart := 37959 },
  { event := event38025
    frameStart := 37959 },
  { event := event38026
    frameStart := 37959 },
  { event := event38027
    frameStart := 37959 },
  { event := event38028
    frameStart := 37959 },
  { event := event38029
    frameStart := 37959 },
  { event := event38030
    frameStart := 37959 },
  { event := event38031
    frameStart := 37959 }
]

def eventLeaf2377 : Array AnnotatedEvent := #[
  { event := event38032
    frameStart := 37959 },
  { event := event38033
    frameStart := 37959 },
  { event := event38034
    frameStart := 37959 },
  { event := event38035
    frameStart := 37959 },
  { event := event38036
    frameStart := 37959 },
  { event := event38037
    frameStart := 37959 },
  { event := event38038
    frameStart := 37959 },
  { event := event38039
    frameStart := 37959 },
  { event := event38040
    frameStart := 37959 },
  { event := event38041
    frameStart := 37959 },
  { event := event38042
    frameStart := 37959 },
  { event := event38043
    frameStart := 37959 },
  { event := event38044
    frameStart := 37959 },
  { event := event38045
    frameStart := 37959 },
  { event := event38046
    frameStart := 37959 },
  { event := event38047
    frameStart := 37959 }
]

def eventLeaf2378 : Array AnnotatedEvent := #[
  { event := event38048
    frameStart := 37959 },
  { event := event38049
    frameStart := 37959 },
  { event := event38050
    frameStart := 37959 },
  { event := event38051
    frameStart := 37959 },
  { event := event38052
    frameStart := 37959 },
  { event := event38053
    frameStart := 37959 },
  { event := event38054
    frameStart := 37959 },
  { event := event38055
    frameStart := 37959 },
  { event := event38056
    frameStart := 37959 },
  { event := event38057
    frameStart := 37959 },
  { event := event38058
    frameStart := 37959 },
  { event := event38059
    frameStart := 37959 },
  { event := event38060
    frameStart := 37959 },
  { event := event38061
    frameStart := 37959 },
  { event := event38062
    frameStart := 37959 },
  { event := event38063
    frameStart := 37959 }
]

def eventLeaf2379 : Array AnnotatedEvent := #[
  { event := event38064
    frameStart := 37959 },
  { event := event38065
    frameStart := 37959 },
  { event := event38066
    frameStart := 37959 },
  { event := event38067
    frameStart := 37959 },
  { event := event38068
    frameStart := 37959 },
  { event := event38069
    frameStart := 37959 },
  { event := event38070
    frameStart := 37959 },
  { event := event38071
    frameStart := 37959 },
  { event := event38072
    frameStart := 37959 },
  { event := event38073
    frameStart := 37959 },
  { event := event38074
    frameStart := 37959 },
  { event := event38075
    frameStart := 37959 },
  { event := event38076
    frameStart := 37959 },
  { event := event38077
    frameStart := 0 },
  { event := event38078
    frameStart := 0 },
  { event := event38079
    frameStart := 0 }
]

def eventLeaf2380 : Array AnnotatedEvent := #[
  { event := event38080
    frameStart := 0 },
  { event := event38081
    frameStart := 0 },
  { event := event38082
    frameStart := 0 },
  { event := event38083
    frameStart := 0 },
  { event := event38084
    frameStart := 0 },
  { event := event38085
    frameStart := 0 },
  { event := event38086
    frameStart := 0 },
  { event := event38087
    frameStart := 0 },
  { event := event38088
    frameStart := 0 },
  { event := event38089
    frameStart := 0 },
  { event := event38090
    frameStart := 0 },
  { event := event38091
    frameStart := 0 },
  { event := event38092
    frameStart := 0 },
  { event := event38093
    frameStart := 0 },
  { event := event38094
    frameStart := 0 },
  { event := event38095
    frameStart := 0 }
]

def eventLeaf2381 : Array AnnotatedEvent := #[
  { event := event38096
    frameStart := 0 },
  { event := event38097
    frameStart := 0 },
  { event := event38098
    frameStart := 0 },
  { event := event38099
    frameStart := 0 },
  { event := event38100
    frameStart := 0 },
  { event := event38101
    frameStart := 0 },
  { event := event38102
    frameStart := 0 },
  { event := event38103
    frameStart := 0 },
  { event := event38104
    frameStart := 0 },
  { event := event38105
    frameStart := 0 },
  { event := event38106
    frameStart := 0 },
  { event := event38107
    frameStart := 0 },
  { event := event38108
    frameStart := 0 },
  { event := event38109
    frameStart := 0 },
  { event := event38110
    frameStart := 0 },
  { event := event38111
    frameStart := 0 }
]

def eventLeaf2382 : Array AnnotatedEvent := #[
  { event := event38112
    frameStart := 0 },
  { event := event38113
    frameStart := 0 },
  { event := event38114
    frameStart := 38114 },
  { event := event38115
    frameStart := 38114 },
  { event := event38116
    frameStart := 38114 },
  { event := event38117
    frameStart := 38114 },
  { event := event38118
    frameStart := 38114 },
  { event := event38119
    frameStart := 38114 },
  { event := event38120
    frameStart := 38114 },
  { event := event38121
    frameStart := 38114 },
  { event := event38122
    frameStart := 38114 },
  { event := event38123
    frameStart := 38114 },
  { event := event38124
    frameStart := 38114 },
  { event := event38125
    frameStart := 38114 },
  { event := event38126
    frameStart := 38114 },
  { event := event38127
    frameStart := 38114 }
]

def eventLeaf2383 : Array AnnotatedEvent := #[
  { event := event38128
    frameStart := 38114 },
  { event := event38129
    frameStart := 38114 },
  { event := event38130
    frameStart := 38114 },
  { event := event38131
    frameStart := 38114 },
  { event := event38132
    frameStart := 38114 },
  { event := event38133
    frameStart := 38114 },
  { event := event38134
    frameStart := 38114 },
  { event := event38135
    frameStart := 38114 },
  { event := event38136
    frameStart := 38114 },
  { event := event38137
    frameStart := 38114 },
  { event := event38138
    frameStart := 38114 },
  { event := event38139
    frameStart := 38114 },
  { event := event38140
    frameStart := 38114 },
  { event := event38141
    frameStart := 38114 },
  { event := event38142
    frameStart := 38114 },
  { event := event38143
    frameStart := 38114 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events148
