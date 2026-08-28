import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events320

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event81920 : Event := .predecessor (⟨.program ⟨257⟩, ⟨55293⟩⟩) 1 ⟨55292⟩ 81895

def event81921 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨55293⟩⟩) (.sum [.predecessor 0 81919 .coefficient, .predecessor 1 81920 .coefficient])

def exact81922RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨24842⟩⟩, ⟨.program ⟨257⟩, ⟨53687⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact81922RawTermsValid :
    exact81922RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event81922 : Event := .resultExact (⟨.program ⟨257⟩, ⟨55293⟩⟩) exact81922RawTerms .large 81921 .exactZero (none)

def event81923 : Event := .predecessor (⟨.program ⟨257⟩, ⟨55568⟩⟩) 0 ⟨55293⟩ 81922

def event81924 : Event := .predecessor (⟨.program ⟨257⟩, ⟨55568⟩⟩) 1 ⟨55565⟩ 81879

def event81925 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨55568⟩⟩) (.product (.predecessor 0 81923 .coefficient) (.predecessor 1 81924 .coefficient) (⟨false, false, none, none, none⟩))

def event81926 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨55568⟩⟩, .operator (⟨81922, 0⟩, ⟨81879, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55565⟩⟩]⟩, (1)⟩)

def event81927 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨55568⟩⟩, .operator (⟨81922, 1⟩, ⟨81879, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨24842⟩⟩, ⟨.program ⟨257⟩, ⟨53687⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨55565⟩⟩]⟩, (-1)⟩)

def event81928 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨55568⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨24842⟩⟩, ⟨.program ⟨257⟩, ⟨53687⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨55565⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨55565⟩⟩) ⟨55025⟩ 81876)

def event81929 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨55568⟩⟩, .relation 81928 0, ⟨[⟨.program ⟨257⟩, ⟨24842⟩⟩, ⟨.program ⟨257⟩, ⟨53687⟩⟩], [⟨.program ⟨257⟩, ⟨55025⟩⟩]⟩, (-1)⟩)

def exact81930RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55565⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨24842⟩⟩, ⟨.program ⟨257⟩, ⟨53687⟩⟩], [⟨.program ⟨257⟩, ⟨55025⟩⟩]⟩, (-1)⟩]

theorem exact81930RawTermsValid :
    exact81930RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event81930 : Event := .resultExact (⟨.program ⟨257⟩, ⟨55568⟩⟩) exact81930RawTerms .large 81925 .exactZero (none)

def event81931 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53916⟩⟩) 0 ⟨53689⟩ 81868

def event81932 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨53916⟩⟩) (.authority (.programFamilyFact))

def exact81933RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨53916⟩⟩], []⟩, (1)⟩]

theorem exact81933RawTermsValid :
    exact81933RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event81933 : Event := .resultExact (⟨.program ⟨257⟩, ⟨53916⟩⟩) exact81933RawTerms (.finite 12) 81932 .exactZero (none)

def event81934 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53918⟩⟩) 0 ⟨6908⟩ 81890

def event81935 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53918⟩⟩) 1 ⟨53916⟩ 81933

def event81936 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨53918⟩⟩) (.product (.predecessor 0 81934 .coefficient) (.predecessor 1 81935 .coefficient) (⟨false, true, none, none, some 1⟩))

def event81937 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨53918⟩⟩, .operator (⟨81890, 0⟩, ⟨81933, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨53916⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact81938RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨53916⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact81938RawTermsValid :
    exact81938RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event81938 : Event := .resultExact (⟨.program ⟨257⟩, ⟨53918⟩⟩) exact81938RawTerms .large 81936 .exactZero (none)

def event81939 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7184⟩⟩) 0 ⟨7177⟩ 81872

def event81940 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7184⟩⟩) (.authority (.operator))

def exact81941RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7184⟩⟩]⟩, (1)⟩]

theorem exact81941RawTermsValid :
    exact81941RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event81941 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7184⟩⟩) exact81941RawTerms .large 81940 .exactZero (none)

def event81942 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53919⟩⟩) 0 ⟨7184⟩ 81941

def event81943 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53919⟩⟩) 1 ⟨53918⟩ 81938

def event81944 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨53919⟩⟩) (.sum [.predecessor 0 81942 .coefficient, .predecessor 1 81943 .coefficient])

def exact81945RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7184⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨53916⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact81945RawTermsValid :
    exact81945RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event81945 : Event := .resultExact (⟨.program ⟨257⟩, ⟨53919⟩⟩) exact81945RawTerms .large 81944 .exactZero (none)

def event81946 : Event := .predecessor (⟨.program ⟨257⟩, ⟨55569⟩⟩) 0 ⟨53919⟩ 81945

def event81947 : Event := .predecessor (⟨.program ⟨257⟩, ⟨55569⟩⟩) 1 ⟨55568⟩ 81930

def event81948 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨55569⟩⟩) (.sum [.predecessor 0 81946 .coefficient, .predecessor 1 81947 .coefficient])

def exact81949RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7184⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55565⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨24842⟩⟩, ⟨.program ⟨257⟩, ⟨53687⟩⟩], [⟨.program ⟨257⟩, ⟨55025⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨53916⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact81949RawTermsValid :
    exact81949RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event81949 : Event := .resultExact (⟨.program ⟨257⟩, ⟨55569⟩⟩) exact81949RawTerms .large 81948 .exactZero (none)

def event81950 : Event := .preFoldPolynomial 81949 [⟨⟨[], [⟨.program ⟨257⟩, ⟨7184⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55565⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨24842⟩⟩, ⟨.program ⟨257⟩, ⟨53687⟩⟩], [⟨.program ⟨257⟩, ⟨55025⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨53916⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩] .exactZero none

def exact81951RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7184⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55565⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨24842⟩⟩, ⟨.program ⟨257⟩, ⟨53687⟩⟩], [⟨.program ⟨257⟩, ⟨55025⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨53916⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

def event81951 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨55569⟩⟩) 81950 exact81951RawTerms .large 81948 .exactZero (none)

def event81952 : Event := .specializationComputed (⟨.program ⟨257⟩, ⟨53689⟩⟩) ⟨⟨63⟩, ⟨41⟩, ⟨135⟩⟩ ⟨81786, 81952⟩

def event81953 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨54492⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54489⟩⟩]⟩) (1) 0 2 (.universal 81952 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54489⟩⟩]⟩) (none) 81951)

def event81954 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨54492⟩⟩, .relation 81953 0, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7184⟩⟩]⟩, (1)⟩)

def event81955 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨54492⟩⟩, .relation 81953 1, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55565⟩⟩]⟩, (-1)⟩)

def event81956 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨54492⟩⟩, .relation 81953 2, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨24842⟩⟩, ⟨.program ⟨257⟩, ⟨53687⟩⟩], [⟨.program ⟨257⟩, ⟨55025⟩⟩]⟩, (1)⟩)

def event81957 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨54492⟩⟩, .relation 81953 3, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨53916⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact81958RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7184⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55565⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨24842⟩⟩, ⟨.program ⟨257⟩, ⟨53687⟩⟩], [⟨.program ⟨257⟩, ⟨55025⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨53916⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact81958RawTermsValid :
    exact81958RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event81958 : Event := .resultExact (⟨.program ⟨257⟩, ⟨54492⟩⟩) exact81958RawTerms .large 81782 (.finite 202072841853861888) (some (81784))

def event81959 : Event := .predecessor (⟨.program ⟨257⟩, ⟨55567⟩⟩) 0 ⟨54492⟩ 81958

def event81960 : Event := .predecessor (⟨.program ⟨257⟩, ⟨55567⟩⟩) 1 ⟨55566⟩ 81772

def event81961 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨55567⟩⟩) (.sum [.predecessor 0 81959 .coefficient, .predecessor 1 81960 .coefficient])

def event81962 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨55567⟩⟩, .operator (⟨81958, 2⟩, ⟨81772, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨24842⟩⟩, ⟨.program ⟨257⟩, ⟨53687⟩⟩], [⟨.program ⟨257⟩, ⟨55025⟩⟩]⟩, (-1)⟩)

def event81963 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨55567⟩⟩, .operator (⟨81958, 1⟩, ⟨81772, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55565⟩⟩]⟩, (1)⟩)

def event81964 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨55567⟩⟩) (.sum [.result 81958 .summary, .result 81772 .summary])

def exact81965RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7184⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨53916⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact81965RawTermsValid :
    exact81965RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event81965 : Event := .resultExact (⟨.program ⟨257⟩, ⟨55567⟩⟩) exact81965RawTerms .large 81961 (.finite 2997907760060573155328) (some (81964))

def event81966 : Event := .predecessor (⟨.program ⟨257⟩, ⟨56120⟩⟩) 0 ⟨55567⟩ 81965

def event81967 : Event := .predecessor (⟨.program ⟨257⟩, ⟨56120⟩⟩) 1 ⟨56118⟩ 81688

def event81968 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨56120⟩⟩) (.product (.predecessor 0 81966 .coefficient) (.predecessor 1 81967 .coefficient) (⟨false, false, none, none, none⟩))

def event81969 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨56120⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨56118⟩⟩]⟩) [⟨.result 81688 .coefficient, false, none⟩])

def event81970 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨56120⟩⟩) (.product (.result 81965 .summary) (.transfer 81969) (⟨false, false, none, none, none⟩))

def event81971 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨56120⟩⟩, .operator (⟨81965, 0⟩, ⟨81688, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7184⟩⟩, ⟨.program ⟨257⟩, ⟨56118⟩⟩]⟩, (1)⟩)

def event81972 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨56120⟩⟩, .operator (⟨81965, 1⟩, ⟨81688, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨53916⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨56118⟩⟩]⟩, (-1)⟩)

def event81973 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨56120⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨53916⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨56118⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨56118⟩⟩) ⟨55195⟩ 81685)

def event81974 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨56120⟩⟩, .relation 81973 0, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨53916⟩⟩], [⟨.program ⟨257⟩, ⟨55195⟩⟩]⟩, (-1)⟩)

def exact81975RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7184⟩⟩, ⟨.program ⟨257⟩, ⟨56118⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨53916⟩⟩], [⟨.program ⟨257⟩, ⟨55195⟩⟩]⟩, (-1)⟩]

theorem exact81975RawTermsValid :
    exact81975RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event81975 : Event := .resultExact (⟨.program ⟨257⟩, ⟨56120⟩⟩) exact81975RawTerms .large 81968 (.finite 32189789464711941702873220382720) (some (81970))

def event81976 : Event := .predecessor (⟨.program ⟨257⟩, ⟨54856⟩⟩) 0 ⟨53917⟩ 3379

def event81977 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨54856⟩⟩) (.authority (.relationPreimageSource ⟨68⟩))

def exact81978RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨54856⟩⟩]⟩, (1)⟩]

theorem exact81978RawTermsValid :
    exact81978RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event81978 : Event := .resultExact (⟨.program ⟨257⟩, ⟨54856⟩⟩) exact81978RawTerms (.finite 5647228698) 81977 .exactZero (none)

def event81979 : Event := .predecessor (⟨.program ⟨257⟩, ⟨54858⟩⟩) 0 ⟨54856⟩ 81978

def event81980 : Event := .predecessor (⟨.program ⟨257⟩, ⟨54858⟩⟩) 1 ⟨2370⟩ 4

def event81981 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨54858⟩⟩) (.scale (.predecessor 0 81979 .coefficient) (.value (.predecessor 1 81980 .coefficient)))

def exact81982RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨54856⟩⟩]⟩, (1)⟩]

theorem exact81982RawTermsValid :
    exact81982RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event81982 : Event := .resultExact (⟨.program ⟨257⟩, ⟨54858⟩⟩) exact81982RawTerms (.finite 5647228698) 81981 .exactZero (none)

def event81983 : Event := .predecessor (⟨.program ⟨257⟩, ⟨54859⟩⟩) 0 ⟨10368⟩ 75995

def event81984 : Event := .predecessor (⟨.program ⟨257⟩, ⟨54859⟩⟩) 1 ⟨54858⟩ 81982

def event81985 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨54859⟩⟩) (.product (.predecessor 0 81983 .coefficient) (.predecessor 1 81984 .coefficient) (⟨false, false, none, none, none⟩))

def event81986 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨54859⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨54856⟩⟩]⟩) [⟨.result 81978 .coefficient, false, none⟩])

def event81987 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨54859⟩⟩) (.product (.result 75995 .summary) (.transfer 81986) (⟨false, false, none, none, none⟩))

def event81988 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨54859⟩⟩, .operator (⟨75995, 0⟩, ⟨81982, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54856⟩⟩]⟩, (1)⟩)

def event81989 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨54857⟩⟩)

def event81990 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event81991 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event81992 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10267⟩⟩) (.authority (.operator))

def event81993 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10267⟩⟩) (.finite 15)

def event81994 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event81995 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event81996 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event81997 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event81998 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 81997

def event81999 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 81995

def event82000 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 81998 .coefficient) (.value (.predecessor 1 81999 .coefficient)))

def event82001 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event82002 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10269⟩⟩) 0 ⟨392⟩ 82001

def event82003 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10269⟩⟩) 1 ⟨10267⟩ 81993

def event82004 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10269⟩⟩) (.sum [.predecessor 0 82002 .coefficient, .predecessor 1 82003 .coefficient])

def event82005 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10269⟩⟩) (.finite 655355)

def event82006 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10325⟩⟩) 0 ⟨10269⟩ 82005

def event82007 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10325⟩⟩) 1 ⟨5426⟩ 81991

def event82008 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10325⟩⟩) (.identity (.predecessor 1 82007 .coefficient))

def event82009 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10325⟩⟩) (.finite 655360)

def event82010 : Event := .predecessor (⟨.program ⟨257⟩, ⟨24842⟩⟩) 0 ⟨10325⟩ 82009

def event82011 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨24842⟩⟩) (.authority (.programFamilyFact))

def exact82012RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨24842⟩⟩], []⟩, (1)⟩]

theorem exact82012RawTermsValid :
    exact82012RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82012 : Event := .resultExact (⟨.program ⟨257⟩, ⟨24842⟩⟩) exact82012RawTerms (.finite 12) 82011 .exactZero (none)

def event82013 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53687⟩⟩) 0 ⟨10325⟩ 82009

def event82014 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨53687⟩⟩) (.authority (.programFamilyFact))

def exact82015RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨53687⟩⟩], []⟩, (1)⟩]

theorem exact82015RawTermsValid :
    exact82015RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82015 : Event := .resultExact (⟨.program ⟨257⟩, ⟨53687⟩⟩) exact82015RawTerms (.finite 12) 82014 .exactZero (none)

def event82016 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53688⟩⟩) 0 ⟨53687⟩ 82015

def event82017 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53688⟩⟩) 1 ⟨24842⟩ 82012

def event82018 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨53688⟩⟩) (.product (.predecessor 0 82016 .coefficient) (.predecessor 1 82017 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event82019 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨53688⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨24842⟩⟩, ⟨.program ⟨257⟩, ⟨53687⟩⟩], []⟩) [⟨.result 82015 .coefficient, true, some 1⟩, ⟨.result 82012 .coefficient, true, some 1⟩])

def event82020 : Event := .survivorFold (1) 82019

def exact82021RawTerms : List Term := []

theorem exact82021RawTermsValid :
    exact82021RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82021 : Event := .resultExact (⟨.program ⟨257⟩, ⟨53688⟩⟩) exact82021RawTerms (.finite 144) 82018 (.finite 144) (some (82019))

def event82022 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53689⟩⟩) 0 ⟨53688⟩ 82021

def event82023 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨53689⟩⟩) (.identity (.predecessor 0 82022 .coefficient))

def event82024 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨53689⟩⟩) (.finite 144)

def event82025 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53916⟩⟩) 0 ⟨53689⟩ 82024

def event82026 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨53916⟩⟩) (.authority (.programFamilyFact))

def exact82027RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨53916⟩⟩], []⟩, (1)⟩]

theorem exact82027RawTermsValid :
    exact82027RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82027 : Event := .resultExact (⟨.program ⟨257⟩, ⟨53916⟩⟩) exact82027RawTerms (.finite 12) 82026 .exactZero (none)

def event82028 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53917⟩⟩) 0 ⟨53916⟩ 82027

def event82029 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨53917⟩⟩) (.identity (.predecessor 0 82028 .coefficient))

def event82030 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨53917⟩⟩) (.finite 12)

def event82031 : Event := .predecessor (⟨.program ⟨257⟩, ⟨54856⟩⟩) 0 ⟨53917⟩ 82030

def event82032 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨54856⟩⟩) (.authority (.relationPreimageSource ⟨68⟩))

def exact82033RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨54856⟩⟩]⟩, (1)⟩]

theorem exact82033RawTermsValid :
    exact82033RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82033 : Event := .resultExact (⟨.program ⟨257⟩, ⟨54856⟩⟩) exact82033RawTerms (.finite 5647228698) 82032 .exactZero (none)

def event82034 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35⟩⟩) (.authority (.operator))

def exact82035RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩]⟩, (1)⟩]

theorem exact82035RawTermsValid :
    exact82035RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82035 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35⟩⟩) exact82035RawTerms .large 82034 .exactZero (none)

def event82036 : Event := .predecessor (⟨.program ⟨257⟩, ⟨54857⟩⟩) 0 ⟨35⟩ 82035

def event82037 : Event := .predecessor (⟨.program ⟨257⟩, ⟨54857⟩⟩) 1 ⟨54856⟩ 82033

def event82038 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨54857⟩⟩) (.product (.predecessor 0 82036 .coefficient) (.predecessor 1 82037 .coefficient) (⟨false, false, none, none, none⟩))

def event82039 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨54857⟩⟩, .operator (⟨82035, 0⟩, ⟨82033, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54856⟩⟩]⟩, (1)⟩)

def exact82040RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54856⟩⟩]⟩, (1)⟩]

theorem exact82040RawTermsValid :
    exact82040RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82040 : Event := .resultExact (⟨.program ⟨257⟩, ⟨54857⟩⟩) exact82040RawTerms .large 82038 .exactZero (none)

def event82041 : Event := .preFoldPolynomial 82040 [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54856⟩⟩]⟩, (1)⟩] .exactZero none

def exact82042RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54856⟩⟩]⟩, (1)⟩]

def event82042 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨54857⟩⟩) 82041 exact82042RawTerms .large 82038 .exactZero (none)

def event82043 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨56123⟩⟩)

def event82044 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event82045 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event82046 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10267⟩⟩) (.authority (.operator))

def event82047 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10267⟩⟩) (.finite 15)

def event82048 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event82049 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event82050 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event82051 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event82052 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 82051

def event82053 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 82049

def event82054 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 82052 .coefficient) (.value (.predecessor 1 82053 .coefficient)))

def event82055 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event82056 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10269⟩⟩) 0 ⟨392⟩ 82055

def event82057 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10269⟩⟩) 1 ⟨10267⟩ 82047

def event82058 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10269⟩⟩) (.sum [.predecessor 0 82056 .coefficient, .predecessor 1 82057 .coefficient])

def event82059 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10269⟩⟩) (.finite 655355)

def event82060 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10325⟩⟩) 0 ⟨10269⟩ 82059

def event82061 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10325⟩⟩) 1 ⟨5426⟩ 82045

def event82062 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10325⟩⟩) (.identity (.predecessor 1 82061 .coefficient))

def event82063 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10325⟩⟩) (.finite 655360)

def event82064 : Event := .predecessor (⟨.program ⟨257⟩, ⟨24842⟩⟩) 0 ⟨10325⟩ 82063

def event82065 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨24842⟩⟩) (.authority (.programFamilyFact))

def exact82066RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨24842⟩⟩], []⟩, (1)⟩]

theorem exact82066RawTermsValid :
    exact82066RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82066 : Event := .resultExact (⟨.program ⟨257⟩, ⟨24842⟩⟩) exact82066RawTerms (.finite 12) 82065 .exactZero (none)

def event82067 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53687⟩⟩) 0 ⟨10325⟩ 82063

def event82068 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨53687⟩⟩) (.authority (.programFamilyFact))

def exact82069RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨53687⟩⟩], []⟩, (1)⟩]

theorem exact82069RawTermsValid :
    exact82069RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82069 : Event := .resultExact (⟨.program ⟨257⟩, ⟨53687⟩⟩) exact82069RawTerms (.finite 12) 82068 .exactZero (none)

def event82070 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53688⟩⟩) 0 ⟨53687⟩ 82069

def event82071 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53688⟩⟩) 1 ⟨24842⟩ 82066

def event82072 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨53688⟩⟩) (.product (.predecessor 0 82070 .coefficient) (.predecessor 1 82071 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event82073 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨53688⟩⟩, .operator (⟨82069, 0⟩, ⟨82066, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨24842⟩⟩, ⟨.program ⟨257⟩, ⟨53687⟩⟩], []⟩, (1)⟩)

def exact82074RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨24842⟩⟩, ⟨.program ⟨257⟩, ⟨53687⟩⟩], []⟩, (1)⟩]

theorem exact82074RawTermsValid :
    exact82074RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82074 : Event := .resultExact (⟨.program ⟨257⟩, ⟨53688⟩⟩) exact82074RawTerms (.finite 144) 82072 .exactZero (none)

def event82075 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53689⟩⟩) 0 ⟨53688⟩ 82074

def event82076 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨53689⟩⟩) (.identity (.predecessor 0 82075 .coefficient))

def event82077 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨53689⟩⟩) (.finite 144)

def event82078 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53916⟩⟩) 0 ⟨53689⟩ 82077

def event82079 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨53916⟩⟩) (.authority (.programFamilyFact))

def exact82080RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨53916⟩⟩], []⟩, (1)⟩]

theorem exact82080RawTermsValid :
    exact82080RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82080 : Event := .resultExact (⟨.program ⟨257⟩, ⟨53916⟩⟩) exact82080RawTerms (.finite 12) 82079 .exactZero (none)

def event82081 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53917⟩⟩) 0 ⟨53916⟩ 82080

def event82082 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨53917⟩⟩) (.identity (.predecessor 0 82081 .coefficient))

def event82083 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨53917⟩⟩) (.finite 12)

def event82084 : Event := .predecessor (⟨.program ⟨257⟩, ⟨55193⟩⟩) 0 ⟨53917⟩ 82083

def event82085 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨55193⟩⟩) (.authority (.programFamilyFact))

def event82086 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨55193⟩⟩) (.finite 3720)

def event82087 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨7177⟩⟩) .missing

def event82088 : Event := .predecessor (⟨.program ⟨257⟩, ⟨55195⟩⟩) 0 ⟨7177⟩ 82087

def event82089 : Event := .predecessor (⟨.program ⟨257⟩, ⟨55195⟩⟩) 1 ⟨55193⟩ 82086

def event82090 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨55195⟩⟩) (.authority (.operator))

def exact82091RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨55195⟩⟩]⟩, (1)⟩]

theorem exact82091RawTermsValid :
    exact82091RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82091 : Event := .resultExact (⟨.program ⟨257⟩, ⟨55195⟩⟩) exact82091RawTerms .large 82090 .exactZero (none)

def event82092 : Event := .predecessor (⟨.program ⟨257⟩, ⟨56118⟩⟩) 0 ⟨55195⟩ 82091

def event82093 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨56118⟩⟩) (.authority (.operator))

def exact82094RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨56118⟩⟩]⟩, (1)⟩]

theorem exact82094RawTermsValid :
    exact82094RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82094 : Event := .resultExact (⟨.program ⟨257⟩, ⟨56118⟩⟩) exact82094RawTerms (.finite 8192) 82093 .exactZero (none)

def event82095 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨136⟩⟩) (.authority (.operator))

def event82096 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨136⟩⟩) .exactZero

def event82097 : Event := .predecessor (⟨.program ⟨257⟩, ⟨55370⟩⟩) 0 ⟨53917⟩ 82083

def event82098 : Event := .predecessor (⟨.program ⟨257⟩, ⟨55370⟩⟩) 1 ⟨136⟩ 82096

def event82099 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨55370⟩⟩) (.sum [.predecessor 0 82097 .coefficient, .predecessor 1 82098 .coefficient])

def event82100 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨55370⟩⟩) (.finite 12)

def event82101 : Event := .predecessor (⟨.program ⟨257⟩, ⟨55371⟩⟩) 0 ⟨55370⟩ 82100

def event82102 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨55371⟩⟩) (.identity (.predecessor 0 82101 .coefficient))

def exact82103RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨53916⟩⟩], []⟩, (1)⟩]

theorem exact82103RawTermsValid :
    exact82103RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82103 : Event := .resultExact (⟨.program ⟨257⟩, ⟨55371⟩⟩) exact82103RawTerms (.finite 12) 82102 .exactZero (none)

def event82104 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6908⟩⟩) (.authority (.factStore))

def exact82105RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact82105RawTermsValid :
    exact82105RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82105 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6908⟩⟩) exact82105RawTerms .large 82104 .exactZero (none)

def event82106 : Event := .predecessor (⟨.program ⟨257⟩, ⟨55372⟩⟩) 0 ⟨6908⟩ 82105

def event82107 : Event := .predecessor (⟨.program ⟨257⟩, ⟨55372⟩⟩) 1 ⟨55371⟩ 82103

def event82108 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨55372⟩⟩) (.product (.predecessor 0 82106 .coefficient) (.predecessor 1 82107 .coefficient) (⟨false, false, none, none, none⟩))

def event82109 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨55372⟩⟩, .operator (⟨82105, 0⟩, ⟨82103, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨53916⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact82110RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨53916⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact82110RawTermsValid :
    exact82110RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82110 : Event := .resultExact (⟨.program ⟨257⟩, ⟨55372⟩⟩) exact82110RawTerms .large 82108 .exactZero (none)

def event82111 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7184⟩⟩) 0 ⟨7177⟩ 82087

def event82112 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7184⟩⟩) (.authority (.operator))

def exact82113RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7184⟩⟩]⟩, (1)⟩]

theorem exact82113RawTermsValid :
    exact82113RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82113 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7184⟩⟩) exact82113RawTerms .large 82112 .exactZero (none)

def event82114 : Event := .predecessor (⟨.program ⟨257⟩, ⟨55373⟩⟩) 0 ⟨7184⟩ 82113

def event82115 : Event := .predecessor (⟨.program ⟨257⟩, ⟨55373⟩⟩) 1 ⟨55372⟩ 82110

def event82116 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨55373⟩⟩) (.sum [.predecessor 0 82114 .coefficient, .predecessor 1 82115 .coefficient])

def exact82117RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7184⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨53916⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact82117RawTermsValid :
    exact82117RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82117 : Event := .resultExact (⟨.program ⟨257⟩, ⟨55373⟩⟩) exact82117RawTerms .large 82116 .exactZero (none)

def event82118 : Event := .predecessor (⟨.program ⟨257⟩, ⟨56119⟩⟩) 0 ⟨55373⟩ 82117

def event82119 : Event := .predecessor (⟨.program ⟨257⟩, ⟨56119⟩⟩) 1 ⟨56118⟩ 82094

def event82120 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨56119⟩⟩) (.product (.predecessor 0 82118 .coefficient) (.predecessor 1 82119 .coefficient) (⟨false, false, none, none, none⟩))

def event82121 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨56119⟩⟩, .operator (⟨82117, 0⟩, ⟨82094, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7184⟩⟩, ⟨.program ⟨257⟩, ⟨56118⟩⟩]⟩, (1)⟩)

def event82122 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨56119⟩⟩, .operator (⟨82117, 1⟩, ⟨82094, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨53916⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨56118⟩⟩]⟩, (-1)⟩)

def event82123 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨56119⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨53916⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨56118⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨56118⟩⟩) ⟨55195⟩ 82091)

def event82124 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨56119⟩⟩, .relation 82123 0, ⟨[⟨.program ⟨257⟩, ⟨53916⟩⟩], [⟨.program ⟨257⟩, ⟨55195⟩⟩]⟩, (-1)⟩)

def exact82125RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7184⟩⟩, ⟨.program ⟨257⟩, ⟨56118⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨53916⟩⟩], [⟨.program ⟨257⟩, ⟨55195⟩⟩]⟩, (-1)⟩]

theorem exact82125RawTermsValid :
    exact82125RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82125 : Event := .resultExact (⟨.program ⟨257⟩, ⟨56119⟩⟩) exact82125RawTerms .large 82120 .exactZero (none)

def event82126 : Event := .predecessor (⟨.program ⟨257⟩, ⟨54255⟩⟩) 0 ⟨53917⟩ 82083

def event82127 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨54255⟩⟩) (.authority (.programFamilyFact))

def exact82128RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨54255⟩⟩], []⟩, (1)⟩]

theorem exact82128RawTermsValid :
    exact82128RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82128 : Event := .resultExact (⟨.program ⟨257⟩, ⟨54255⟩⟩) exact82128RawTerms (.finite 59) 82127 .exactZero (none)

def event82129 : Event := .predecessor (⟨.program ⟨257⟩, ⟨54257⟩⟩) 0 ⟨6908⟩ 82105

def event82130 : Event := .predecessor (⟨.program ⟨257⟩, ⟨54257⟩⟩) 1 ⟨54255⟩ 82128

def event82131 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨54257⟩⟩) (.product (.predecessor 0 82129 .coefficient) (.predecessor 1 82130 .coefficient) (⟨false, true, none, none, some 1⟩))

def event82132 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨54257⟩⟩, .operator (⟨82105, 0⟩, ⟨82128, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨54255⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact82133RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨54255⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact82133RawTermsValid :
    exact82133RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82133 : Event := .resultExact (⟨.program ⟨257⟩, ⟨54257⟩⟩) exact82133RawTerms .large 82131 .exactZero (none)

def event82134 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7208⟩⟩) 0 ⟨7177⟩ 82087

def event82135 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7208⟩⟩) (.authority (.operator))

def exact82136RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩]

theorem exact82136RawTermsValid :
    exact82136RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82136 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7208⟩⟩) exact82136RawTerms .large 82135 .exactZero (none)

def event82137 : Event := .predecessor (⟨.program ⟨257⟩, ⟨54258⟩⟩) 0 ⟨7208⟩ 82136

def event82138 : Event := .predecessor (⟨.program ⟨257⟩, ⟨54258⟩⟩) 1 ⟨54257⟩ 82133

def event82139 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨54258⟩⟩) (.sum [.predecessor 0 82137 .coefficient, .predecessor 1 82138 .coefficient])

def exact82140RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨54255⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact82140RawTermsValid :
    exact82140RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82140 : Event := .resultExact (⟨.program ⟨257⟩, ⟨54258⟩⟩) exact82140RawTerms .large 82139 .exactZero (none)

def event82141 : Event := .predecessor (⟨.program ⟨257⟩, ⟨56123⟩⟩) 0 ⟨54258⟩ 82140

def event82142 : Event := .predecessor (⟨.program ⟨257⟩, ⟨56123⟩⟩) 1 ⟨56119⟩ 82125

def event82143 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨56123⟩⟩) (.sum [.predecessor 0 82141 .coefficient, .predecessor 1 82142 .coefficient])

def exact82144RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7184⟩⟩, ⟨.program ⟨257⟩, ⟨56118⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨53916⟩⟩], [⟨.program ⟨257⟩, ⟨55195⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨54255⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact82144RawTermsValid :
    exact82144RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82144 : Event := .resultExact (⟨.program ⟨257⟩, ⟨56123⟩⟩) exact82144RawTerms .large 82143 .exactZero (none)

def event82145 : Event := .preFoldPolynomial 82144 [⟨⟨[], [⟨.program ⟨257⟩, ⟨7184⟩⟩, ⟨.program ⟨257⟩, ⟨56118⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨53916⟩⟩], [⟨.program ⟨257⟩, ⟨55195⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨54255⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩] .exactZero none

def exact82146RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7184⟩⟩, ⟨.program ⟨257⟩, ⟨56118⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨53916⟩⟩], [⟨.program ⟨257⟩, ⟨55195⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨54255⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

def event82146 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨56123⟩⟩) 82145 exact82146RawTerms .large 82143 .exactZero (none)

def event82147 : Event := .specializationComputed (⟨.program ⟨257⟩, ⟨53917⟩⟩) ⟨⟨87⟩, ⟨68⟩, ⟨135⟩⟩ ⟨81989, 82147⟩

def event82148 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨54859⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54856⟩⟩]⟩) (1) 0 2 (.universal 82147 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54856⟩⟩]⟩) (none) 82146)

def event82149 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨54859⟩⟩, .relation 82148 1, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩)

def event82150 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨54859⟩⟩, .relation 82148 0, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7184⟩⟩, ⟨.program ⟨257⟩, ⟨56118⟩⟩]⟩, (-1)⟩)

def event82151 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨54859⟩⟩, .relation 82148 2, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨53916⟩⟩], [⟨.program ⟨257⟩, ⟨55195⟩⟩]⟩, (1)⟩)

def event82152 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨54859⟩⟩, .relation 82148 3, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨54255⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact82153RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7184⟩⟩, ⟨.program ⟨257⟩, ⟨56118⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨53916⟩⟩], [⟨.program ⟨257⟩, ⟨55195⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨54255⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact82153RawTermsValid :
    exact82153RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82153 : Event := .resultExact (⟨.program ⟨257⟩, ⟨54859⟩⟩) exact82153RawTerms .large 81985 (.finite 202072841853861888) (some (81987))

def event82154 : Event := .predecessor (⟨.program ⟨257⟩, ⟨56121⟩⟩) 0 ⟨54859⟩ 82153

def event82155 : Event := .predecessor (⟨.program ⟨257⟩, ⟨56121⟩⟩) 1 ⟨56120⟩ 81975

def event82156 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨56121⟩⟩) (.sum [.predecessor 0 82154 .coefficient, .predecessor 1 82155 .coefficient])

def event82157 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨56121⟩⟩, .operator (⟨82153, 0⟩, ⟨81975, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7184⟩⟩, ⟨.program ⟨257⟩, ⟨56118⟩⟩]⟩, (1)⟩)

def event82158 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨56121⟩⟩, .operator (⟨82153, 2⟩, ⟨81975, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨53916⟩⟩], [⟨.program ⟨257⟩, ⟨55195⟩⟩]⟩, (-1)⟩)

def event82159 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨56121⟩⟩) (.sum [.result 82153 .summary, .result 81975 .summary])

def exact82160RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨54255⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact82160RawTermsValid :
    exact82160RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82160 : Event := .resultExact (⟨.program ⟨257⟩, ⟨56121⟩⟩) exact82160RawTerms .large 82156 (.finite 32189789464712143775715074244608) (some (82159))

def event82161 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52213⟩⟩) 0 ⟨50937⟩ 3402

def event82162 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨52213⟩⟩) (.authority (.programFamilyFact))

def event82163 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨52213⟩⟩) (.finite 3720)

def event82164 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52215⟩⟩) 0 ⟨7177⟩ 15500

def event82165 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52215⟩⟩) 1 ⟨52213⟩ 82163

def event82166 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨52215⟩⟩) (.authority (.operator))

def exact82167RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨52215⟩⟩]⟩, (1)⟩]

theorem exact82167RawTermsValid :
    exact82167RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82167 : Event := .resultExact (⟨.program ⟨257⟩, ⟨52215⟩⟩) exact82167RawTerms .large 82166 .exactZero (none)

def event82168 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53138⟩⟩) 0 ⟨52215⟩ 82167

def event82169 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨53138⟩⟩) (.authority (.operator))

def exact82170RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨53138⟩⟩]⟩, (1)⟩]

theorem exact82170RawTermsValid :
    exact82170RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82170 : Event := .resultExact (⟨.program ⟨257⟩, ⟨53138⟩⟩) exact82170RawTerms (.finite 8192) 82169 .exactZero (none)

def event82171 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52044⟩⟩) 0 ⟨50709⟩ 3396

def event82172 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨52044⟩⟩) (.authority (.programFamilyFact))

def event82173 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨52044⟩⟩) (.finite 3720)

def event82174 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52045⟩⟩) 0 ⟨7177⟩ 15500

def event82175 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52045⟩⟩) 1 ⟨52044⟩ 82173

def eventLeaf5120 : Array AnnotatedEvent := #[
  { event := event81920
    frameStart := 81834 },
  { event := event81921
    frameStart := 81834 },
  { event := event81922
    frameStart := 81834 },
  { event := event81923
    frameStart := 81834 },
  { event := event81924
    frameStart := 81834 },
  { event := event81925
    frameStart := 81834 },
  { event := event81926
    frameStart := 81834 },
  { event := event81927
    frameStart := 81834 },
  { event := event81928
    frameStart := 81834 },
  { event := event81929
    frameStart := 81834 },
  { event := event81930
    frameStart := 81834 },
  { event := event81931
    frameStart := 81834 },
  { event := event81932
    frameStart := 81834 },
  { event := event81933
    frameStart := 81834 },
  { event := event81934
    frameStart := 81834 },
  { event := event81935
    frameStart := 81834 }
]

def eventLeaf5121 : Array AnnotatedEvent := #[
  { event := event81936
    frameStart := 81834 },
  { event := event81937
    frameStart := 81834 },
  { event := event81938
    frameStart := 81834 },
  { event := event81939
    frameStart := 81834 },
  { event := event81940
    frameStart := 81834 },
  { event := event81941
    frameStart := 81834 },
  { event := event81942
    frameStart := 81834 },
  { event := event81943
    frameStart := 81834 },
  { event := event81944
    frameStart := 81834 },
  { event := event81945
    frameStart := 81834 },
  { event := event81946
    frameStart := 81834 },
  { event := event81947
    frameStart := 81834 },
  { event := event81948
    frameStart := 81834 },
  { event := event81949
    frameStart := 81834 },
  { event := event81950
    frameStart := 81834 },
  { event := event81951
    frameStart := 81834 }
]

def eventLeaf5122 : Array AnnotatedEvent := #[
  { event := event81952
    frameStart := 0 },
  { event := event81953
    frameStart := 0 },
  { event := event81954
    frameStart := 0 },
  { event := event81955
    frameStart := 0 },
  { event := event81956
    frameStart := 0 },
  { event := event81957
    frameStart := 0 },
  { event := event81958
    frameStart := 0 },
  { event := event81959
    frameStart := 0 },
  { event := event81960
    frameStart := 0 },
  { event := event81961
    frameStart := 0 },
  { event := event81962
    frameStart := 0 },
  { event := event81963
    frameStart := 0 },
  { event := event81964
    frameStart := 0 },
  { event := event81965
    frameStart := 0 },
  { event := event81966
    frameStart := 0 },
  { event := event81967
    frameStart := 0 }
]

def eventLeaf5123 : Array AnnotatedEvent := #[
  { event := event81968
    frameStart := 0 },
  { event := event81969
    frameStart := 0 },
  { event := event81970
    frameStart := 0 },
  { event := event81971
    frameStart := 0 },
  { event := event81972
    frameStart := 0 },
  { event := event81973
    frameStart := 0 },
  { event := event81974
    frameStart := 0 },
  { event := event81975
    frameStart := 0 },
  { event := event81976
    frameStart := 0 },
  { event := event81977
    frameStart := 0 },
  { event := event81978
    frameStart := 0 },
  { event := event81979
    frameStart := 0 },
  { event := event81980
    frameStart := 0 },
  { event := event81981
    frameStart := 0 },
  { event := event81982
    frameStart := 0 },
  { event := event81983
    frameStart := 0 }
]

def eventLeaf5124 : Array AnnotatedEvent := #[
  { event := event81984
    frameStart := 0 },
  { event := event81985
    frameStart := 0 },
  { event := event81986
    frameStart := 0 },
  { event := event81987
    frameStart := 0 },
  { event := event81988
    frameStart := 0 },
  { event := event81989
    frameStart := 81989 },
  { event := event81990
    frameStart := 81989 },
  { event := event81991
    frameStart := 81989 },
  { event := event81992
    frameStart := 81989 },
  { event := event81993
    frameStart := 81989 },
  { event := event81994
    frameStart := 81989 },
  { event := event81995
    frameStart := 81989 },
  { event := event81996
    frameStart := 81989 },
  { event := event81997
    frameStart := 81989 },
  { event := event81998
    frameStart := 81989 },
  { event := event81999
    frameStart := 81989 }
]

def eventLeaf5125 : Array AnnotatedEvent := #[
  { event := event82000
    frameStart := 81989 },
  { event := event82001
    frameStart := 81989 },
  { event := event82002
    frameStart := 81989 },
  { event := event82003
    frameStart := 81989 },
  { event := event82004
    frameStart := 81989 },
  { event := event82005
    frameStart := 81989 },
  { event := event82006
    frameStart := 81989 },
  { event := event82007
    frameStart := 81989 },
  { event := event82008
    frameStart := 81989 },
  { event := event82009
    frameStart := 81989 },
  { event := event82010
    frameStart := 81989 },
  { event := event82011
    frameStart := 81989 },
  { event := event82012
    frameStart := 81989 },
  { event := event82013
    frameStart := 81989 },
  { event := event82014
    frameStart := 81989 },
  { event := event82015
    frameStart := 81989 }
]

def eventLeaf5126 : Array AnnotatedEvent := #[
  { event := event82016
    frameStart := 81989 },
  { event := event82017
    frameStart := 81989 },
  { event := event82018
    frameStart := 81989 },
  { event := event82019
    frameStart := 81989 },
  { event := event82020
    frameStart := 81989 },
  { event := event82021
    frameStart := 81989 },
  { event := event82022
    frameStart := 81989 },
  { event := event82023
    frameStart := 81989 },
  { event := event82024
    frameStart := 81989 },
  { event := event82025
    frameStart := 81989 },
  { event := event82026
    frameStart := 81989 },
  { event := event82027
    frameStart := 81989 },
  { event := event82028
    frameStart := 81989 },
  { event := event82029
    frameStart := 81989 },
  { event := event82030
    frameStart := 81989 },
  { event := event82031
    frameStart := 81989 }
]

def eventLeaf5127 : Array AnnotatedEvent := #[
  { event := event82032
    frameStart := 81989 },
  { event := event82033
    frameStart := 81989 },
  { event := event82034
    frameStart := 81989 },
  { event := event82035
    frameStart := 81989 },
  { event := event82036
    frameStart := 81989 },
  { event := event82037
    frameStart := 81989 },
  { event := event82038
    frameStart := 81989 },
  { event := event82039
    frameStart := 81989 },
  { event := event82040
    frameStart := 81989 },
  { event := event82041
    frameStart := 81989 },
  { event := event82042
    frameStart := 81989 },
  { event := event82043
    frameStart := 82043 },
  { event := event82044
    frameStart := 82043 },
  { event := event82045
    frameStart := 82043 },
  { event := event82046
    frameStart := 82043 },
  { event := event82047
    frameStart := 82043 }
]

def eventLeaf5128 : Array AnnotatedEvent := #[
  { event := event82048
    frameStart := 82043 },
  { event := event82049
    frameStart := 82043 },
  { event := event82050
    frameStart := 82043 },
  { event := event82051
    frameStart := 82043 },
  { event := event82052
    frameStart := 82043 },
  { event := event82053
    frameStart := 82043 },
  { event := event82054
    frameStart := 82043 },
  { event := event82055
    frameStart := 82043 },
  { event := event82056
    frameStart := 82043 },
  { event := event82057
    frameStart := 82043 },
  { event := event82058
    frameStart := 82043 },
  { event := event82059
    frameStart := 82043 },
  { event := event82060
    frameStart := 82043 },
  { event := event82061
    frameStart := 82043 },
  { event := event82062
    frameStart := 82043 },
  { event := event82063
    frameStart := 82043 }
]

def eventLeaf5129 : Array AnnotatedEvent := #[
  { event := event82064
    frameStart := 82043 },
  { event := event82065
    frameStart := 82043 },
  { event := event82066
    frameStart := 82043 },
  { event := event82067
    frameStart := 82043 },
  { event := event82068
    frameStart := 82043 },
  { event := event82069
    frameStart := 82043 },
  { event := event82070
    frameStart := 82043 },
  { event := event82071
    frameStart := 82043 },
  { event := event82072
    frameStart := 82043 },
  { event := event82073
    frameStart := 82043 },
  { event := event82074
    frameStart := 82043 },
  { event := event82075
    frameStart := 82043 },
  { event := event82076
    frameStart := 82043 },
  { event := event82077
    frameStart := 82043 },
  { event := event82078
    frameStart := 82043 },
  { event := event82079
    frameStart := 82043 }
]

def eventLeaf5130 : Array AnnotatedEvent := #[
  { event := event82080
    frameStart := 82043 },
  { event := event82081
    frameStart := 82043 },
  { event := event82082
    frameStart := 82043 },
  { event := event82083
    frameStart := 82043 },
  { event := event82084
    frameStart := 82043 },
  { event := event82085
    frameStart := 82043 },
  { event := event82086
    frameStart := 82043 },
  { event := event82087
    frameStart := 82043 },
  { event := event82088
    frameStart := 82043 },
  { event := event82089
    frameStart := 82043 },
  { event := event82090
    frameStart := 82043 },
  { event := event82091
    frameStart := 82043 },
  { event := event82092
    frameStart := 82043 },
  { event := event82093
    frameStart := 82043 },
  { event := event82094
    frameStart := 82043 },
  { event := event82095
    frameStart := 82043 }
]

def eventLeaf5131 : Array AnnotatedEvent := #[
  { event := event82096
    frameStart := 82043 },
  { event := event82097
    frameStart := 82043 },
  { event := event82098
    frameStart := 82043 },
  { event := event82099
    frameStart := 82043 },
  { event := event82100
    frameStart := 82043 },
  { event := event82101
    frameStart := 82043 },
  { event := event82102
    frameStart := 82043 },
  { event := event82103
    frameStart := 82043 },
  { event := event82104
    frameStart := 82043 },
  { event := event82105
    frameStart := 82043 },
  { event := event82106
    frameStart := 82043 },
  { event := event82107
    frameStart := 82043 },
  { event := event82108
    frameStart := 82043 },
  { event := event82109
    frameStart := 82043 },
  { event := event82110
    frameStart := 82043 },
  { event := event82111
    frameStart := 82043 }
]

def eventLeaf5132 : Array AnnotatedEvent := #[
  { event := event82112
    frameStart := 82043 },
  { event := event82113
    frameStart := 82043 },
  { event := event82114
    frameStart := 82043 },
  { event := event82115
    frameStart := 82043 },
  { event := event82116
    frameStart := 82043 },
  { event := event82117
    frameStart := 82043 },
  { event := event82118
    frameStart := 82043 },
  { event := event82119
    frameStart := 82043 },
  { event := event82120
    frameStart := 82043 },
  { event := event82121
    frameStart := 82043 },
  { event := event82122
    frameStart := 82043 },
  { event := event82123
    frameStart := 82043 },
  { event := event82124
    frameStart := 82043 },
  { event := event82125
    frameStart := 82043 },
  { event := event82126
    frameStart := 82043 },
  { event := event82127
    frameStart := 82043 }
]

def eventLeaf5133 : Array AnnotatedEvent := #[
  { event := event82128
    frameStart := 82043 },
  { event := event82129
    frameStart := 82043 },
  { event := event82130
    frameStart := 82043 },
  { event := event82131
    frameStart := 82043 },
  { event := event82132
    frameStart := 82043 },
  { event := event82133
    frameStart := 82043 },
  { event := event82134
    frameStart := 82043 },
  { event := event82135
    frameStart := 82043 },
  { event := event82136
    frameStart := 82043 },
  { event := event82137
    frameStart := 82043 },
  { event := event82138
    frameStart := 82043 },
  { event := event82139
    frameStart := 82043 },
  { event := event82140
    frameStart := 82043 },
  { event := event82141
    frameStart := 82043 },
  { event := event82142
    frameStart := 82043 },
  { event := event82143
    frameStart := 82043 }
]

def eventLeaf5134 : Array AnnotatedEvent := #[
  { event := event82144
    frameStart := 82043 },
  { event := event82145
    frameStart := 82043 },
  { event := event82146
    frameStart := 82043 },
  { event := event82147
    frameStart := 0 },
  { event := event82148
    frameStart := 0 },
  { event := event82149
    frameStart := 0 },
  { event := event82150
    frameStart := 0 },
  { event := event82151
    frameStart := 0 },
  { event := event82152
    frameStart := 0 },
  { event := event82153
    frameStart := 0 },
  { event := event82154
    frameStart := 0 },
  { event := event82155
    frameStart := 0 },
  { event := event82156
    frameStart := 0 },
  { event := event82157
    frameStart := 0 },
  { event := event82158
    frameStart := 0 },
  { event := event82159
    frameStart := 0 }
]

def eventLeaf5135 : Array AnnotatedEvent := #[
  { event := event82160
    frameStart := 0 },
  { event := event82161
    frameStart := 0 },
  { event := event82162
    frameStart := 0 },
  { event := event82163
    frameStart := 0 },
  { event := event82164
    frameStart := 0 },
  { event := event82165
    frameStart := 0 },
  { event := event82166
    frameStart := 0 },
  { event := event82167
    frameStart := 0 },
  { event := event82168
    frameStart := 0 },
  { event := event82169
    frameStart := 0 },
  { event := event82170
    frameStart := 0 },
  { event := event82171
    frameStart := 0 },
  { event := event82172
    frameStart := 0 },
  { event := event82173
    frameStart := 0 },
  { event := event82174
    frameStart := 0 },
  { event := event82175
    frameStart := 0 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events320
