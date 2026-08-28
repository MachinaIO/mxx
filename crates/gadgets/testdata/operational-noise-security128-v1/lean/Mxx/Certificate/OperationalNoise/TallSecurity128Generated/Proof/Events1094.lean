import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events1094

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event280064 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17522⟩⟩) (.authority (.operator))

def exact280065RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨17522⟩⟩]⟩, (1)⟩]

theorem exact280065RawTermsValid :
    exact280065RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event280065 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17522⟩⟩) exact280065RawTerms (.finite 8192) 280064 .exactZero (none)

def event280066 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨136⟩⟩) (.authority (.operator))

def event280067 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨136⟩⟩) .exactZero

def event280068 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17174⟩⟩) 0 ⟨15723⟩ 280054

def event280069 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17174⟩⟩) 1 ⟨136⟩ 280067

def event280070 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17174⟩⟩) (.sum [.predecessor 0 280068 .coefficient, .predecessor 1 280069 .coefficient])

def event280071 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨17174⟩⟩) (.finite 2)

def event280072 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17175⟩⟩) 0 ⟨17174⟩ 280071

def event280073 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17175⟩⟩) (.identity (.predecessor 0 280072 .coefficient))

def exact280074RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15722⟩⟩], []⟩, (1)⟩]

theorem exact280074RawTermsValid :
    exact280074RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event280074 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17175⟩⟩) exact280074RawTerms (.finite 2) 280073 .exactZero (none)

def event280075 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6908⟩⟩) (.authority (.factStore))

def exact280076RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact280076RawTermsValid :
    exact280076RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event280076 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6908⟩⟩) exact280076RawTerms .large 280075 .exactZero (none)

def event280077 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17176⟩⟩) 0 ⟨6908⟩ 280076

def event280078 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17176⟩⟩) 1 ⟨17175⟩ 280074

def event280079 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17176⟩⟩) (.product (.predecessor 0 280077 .coefficient) (.predecessor 1 280078 .coefficient) (⟨false, false, none, none, none⟩))

def event280080 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17176⟩⟩, .operator (⟨280076, 0⟩, ⟨280074, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨15722⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact280081RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15722⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact280081RawTermsValid :
    exact280081RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event280081 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17176⟩⟩) exact280081RawTerms .large 280079 .exactZero (none)

def event280082 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7179⟩⟩) 0 ⟨7177⟩ 280058

def event280083 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7179⟩⟩) (.authority (.operator))

def exact280084RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩]⟩, (1)⟩]

theorem exact280084RawTermsValid :
    exact280084RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event280084 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7179⟩⟩) exact280084RawTerms .large 280083 .exactZero (none)

def event280085 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17177⟩⟩) 0 ⟨7179⟩ 280084

def event280086 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17177⟩⟩) 1 ⟨17176⟩ 280081

def event280087 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17177⟩⟩) (.sum [.predecessor 0 280085 .coefficient, .predecessor 1 280086 .coefficient])

def exact280088RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15722⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact280088RawTermsValid :
    exact280088RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event280088 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17177⟩⟩) exact280088RawTerms .large 280087 .exactZero (none)

def event280089 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17523⟩⟩) 0 ⟨17177⟩ 280088

def event280090 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17523⟩⟩) 1 ⟨17522⟩ 280065

def event280091 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17523⟩⟩) (.product (.predecessor 0 280089 .coefficient) (.predecessor 1 280090 .coefficient) (⟨false, false, none, none, none⟩))

def event280092 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17523⟩⟩, .operator (⟨280088, 0⟩, ⟨280065, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17522⟩⟩]⟩, (1)⟩)

def event280093 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17523⟩⟩, .operator (⟨280088, 1⟩, ⟨280065, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨15722⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17522⟩⟩]⟩, (-1)⟩)

def event280094 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨17523⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨15722⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17522⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨17522⟩⟩) ⟨16925⟩ 280062)

def event280095 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17523⟩⟩, .relation 280094 0, ⟨[⟨.program ⟨257⟩, ⟨15722⟩⟩], [⟨.program ⟨257⟩, ⟨16925⟩⟩]⟩, (-1)⟩)

def exact280096RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17522⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15722⟩⟩], [⟨.program ⟨257⟩, ⟨16925⟩⟩]⟩, (-1)⟩]

theorem exact280096RawTermsValid :
    exact280096RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event280096 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17523⟩⟩) exact280096RawTerms .large 280091 .exactZero (none)

def event280097 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15898⟩⟩) 0 ⟨15723⟩ 280054

def event280098 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15898⟩⟩) (.authority (.programFamilyFact))

def exact280099RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15898⟩⟩], []⟩, (1)⟩]

theorem exact280099RawTermsValid :
    exact280099RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event280099 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15898⟩⟩) exact280099RawTerms (.finite 2) 280098 .exactZero (none)

def event280100 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15901⟩⟩) 0 ⟨6908⟩ 280076

def event280101 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15901⟩⟩) 1 ⟨15898⟩ 280099

def event280102 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15901⟩⟩) (.product (.predecessor 0 280100 .coefficient) (.predecessor 1 280101 .coefficient) (⟨false, true, none, none, some 1⟩))

def event280103 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨15901⟩⟩, .operator (⟨280076, 0⟩, ⟨280099, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨15898⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact280104RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15898⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact280104RawTermsValid :
    exact280104RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event280104 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15901⟩⟩) exact280104RawTerms .large 280102 .exactZero (none)

def event280105 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7197⟩⟩) 0 ⟨7177⟩ 280058

def event280106 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7197⟩⟩) (.authority (.operator))

def exact280107RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7197⟩⟩]⟩, (1)⟩]

theorem exact280107RawTermsValid :
    exact280107RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event280107 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7197⟩⟩) exact280107RawTerms .large 280106 .exactZero (none)

def event280108 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15902⟩⟩) 0 ⟨7197⟩ 280107

def event280109 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15902⟩⟩) 1 ⟨15901⟩ 280104

def event280110 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15902⟩⟩) (.sum [.predecessor 0 280108 .coefficient, .predecessor 1 280109 .coefficient])

def exact280111RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7197⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15898⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact280111RawTermsValid :
    exact280111RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event280111 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15902⟩⟩) exact280111RawTerms .large 280110 .exactZero (none)

def event280112 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17528⟩⟩) 0 ⟨15902⟩ 280111

def event280113 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17528⟩⟩) 1 ⟨17523⟩ 280096

def event280114 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17528⟩⟩) (.sum [.predecessor 0 280112 .coefficient, .predecessor 1 280113 .coefficient])

def exact280115RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17522⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7197⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15722⟩⟩], [⟨.program ⟨257⟩, ⟨16925⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15898⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact280115RawTermsValid :
    exact280115RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event280115 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17528⟩⟩) exact280115RawTerms .large 280114 .exactZero (none)

def event280116 : Event := .preFoldPolynomial 280115 [⟨⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17522⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7197⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15722⟩⟩], [⟨.program ⟨257⟩, ⟨16925⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15898⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩] .exactZero none

def exact280117RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17522⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7197⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15722⟩⟩], [⟨.program ⟨257⟩, ⟨16925⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15898⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

def event280117 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨17528⟩⟩) 280116 exact280117RawTerms .large 280114 .exactZero (none)

def event280118 : Event := .specializationComputed (⟨.program ⟨257⟩, ⟨15723⟩⟩) ⟨⟨76⟩, ⟨56⟩, ⟨135⟩⟩ ⟨279960, 280118⟩

def event280119 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨16429⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16426⟩⟩]⟩) (1) 0 2 (.universal 280118 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16426⟩⟩]⟩) (none) 280117)

def event280120 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨16429⟩⟩, .relation 280119 1, ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩]⟩, (1)⟩)

def event280121 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨16429⟩⟩, .relation 280119 0, ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17522⟩⟩]⟩, (-1)⟩)

def event280122 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨16429⟩⟩, .relation 280119 2, ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨15722⟩⟩], [⟨.program ⟨257⟩, ⟨16925⟩⟩]⟩, (1)⟩)

def event280123 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨16429⟩⟩, .relation 280119 3, ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨15898⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact280124RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17522⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨15722⟩⟩], [⟨.program ⟨257⟩, ⟨16925⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨15898⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact280124RawTermsValid :
    exact280124RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event280124 : Event := .resultExact (⟨.program ⟨257⟩, ⟨16429⟩⟩) exact280124RawTerms .large 279956 (.finite 202072841853861888) (some (279958))

def event280125 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17525⟩⟩) 0 ⟨16429⟩ 280124

def event280126 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17525⟩⟩) 1 ⟨17524⟩ 279946

def event280127 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17525⟩⟩) (.sum [.predecessor 0 280125 .coefficient, .predecessor 1 280126 .coefficient])

def event280128 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17525⟩⟩, .operator (⟨280124, 0⟩, ⟨279946, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17522⟩⟩]⟩, (1)⟩)

def event280129 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17525⟩⟩, .operator (⟨280124, 2⟩, ⟨279946, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨15722⟩⟩], [⟨.program ⟨257⟩, ⟨16925⟩⟩]⟩, (-1)⟩)

def event280130 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17525⟩⟩) (.sum [.result 280124 .summary, .result 279946 .summary])

def exact280131RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨15898⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact280131RawTermsValid :
    exact280131RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event280131 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17525⟩⟩) exact280131RawTerms .large 280127 (.finite 32188807212483706889510625476608) (some (280130))

def event280132 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17526⟩⟩) 0 ⟨17525⟩ 280131

def event280133 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17526⟩⟩) 1 ⟨7172⟩ 15882

def event280134 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17526⟩⟩) (.product (.predecessor 0 280132 .coefficient) (.predecessor 1 280133 .coefficient) (⟨false, false, none, none, none⟩))

def event280135 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17526⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩) [⟨.result 15878 .coefficient, false, none⟩])

def event280136 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17526⟩⟩) (.product (.result 280131 .summary) (.transfer 280135) (⟨false, false, none, none, none⟩))

def event280137 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17526⟩⟩, .operator (⟨280131, 0⟩, ⟨15882, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩)

def event280138 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17526⟩⟩, .operator (⟨280131, 1⟩, ⟨15882, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨15898⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (-1)⟩)

def event280139 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨17526⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨15898⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨7171⟩⟩) ⟨7051⟩ 15875)

def event280140 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17526⟩⟩, .relation 280139 0, ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15898⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact280141RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15898⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact280141RawTermsValid :
    exact280141RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event280141 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17526⟩⟩) exact280141RawTerms .large 280134 (.finite 345624685687166110058245054666339432529920) (some (280136))

def event280142 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7060⟩⟩) 0 ⟨6727⟩ 723

def event280143 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7060⟩⟩) 1 ⟨6915⟩ 266028

def event280144 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7060⟩⟩) (.tensor (.predecessor 0 280142 .coefficient) (.predecessor 1 280143 .coefficient) true false)

def event280145 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨7060⟩⟩, .operator (⟨723, 0⟩, ⟨266028, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6727⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact280146RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6727⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact280146RawTermsValid :
    exact280146RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event280146 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7060⟩⟩) exact280146RawTerms .large 280144 .exactZero (none)

def event280147 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7648⟩⟩) 0 ⟨5447⟩ 265898

def event280148 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7648⟩⟩) 1 ⟨7292⟩ 15896

def event280149 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7648⟩⟩) (.product (.predecessor 0 280147 .coefficient) (.predecessor 1 280148 .coefficient) (⟨false, false, none, none, none⟩))

def event280150 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨7648⟩⟩, .operator (⟨265898, 0⟩, ⟨15896, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7292⟩⟩]⟩, (1)⟩)

def exact280151RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7292⟩⟩]⟩, (1)⟩]

theorem exact280151RawTermsValid :
    exact280151RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event280151 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7648⟩⟩) exact280151RawTerms .large 280149 .exactZero (none)

def event280152 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9289⟩⟩) 0 ⟨7648⟩ 280151

def event280153 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9289⟩⟩) 1 ⟨7060⟩ 280146

def event280154 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9289⟩⟩) (.sum [.predecessor 0 280152 .coefficient, .predecessor 1 280153 .coefficient])

def exact280155RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7292⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6727⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact280155RawTermsValid :
    exact280155RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event280155 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9289⟩⟩) exact280155RawTerms .large 280154 .exactZero (none)

def event280156 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9290⟩⟩) 0 ⟨9289⟩ 280155

def event280157 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9290⟩⟩) 1 ⟨118⟩ 31516

def event280158 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9290⟩⟩) (.sum [.predecessor 0 280156 .coefficient, .predecessor 1 280157 .coefficient])

def event280159 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9290⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨118⟩⟩]⟩) [⟨.result 31516 .coefficient, false, none⟩])

def event280160 : Event := .survivorFold (1) 280159

def exact280161RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7292⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6727⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact280161RawTermsValid :
    exact280161RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event280161 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9290⟩⟩) exact280161RawTerms .large 280158 (.finite 26) (some (280159))

def event280162 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9452⟩⟩) 0 ⟨9290⟩ 280161

def event280163 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9452⟩⟩) 1 ⟨9290⟩ 280161

def event280164 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9452⟩⟩) (.sum [.predecessor 0 280162 .coefficient, .predecessor 1 280163 .coefficient])

def event280165 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨9452⟩⟩, .operator (⟨280161, 1⟩, ⟨280161, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6727⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def event280166 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨9452⟩⟩, .operator (⟨280161, 0⟩, ⟨280161, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7292⟩⟩]⟩, (-1)⟩)

def event280167 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9452⟩⟩) (.sum [.result 280161 .summary, .result 280161 .summary])

def exact280168RawTerms : List Term := []

theorem exact280168RawTermsValid :
    exact280168RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event280168 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9452⟩⟩) exact280168RawTerms .large 280164 (.finite 52) (some (280167))

def event280169 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17527⟩⟩) 0 ⟨9452⟩ 280168

def event280170 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17527⟩⟩) 1 ⟨17526⟩ 280141

def event280171 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17527⟩⟩) (.sum [.predecessor 0 280169 .coefficient, .predecessor 1 280170 .coefficient])

def event280172 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17527⟩⟩) (.sum [.result 280168 .summary, .result 280141 .summary])

def exact280173RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15898⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact280173RawTermsValid :
    exact280173RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event280173 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17527⟩⟩) exact280173RawTerms .large 280171 (.finite 345624685687166110058245054666339432529972) (some (280172))

def event280174 : Event := .predecessor (⟨.program ⟨257⟩, ⟨20393⟩⟩) 0 ⟨17527⟩ 280173

def event280175 : Event := .predecessor (⟨.program ⟨257⟩, ⟨20393⟩⟩) 1 ⟨20392⟩ 279929

def event280176 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨20393⟩⟩) (.sum [.predecessor 0 280174 .coefficient, .predecessor 1 280175 .coefficient])

def event280177 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨20393⟩⟩) (.sum [.result 280173 .summary, .result 279929 .summary])

def exact280178RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7199⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18704⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15898⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact280178RawTermsValid :
    exact280178RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event280178 : Event := .resultExact (⟨.program ⟨257⟩, ⟨20393⟩⟩) exact280178RawTerms .large 280176 (.finite 691250426059631610003352154589745737891892) (some (280177))

def event280179 : Event := .predecessor (⟨.program ⟨257⟩, ⟨23613⟩⟩) 0 ⟨20393⟩ 280178

def event280180 : Event := .predecessor (⟨.program ⟨257⟩, ⟨23613⟩⟩) 1 ⟨23612⟩ 279717

def event280181 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨23613⟩⟩) (.sum [.predecessor 0 280179 .coefficient, .predecessor 1 280180 .coefficient])

def event280182 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨23613⟩⟩) (.sum [.result 280178 .summary, .result 279717 .summary])

def exact280183RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7199⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨21924⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18704⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15898⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact280183RawTermsValid :
    exact280183RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event280183 : Event := .resultExact (⟨.program ⟨257⟩, ⟨23613⟩⟩) exact280183RawTerms .large 280181 (.finite 1036877221117396499835321299770218916085812) (some (280182))

def event280184 : Event := .predecessor (⟨.program ⟨257⟩, ⟨33633⟩⟩) 0 ⟨23613⟩ 280183

def event280185 : Event := .predecessor (⟨.program ⟨257⟩, ⟨33633⟩⟩) 1 ⟨33632⟩ 279505

def event280186 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨33633⟩⟩) (.sum [.predecessor 0 280184 .coefficient, .predecessor 1 280185 .coefficient])

def event280187 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨33633⟩⟩) (.sum [.result 280183 .summary, .result 279505 .summary])

def exact280188RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7199⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7203⟩⟩, ⟨.program ⟨257⟩, ⟨7145⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨31944⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨21924⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18704⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15898⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact280188RawTermsValid :
    exact280188RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event280188 : Event := .resultExact (⟨.program ⟨257⟩, ⟨33633⟩⟩) exact280188RawTerms .large 280186 (.finite 1382506125545760169441014535464825839943732) (some (280187))

def event280189 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52693⟩⟩) 0 ⟨33633⟩ 280188

def event280190 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52693⟩⟩) 1 ⟨52692⟩ 279293

def event280191 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨52693⟩⟩) (.sum [.predecessor 0 280189 .coefficient, .predecessor 1 280190 .coefficient])

def event280192 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨52693⟩⟩) (.sum [.result 280188 .summary, .result 279293 .summary])

def exact280193RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7199⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7203⟩⟩, ⟨.program ⟨257⟩, ⟨7145⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7205⟩⟩, ⟨.program ⟨257⟩, ⟨7131⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51008⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨31944⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨21924⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18704⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15898⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact280193RawTermsValid :
    exact280193RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event280193 : Event := .resultExact (⟨.program ⟨257⟩, ⟨52693⟩⟩) exact280193RawTerms .large 280191 (.finite 1728139248715321398594155952187700255129652) (some (280192))

def event280194 : Event := .predecessor (⟨.program ⟨257⟩, ⟨55673⟩⟩) 0 ⟨52693⟩ 280193

def event280195 : Event := .predecessor (⟨.program ⟨257⟩, ⟨55673⟩⟩) 1 ⟨55672⟩ 279081

def event280196 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨55673⟩⟩) (.sum [.predecessor 0 280194 .coefficient, .predecessor 1 280195 .coefficient])

def event280197 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨55673⟩⟩) (.sum [.result 280193 .summary, .result 279081 .summary])

def exact280198RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7199⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7203⟩⟩, ⟨.program ⟨257⟩, ⟨7145⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7205⟩⟩, ⟨.program ⟨257⟩, ⟨7131⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7207⟩⟩, ⟨.program ⟨257⟩, ⟨7125⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨53988⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51008⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨31944⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨21924⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18704⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15898⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact280198RawTermsValid :
    exact280198RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event280198 : Event := .resultExact (⟨.program ⟨257⟩, ⟨55673⟩⟩) exact280198RawTerms .large 280196 (.finite 2073774481255481407521021459424708415979572) (some (280197))

def event280199 : Event := .predecessor (⟨.program ⟨257⟩, ⟨58653⟩⟩) 0 ⟨55673⟩ 280198

def event280200 : Event := .predecessor (⟨.program ⟨257⟩, ⟨58653⟩⟩) 1 ⟨58652⟩ 278869

def event280201 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨58653⟩⟩) (.sum [.predecessor 0 280199 .coefficient, .predecessor 1 280200 .coefficient])

def event280202 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨58653⟩⟩) (.sum [.result 280198 .summary, .result 278869 .summary])

def exact280203RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7199⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7203⟩⟩, ⟨.program ⟨257⟩, ⟨7145⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7205⟩⟩, ⟨.program ⟨257⟩, ⟨7131⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7207⟩⟩, ⟨.program ⟨257⟩, ⟨7125⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7209⟩⟩, ⟨.program ⟨257⟩, ⟨7107⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨56968⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨53988⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51008⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨31944⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨21924⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18704⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15898⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact280203RawTermsValid :
    exact280203RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event280203 : Event := .resultExact (⟨.program ⟨257⟩, ⟨58653⟩⟩) exact280203RawTerms .large 280201 (.finite 2419413932536838975995335147689984068157492) (some (280202))

def event280204 : Event := .predecessor (⟨.program ⟨257⟩, ⟨61633⟩⟩) 0 ⟨58653⟩ 280203

def event280205 : Event := .predecessor (⟨.program ⟨257⟩, ⟨61633⟩⟩) 1 ⟨61632⟩ 278657

def event280206 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨61633⟩⟩) (.sum [.predecessor 0 280204 .coefficient, .predecessor 1 280205 .coefficient])

def event280207 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨61633⟩⟩) (.sum [.result 280203 .summary, .result 278657 .summary])

def exact280208RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7199⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7203⟩⟩, ⟨.program ⟨257⟩, ⟨7145⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7205⟩⟩, ⟨.program ⟨257⟩, ⟨7131⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7207⟩⟩, ⟨.program ⟨257⟩, ⟨7125⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7209⟩⟩, ⟨.program ⟨257⟩, ⟨7107⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7211⟩⟩, ⟨.program ⟨257⟩, ⟨7103⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨59948⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨56968⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨53988⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51008⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨31944⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨21924⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18704⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15898⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact280208RawTermsValid :
    exact280208RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event280208 : Event := .resultExact (⟨.program ⟨257⟩, ⟨61633⟩⟩) exact280208RawTerms .large 280206 (.finite 2765055493188795324243372926469393465999412) (some (280207))

def event280209 : Event := .predecessor (⟨.program ⟨257⟩, ⟨64613⟩⟩) 0 ⟨61633⟩ 280208

def event280210 : Event := .predecessor (⟨.program ⟨257⟩, ⟨64613⟩⟩) 1 ⟨64612⟩ 278445

def event280211 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨64613⟩⟩) (.sum [.predecessor 0 280209 .coefficient, .predecessor 1 280210 .coefficient])

def event280212 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨64613⟩⟩) (.sum [.result 280208 .summary, .result 278445 .summary])

def exact280213RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7199⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7203⟩⟩, ⟨.program ⟨257⟩, ⟨7145⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7205⟩⟩, ⟨.program ⟨257⟩, ⟨7131⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7207⟩⟩, ⟨.program ⟨257⟩, ⟨7125⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7209⟩⟩, ⟨.program ⟨257⟩, ⟨7107⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7211⟩⟩, ⟨.program ⟨257⟩, ⟨7103⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7213⟩⟩, ⟨.program ⟨257⟩, ⟨7099⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨62928⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨59948⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨56968⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨53988⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51008⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨31944⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨21924⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18704⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15898⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact280213RawTermsValid :
    exact280213RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event280213 : Event := .resultExact (⟨.program ⟨257⟩, ⟨64613⟩⟩) exact280213RawTerms .large 280211 (.finite 3110701272581949232038858886277070355169332) (some (280212))

def event280214 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69510⟩⟩) 0 ⟨64613⟩ 280213

def event280215 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69510⟩⟩) 1 ⟨69509⟩ 278233

def event280216 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69510⟩⟩) (.sum [.predecessor 0 280214 .coefficient, .predecessor 1 280215 .coefficient])

def event280217 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69510⟩⟩) (.sum [.result 280213 .summary, .result 278233 .summary])

def exact280218RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7199⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7203⟩⟩, ⟨.program ⟨257⟩, ⟨7145⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7205⟩⟩, ⟨.program ⟨257⟩, ⟨7131⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7207⟩⟩, ⟨.program ⟨257⟩, ⟨7125⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7209⟩⟩, ⟨.program ⟨257⟩, ⟨7107⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7211⟩⟩, ⟨.program ⟨257⟩, ⟨7103⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7213⟩⟩, ⟨.program ⟨257⟩, ⟨7099⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7215⟩⟩, ⟨.program ⟨257⟩, ⟨7173⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨62928⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨59948⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨56968⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨53988⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51008⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨31944⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨21924⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18704⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15898⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66006⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact280218RawTermsValid :
    exact280218RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event280218 : Event := .resultExact (⟨.program ⟨257⟩, ⟨69510⟩⟩) exact280218RawTerms .large 280216 (.finite 3456353380086899479155517117627148481331252) (some (280217))

def event280219 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69511⟩⟩) 0 ⟨69510⟩ 280218

def event280220 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69511⟩⟩) 1 ⟨28080⟩ 278021

def event280221 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69511⟩⟩) (.sum [.predecessor 0 280219 .coefficient, .predecessor 1 280220 .coefficient])

def event280222 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69511⟩⟩) (.sum [.result 280218 .summary, .result 278021 .summary])

def exact280223RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7199⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7203⟩⟩, ⟨.program ⟨257⟩, ⟨7145⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7205⟩⟩, ⟨.program ⟨257⟩, ⟨7131⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7207⟩⟩, ⟨.program ⟨257⟩, ⟨7125⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7209⟩⟩, ⟨.program ⟨257⟩, ⟨7107⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7211⟩⟩, ⟨.program ⟨257⟩, ⟨7103⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7213⟩⟩, ⟨.program ⟨257⟩, ⟨7099⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7215⟩⟩, ⟨.program ⟨257⟩, ⟨7173⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7217⟩⟩, ⟨.program ⟨257⟩, ⟨7169⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨62928⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨59948⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨56968⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨53988⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51008⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨31944⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨21924⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18704⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26515⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15898⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66006⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact280223RawTermsValid :
    exact280223RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event280223 : Event := .resultExact (⟨.program ⟨257⟩, ⟨69511⟩⟩) exact280223RawTerms .large 280221 (.finite 3802007596962448506045899439491360353157172) (some (280222))

def event280224 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69512⟩⟩) 0 ⟨69511⟩ 280223

def event280225 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69512⟩⟩) 1 ⟨30760⟩ 277809

def event280226 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69512⟩⟩) (.sum [.predecessor 0 280224 .coefficient, .predecessor 1 280225 .coefficient])

def event280227 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69512⟩⟩) (.sum [.result 280223 .summary, .result 277809 .summary])

def exact280228RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7199⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7203⟩⟩, ⟨.program ⟨257⟩, ⟨7145⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7205⟩⟩, ⟨.program ⟨257⟩, ⟨7131⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7207⟩⟩, ⟨.program ⟨257⟩, ⟨7125⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7209⟩⟩, ⟨.program ⟨257⟩, ⟨7107⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7211⟩⟩, ⟨.program ⟨257⟩, ⟨7103⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7213⟩⟩, ⟨.program ⟨257⟩, ⟨7099⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7215⟩⟩, ⟨.program ⟨257⟩, ⟨7173⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7217⟩⟩, ⟨.program ⟨257⟩, ⟨7169⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7219⟩⟩, ⟨.program ⟨257⟩, ⟨7167⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨62928⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨59948⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨56968⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨53988⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51008⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨31944⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨21924⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18704⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29195⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26515⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15898⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66006⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact280228RawTermsValid :
    exact280228RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event280228 : Event := .resultExact (⟨.program ⟨257⟩, ⟨69512⟩⟩) exact280228RawTerms .large 280226 (.finite 4147668141949793872257454032897973461975092) (some (280227))

def event280229 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69513⟩⟩) 0 ⟨69512⟩ 280228

def event280230 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69513⟩⟩) 1 ⟨36420⟩ 277597

def event280231 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69513⟩⟩) (.sum [.predecessor 0 280229 .coefficient, .predecessor 1 280230 .coefficient])

def event280232 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69513⟩⟩) (.sum [.result 280228 .summary, .result 277597 .summary])

def exact280233RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7199⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7203⟩⟩, ⟨.program ⟨257⟩, ⟨7145⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7205⟩⟩, ⟨.program ⟨257⟩, ⟨7131⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7207⟩⟩, ⟨.program ⟨257⟩, ⟨7125⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7209⟩⟩, ⟨.program ⟨257⟩, ⟨7107⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7211⟩⟩, ⟨.program ⟨257⟩, ⟨7103⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7213⟩⟩, ⟨.program ⟨257⟩, ⟨7099⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7215⟩⟩, ⟨.program ⟨257⟩, ⟨7173⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7217⟩⟩, ⟨.program ⟨257⟩, ⟨7169⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7219⟩⟩, ⟨.program ⟨257⟩, ⟨7167⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7221⟩⟩, ⟨.program ⟨257⟩, ⟨7163⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨62928⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨59948⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨56968⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨53988⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51008⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨31944⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨21924⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34852⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18704⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29195⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26515⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15898⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66006⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact280233RawTermsValid :
    exact280233RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event280233 : Event := .resultExact (⟨.program ⟨257⟩, ⟨69513⟩⟩) exact280233RawTerms .large 280231 (.finite 4493332905678336798016456807332854062121012) (some (280232))

def event280234 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69514⟩⟩) 0 ⟨69513⟩ 280233

def event280235 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69514⟩⟩) 1 ⟨39100⟩ 277385

def event280236 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69514⟩⟩) (.sum [.predecessor 0 280234 .coefficient, .predecessor 1 280235 .coefficient])

def event280237 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69514⟩⟩) (.sum [.result 280233 .summary, .result 277385 .summary])

def exact280238RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7199⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7203⟩⟩, ⟨.program ⟨257⟩, ⟨7145⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7205⟩⟩, ⟨.program ⟨257⟩, ⟨7131⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7207⟩⟩, ⟨.program ⟨257⟩, ⟨7125⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7209⟩⟩, ⟨.program ⟨257⟩, ⟨7107⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7211⟩⟩, ⟨.program ⟨257⟩, ⟨7103⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7213⟩⟩, ⟨.program ⟨257⟩, ⟨7099⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7215⟩⟩, ⟨.program ⟨257⟩, ⟨7173⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7217⟩⟩, ⟨.program ⟨257⟩, ⟨7169⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7219⟩⟩, ⟨.program ⟨257⟩, ⟨7167⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7221⟩⟩, ⟨.program ⟨257⟩, ⟨7163⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7223⟩⟩, ⟨.program ⟨257⟩, ⟨7161⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨62928⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨59948⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨56968⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨53988⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51008⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨31944⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨21924⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37532⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34852⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18704⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29195⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26515⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15898⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66006⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact280238RawTermsValid :
    exact280238RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event280238 : Event := .resultExact (⟨.program ⟨257⟩, ⟨69514⟩⟩) exact280238RawTerms .large 280236 (.finite 4838999778777478503549183672281868407930932) (some (280237))

def event280239 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69515⟩⟩) 0 ⟨69514⟩ 280238

def event280240 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69515⟩⟩) 1 ⟨41780⟩ 277173

def event280241 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69515⟩⟩) (.sum [.predecessor 0 280239 .coefficient, .predecessor 1 280240 .coefficient])

def event280242 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69515⟩⟩) (.sum [.result 280238 .summary, .result 277173 .summary])

def exact280243RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7199⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7203⟩⟩, ⟨.program ⟨257⟩, ⟨7145⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7205⟩⟩, ⟨.program ⟨257⟩, ⟨7131⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7207⟩⟩, ⟨.program ⟨257⟩, ⟨7125⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7209⟩⟩, ⟨.program ⟨257⟩, ⟨7107⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7211⟩⟩, ⟨.program ⟨257⟩, ⟨7103⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7213⟩⟩, ⟨.program ⟨257⟩, ⟨7099⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7215⟩⟩, ⟨.program ⟨257⟩, ⟨7173⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7217⟩⟩, ⟨.program ⟨257⟩, ⟨7169⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7219⟩⟩, ⟨.program ⟨257⟩, ⟨7167⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7221⟩⟩, ⟨.program ⟨257⟩, ⟨7163⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7223⟩⟩, ⟨.program ⟨257⟩, ⟨7161⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7225⟩⟩, ⟨.program ⟨257⟩, ⟨7159⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨62928⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨59948⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨56968⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨53988⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51008⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨31944⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨21924⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40215⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37532⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34852⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18704⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29195⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26515⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15898⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66006⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact280243RawTermsValid :
    exact280243RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event280243 : Event := .resultExact (⟨.program ⟨257⟩, ⟨69515⟩⟩) exact280243RawTerms .large 280241 (.finite 5184670870617817768629358718259150245068852) (some (280242))

def event280244 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69516⟩⟩) 0 ⟨69515⟩ 280243

def event280245 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69516⟩⟩) 1 ⟨44460⟩ 276961

def event280246 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69516⟩⟩) (.sum [.predecessor 0 280244 .coefficient, .predecessor 1 280245 .coefficient])

def event280247 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69516⟩⟩) (.sum [.result 280243 .summary, .result 276961 .summary])

def exact280248RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7199⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7203⟩⟩, ⟨.program ⟨257⟩, ⟨7145⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7205⟩⟩, ⟨.program ⟨257⟩, ⟨7131⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7207⟩⟩, ⟨.program ⟨257⟩, ⟨7125⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7209⟩⟩, ⟨.program ⟨257⟩, ⟨7107⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7211⟩⟩, ⟨.program ⟨257⟩, ⟨7103⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7213⟩⟩, ⟨.program ⟨257⟩, ⟨7099⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7215⟩⟩, ⟨.program ⟨257⟩, ⟨7173⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7217⟩⟩, ⟨.program ⟨257⟩, ⟨7169⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7219⟩⟩, ⟨.program ⟨257⟩, ⟨7167⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7221⟩⟩, ⟨.program ⟨257⟩, ⟨7163⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7223⟩⟩, ⟨.program ⟨257⟩, ⟨7161⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7225⟩⟩, ⟨.program ⟨257⟩, ⟨7159⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7227⟩⟩, ⟨.program ⟨257⟩, ⟨7153⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨62928⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨59948⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨56968⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨53988⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51008⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨31944⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨42895⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨21924⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40215⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37532⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34852⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18704⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29195⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26515⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15898⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66006⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact280248RawTermsValid :
    exact280248RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event280248 : Event := .resultExact (⟨.program ⟨257⟩, ⟨69516⟩⟩) exact280248RawTerms .large 280246 (.finite 5530348290569953373030706035778833319198772) (some (280247))

def event280249 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69517⟩⟩) 0 ⟨69516⟩ 280248

def event280250 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69517⟩⟩) 1 ⟨47140⟩ 276749

def event280251 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69517⟩⟩) (.sum [.predecessor 0 280249 .coefficient, .predecessor 1 280250 .coefficient])

def event280252 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69517⟩⟩) (.sum [.result 280248 .summary, .result 276749 .summary])

def exact280253RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7199⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7203⟩⟩, ⟨.program ⟨257⟩, ⟨7145⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7205⟩⟩, ⟨.program ⟨257⟩, ⟨7131⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7207⟩⟩, ⟨.program ⟨257⟩, ⟨7125⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7209⟩⟩, ⟨.program ⟨257⟩, ⟨7107⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7211⟩⟩, ⟨.program ⟨257⟩, ⟨7103⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7213⟩⟩, ⟨.program ⟨257⟩, ⟨7099⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7215⟩⟩, ⟨.program ⟨257⟩, ⟨7173⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7217⟩⟩, ⟨.program ⟨257⟩, ⟨7169⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7219⟩⟩, ⟨.program ⟨257⟩, ⟨7167⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7221⟩⟩, ⟨.program ⟨257⟩, ⟨7163⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7223⟩⟩, ⟨.program ⟨257⟩, ⟨7161⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7225⟩⟩, ⟨.program ⟨257⟩, ⟨7159⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7227⟩⟩, ⟨.program ⟨257⟩, ⟨7153⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7229⟩⟩, ⟨.program ⟨257⟩, ⟨7151⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨62928⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨59948⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨56968⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨53988⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51008⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨31944⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45572⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨42895⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨21924⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40215⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37532⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34852⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18704⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29195⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26515⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15898⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66006⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact280253RawTermsValid :
    exact280253RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event280253 : Event := .resultExact (⟨.program ⟨257⟩, ⟨69517⟩⟩) exact280253RawTerms .large 280251 (.finite 5876032038633885316753225624840917630320692) (some (280252))

def event280254 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69518⟩⟩) 0 ⟨69517⟩ 280253

def event280255 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69518⟩⟩) 1 ⟨49820⟩ 276537

def event280256 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69518⟩⟩) (.sum [.predecessor 0 280254 .coefficient, .predecessor 1 280255 .coefficient])

def event280257 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69518⟩⟩) (.sum [.result 280253 .summary, .result 276537 .summary])

def exact280258RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7199⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7203⟩⟩, ⟨.program ⟨257⟩, ⟨7145⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7205⟩⟩, ⟨.program ⟨257⟩, ⟨7131⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7207⟩⟩, ⟨.program ⟨257⟩, ⟨7125⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7209⟩⟩, ⟨.program ⟨257⟩, ⟨7107⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7211⟩⟩, ⟨.program ⟨257⟩, ⟨7103⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7213⟩⟩, ⟨.program ⟨257⟩, ⟨7099⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7215⟩⟩, ⟨.program ⟨257⟩, ⟨7173⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7217⟩⟩, ⟨.program ⟨257⟩, ⟨7169⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7219⟩⟩, ⟨.program ⟨257⟩, ⟨7167⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7221⟩⟩, ⟨.program ⟨257⟩, ⟨7163⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7223⟩⟩, ⟨.program ⟨257⟩, ⟨7161⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7225⟩⟩, ⟨.program ⟨257⟩, ⟨7159⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7227⟩⟩, ⟨.program ⟨257⟩, ⟨7153⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7229⟩⟩, ⟨.program ⟨257⟩, ⟨7151⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7231⟩⟩, ⟨.program ⟨257⟩, ⟨7147⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨62928⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨59948⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨56968⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨53988⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51008⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨31944⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48252⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45572⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨42895⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨21924⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40215⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37532⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34852⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18704⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29195⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26515⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15898⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66006⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact280258RawTermsValid :
    exact280258RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event280258 : Event := .resultExact (⟨.program ⟨257⟩, ⟨69518⟩⟩) exact280258RawTerms .large 280256 (.finite 6221717896068416040249469304417135687106612) (some (280257))

def event280259 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70985⟩⟩) 0 ⟨69518⟩ 280258

def event280260 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70985⟩⟩) 1 ⟨70983⟩ 276325

def event280261 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70985⟩⟩) (.sum [.predecessor 0 280259 .coefficient, .predecessor 1 280260 .coefficient])

def event280262 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70985⟩⟩) (.sum [.result 280258 .summary, .result 276325 .summary])

def exact280263RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7199⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7203⟩⟩, ⟨.program ⟨257⟩, ⟨7145⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7205⟩⟩, ⟨.program ⟨257⟩, ⟨7131⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7207⟩⟩, ⟨.program ⟨257⟩, ⟨7125⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7209⟩⟩, ⟨.program ⟨257⟩, ⟨7107⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7211⟩⟩, ⟨.program ⟨257⟩, ⟨7103⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7213⟩⟩, ⟨.program ⟨257⟩, ⟨7099⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7215⟩⟩, ⟨.program ⟨257⟩, ⟨7173⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7217⟩⟩, ⟨.program ⟨257⟩, ⟨7169⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7219⟩⟩, ⟨.program ⟨257⟩, ⟨7167⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7221⟩⟩, ⟨.program ⟨257⟩, ⟨7163⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7223⟩⟩, ⟨.program ⟨257⟩, ⟨7161⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7225⟩⟩, ⟨.program ⟨257⟩, ⟨7159⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7227⟩⟩, ⟨.program ⟨257⟩, ⟨7153⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7229⟩⟩, ⟨.program ⟨257⟩, ⟨7151⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7231⟩⟩, ⟨.program ⟨257⟩, ⟨7147⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7233⟩⟩, ⟨.program ⟨257⟩, ⟨7139⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨62928⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨59948⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨56968⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨53988⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51008⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨67300⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨31944⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48252⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45572⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨42895⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨21924⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40215⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37532⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34852⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18704⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29195⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26515⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15898⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66006⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact280263RawTermsValid :
    exact280263RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event280263 : Event := .resultExact (⟨.program ⟨257⟩, ⟨70985⟩⟩) exact280263RawTerms .large 280261 (.finite 66805187227601152574551644069558752530002096506798132) (some (280262))

def event280264 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨55⟩⟩) (.authority (.operator))

def exact280265RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨55⟩⟩]⟩, (1)⟩]

theorem exact280265RawTermsValid :
    exact280265RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event280265 : Event := .resultExact (⟨.program ⟨257⟩, ⟨55⟩⟩) exact280265RawTerms (.finite 26) 280264 .exactZero (none)

def event280266 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7418⟩⟩) 0 ⟨2377⟩ 27

def event280267 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7418⟩⟩) 1 ⟨7268⟩ 16667

def event280268 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7418⟩⟩) (.product (.predecessor 0 280266 .coefficient) (.predecessor 1 280267 .coefficient) (⟨false, false, none, none, none⟩))

def event280269 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨7418⟩⟩, .operator (⟨27, 0⟩, ⟨16667, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7268⟩⟩]⟩, (1)⟩)

def exact280270RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7268⟩⟩]⟩, (1)⟩]

theorem exact280270RawTermsValid :
    exact280270RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event280270 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7418⟩⟩) exact280270RawTerms .large 280268 .exactZero (none)

def event280271 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9239⟩⟩) 0 ⟨7418⟩ 280270

def event280272 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9239⟩⟩) 1 ⟨6915⟩ 266028

def event280273 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9239⟩⟩) (.sum [.predecessor 0 280271 .coefficient, .predecessor 1 280272 .coefficient])

def exact280274RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7268⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact280274RawTermsValid :
    exact280274RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event280274 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9239⟩⟩) exact280274RawTerms .large 280273 .exactZero (none)

def event280275 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9240⟩⟩) 0 ⟨9239⟩ 280274

def event280276 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9240⟩⟩) 1 ⟨55⟩ 280265

def event280277 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9240⟩⟩) (.sum [.predecessor 0 280275 .coefficient, .predecessor 1 280276 .coefficient])

def event280278 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9240⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨55⟩⟩]⟩) [⟨.result 280265 .coefficient, false, none⟩])

def event280279 : Event := .survivorFold (1) 280278

def exact280280RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7268⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact280280RawTermsValid :
    exact280280RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event280280 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9240⟩⟩) exact280280RawTerms .large 280277 (.finite 26) (some (280278))

def event280281 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9630⟩⟩) 0 ⟨9240⟩ 280280

def event280282 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9630⟩⟩) 1 ⟨9584⟩ 15984

def event280283 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9630⟩⟩) (.product (.predecessor 0 280281 .coefficient) (.predecessor 1 280282 .coefficient) (⟨false, false, none, none, none⟩))

def event280284 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9630⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨9583⟩⟩]⟩) [⟨.result 15980 .coefficient, false, none⟩])

def event280285 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9630⟩⟩) (.product (.result 280280 .summary) (.transfer 280284) (⟨false, false, none, none, none⟩))

def event280286 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨9630⟩⟩, .operator (⟨280280, 1⟩, ⟨15984, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩]⟩, (-1)⟩)

def event280287 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨9630⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨9583⟩⟩) ⟨9443⟩ 15977)

def event280288 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨9630⟩⟩, .relation 280287 18, ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7233⟩⟩, ⟨.program ⟨257⟩, ⟨7139⟩⟩]⟩, (1)⟩)

def event280289 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨9630⟩⟩, .relation 280287 17, ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7231⟩⟩, ⟨.program ⟨257⟩, ⟨7147⟩⟩]⟩, (-1)⟩)

def event280290 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨9630⟩⟩, .relation 280287 16, ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7229⟩⟩, ⟨.program ⟨257⟩, ⟨7151⟩⟩]⟩, (-1)⟩)

def event280291 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨9630⟩⟩, .relation 280287 15, ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7227⟩⟩, ⟨.program ⟨257⟩, ⟨7153⟩⟩]⟩, (-1)⟩)

def event280292 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨9630⟩⟩, .relation 280287 14, ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7225⟩⟩, ⟨.program ⟨257⟩, ⟨7159⟩⟩]⟩, (-1)⟩)

def event280293 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨9630⟩⟩, .relation 280287 13, ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7223⟩⟩, ⟨.program ⟨257⟩, ⟨7161⟩⟩]⟩, (-1)⟩)

def event280294 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨9630⟩⟩, .relation 280287 12, ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7221⟩⟩, ⟨.program ⟨257⟩, ⟨7163⟩⟩]⟩, (-1)⟩)

def event280295 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨9630⟩⟩, .relation 280287 11, ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7219⟩⟩, ⟨.program ⟨257⟩, ⟨7167⟩⟩]⟩, (-1)⟩)

def event280296 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨9630⟩⟩, .relation 280287 10, ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7217⟩⟩, ⟨.program ⟨257⟩, ⟨7169⟩⟩]⟩, (-1)⟩)

def event280297 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨9630⟩⟩, .relation 280287 9, ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7215⟩⟩, ⟨.program ⟨257⟩, ⟨7173⟩⟩]⟩, (-1)⟩)

def event280298 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨9630⟩⟩, .relation 280287 8, ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7213⟩⟩, ⟨.program ⟨257⟩, ⟨7099⟩⟩]⟩, (-1)⟩)

def event280299 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨9630⟩⟩, .relation 280287 7, ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7211⟩⟩, ⟨.program ⟨257⟩, ⟨7103⟩⟩]⟩, (-1)⟩)

def event280300 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨9630⟩⟩, .relation 280287 6, ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7209⟩⟩, ⟨.program ⟨257⟩, ⟨7107⟩⟩]⟩, (-1)⟩)

def event280301 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨9630⟩⟩, .relation 280287 5, ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7207⟩⟩, ⟨.program ⟨257⟩, ⟨7125⟩⟩]⟩, (-1)⟩)

def event280302 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨9630⟩⟩, .relation 280287 4, ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7205⟩⟩, ⟨.program ⟨257⟩, ⟨7131⟩⟩]⟩, (-1)⟩)

def event280303 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨9630⟩⟩, .relation 280287 3, ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7203⟩⟩, ⟨.program ⟨257⟩, ⟨7145⟩⟩]⟩, (-1)⟩)

def event280304 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨9630⟩⟩, .relation 280287 2, ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩, (-1)⟩)

def event280305 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨9630⟩⟩, .relation 280287 1, ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7199⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩, (-1)⟩)

def event280306 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨9630⟩⟩, .relation 280287 0, ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (-1)⟩)

def event280307 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨9630⟩⟩, .operator (⟨280280, 0⟩, ⟨15984, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7268⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩]⟩, (1)⟩)

def exact280308RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7268⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7199⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7203⟩⟩, ⟨.program ⟨257⟩, ⟨7145⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7205⟩⟩, ⟨.program ⟨257⟩, ⟨7131⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7207⟩⟩, ⟨.program ⟨257⟩, ⟨7125⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7209⟩⟩, ⟨.program ⟨257⟩, ⟨7107⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7211⟩⟩, ⟨.program ⟨257⟩, ⟨7103⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7213⟩⟩, ⟨.program ⟨257⟩, ⟨7099⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7215⟩⟩, ⟨.program ⟨257⟩, ⟨7173⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7217⟩⟩, ⟨.program ⟨257⟩, ⟨7169⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7219⟩⟩, ⟨.program ⟨257⟩, ⟨7167⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7221⟩⟩, ⟨.program ⟨257⟩, ⟨7163⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7223⟩⟩, ⟨.program ⟨257⟩, ⟨7161⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7225⟩⟩, ⟨.program ⟨257⟩, ⟨7159⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7227⟩⟩, ⟨.program ⟨257⟩, ⟨7153⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7229⟩⟩, ⟨.program ⟨257⟩, ⟨7151⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7231⟩⟩, ⟨.program ⟨257⟩, ⟨7147⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7233⟩⟩, ⟨.program ⟨257⟩, ⟨7139⟩⟩]⟩, (1)⟩]

theorem exact280308RawTermsValid :
    exact280308RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event280308 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9630⟩⟩) exact280308RawTerms .large 280283 (.finite 279172874240) (some (280285))

def event280309 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70986⟩⟩) 0 ⟨9630⟩ 280308

def event280310 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70986⟩⟩) 1 ⟨70985⟩ 280263

def event280311 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70986⟩⟩) (.sum [.predecessor 0 280309 .coefficient, .predecessor 1 280310 .coefficient])

def event280312 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70986⟩⟩, .operator (⟨280308, 19⟩, ⟨280263, 18⟩), ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7233⟩⟩, ⟨.program ⟨257⟩, ⟨7139⟩⟩]⟩, (-1)⟩)

def event280313 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70986⟩⟩, .operator (⟨280308, 18⟩, ⟨280263, 17⟩), ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7231⟩⟩, ⟨.program ⟨257⟩, ⟨7147⟩⟩]⟩, (1)⟩)

def event280314 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70986⟩⟩, .operator (⟨280308, 17⟩, ⟨280263, 16⟩), ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7229⟩⟩, ⟨.program ⟨257⟩, ⟨7151⟩⟩]⟩, (1)⟩)

def event280315 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70986⟩⟩, .operator (⟨280308, 16⟩, ⟨280263, 15⟩), ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7227⟩⟩, ⟨.program ⟨257⟩, ⟨7153⟩⟩]⟩, (1)⟩)

def event280316 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70986⟩⟩, .operator (⟨280308, 15⟩, ⟨280263, 14⟩), ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7225⟩⟩, ⟨.program ⟨257⟩, ⟨7159⟩⟩]⟩, (1)⟩)

def event280317 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70986⟩⟩, .operator (⟨280308, 14⟩, ⟨280263, 13⟩), ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7223⟩⟩, ⟨.program ⟨257⟩, ⟨7161⟩⟩]⟩, (1)⟩)

def event280318 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70986⟩⟩, .operator (⟨280308, 13⟩, ⟨280263, 12⟩), ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7221⟩⟩, ⟨.program ⟨257⟩, ⟨7163⟩⟩]⟩, (1)⟩)

def event280319 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70986⟩⟩, .operator (⟨280308, 12⟩, ⟨280263, 11⟩), ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7219⟩⟩, ⟨.program ⟨257⟩, ⟨7167⟩⟩]⟩, (1)⟩)

def eventLeaf17504 : Array AnnotatedEvent := #[
  { event := event280064
    frameStart := 280014 },
  { event := event280065
    frameStart := 280014 },
  { event := event280066
    frameStart := 280014 },
  { event := event280067
    frameStart := 280014 },
  { event := event280068
    frameStart := 280014 },
  { event := event280069
    frameStart := 280014 },
  { event := event280070
    frameStart := 280014 },
  { event := event280071
    frameStart := 280014 },
  { event := event280072
    frameStart := 280014 },
  { event := event280073
    frameStart := 280014 },
  { event := event280074
    frameStart := 280014 },
  { event := event280075
    frameStart := 280014 },
  { event := event280076
    frameStart := 280014 },
  { event := event280077
    frameStart := 280014 },
  { event := event280078
    frameStart := 280014 },
  { event := event280079
    frameStart := 280014 }
]

def eventLeaf17505 : Array AnnotatedEvent := #[
  { event := event280080
    frameStart := 280014 },
  { event := event280081
    frameStart := 280014 },
  { event := event280082
    frameStart := 280014 },
  { event := event280083
    frameStart := 280014 },
  { event := event280084
    frameStart := 280014 },
  { event := event280085
    frameStart := 280014 },
  { event := event280086
    frameStart := 280014 },
  { event := event280087
    frameStart := 280014 },
  { event := event280088
    frameStart := 280014 },
  { event := event280089
    frameStart := 280014 },
  { event := event280090
    frameStart := 280014 },
  { event := event280091
    frameStart := 280014 },
  { event := event280092
    frameStart := 280014 },
  { event := event280093
    frameStart := 280014 },
  { event := event280094
    frameStart := 280014 },
  { event := event280095
    frameStart := 280014 }
]

def eventLeaf17506 : Array AnnotatedEvent := #[
  { event := event280096
    frameStart := 280014 },
  { event := event280097
    frameStart := 280014 },
  { event := event280098
    frameStart := 280014 },
  { event := event280099
    frameStart := 280014 },
  { event := event280100
    frameStart := 280014 },
  { event := event280101
    frameStart := 280014 },
  { event := event280102
    frameStart := 280014 },
  { event := event280103
    frameStart := 280014 },
  { event := event280104
    frameStart := 280014 },
  { event := event280105
    frameStart := 280014 },
  { event := event280106
    frameStart := 280014 },
  { event := event280107
    frameStart := 280014 },
  { event := event280108
    frameStart := 280014 },
  { event := event280109
    frameStart := 280014 },
  { event := event280110
    frameStart := 280014 },
  { event := event280111
    frameStart := 280014 }
]

def eventLeaf17507 : Array AnnotatedEvent := #[
  { event := event280112
    frameStart := 280014 },
  { event := event280113
    frameStart := 280014 },
  { event := event280114
    frameStart := 280014 },
  { event := event280115
    frameStart := 280014 },
  { event := event280116
    frameStart := 280014 },
  { event := event280117
    frameStart := 280014 },
  { event := event280118
    frameStart := 0 },
  { event := event280119
    frameStart := 0 },
  { event := event280120
    frameStart := 0 },
  { event := event280121
    frameStart := 0 },
  { event := event280122
    frameStart := 0 },
  { event := event280123
    frameStart := 0 },
  { event := event280124
    frameStart := 0 },
  { event := event280125
    frameStart := 0 },
  { event := event280126
    frameStart := 0 },
  { event := event280127
    frameStart := 0 }
]

def eventLeaf17508 : Array AnnotatedEvent := #[
  { event := event280128
    frameStart := 0 },
  { event := event280129
    frameStart := 0 },
  { event := event280130
    frameStart := 0 },
  { event := event280131
    frameStart := 0 },
  { event := event280132
    frameStart := 0 },
  { event := event280133
    frameStart := 0 },
  { event := event280134
    frameStart := 0 },
  { event := event280135
    frameStart := 0 },
  { event := event280136
    frameStart := 0 },
  { event := event280137
    frameStart := 0 },
  { event := event280138
    frameStart := 0 },
  { event := event280139
    frameStart := 0 },
  { event := event280140
    frameStart := 0 },
  { event := event280141
    frameStart := 0 },
  { event := event280142
    frameStart := 0 },
  { event := event280143
    frameStart := 0 }
]

def eventLeaf17509 : Array AnnotatedEvent := #[
  { event := event280144
    frameStart := 0 },
  { event := event280145
    frameStart := 0 },
  { event := event280146
    frameStart := 0 },
  { event := event280147
    frameStart := 0 },
  { event := event280148
    frameStart := 0 },
  { event := event280149
    frameStart := 0 },
  { event := event280150
    frameStart := 0 },
  { event := event280151
    frameStart := 0 },
  { event := event280152
    frameStart := 0 },
  { event := event280153
    frameStart := 0 },
  { event := event280154
    frameStart := 0 },
  { event := event280155
    frameStart := 0 },
  { event := event280156
    frameStart := 0 },
  { event := event280157
    frameStart := 0 },
  { event := event280158
    frameStart := 0 },
  { event := event280159
    frameStart := 0 }
]

def eventLeaf17510 : Array AnnotatedEvent := #[
  { event := event280160
    frameStart := 0 },
  { event := event280161
    frameStart := 0 },
  { event := event280162
    frameStart := 0 },
  { event := event280163
    frameStart := 0 },
  { event := event280164
    frameStart := 0 },
  { event := event280165
    frameStart := 0 },
  { event := event280166
    frameStart := 0 },
  { event := event280167
    frameStart := 0 },
  { event := event280168
    frameStart := 0 },
  { event := event280169
    frameStart := 0 },
  { event := event280170
    frameStart := 0 },
  { event := event280171
    frameStart := 0 },
  { event := event280172
    frameStart := 0 },
  { event := event280173
    frameStart := 0 },
  { event := event280174
    frameStart := 0 },
  { event := event280175
    frameStart := 0 }
]

def eventLeaf17511 : Array AnnotatedEvent := #[
  { event := event280176
    frameStart := 0 },
  { event := event280177
    frameStart := 0 },
  { event := event280178
    frameStart := 0 },
  { event := event280179
    frameStart := 0 },
  { event := event280180
    frameStart := 0 },
  { event := event280181
    frameStart := 0 },
  { event := event280182
    frameStart := 0 },
  { event := event280183
    frameStart := 0 },
  { event := event280184
    frameStart := 0 },
  { event := event280185
    frameStart := 0 },
  { event := event280186
    frameStart := 0 },
  { event := event280187
    frameStart := 0 },
  { event := event280188
    frameStart := 0 },
  { event := event280189
    frameStart := 0 },
  { event := event280190
    frameStart := 0 },
  { event := event280191
    frameStart := 0 }
]

def eventLeaf17512 : Array AnnotatedEvent := #[
  { event := event280192
    frameStart := 0 },
  { event := event280193
    frameStart := 0 },
  { event := event280194
    frameStart := 0 },
  { event := event280195
    frameStart := 0 },
  { event := event280196
    frameStart := 0 },
  { event := event280197
    frameStart := 0 },
  { event := event280198
    frameStart := 0 },
  { event := event280199
    frameStart := 0 },
  { event := event280200
    frameStart := 0 },
  { event := event280201
    frameStart := 0 },
  { event := event280202
    frameStart := 0 },
  { event := event280203
    frameStart := 0 },
  { event := event280204
    frameStart := 0 },
  { event := event280205
    frameStart := 0 },
  { event := event280206
    frameStart := 0 },
  { event := event280207
    frameStart := 0 }
]

def eventLeaf17513 : Array AnnotatedEvent := #[
  { event := event280208
    frameStart := 0 },
  { event := event280209
    frameStart := 0 },
  { event := event280210
    frameStart := 0 },
  { event := event280211
    frameStart := 0 },
  { event := event280212
    frameStart := 0 },
  { event := event280213
    frameStart := 0 },
  { event := event280214
    frameStart := 0 },
  { event := event280215
    frameStart := 0 },
  { event := event280216
    frameStart := 0 },
  { event := event280217
    frameStart := 0 },
  { event := event280218
    frameStart := 0 },
  { event := event280219
    frameStart := 0 },
  { event := event280220
    frameStart := 0 },
  { event := event280221
    frameStart := 0 },
  { event := event280222
    frameStart := 0 },
  { event := event280223
    frameStart := 0 }
]

def eventLeaf17514 : Array AnnotatedEvent := #[
  { event := event280224
    frameStart := 0 },
  { event := event280225
    frameStart := 0 },
  { event := event280226
    frameStart := 0 },
  { event := event280227
    frameStart := 0 },
  { event := event280228
    frameStart := 0 },
  { event := event280229
    frameStart := 0 },
  { event := event280230
    frameStart := 0 },
  { event := event280231
    frameStart := 0 },
  { event := event280232
    frameStart := 0 },
  { event := event280233
    frameStart := 0 },
  { event := event280234
    frameStart := 0 },
  { event := event280235
    frameStart := 0 },
  { event := event280236
    frameStart := 0 },
  { event := event280237
    frameStart := 0 },
  { event := event280238
    frameStart := 0 },
  { event := event280239
    frameStart := 0 }
]

def eventLeaf17515 : Array AnnotatedEvent := #[
  { event := event280240
    frameStart := 0 },
  { event := event280241
    frameStart := 0 },
  { event := event280242
    frameStart := 0 },
  { event := event280243
    frameStart := 0 },
  { event := event280244
    frameStart := 0 },
  { event := event280245
    frameStart := 0 },
  { event := event280246
    frameStart := 0 },
  { event := event280247
    frameStart := 0 },
  { event := event280248
    frameStart := 0 },
  { event := event280249
    frameStart := 0 },
  { event := event280250
    frameStart := 0 },
  { event := event280251
    frameStart := 0 },
  { event := event280252
    frameStart := 0 },
  { event := event280253
    frameStart := 0 },
  { event := event280254
    frameStart := 0 },
  { event := event280255
    frameStart := 0 }
]

def eventLeaf17516 : Array AnnotatedEvent := #[
  { event := event280256
    frameStart := 0 },
  { event := event280257
    frameStart := 0 },
  { event := event280258
    frameStart := 0 },
  { event := event280259
    frameStart := 0 },
  { event := event280260
    frameStart := 0 },
  { event := event280261
    frameStart := 0 },
  { event := event280262
    frameStart := 0 },
  { event := event280263
    frameStart := 0 },
  { event := event280264
    frameStart := 0 },
  { event := event280265
    frameStart := 0 },
  { event := event280266
    frameStart := 0 },
  { event := event280267
    frameStart := 0 },
  { event := event280268
    frameStart := 0 },
  { event := event280269
    frameStart := 0 },
  { event := event280270
    frameStart := 0 },
  { event := event280271
    frameStart := 0 }
]

def eventLeaf17517 : Array AnnotatedEvent := #[
  { event := event280272
    frameStart := 0 },
  { event := event280273
    frameStart := 0 },
  { event := event280274
    frameStart := 0 },
  { event := event280275
    frameStart := 0 },
  { event := event280276
    frameStart := 0 },
  { event := event280277
    frameStart := 0 },
  { event := event280278
    frameStart := 0 },
  { event := event280279
    frameStart := 0 },
  { event := event280280
    frameStart := 0 },
  { event := event280281
    frameStart := 0 },
  { event := event280282
    frameStart := 0 },
  { event := event280283
    frameStart := 0 },
  { event := event280284
    frameStart := 0 },
  { event := event280285
    frameStart := 0 },
  { event := event280286
    frameStart := 0 },
  { event := event280287
    frameStart := 0 }
]

def eventLeaf17518 : Array AnnotatedEvent := #[
  { event := event280288
    frameStart := 0 },
  { event := event280289
    frameStart := 0 },
  { event := event280290
    frameStart := 0 },
  { event := event280291
    frameStart := 0 },
  { event := event280292
    frameStart := 0 },
  { event := event280293
    frameStart := 0 },
  { event := event280294
    frameStart := 0 },
  { event := event280295
    frameStart := 0 },
  { event := event280296
    frameStart := 0 },
  { event := event280297
    frameStart := 0 },
  { event := event280298
    frameStart := 0 },
  { event := event280299
    frameStart := 0 },
  { event := event280300
    frameStart := 0 },
  { event := event280301
    frameStart := 0 },
  { event := event280302
    frameStart := 0 },
  { event := event280303
    frameStart := 0 }
]

def eventLeaf17519 : Array AnnotatedEvent := #[
  { event := event280304
    frameStart := 0 },
  { event := event280305
    frameStart := 0 },
  { event := event280306
    frameStart := 0 },
  { event := event280307
    frameStart := 0 },
  { event := event280308
    frameStart := 0 },
  { event := event280309
    frameStart := 0 },
  { event := event280310
    frameStart := 0 },
  { event := event280311
    frameStart := 0 },
  { event := event280312
    frameStart := 0 },
  { event := event280313
    frameStart := 0 },
  { event := event280314
    frameStart := 0 },
  { event := event280315
    frameStart := 0 },
  { event := event280316
    frameStart := 0 },
  { event := event280317
    frameStart := 0 },
  { event := event280318
    frameStart := 0 },
  { event := event280319
    frameStart := 0 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events1094
