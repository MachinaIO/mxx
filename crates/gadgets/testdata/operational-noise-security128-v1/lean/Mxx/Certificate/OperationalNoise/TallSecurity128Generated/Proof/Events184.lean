import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events184

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event47104 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48469⟩⟩) 1 ⟨48468⟩ 47099

def event47105 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48469⟩⟩) (.sum [.predecessor 0 47103 .coefficient, .predecessor 1 47104 .coefficient])

def exact47106RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7232⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨48467⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact47106RawTermsValid :
    exact47106RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event47106 : Event := .resultExact (⟨.program ⟨257⟩, ⟨48469⟩⟩) exact47106RawTerms .large 47105 .exactZero (none)

def event47107 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50233⟩⟩) 0 ⟨48469⟩ 47106

def event47108 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50233⟩⟩) 1 ⟨50230⟩ 47091

def event47109 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50233⟩⟩) (.sum [.predecessor 0 47107 .coefficient, .predecessor 1 47108 .coefficient])

def exact47110RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7196⟩⟩, ⟨.program ⟨257⟩, ⟨50229⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7232⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨48212⟩⟩], [⟨.program ⟨257⟩, ⟨49373⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨48467⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact47110RawTermsValid :
    exact47110RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event47110 : Event := .resultExact (⟨.program ⟨257⟩, ⟨50233⟩⟩) exact47110RawTerms .large 47109 .exactZero (none)

def event47111 : Event := .preFoldPolynomial 47110 [⟨⟨[], [⟨.program ⟨257⟩, ⟨7196⟩⟩, ⟨.program ⟨257⟩, ⟨50229⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7232⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨48212⟩⟩], [⟨.program ⟨257⟩, ⟨49373⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨48467⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩] .exactZero none

def exact47112RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7196⟩⟩, ⟨.program ⟨257⟩, ⟨50229⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7232⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨48212⟩⟩], [⟨.program ⟨257⟩, ⟨49373⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨48467⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

def event47112 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨50233⟩⟩) 47111 exact47112RawTerms .large 47109 .exactZero (none)

def event47113 : Event := .specializationComputed (⟨.program ⟨257⟩, ⟨48213⟩⟩) ⟨⟨111⟩, ⟨94⟩, ⟨135⟩⟩ ⟨46955, 47113⟩

def event47114 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨49059⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨49056⟩⟩]⟩) (1) 0 2 (.universal 47113 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨49056⟩⟩]⟩) (none) 47112)

def event47115 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨49059⟩⟩, .relation 47114 1, ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7232⟩⟩]⟩, (1)⟩)

def event47116 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨49059⟩⟩, .relation 47114 0, ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7196⟩⟩, ⟨.program ⟨257⟩, ⟨50229⟩⟩]⟩, (-1)⟩)

def event47117 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨49059⟩⟩, .relation 47114 2, ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨48212⟩⟩], [⟨.program ⟨257⟩, ⟨49373⟩⟩]⟩, (1)⟩)

def event47118 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨49059⟩⟩, .relation 47114 3, ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨48467⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact47119RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7196⟩⟩, ⟨.program ⟨257⟩, ⟨50229⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7232⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨48212⟩⟩], [⟨.program ⟨257⟩, ⟨49373⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨48467⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact47119RawTermsValid :
    exact47119RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event47119 : Event := .resultExact (⟨.program ⟨257⟩, ⟨49059⟩⟩) exact47119RawTerms .large 46951 (.finite 202072841853861888) (some (46953))

def event47120 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50232⟩⟩) 0 ⟨49059⟩ 47119

def event47121 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50232⟩⟩) 1 ⟨50231⟩ 46941

def event47122 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50232⟩⟩) (.sum [.predecessor 0 47120 .coefficient, .predecessor 1 47121 .coefficient])

def event47123 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨50232⟩⟩, .operator (⟨47119, 0⟩, ⟨46941, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7196⟩⟩, ⟨.program ⟨257⟩, ⟨50229⟩⟩]⟩, (1)⟩)

def event47124 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨50232⟩⟩, .operator (⟨47119, 2⟩, ⟨46941, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨48212⟩⟩], [⟨.program ⟨257⟩, ⟨49373⟩⟩]⟩, (-1)⟩)

def event47125 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50232⟩⟩) (.sum [.result 47119 .summary, .result 46941 .summary])

def exact47126RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7232⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨48467⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact47126RawTermsValid :
    exact47126RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event47126 : Event := .resultExact (⟨.program ⟨257⟩, ⟨50232⟩⟩) exact47126RawTerms .large 47122 (.finite 32194504275408640829496428331008) (some (47125))

def event47127 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46691⟩⟩) 0 ⟨45533⟩ 1630

def event47128 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨46691⟩⟩) (.authority (.programFamilyFact))

def event47129 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨46691⟩⟩) (.finite 3720)

def event47130 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46693⟩⟩) 0 ⟨7177⟩ 15500

def event47131 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46693⟩⟩) 1 ⟨46691⟩ 47129

def event47132 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨46693⟩⟩) (.authority (.operator))

def exact47133RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨46693⟩⟩]⟩, (1)⟩]

theorem exact47133RawTermsValid :
    exact47133RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event47133 : Event := .resultExact (⟨.program ⟨257⟩, ⟨46693⟩⟩) exact47133RawTerms .large 47132 .exactZero (none)

def event47134 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47549⟩⟩) 0 ⟨46693⟩ 47133

def event47135 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47549⟩⟩) (.authority (.operator))

def exact47136RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨47549⟩⟩]⟩, (1)⟩]

theorem exact47136RawTermsValid :
    exact47136RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event47136 : Event := .resultExact (⟨.program ⟨257⟩, ⟨47549⟩⟩) exact47136RawTerms (.finite 8192) 47135 .exactZero (none)

def event47137 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46516⟩⟩) 0 ⟨45348⟩ 1624

def event47138 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨46516⟩⟩) (.authority (.programFamilyFact))

def event47139 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨46516⟩⟩) (.finite 3720)

def event47140 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46517⟩⟩) 0 ⟨7177⟩ 15500

def event47141 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46517⟩⟩) 1 ⟨46516⟩ 47139

def event47142 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨46517⟩⟩) (.authority (.operator))

def exact47143RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨46517⟩⟩]⟩, (1)⟩]

theorem exact47143RawTermsValid :
    exact47143RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event47143 : Event := .resultExact (⟨.program ⟨257⟩, ⟨46517⟩⟩) exact47143RawTerms .large 47142 .exactZero (none)

def event47144 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47067⟩⟩) 0 ⟨46517⟩ 47143

def event47145 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47067⟩⟩) (.authority (.operator))

def exact47146RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨47067⟩⟩]⟩, (1)⟩]

theorem exact47146RawTermsValid :
    exact47146RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event47146 : Event := .resultExact (⟨.program ⟨257⟩, ⟨47067⟩⟩) exact47146RawTerms (.finite 8192) 47145 .exactZero (none)

def event47147 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45349⟩⟩) 0 ⟨45346⟩ 1613

def event47148 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45349⟩⟩) 1 ⟨11176⟩ 46653

def event47149 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45349⟩⟩) (.tensor (.predecessor 0 47147 .coefficient) (.predecessor 1 47148 .coefficient) true false)

def event47150 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨45349⟩⟩, .operator (⟨1613, 0⟩, ⟨46653, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨45346⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact47151RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨45346⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact47151RawTermsValid :
    exact47151RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event47151 : Event := .resultExact (⟨.program ⟨257⟩, ⟨45349⟩⟩) exact47151RawTerms .large 47149 .exactZero (none)

def event47152 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11190⟩⟩) 0 ⟨11175⟩ 46523

def event47153 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11190⟩⟩) 1 ⟨7284⟩ 17581

def event47154 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨11190⟩⟩) (.product (.predecessor 0 47152 .coefficient) (.predecessor 1 47153 .coefficient) (⟨false, false, none, none, none⟩))

def event47155 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨11190⟩⟩, .operator (⟨46523, 0⟩, ⟨17581, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7284⟩⟩]⟩, (1)⟩)

def exact47156RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7284⟩⟩]⟩, (1)⟩]

theorem exact47156RawTermsValid :
    exact47156RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event47156 : Event := .resultExact (⟨.program ⟨257⟩, ⟨11190⟩⟩) exact47156RawTerms .large 47154 .exactZero (none)

def event47157 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45350⟩⟩) 0 ⟨11190⟩ 47156

def event47158 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45350⟩⟩) 1 ⟨45349⟩ 47151

def event47159 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45350⟩⟩) (.sum [.predecessor 0 47157 .coefficient, .predecessor 1 47158 .coefficient])

def exact47160RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7284⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨45346⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact47160RawTermsValid :
    exact47160RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event47160 : Event := .resultExact (⟨.program ⟨257⟩, ⟨45350⟩⟩) exact47160RawTerms .large 47159 .exactZero (none)

def event47161 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45351⟩⟩) 0 ⟨45350⟩ 47160

def event47162 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45351⟩⟩) 1 ⟨110⟩ 17573

def event47163 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45351⟩⟩) (.sum [.predecessor 0 47161 .coefficient, .predecessor 1 47162 .coefficient])

def event47164 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45351⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨110⟩⟩]⟩) [⟨.result 17573 .coefficient, false, none⟩])

def event47165 : Event := .survivorFold (1) 47164

def exact47166RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7284⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨45346⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact47166RawTermsValid :
    exact47166RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event47166 : Event := .resultExact (⟨.program ⟨257⟩, ⟨45351⟩⟩) exact47166RawTerms .large 47163 (.finite 26) (some (47164))

def event47167 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45352⟩⟩) 0 ⟨45351⟩ 47166

def event47168 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45352⟩⟩) 1 ⟨14901⟩ 1616

def event47169 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45352⟩⟩) (.product (.predecessor 0 47167 .coefficient) (.predecessor 1 47168 .coefficient) (⟨false, true, none, none, some 1⟩))

def event47170 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45352⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨14901⟩⟩], []⟩) [⟨.result 1616 .coefficient, true, some 1⟩])

def event47171 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45352⟩⟩) (.product (.result 47166 .summary) (.transfer 47170) (⟨false, false, none, none, none⟩))

def event47172 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨45352⟩⟩, .operator (⟨47166, 1⟩, ⟨1616, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨14901⟩⟩, ⟨.program ⟨257⟩, ⟨45346⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def event47173 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨45352⟩⟩, .operator (⟨47166, 0⟩, ⟨1616, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨14901⟩⟩], [⟨.program ⟨257⟩, ⟨7284⟩⟩]⟩, (1)⟩)

def exact47174RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨14901⟩⟩], [⟨.program ⟨257⟩, ⟨7284⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨14901⟩⟩, ⟨.program ⟨257⟩, ⟨45346⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact47174RawTermsValid :
    exact47174RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event47174 : Event := .resultExact (⟨.program ⟨257⟩, ⟨45352⟩⟩) exact47174RawTerms .large 47169 (.finite 49414144) (some (47171))

def event47175 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14902⟩⟩) 0 ⟨14901⟩ 1616

def event47176 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14902⟩⟩) 1 ⟨11176⟩ 46653

def event47177 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14902⟩⟩) (.tensor (.predecessor 0 47175 .coefficient) (.predecessor 1 47176 .coefficient) true false)

def event47178 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨14902⟩⟩, .operator (⟨1616, 0⟩, ⟨46653, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨14901⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact47179RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨14901⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact47179RawTermsValid :
    exact47179RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event47179 : Event := .resultExact (⟨.program ⟨257⟩, ⟨14902⟩⟩) exact47179RawTerms .large 47177 .exactZero (none)

def event47180 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11207⟩⟩) 0 ⟨11175⟩ 46523

def event47181 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11207⟩⟩) 1 ⟨7301⟩ 17622

def event47182 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨11207⟩⟩) (.product (.predecessor 0 47180 .coefficient) (.predecessor 1 47181 .coefficient) (⟨false, false, none, none, none⟩))

def event47183 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨11207⟩⟩, .operator (⟨46523, 0⟩, ⟨17622, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7301⟩⟩]⟩, (1)⟩)

def exact47184RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7301⟩⟩]⟩, (1)⟩]

theorem exact47184RawTermsValid :
    exact47184RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event47184 : Event := .resultExact (⟨.program ⟨257⟩, ⟨11207⟩⟩) exact47184RawTerms .large 47182 .exactZero (none)

def event47185 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14903⟩⟩) 0 ⟨11207⟩ 47184

def event47186 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14903⟩⟩) 1 ⟨14902⟩ 47179

def event47187 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14903⟩⟩) (.sum [.predecessor 0 47185 .coefficient, .predecessor 1 47186 .coefficient])

def exact47188RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7301⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨14901⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact47188RawTermsValid :
    exact47188RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event47188 : Event := .resultExact (⟨.program ⟨257⟩, ⟨14903⟩⟩) exact47188RawTerms .large 47187 .exactZero (none)

def event47189 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14904⟩⟩) 0 ⟨14903⟩ 47188

def event47190 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14904⟩⟩) 1 ⟨127⟩ 17614

def event47191 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14904⟩⟩) (.sum [.predecessor 0 47189 .coefficient, .predecessor 1 47190 .coefficient])

def event47192 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14904⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨127⟩⟩]⟩) [⟨.result 17614 .coefficient, false, none⟩])

def event47193 : Event := .survivorFold (1) 47192

def exact47194RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7301⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨14901⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact47194RawTermsValid :
    exact47194RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event47194 : Event := .resultExact (⟨.program ⟨257⟩, ⟨14904⟩⟩) exact47194RawTerms .large 47191 (.finite 26) (some (47192))

def event47195 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14905⟩⟩) 0 ⟨14904⟩ 47194

def event47196 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14905⟩⟩) 1 ⟨9563⟩ 17611

def event47197 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14905⟩⟩) (.product (.predecessor 0 47195 .coefficient) (.predecessor 1 47196 .coefficient) (⟨false, false, none, none, none⟩))

def event47198 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14905⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨9562⟩⟩]⟩) [⟨.result 17607 .coefficient, false, none⟩])

def event47199 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14905⟩⟩) (.product (.result 47194 .summary) (.transfer 47198) (⟨false, false, none, none, none⟩))

def event47200 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨14905⟩⟩, .operator (⟨47194, 1⟩, ⟨17611, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨14901⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩]⟩, (-1)⟩)

def event47201 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨14905⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨14901⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨9562⟩⟩) ⟨7284⟩ 17581)

def event47202 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨14905⟩⟩, .relation 47201 0, ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨14901⟩⟩], [⟨.program ⟨257⟩, ⟨7284⟩⟩]⟩, (-1)⟩)

def event47203 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨14905⟩⟩, .operator (⟨47194, 0⟩, ⟨17611, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩]⟩, (1)⟩)

def exact47204RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨14901⟩⟩], [⟨.program ⟨257⟩, ⟨7284⟩⟩]⟩, (-1)⟩]

theorem exact47204RawTermsValid :
    exact47204RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event47204 : Event := .resultExact (⟨.program ⟨257⟩, ⟨14905⟩⟩) exact47204RawTerms .large 47197 (.finite 279172874240) (some (47199))

def event47205 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45353⟩⟩) 0 ⟨14905⟩ 47204

def event47206 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45353⟩⟩) 1 ⟨45352⟩ 47174

def event47207 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45353⟩⟩) (.sum [.predecessor 0 47205 .coefficient, .predecessor 1 47206 .coefficient])

def event47208 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨45353⟩⟩, .operator (⟨47204, 1⟩, ⟨47174, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨14901⟩⟩], [⟨.program ⟨257⟩, ⟨7284⟩⟩]⟩, (1)⟩)

def event47209 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45353⟩⟩) (.sum [.result 47204 .summary, .result 47174 .summary])

def exact47210RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨14901⟩⟩, ⟨.program ⟨257⟩, ⟨45346⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact47210RawTermsValid :
    exact47210RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event47210 : Event := .resultExact (⟨.program ⟨257⟩, ⟨45353⟩⟩) exact47210RawTerms .large 47207 (.finite 279222288384) (some (47209))

def event47211 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47068⟩⟩) 0 ⟨45353⟩ 47210

def event47212 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47068⟩⟩) 1 ⟨47067⟩ 47146

def event47213 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47068⟩⟩) (.product (.predecessor 0 47211 .coefficient) (.predecessor 1 47212 .coefficient) (⟨false, false, none, none, none⟩))

def event47214 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47068⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨47067⟩⟩]⟩) [⟨.result 47146 .coefficient, false, none⟩])

def event47215 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47068⟩⟩) (.product (.result 47210 .summary) (.transfer 47214) (⟨false, false, none, none, none⟩))

def event47216 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨47068⟩⟩, .operator (⟨47210, 1⟩, ⟨47146, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨14901⟩⟩, ⟨.program ⟨257⟩, ⟨45346⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨47067⟩⟩]⟩, (-1)⟩)

def event47217 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨47068⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨14901⟩⟩, ⟨.program ⟨257⟩, ⟨45346⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨47067⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨47067⟩⟩) ⟨46517⟩ 47143)

def event47218 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨47068⟩⟩, .relation 47217 0, ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨14901⟩⟩, ⟨.program ⟨257⟩, ⟨45346⟩⟩], [⟨.program ⟨257⟩, ⟨46517⟩⟩]⟩, (-1)⟩)

def event47219 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨47068⟩⟩, .operator (⟨47210, 0⟩, ⟨47146, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩, ⟨.program ⟨257⟩, ⟨47067⟩⟩]⟩, (1)⟩)

def exact47220RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩, ⟨.program ⟨257⟩, ⟨47067⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨14901⟩⟩, ⟨.program ⟨257⟩, ⟨45346⟩⟩], [⟨.program ⟨257⟩, ⟨46517⟩⟩]⟩, (-1)⟩]

theorem exact47220RawTermsValid :
    exact47220RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event47220 : Event := .resultExact (⟨.program ⟨257⟩, ⟨47068⟩⟩) exact47220RawTerms .large 47213 (.finite 2998126492308901724160) (some (47215))

def event47221 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45989⟩⟩) 0 ⟨45348⟩ 1624

def event47222 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45989⟩⟩) (.authority (.relationPreimageSource ⟨53⟩))

def exact47223RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨45989⟩⟩]⟩, (1)⟩]

theorem exact47223RawTermsValid :
    exact47223RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event47223 : Event := .resultExact (⟨.program ⟨257⟩, ⟨45989⟩⟩) exact47223RawTerms (.finite 5647228698) 47222 .exactZero (none)

def event47224 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45991⟩⟩) 0 ⟨45989⟩ 47223

def event47225 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45991⟩⟩) 1 ⟨2370⟩ 4

def event47226 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45991⟩⟩) (.scale (.predecessor 0 47224 .coefficient) (.value (.predecessor 1 47225 .coefficient)))

def exact47227RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨45989⟩⟩]⟩, (1)⟩]

theorem exact47227RawTermsValid :
    exact47227RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event47227 : Event := .resultExact (⟨.program ⟨257⟩, ⟨45991⟩⟩) exact47227RawTerms (.finite 5647228698) 47226 .exactZero (none)

def event47228 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45992⟩⟩) 0 ⟨11216⟩ 46745

def event47229 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45992⟩⟩) 1 ⟨45991⟩ 47227

def event47230 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45992⟩⟩) (.product (.predecessor 0 47228 .coefficient) (.predecessor 1 47229 .coefficient) (⟨false, false, none, none, none⟩))

def event47231 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45992⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨45989⟩⟩]⟩) [⟨.result 47223 .coefficient, false, none⟩])

def event47232 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45992⟩⟩) (.product (.result 46745 .summary) (.transfer 47231) (⟨false, false, none, none, none⟩))

def event47233 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨45992⟩⟩, .operator (⟨46745, 0⟩, ⟨47227, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨45989⟩⟩]⟩, (1)⟩)

def event47234 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨45990⟩⟩)

def event47235 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event47236 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event47237 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨11115⟩⟩) (.authority (.operator))

def event47238 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨11115⟩⟩) (.finite 17)

def event47239 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event47240 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event47241 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event47242 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event47243 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 47242

def event47244 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 47240

def event47245 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 47243 .coefficient) (.value (.predecessor 1 47244 .coefficient)))

def event47246 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event47247 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11117⟩⟩) 0 ⟨392⟩ 47246

def event47248 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11117⟩⟩) 1 ⟨11115⟩ 47238

def event47249 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨11117⟩⟩) (.sum [.predecessor 0 47247 .coefficient, .predecessor 1 47248 .coefficient])

def event47250 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨11117⟩⟩) (.finite 655357)

def event47251 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11173⟩⟩) 0 ⟨11117⟩ 47250

def event47252 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11173⟩⟩) 1 ⟨5426⟩ 47236

def event47253 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨11173⟩⟩) (.identity (.predecessor 1 47252 .coefficient))

def event47254 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨11173⟩⟩) (.finite 655360)

def event47255 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45346⟩⟩) 0 ⟨11173⟩ 47254

def event47256 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45346⟩⟩) (.authority (.programFamilyFact))

def exact47257RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨45346⟩⟩], []⟩, (1)⟩]

theorem exact47257RawTermsValid :
    exact47257RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event47257 : Event := .resultExact (⟨.program ⟨257⟩, ⟨45346⟩⟩) exact47257RawTerms (.finite 58) 47256 .exactZero (none)

def event47258 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14901⟩⟩) 0 ⟨11173⟩ 47254

def event47259 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14901⟩⟩) (.authority (.programFamilyFact))

def exact47260RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14901⟩⟩], []⟩, (1)⟩]

theorem exact47260RawTermsValid :
    exact47260RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event47260 : Event := .resultExact (⟨.program ⟨257⟩, ⟨14901⟩⟩) exact47260RawTerms (.finite 58) 47259 .exactZero (none)

def event47261 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45347⟩⟩) 0 ⟨14901⟩ 47260

def event47262 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45347⟩⟩) 1 ⟨45346⟩ 47257

def event47263 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45347⟩⟩) (.product (.predecessor 0 47261 .coefficient) (.predecessor 1 47262 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event47264 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45347⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨14901⟩⟩, ⟨.program ⟨257⟩, ⟨45346⟩⟩], []⟩) [⟨.result 47260 .coefficient, true, some 1⟩, ⟨.result 47257 .coefficient, true, some 1⟩])

def event47265 : Event := .survivorFold (1) 47264

def exact47266RawTerms : List Term := []

theorem exact47266RawTermsValid :
    exact47266RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event47266 : Event := .resultExact (⟨.program ⟨257⟩, ⟨45347⟩⟩) exact47266RawTerms (.finite 3364) 47263 (.finite 3364) (some (47264))

def event47267 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45348⟩⟩) 0 ⟨45347⟩ 47266

def event47268 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45348⟩⟩) (.identity (.predecessor 0 47267 .coefficient))

def event47269 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨45348⟩⟩) (.finite 3364)

def event47270 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45989⟩⟩) 0 ⟨45348⟩ 47269

def event47271 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45989⟩⟩) (.authority (.relationPreimageSource ⟨53⟩))

def exact47272RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨45989⟩⟩]⟩, (1)⟩]

theorem exact47272RawTermsValid :
    exact47272RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event47272 : Event := .resultExact (⟨.program ⟨257⟩, ⟨45989⟩⟩) exact47272RawTerms (.finite 5647228698) 47271 .exactZero (none)

def event47273 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35⟩⟩) (.authority (.operator))

def exact47274RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩]⟩, (1)⟩]

theorem exact47274RawTermsValid :
    exact47274RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event47274 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35⟩⟩) exact47274RawTerms .large 47273 .exactZero (none)

def event47275 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45990⟩⟩) 0 ⟨35⟩ 47274

def event47276 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45990⟩⟩) 1 ⟨45989⟩ 47272

def event47277 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45990⟩⟩) (.product (.predecessor 0 47275 .coefficient) (.predecessor 1 47276 .coefficient) (⟨false, false, none, none, none⟩))

def event47278 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨45990⟩⟩, .operator (⟨47274, 0⟩, ⟨47272, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨45989⟩⟩]⟩, (1)⟩)

def exact47279RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨45989⟩⟩]⟩, (1)⟩]

theorem exact47279RawTermsValid :
    exact47279RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event47279 : Event := .resultExact (⟨.program ⟨257⟩, ⟨45990⟩⟩) exact47279RawTerms .large 47277 .exactZero (none)

def event47280 : Event := .preFoldPolynomial 47279 [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨45989⟩⟩]⟩, (1)⟩] .exactZero none

def exact47281RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨45989⟩⟩]⟩, (1)⟩]

def event47281 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨45990⟩⟩) 47280 exact47281RawTerms .large 47277 .exactZero (none)

def event47282 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨47071⟩⟩)

def event47283 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event47284 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event47285 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨11115⟩⟩) (.authority (.operator))

def event47286 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨11115⟩⟩) (.finite 17)

def event47287 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event47288 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event47289 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event47290 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event47291 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 47290

def event47292 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 47288

def event47293 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 47291 .coefficient) (.value (.predecessor 1 47292 .coefficient)))

def event47294 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event47295 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11117⟩⟩) 0 ⟨392⟩ 47294

def event47296 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11117⟩⟩) 1 ⟨11115⟩ 47286

def event47297 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨11117⟩⟩) (.sum [.predecessor 0 47295 .coefficient, .predecessor 1 47296 .coefficient])

def event47298 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨11117⟩⟩) (.finite 655357)

def event47299 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11173⟩⟩) 0 ⟨11117⟩ 47298

def event47300 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11173⟩⟩) 1 ⟨5426⟩ 47284

def event47301 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨11173⟩⟩) (.identity (.predecessor 1 47300 .coefficient))

def event47302 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨11173⟩⟩) (.finite 655360)

def event47303 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45346⟩⟩) 0 ⟨11173⟩ 47302

def event47304 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45346⟩⟩) (.authority (.programFamilyFact))

def exact47305RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨45346⟩⟩], []⟩, (1)⟩]

theorem exact47305RawTermsValid :
    exact47305RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event47305 : Event := .resultExact (⟨.program ⟨257⟩, ⟨45346⟩⟩) exact47305RawTerms (.finite 58) 47304 .exactZero (none)

def event47306 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14901⟩⟩) 0 ⟨11173⟩ 47302

def event47307 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14901⟩⟩) (.authority (.programFamilyFact))

def exact47308RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14901⟩⟩], []⟩, (1)⟩]

theorem exact47308RawTermsValid :
    exact47308RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event47308 : Event := .resultExact (⟨.program ⟨257⟩, ⟨14901⟩⟩) exact47308RawTerms (.finite 58) 47307 .exactZero (none)

def event47309 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45347⟩⟩) 0 ⟨14901⟩ 47308

def event47310 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45347⟩⟩) 1 ⟨45346⟩ 47305

def event47311 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45347⟩⟩) (.product (.predecessor 0 47309 .coefficient) (.predecessor 1 47310 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event47312 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨45347⟩⟩, .operator (⟨47308, 0⟩, ⟨47305, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨14901⟩⟩, ⟨.program ⟨257⟩, ⟨45346⟩⟩], []⟩, (1)⟩)

def exact47313RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14901⟩⟩, ⟨.program ⟨257⟩, ⟨45346⟩⟩], []⟩, (1)⟩]

theorem exact47313RawTermsValid :
    exact47313RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event47313 : Event := .resultExact (⟨.program ⟨257⟩, ⟨45347⟩⟩) exact47313RawTerms (.finite 3364) 47311 .exactZero (none)

def event47314 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45348⟩⟩) 0 ⟨45347⟩ 47313

def event47315 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45348⟩⟩) (.identity (.predecessor 0 47314 .coefficient))

def event47316 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨45348⟩⟩) (.finite 3364)

def event47317 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46516⟩⟩) 0 ⟨45348⟩ 47316

def event47318 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨46516⟩⟩) (.authority (.programFamilyFact))

def event47319 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨46516⟩⟩) (.finite 3720)

def event47320 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨7177⟩⟩) .missing

def event47321 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46517⟩⟩) 0 ⟨7177⟩ 47320

def event47322 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46517⟩⟩) 1 ⟨46516⟩ 47319

def event47323 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨46517⟩⟩) (.authority (.operator))

def exact47324RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨46517⟩⟩]⟩, (1)⟩]

theorem exact47324RawTermsValid :
    exact47324RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event47324 : Event := .resultExact (⟨.program ⟨257⟩, ⟨46517⟩⟩) exact47324RawTerms .large 47323 .exactZero (none)

def event47325 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47067⟩⟩) 0 ⟨46517⟩ 47324

def event47326 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47067⟩⟩) (.authority (.operator))

def exact47327RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨47067⟩⟩]⟩, (1)⟩]

theorem exact47327RawTermsValid :
    exact47327RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event47327 : Event := .resultExact (⟨.program ⟨257⟩, ⟨47067⟩⟩) exact47327RawTerms (.finite 8192) 47326 .exactZero (none)

def event47328 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨136⟩⟩) (.authority (.operator))

def event47329 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨136⟩⟩) .exactZero

def event47330 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46778⟩⟩) 0 ⟨45348⟩ 47316

def event47331 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46778⟩⟩) 1 ⟨136⟩ 47329

def event47332 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨46778⟩⟩) (.sum [.predecessor 0 47330 .coefficient, .predecessor 1 47331 .coefficient])

def event47333 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨46778⟩⟩) (.finite 3364)

def event47334 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46779⟩⟩) 0 ⟨46778⟩ 47333

def event47335 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨46779⟩⟩) (.identity (.predecessor 0 47334 .coefficient))

def exact47336RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14901⟩⟩, ⟨.program ⟨257⟩, ⟨45346⟩⟩], []⟩, (1)⟩]

theorem exact47336RawTermsValid :
    exact47336RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event47336 : Event := .resultExact (⟨.program ⟨257⟩, ⟨46779⟩⟩) exact47336RawTerms (.finite 3364) 47335 .exactZero (none)

def event47337 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6908⟩⟩) (.authority (.factStore))

def exact47338RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact47338RawTermsValid :
    exact47338RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event47338 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6908⟩⟩) exact47338RawTerms .large 47337 .exactZero (none)

def event47339 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46780⟩⟩) 0 ⟨6908⟩ 47338

def event47340 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46780⟩⟩) 1 ⟨46779⟩ 47336

def event47341 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨46780⟩⟩) (.product (.predecessor 0 47339 .coefficient) (.predecessor 1 47340 .coefficient) (⟨false, false, none, none, none⟩))

def event47342 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨46780⟩⟩, .operator (⟨47338, 0⟩, ⟨47336, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨14901⟩⟩, ⟨.program ⟨257⟩, ⟨45346⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact47343RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14901⟩⟩, ⟨.program ⟨257⟩, ⟨45346⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact47343RawTermsValid :
    exact47343RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event47343 : Event := .resultExact (⟨.program ⟨257⟩, ⟨46780⟩⟩) exact47343RawTerms .large 47341 .exactZero (none)

def event47344 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨2370⟩⟩) (.authority (.operator))

def event47345 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨2370⟩⟩) (.finite 1)

def event47346 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7178⟩⟩) 0 ⟨7177⟩ 47320

def event47347 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7178⟩⟩) (.authority (.operator))

def exact47348RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7178⟩⟩]⟩, (1)⟩]

theorem exact47348RawTermsValid :
    exact47348RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event47348 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7178⟩⟩) exact47348RawTerms .large 47347 .exactZero (none)

def event47349 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7284⟩⟩) 0 ⟨7178⟩ 47348

def event47350 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7284⟩⟩) (.identity (.predecessor 0 47349 .coefficient))

def exact47351RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7284⟩⟩]⟩, (1)⟩]

theorem exact47351RawTermsValid :
    exact47351RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event47351 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7284⟩⟩) exact47351RawTerms .large 47350 .exactZero (none)

def event47352 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9562⟩⟩) 0 ⟨7284⟩ 47351

def event47353 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9562⟩⟩) (.authority (.operator))

def exact47354RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨9562⟩⟩]⟩, (1)⟩]

theorem exact47354RawTermsValid :
    exact47354RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event47354 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9562⟩⟩) exact47354RawTerms (.finite 8192) 47353 .exactZero (none)

def event47355 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9563⟩⟩) 0 ⟨9562⟩ 47354

def event47356 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9563⟩⟩) 1 ⟨2370⟩ 47345

def event47357 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9563⟩⟩) (.scale (.predecessor 0 47355 .coefficient) (.value (.predecessor 1 47356 .coefficient)))

def exact47358RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨9562⟩⟩]⟩, (1)⟩]

theorem exact47358RawTermsValid :
    exact47358RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event47358 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9563⟩⟩) exact47358RawTerms (.finite 8192) 47357 .exactZero (none)

def event47359 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7301⟩⟩) 0 ⟨7178⟩ 47348

def eventLeaf2944 : Array AnnotatedEvent := #[
  { event := event47104
    frameStart := 47009 },
  { event := event47105
    frameStart := 47009 },
  { event := event47106
    frameStart := 47009 },
  { event := event47107
    frameStart := 47009 },
  { event := event47108
    frameStart := 47009 },
  { event := event47109
    frameStart := 47009 },
  { event := event47110
    frameStart := 47009 },
  { event := event47111
    frameStart := 47009 },
  { event := event47112
    frameStart := 47009 },
  { event := event47113
    frameStart := 0 },
  { event := event47114
    frameStart := 0 },
  { event := event47115
    frameStart := 0 },
  { event := event47116
    frameStart := 0 },
  { event := event47117
    frameStart := 0 },
  { event := event47118
    frameStart := 0 },
  { event := event47119
    frameStart := 0 }
]

def eventLeaf2945 : Array AnnotatedEvent := #[
  { event := event47120
    frameStart := 0 },
  { event := event47121
    frameStart := 0 },
  { event := event47122
    frameStart := 0 },
  { event := event47123
    frameStart := 0 },
  { event := event47124
    frameStart := 0 },
  { event := event47125
    frameStart := 0 },
  { event := event47126
    frameStart := 0 },
  { event := event47127
    frameStart := 0 },
  { event := event47128
    frameStart := 0 },
  { event := event47129
    frameStart := 0 },
  { event := event47130
    frameStart := 0 },
  { event := event47131
    frameStart := 0 },
  { event := event47132
    frameStart := 0 },
  { event := event47133
    frameStart := 0 },
  { event := event47134
    frameStart := 0 },
  { event := event47135
    frameStart := 0 }
]

def eventLeaf2946 : Array AnnotatedEvent := #[
  { event := event47136
    frameStart := 0 },
  { event := event47137
    frameStart := 0 },
  { event := event47138
    frameStart := 0 },
  { event := event47139
    frameStart := 0 },
  { event := event47140
    frameStart := 0 },
  { event := event47141
    frameStart := 0 },
  { event := event47142
    frameStart := 0 },
  { event := event47143
    frameStart := 0 },
  { event := event47144
    frameStart := 0 },
  { event := event47145
    frameStart := 0 },
  { event := event47146
    frameStart := 0 },
  { event := event47147
    frameStart := 0 },
  { event := event47148
    frameStart := 0 },
  { event := event47149
    frameStart := 0 },
  { event := event47150
    frameStart := 0 },
  { event := event47151
    frameStart := 0 }
]

def eventLeaf2947 : Array AnnotatedEvent := #[
  { event := event47152
    frameStart := 0 },
  { event := event47153
    frameStart := 0 },
  { event := event47154
    frameStart := 0 },
  { event := event47155
    frameStart := 0 },
  { event := event47156
    frameStart := 0 },
  { event := event47157
    frameStart := 0 },
  { event := event47158
    frameStart := 0 },
  { event := event47159
    frameStart := 0 },
  { event := event47160
    frameStart := 0 },
  { event := event47161
    frameStart := 0 },
  { event := event47162
    frameStart := 0 },
  { event := event47163
    frameStart := 0 },
  { event := event47164
    frameStart := 0 },
  { event := event47165
    frameStart := 0 },
  { event := event47166
    frameStart := 0 },
  { event := event47167
    frameStart := 0 }
]

def eventLeaf2948 : Array AnnotatedEvent := #[
  { event := event47168
    frameStart := 0 },
  { event := event47169
    frameStart := 0 },
  { event := event47170
    frameStart := 0 },
  { event := event47171
    frameStart := 0 },
  { event := event47172
    frameStart := 0 },
  { event := event47173
    frameStart := 0 },
  { event := event47174
    frameStart := 0 },
  { event := event47175
    frameStart := 0 },
  { event := event47176
    frameStart := 0 },
  { event := event47177
    frameStart := 0 },
  { event := event47178
    frameStart := 0 },
  { event := event47179
    frameStart := 0 },
  { event := event47180
    frameStart := 0 },
  { event := event47181
    frameStart := 0 },
  { event := event47182
    frameStart := 0 },
  { event := event47183
    frameStart := 0 }
]

def eventLeaf2949 : Array AnnotatedEvent := #[
  { event := event47184
    frameStart := 0 },
  { event := event47185
    frameStart := 0 },
  { event := event47186
    frameStart := 0 },
  { event := event47187
    frameStart := 0 },
  { event := event47188
    frameStart := 0 },
  { event := event47189
    frameStart := 0 },
  { event := event47190
    frameStart := 0 },
  { event := event47191
    frameStart := 0 },
  { event := event47192
    frameStart := 0 },
  { event := event47193
    frameStart := 0 },
  { event := event47194
    frameStart := 0 },
  { event := event47195
    frameStart := 0 },
  { event := event47196
    frameStart := 0 },
  { event := event47197
    frameStart := 0 },
  { event := event47198
    frameStart := 0 },
  { event := event47199
    frameStart := 0 }
]

def eventLeaf2950 : Array AnnotatedEvent := #[
  { event := event47200
    frameStart := 0 },
  { event := event47201
    frameStart := 0 },
  { event := event47202
    frameStart := 0 },
  { event := event47203
    frameStart := 0 },
  { event := event47204
    frameStart := 0 },
  { event := event47205
    frameStart := 0 },
  { event := event47206
    frameStart := 0 },
  { event := event47207
    frameStart := 0 },
  { event := event47208
    frameStart := 0 },
  { event := event47209
    frameStart := 0 },
  { event := event47210
    frameStart := 0 },
  { event := event47211
    frameStart := 0 },
  { event := event47212
    frameStart := 0 },
  { event := event47213
    frameStart := 0 },
  { event := event47214
    frameStart := 0 },
  { event := event47215
    frameStart := 0 }
]

def eventLeaf2951 : Array AnnotatedEvent := #[
  { event := event47216
    frameStart := 0 },
  { event := event47217
    frameStart := 0 },
  { event := event47218
    frameStart := 0 },
  { event := event47219
    frameStart := 0 },
  { event := event47220
    frameStart := 0 },
  { event := event47221
    frameStart := 0 },
  { event := event47222
    frameStart := 0 },
  { event := event47223
    frameStart := 0 },
  { event := event47224
    frameStart := 0 },
  { event := event47225
    frameStart := 0 },
  { event := event47226
    frameStart := 0 },
  { event := event47227
    frameStart := 0 },
  { event := event47228
    frameStart := 0 },
  { event := event47229
    frameStart := 0 },
  { event := event47230
    frameStart := 0 },
  { event := event47231
    frameStart := 0 }
]

def eventLeaf2952 : Array AnnotatedEvent := #[
  { event := event47232
    frameStart := 0 },
  { event := event47233
    frameStart := 0 },
  { event := event47234
    frameStart := 47234 },
  { event := event47235
    frameStart := 47234 },
  { event := event47236
    frameStart := 47234 },
  { event := event47237
    frameStart := 47234 },
  { event := event47238
    frameStart := 47234 },
  { event := event47239
    frameStart := 47234 },
  { event := event47240
    frameStart := 47234 },
  { event := event47241
    frameStart := 47234 },
  { event := event47242
    frameStart := 47234 },
  { event := event47243
    frameStart := 47234 },
  { event := event47244
    frameStart := 47234 },
  { event := event47245
    frameStart := 47234 },
  { event := event47246
    frameStart := 47234 },
  { event := event47247
    frameStart := 47234 }
]

def eventLeaf2953 : Array AnnotatedEvent := #[
  { event := event47248
    frameStart := 47234 },
  { event := event47249
    frameStart := 47234 },
  { event := event47250
    frameStart := 47234 },
  { event := event47251
    frameStart := 47234 },
  { event := event47252
    frameStart := 47234 },
  { event := event47253
    frameStart := 47234 },
  { event := event47254
    frameStart := 47234 },
  { event := event47255
    frameStart := 47234 },
  { event := event47256
    frameStart := 47234 },
  { event := event47257
    frameStart := 47234 },
  { event := event47258
    frameStart := 47234 },
  { event := event47259
    frameStart := 47234 },
  { event := event47260
    frameStart := 47234 },
  { event := event47261
    frameStart := 47234 },
  { event := event47262
    frameStart := 47234 },
  { event := event47263
    frameStart := 47234 }
]

def eventLeaf2954 : Array AnnotatedEvent := #[
  { event := event47264
    frameStart := 47234 },
  { event := event47265
    frameStart := 47234 },
  { event := event47266
    frameStart := 47234 },
  { event := event47267
    frameStart := 47234 },
  { event := event47268
    frameStart := 47234 },
  { event := event47269
    frameStart := 47234 },
  { event := event47270
    frameStart := 47234 },
  { event := event47271
    frameStart := 47234 },
  { event := event47272
    frameStart := 47234 },
  { event := event47273
    frameStart := 47234 },
  { event := event47274
    frameStart := 47234 },
  { event := event47275
    frameStart := 47234 },
  { event := event47276
    frameStart := 47234 },
  { event := event47277
    frameStart := 47234 },
  { event := event47278
    frameStart := 47234 },
  { event := event47279
    frameStart := 47234 }
]

def eventLeaf2955 : Array AnnotatedEvent := #[
  { event := event47280
    frameStart := 47234 },
  { event := event47281
    frameStart := 47234 },
  { event := event47282
    frameStart := 47282 },
  { event := event47283
    frameStart := 47282 },
  { event := event47284
    frameStart := 47282 },
  { event := event47285
    frameStart := 47282 },
  { event := event47286
    frameStart := 47282 },
  { event := event47287
    frameStart := 47282 },
  { event := event47288
    frameStart := 47282 },
  { event := event47289
    frameStart := 47282 },
  { event := event47290
    frameStart := 47282 },
  { event := event47291
    frameStart := 47282 },
  { event := event47292
    frameStart := 47282 },
  { event := event47293
    frameStart := 47282 },
  { event := event47294
    frameStart := 47282 },
  { event := event47295
    frameStart := 47282 }
]

def eventLeaf2956 : Array AnnotatedEvent := #[
  { event := event47296
    frameStart := 47282 },
  { event := event47297
    frameStart := 47282 },
  { event := event47298
    frameStart := 47282 },
  { event := event47299
    frameStart := 47282 },
  { event := event47300
    frameStart := 47282 },
  { event := event47301
    frameStart := 47282 },
  { event := event47302
    frameStart := 47282 },
  { event := event47303
    frameStart := 47282 },
  { event := event47304
    frameStart := 47282 },
  { event := event47305
    frameStart := 47282 },
  { event := event47306
    frameStart := 47282 },
  { event := event47307
    frameStart := 47282 },
  { event := event47308
    frameStart := 47282 },
  { event := event47309
    frameStart := 47282 },
  { event := event47310
    frameStart := 47282 },
  { event := event47311
    frameStart := 47282 }
]

def eventLeaf2957 : Array AnnotatedEvent := #[
  { event := event47312
    frameStart := 47282 },
  { event := event47313
    frameStart := 47282 },
  { event := event47314
    frameStart := 47282 },
  { event := event47315
    frameStart := 47282 },
  { event := event47316
    frameStart := 47282 },
  { event := event47317
    frameStart := 47282 },
  { event := event47318
    frameStart := 47282 },
  { event := event47319
    frameStart := 47282 },
  { event := event47320
    frameStart := 47282 },
  { event := event47321
    frameStart := 47282 },
  { event := event47322
    frameStart := 47282 },
  { event := event47323
    frameStart := 47282 },
  { event := event47324
    frameStart := 47282 },
  { event := event47325
    frameStart := 47282 },
  { event := event47326
    frameStart := 47282 },
  { event := event47327
    frameStart := 47282 }
]

def eventLeaf2958 : Array AnnotatedEvent := #[
  { event := event47328
    frameStart := 47282 },
  { event := event47329
    frameStart := 47282 },
  { event := event47330
    frameStart := 47282 },
  { event := event47331
    frameStart := 47282 },
  { event := event47332
    frameStart := 47282 },
  { event := event47333
    frameStart := 47282 },
  { event := event47334
    frameStart := 47282 },
  { event := event47335
    frameStart := 47282 },
  { event := event47336
    frameStart := 47282 },
  { event := event47337
    frameStart := 47282 },
  { event := event47338
    frameStart := 47282 },
  { event := event47339
    frameStart := 47282 },
  { event := event47340
    frameStart := 47282 },
  { event := event47341
    frameStart := 47282 },
  { event := event47342
    frameStart := 47282 },
  { event := event47343
    frameStart := 47282 }
]

def eventLeaf2959 : Array AnnotatedEvent := #[
  { event := event47344
    frameStart := 47282 },
  { event := event47345
    frameStart := 47282 },
  { event := event47346
    frameStart := 47282 },
  { event := event47347
    frameStart := 47282 },
  { event := event47348
    frameStart := 47282 },
  { event := event47349
    frameStart := 47282 },
  { event := event47350
    frameStart := 47282 },
  { event := event47351
    frameStart := 47282 },
  { event := event47352
    frameStart := 47282 },
  { event := event47353
    frameStart := 47282 },
  { event := event47354
    frameStart := 47282 },
  { event := event47355
    frameStart := 47282 },
  { event := event47356
    frameStart := 47282 },
  { event := event47357
    frameStart := 47282 },
  { event := event47358
    frameStart := 47282 },
  { event := event47359
    frameStart := 47282 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events184
