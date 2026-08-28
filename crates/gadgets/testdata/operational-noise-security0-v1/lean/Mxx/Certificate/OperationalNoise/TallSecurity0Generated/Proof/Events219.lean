import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events219

open Mxx.Certificate.OperationalNoise
open CertificateABI

def exact56064RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨19460⟩⟩]⟩, (1)⟩]

theorem exact56064RawTermsValid :
    exact56064RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event56064 : Event := .resultExact (⟨.program ⟨214⟩, ⟨19462⟩⟩) exact56064RawTerms (.finite 136065468) 56063 .exactZero (none)

def event56065 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19463⟩⟩) 0 ⟨5547⟩ 50762

def event56066 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19463⟩⟩) 1 ⟨19462⟩ 56064

def event56067 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19463⟩⟩) (.product (.predecessor 0 56065 .coefficient) (.predecessor 1 56066 .coefficient) (⟨false, false, none, none, none⟩))

def event56068 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19463⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨19460⟩⟩]⟩) [⟨.result 56060 .coefficient, false, none⟩])

def event56069 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19463⟩⟩) (.product (.result 50762 .summary) (.transfer 56068) (⟨false, false, none, none, none⟩))

def event56070 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19463⟩⟩, .operator (⟨50762, 0⟩, ⟨56064, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19460⟩⟩]⟩, (1)⟩)

def event56071 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨19461⟩⟩)

def event56072 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event56073 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event56074 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨2875⟩⟩) (.authority (.operator))

def event56075 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨2875⟩⟩) (.finite 3)

def event56076 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event56077 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event56078 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event56079 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event56080 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 56079

def event56081 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 56077

def event56082 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 56080 .coefficient) (.value (.predecessor 1 56081 .coefficient)))

def event56083 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event56084 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5510⟩⟩) 0 ⟨5503⟩ 56083

def event56085 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5510⟩⟩) 1 ⟨2875⟩ 56075

def event56086 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5510⟩⟩) (.sum [.predecessor 0 56084 .coefficient, .predecessor 1 56085 .coefficient])

def event56087 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5510⟩⟩) (.finite 220)

def event56088 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5542⟩⟩) 0 ⟨5510⟩ 56087

def event56089 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5542⟩⟩) 1 ⟨961⟩ 56073

def event56090 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5542⟩⟩) (.identity (.predecessor 1 56089 .coefficient))

def event56091 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5542⟩⟩) (.finite 224)

def event56092 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11389⟩⟩) 0 ⟨5542⟩ 56091

def event56093 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11389⟩⟩) (.authority (.programFamilyFact))

def exact56094RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11389⟩⟩], []⟩, (1)⟩]

theorem exact56094RawTermsValid :
    exact56094RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event56094 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11389⟩⟩) exact56094RawTerms (.finite 16) 56093 .exactZero (none)

def event56095 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13999⟩⟩) 0 ⟨5542⟩ 56091

def event56096 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13999⟩⟩) (.authority (.programFamilyFact))

def exact56097RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨13999⟩⟩], []⟩, (1)⟩]

theorem exact56097RawTermsValid :
    exact56097RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event56097 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13999⟩⟩) exact56097RawTerms (.finite 16) 56096 .exactZero (none)

def event56098 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14000⟩⟩) 0 ⟨13999⟩ 56097

def event56099 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14000⟩⟩) 1 ⟨11389⟩ 56094

def event56100 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14000⟩⟩) (.product (.predecessor 0 56098 .coefficient) (.predecessor 1 56099 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event56101 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14000⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨11389⟩⟩, ⟨.program ⟨214⟩, ⟨13999⟩⟩], []⟩) [⟨.result 56097 .coefficient, true, some 1⟩, ⟨.result 56094 .coefficient, true, some 1⟩])

def event56102 : Event := .survivorFold (1) 56101

def exact56103RawTerms : List Term := []

theorem exact56103RawTermsValid :
    exact56103RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event56103 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14000⟩⟩) exact56103RawTerms (.finite 256) 56100 (.finite 256) (some (56101))

def event56104 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14001⟩⟩) 0 ⟨14000⟩ 56103

def event56105 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14001⟩⟩) (.identity (.predecessor 0 56104 .coefficient))

def event56106 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨14001⟩⟩) (.finite 256)

def event56107 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19460⟩⟩) 0 ⟨14001⟩ 56106

def event56108 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19460⟩⟩) (.authority (.relationPreimageSource ⟨14⟩))

def exact56109RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨19460⟩⟩]⟩, (1)⟩]

theorem exact56109RawTermsValid :
    exact56109RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event56109 : Event := .resultExact (⟨.program ⟨214⟩, ⟨19460⟩⟩) exact56109RawTerms (.finite 136065468) 56108 .exactZero (none)

def event56110 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6⟩⟩) (.authority (.operator))

def exact56111RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩]⟩, (1)⟩]

theorem exact56111RawTermsValid :
    exact56111RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event56111 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6⟩⟩) exact56111RawTerms .large 56110 .exactZero (none)

def event56112 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19461⟩⟩) 0 ⟨6⟩ 56111

def event56113 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19461⟩⟩) 1 ⟨19460⟩ 56109

def event56114 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19461⟩⟩) (.product (.predecessor 0 56112 .coefficient) (.predecessor 1 56113 .coefficient) (⟨false, false, none, none, none⟩))

def event56115 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19461⟩⟩, .operator (⟨56111, 0⟩, ⟨56109, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19460⟩⟩]⟩, (1)⟩)

def exact56116RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19460⟩⟩]⟩, (1)⟩]

theorem exact56116RawTermsValid :
    exact56116RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event56116 : Event := .resultExact (⟨.program ⟨214⟩, ⟨19461⟩⟩) exact56116RawTerms .large 56114 .exactZero (none)

def event56117 : Event := .preFoldPolynomial 56116 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19460⟩⟩]⟩, (1)⟩] .exactZero none

def exact56118RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19460⟩⟩]⟩, (1)⟩]

def event56118 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨19461⟩⟩) 56117 exact56118RawTerms .large 56114 .exactZero (none)

def event56119 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨25998⟩⟩)

def event56120 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event56121 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event56122 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨2875⟩⟩) (.authority (.operator))

def event56123 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨2875⟩⟩) (.finite 3)

def event56124 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event56125 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event56126 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event56127 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event56128 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 56127

def event56129 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 56125

def event56130 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 56128 .coefficient) (.value (.predecessor 1 56129 .coefficient)))

def event56131 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event56132 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5510⟩⟩) 0 ⟨5503⟩ 56131

def event56133 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5510⟩⟩) 1 ⟨2875⟩ 56123

def event56134 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5510⟩⟩) (.sum [.predecessor 0 56132 .coefficient, .predecessor 1 56133 .coefficient])

def event56135 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5510⟩⟩) (.finite 220)

def event56136 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5542⟩⟩) 0 ⟨5510⟩ 56135

def event56137 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5542⟩⟩) 1 ⟨961⟩ 56121

def event56138 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5542⟩⟩) (.identity (.predecessor 1 56137 .coefficient))

def event56139 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5542⟩⟩) (.finite 224)

def event56140 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11389⟩⟩) 0 ⟨5542⟩ 56139

def event56141 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11389⟩⟩) (.authority (.programFamilyFact))

def exact56142RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11389⟩⟩], []⟩, (1)⟩]

theorem exact56142RawTermsValid :
    exact56142RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event56142 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11389⟩⟩) exact56142RawTerms (.finite 16) 56141 .exactZero (none)

def event56143 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13999⟩⟩) 0 ⟨5542⟩ 56139

def event56144 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13999⟩⟩) (.authority (.programFamilyFact))

def exact56145RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨13999⟩⟩], []⟩, (1)⟩]

theorem exact56145RawTermsValid :
    exact56145RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event56145 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13999⟩⟩) exact56145RawTerms (.finite 16) 56144 .exactZero (none)

def event56146 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14000⟩⟩) 0 ⟨13999⟩ 56145

def event56147 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14000⟩⟩) 1 ⟨11389⟩ 56142

def event56148 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14000⟩⟩) (.product (.predecessor 0 56146 .coefficient) (.predecessor 1 56147 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event56149 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨14000⟩⟩, .operator (⟨56145, 0⟩, ⟨56142, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨11389⟩⟩, ⟨.program ⟨214⟩, ⟨13999⟩⟩], []⟩, (1)⟩)

def exact56150RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11389⟩⟩, ⟨.program ⟨214⟩, ⟨13999⟩⟩], []⟩, (1)⟩]

theorem exact56150RawTermsValid :
    exact56150RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event56150 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14000⟩⟩) exact56150RawTerms (.finite 256) 56148 .exactZero (none)

def event56151 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14001⟩⟩) 0 ⟨14000⟩ 56150

def event56152 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14001⟩⟩) (.identity (.predecessor 0 56151 .coefficient))

def event56153 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨14001⟩⟩) (.finite 256)

def event56154 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23543⟩⟩) 0 ⟨14001⟩ 56153

def event56155 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23543⟩⟩) (.authority (.programFamilyFact))

def event56156 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨23543⟩⟩) (.finite 3720)

def event56157 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨6689⟩⟩) .missing

def event56158 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23544⟩⟩) 0 ⟨6689⟩ 56157

def event56159 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23544⟩⟩) 1 ⟨23543⟩ 56156

def event56160 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23544⟩⟩) (.authority (.operator))

def exact56161RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨23544⟩⟩]⟩, (1)⟩]

theorem exact56161RawTermsValid :
    exact56161RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event56161 : Event := .resultExact (⟨.program ⟨214⟩, ⟨23544⟩⟩) exact56161RawTerms .large 56160 .exactZero (none)

def event56162 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25994⟩⟩) 0 ⟨23544⟩ 56161

def event56163 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25994⟩⟩) (.authority (.operator))

def exact56164RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨25994⟩⟩]⟩, (1)⟩]

theorem exact56164RawTermsValid :
    exact56164RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event56164 : Event := .resultExact (⟨.program ⟨214⟩, ⟨25994⟩⟩) exact56164RawTerms (.finite 8192) 56163 .exactZero (none)

def event56165 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨110⟩⟩) (.authority (.operator))

def event56166 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨110⟩⟩) .exactZero

def event56167 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14101⟩⟩) 0 ⟨14001⟩ 56153

def event56168 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14101⟩⟩) 1 ⟨110⟩ 56166

def event56169 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14101⟩⟩) (.sum [.predecessor 0 56167 .coefficient, .predecessor 1 56168 .coefficient])

def event56170 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨14101⟩⟩) (.finite 256)

def event56171 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14102⟩⟩) 0 ⟨14101⟩ 56170

def event56172 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14102⟩⟩) (.identity (.predecessor 0 56171 .coefficient))

def exact56173RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11389⟩⟩, ⟨.program ⟨214⟩, ⟨13999⟩⟩], []⟩, (1)⟩]

theorem exact56173RawTermsValid :
    exact56173RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event56173 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14102⟩⟩) exact56173RawTerms (.finite 256) 56172 .exactZero (none)

def event56174 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6544⟩⟩) (.authority (.factStore))

def exact56175RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact56175RawTermsValid :
    exact56175RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event56175 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6544⟩⟩) exact56175RawTerms .large 56174 .exactZero (none)

def event56176 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14103⟩⟩) 0 ⟨6544⟩ 56175

def event56177 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14103⟩⟩) 1 ⟨14102⟩ 56173

def event56178 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14103⟩⟩) (.product (.predecessor 0 56176 .coefficient) (.predecessor 1 56177 .coefficient) (⟨false, false, none, none, none⟩))

def event56179 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨14103⟩⟩, .operator (⟨56175, 0⟩, ⟨56173, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨11389⟩⟩, ⟨.program ⟨214⟩, ⟨13999⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact56180RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11389⟩⟩, ⟨.program ⟨214⟩, ⟨13999⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact56180RawTermsValid :
    exact56180RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event56180 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14103⟩⟩) exact56180RawTerms .large 56178 .exactZero (none)

def event56181 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨2348⟩⟩) (.authority (.operator))

def event56182 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨2348⟩⟩) (.finite 1)

def event56183 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6757⟩⟩) 0 ⟨6689⟩ 56157

def event56184 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6757⟩⟩) (.authority (.operator))

def exact56185RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6757⟩⟩]⟩, (1)⟩]

theorem exact56185RawTermsValid :
    exact56185RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event56185 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6757⟩⟩) exact56185RawTerms .large 56184 .exactZero (none)

def event56186 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6778⟩⟩) 0 ⟨6757⟩ 56185

def event56187 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6778⟩⟩) (.identity (.predecessor 0 56186 .coefficient))

def exact56188RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6778⟩⟩]⟩, (1)⟩]

theorem exact56188RawTermsValid :
    exact56188RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event56188 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6778⟩⟩) exact56188RawTerms .large 56187 .exactZero (none)

def event56189 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7849⟩⟩) 0 ⟨6778⟩ 56188

def event56190 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7849⟩⟩) (.authority (.operator))

def exact56191RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨7849⟩⟩]⟩, (1)⟩]

theorem exact56191RawTermsValid :
    exact56191RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event56191 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7849⟩⟩) exact56191RawTerms (.finite 8192) 56190 .exactZero (none)

def event56192 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7850⟩⟩) 0 ⟨7849⟩ 56191

def event56193 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7850⟩⟩) 1 ⟨2348⟩ 56182

def event56194 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7850⟩⟩) (.scale (.predecessor 0 56192 .coefficient) (.value (.predecessor 1 56193 .coefficient)))

def exact56195RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨7849⟩⟩]⟩, (1)⟩]

theorem exact56195RawTermsValid :
    exact56195RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event56195 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7850⟩⟩) exact56195RawTerms (.finite 8192) 56194 .exactZero (none)

def event56196 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6758⟩⟩) 0 ⟨6757⟩ 56185

def event56197 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6758⟩⟩) (.identity (.predecessor 0 56196 .coefficient))

def exact56198RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6758⟩⟩]⟩, (1)⟩]

theorem exact56198RawTermsValid :
    exact56198RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event56198 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6758⟩⟩) exact56198RawTerms .large 56197 .exactZero (none)

def event56199 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7851⟩⟩) 0 ⟨6758⟩ 56198

def event56200 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7851⟩⟩) 1 ⟨7850⟩ 56195

def event56201 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7851⟩⟩) (.product (.predecessor 0 56199 .coefficient) (.predecessor 1 56200 .coefficient) (⟨false, false, none, none, none⟩))

def event56202 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7851⟩⟩, .operator (⟨56198, 0⟩, ⟨56195, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6758⟩⟩, ⟨.program ⟨214⟩, ⟨7849⟩⟩]⟩, (1)⟩)

def exact56203RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6758⟩⟩, ⟨.program ⟨214⟩, ⟨7849⟩⟩]⟩, (1)⟩]

theorem exact56203RawTermsValid :
    exact56203RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event56203 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7851⟩⟩) exact56203RawTerms .large 56201 .exactZero (none)

def event56204 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14104⟩⟩) 0 ⟨7851⟩ 56203

def event56205 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14104⟩⟩) 1 ⟨14103⟩ 56180

def event56206 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14104⟩⟩) (.sum [.predecessor 0 56204 .coefficient, .predecessor 1 56205 .coefficient])

def exact56207RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6758⟩⟩, ⟨.program ⟨214⟩, ⟨7849⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨11389⟩⟩, ⟨.program ⟨214⟩, ⟨13999⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact56207RawTermsValid :
    exact56207RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event56207 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14104⟩⟩) exact56207RawTerms .large 56206 .exactZero (none)

def event56208 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25997⟩⟩) 0 ⟨14104⟩ 56207

def event56209 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25997⟩⟩) 1 ⟨25994⟩ 56164

def event56210 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25997⟩⟩) (.product (.predecessor 0 56208 .coefficient) (.predecessor 1 56209 .coefficient) (⟨false, false, none, none, none⟩))

def event56211 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25997⟩⟩, .operator (⟨56207, 0⟩, ⟨56164, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6758⟩⟩, ⟨.program ⟨214⟩, ⟨7849⟩⟩, ⟨.program ⟨214⟩, ⟨25994⟩⟩]⟩, (1)⟩)

def event56212 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25997⟩⟩, .operator (⟨56207, 1⟩, ⟨56164, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨11389⟩⟩, ⟨.program ⟨214⟩, ⟨13999⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25994⟩⟩]⟩, (-1)⟩)

def event56213 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨25997⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨11389⟩⟩, ⟨.program ⟨214⟩, ⟨13999⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25994⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨25994⟩⟩) ⟨23544⟩ 56161)

def event56214 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25997⟩⟩, .relation 56213 0, ⟨[⟨.program ⟨214⟩, ⟨11389⟩⟩, ⟨.program ⟨214⟩, ⟨13999⟩⟩], [⟨.program ⟨214⟩, ⟨23544⟩⟩]⟩, (-1)⟩)

def exact56215RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6758⟩⟩, ⟨.program ⟨214⟩, ⟨7849⟩⟩, ⟨.program ⟨214⟩, ⟨25994⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨11389⟩⟩, ⟨.program ⟨214⟩, ⟨13999⟩⟩], [⟨.program ⟨214⟩, ⟨23544⟩⟩]⟩, (-1)⟩]

theorem exact56215RawTermsValid :
    exact56215RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event56215 : Event := .resultExact (⟨.program ⟨214⟩, ⟨25997⟩⟩) exact56215RawTerms .large 56210 .exactZero (none)

def event56216 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15825⟩⟩) 0 ⟨14001⟩ 56153

def event56217 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15825⟩⟩) (.authority (.programFamilyFact))

def exact56218RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15825⟩⟩], []⟩, (1)⟩]

theorem exact56218RawTermsValid :
    exact56218RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event56218 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15825⟩⟩) exact56218RawTerms (.finite 16) 56217 .exactZero (none)

def event56219 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15827⟩⟩) 0 ⟨6544⟩ 56175

def event56220 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15827⟩⟩) 1 ⟨15825⟩ 56218

def event56221 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15827⟩⟩) (.product (.predecessor 0 56219 .coefficient) (.predecessor 1 56220 .coefficient) (⟨false, true, none, none, some 1⟩))

def event56222 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨15827⟩⟩, .operator (⟨56175, 0⟩, ⟨56218, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨15825⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact56223RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15825⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact56223RawTermsValid :
    exact56223RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event56223 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15827⟩⟩) exact56223RawTerms .large 56221 .exactZero (none)

def event56224 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6696⟩⟩) 0 ⟨6689⟩ 56157

def event56225 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6696⟩⟩) (.authority (.operator))

def exact56226RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6696⟩⟩]⟩, (1)⟩]

theorem exact56226RawTermsValid :
    exact56226RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event56226 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6696⟩⟩) exact56226RawTerms .large 56225 .exactZero (none)

def event56227 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15828⟩⟩) 0 ⟨6696⟩ 56226

def event56228 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15828⟩⟩) 1 ⟨15827⟩ 56223

def event56229 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15828⟩⟩) (.sum [.predecessor 0 56227 .coefficient, .predecessor 1 56228 .coefficient])

def exact56230RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6696⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15825⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact56230RawTermsValid :
    exact56230RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event56230 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15828⟩⟩) exact56230RawTerms .large 56229 .exactZero (none)

def event56231 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25998⟩⟩) 0 ⟨15828⟩ 56230

def event56232 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25998⟩⟩) 1 ⟨25997⟩ 56215

def event56233 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25998⟩⟩) (.sum [.predecessor 0 56231 .coefficient, .predecessor 1 56232 .coefficient])

def exact56234RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6696⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6758⟩⟩, ⟨.program ⟨214⟩, ⟨7849⟩⟩, ⟨.program ⟨214⟩, ⟨25994⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨11389⟩⟩, ⟨.program ⟨214⟩, ⟨13999⟩⟩], [⟨.program ⟨214⟩, ⟨23544⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15825⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact56234RawTermsValid :
    exact56234RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event56234 : Event := .resultExact (⟨.program ⟨214⟩, ⟨25998⟩⟩) exact56234RawTerms .large 56233 .exactZero (none)

def event56235 : Event := .preFoldPolynomial 56234 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6696⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6758⟩⟩, ⟨.program ⟨214⟩, ⟨7849⟩⟩, ⟨.program ⟨214⟩, ⟨25994⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨11389⟩⟩, ⟨.program ⟨214⟩, ⟨13999⟩⟩], [⟨.program ⟨214⟩, ⟨23544⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15825⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩] .exactZero none

def exact56236RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6696⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6758⟩⟩, ⟨.program ⟨214⟩, ⟨7849⟩⟩, ⟨.program ⟨214⟩, ⟨25994⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨11389⟩⟩, ⟨.program ⟨214⟩, ⟨13999⟩⟩], [⟨.program ⟨214⟩, ⟨23544⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15825⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

def event56236 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨25998⟩⟩) 56235 exact56236RawTerms .large 56233 .exactZero (none)

def event56237 : Event := .specializationComputed (⟨.program ⟨214⟩, ⟨14001⟩⟩) ⟨⟨109⟩, ⟨14⟩, ⟨109⟩⟩ ⟨56071, 56237⟩

def event56238 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨19463⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19460⟩⟩]⟩) (1) 0 2 (.universal 56237 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19460⟩⟩]⟩) (none) 56236)

def event56239 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19463⟩⟩, .relation 56238 0, ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6696⟩⟩]⟩, (1)⟩)

def event56240 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19463⟩⟩, .relation 56238 1, ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6758⟩⟩, ⟨.program ⟨214⟩, ⟨7849⟩⟩, ⟨.program ⟨214⟩, ⟨25994⟩⟩]⟩, (-1)⟩)

def event56241 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19463⟩⟩, .relation 56238 2, ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨11389⟩⟩, ⟨.program ⟨214⟩, ⟨13999⟩⟩], [⟨.program ⟨214⟩, ⟨23544⟩⟩]⟩, (1)⟩)

def event56242 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19463⟩⟩, .relation 56238 3, ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨15825⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def exact56243RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6696⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6758⟩⟩, ⟨.program ⟨214⟩, ⟨7849⟩⟩, ⟨.program ⟨214⟩, ⟨25994⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨11389⟩⟩, ⟨.program ⟨214⟩, ⟨13999⟩⟩], [⟨.program ⟨214⟩, ⟨23544⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨15825⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact56243RawTermsValid :
    exact56243RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event56243 : Event := .resultExact (⟨.program ⟨214⟩, ⟨19463⟩⟩) exact56243RawTerms .large 56067 (.finite 1811303510016) (some (56069))

def event56244 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25996⟩⟩) 0 ⟨19463⟩ 56243

def event56245 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25996⟩⟩) 1 ⟨25995⟩ 56057

def event56246 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25996⟩⟩) (.sum [.predecessor 0 56244 .coefficient, .predecessor 1 56245 .coefficient])

def event56247 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25996⟩⟩, .operator (⟨56243, 2⟩, ⟨56057, 1⟩), ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨11389⟩⟩, ⟨.program ⟨214⟩, ⟨13999⟩⟩], [⟨.program ⟨214⟩, ⟨23544⟩⟩]⟩, (-1)⟩)

def event56248 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25996⟩⟩, .operator (⟨56243, 1⟩, ⟨56057, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6758⟩⟩, ⟨.program ⟨214⟩, ⟨7849⟩⟩, ⟨.program ⟨214⟩, ⟨25994⟩⟩]⟩, (1)⟩)

def event56249 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25996⟩⟩) (.sum [.result 56243 .summary, .result 56057 .summary])

def exact56250RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6696⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨15825⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact56250RawTermsValid :
    exact56250RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event56250 : Event := .resultExact (⟨.program ⟨214⟩, ⟨25996⟩⟩) exact56250RawTerms .large 56246 (.finite 352054612209664) (some (56249))

def event56251 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27664⟩⟩) 0 ⟨25996⟩ 56250

def event56252 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27664⟩⟩) 1 ⟨27662⟩ 55973

def event56253 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27664⟩⟩) (.product (.predecessor 0 56251 .coefficient) (.predecessor 1 56252 .coefficient) (⟨false, false, none, none, none⟩))

def event56254 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27664⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨27662⟩⟩]⟩) [⟨.result 55973 .coefficient, false, none⟩])

def event56255 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27664⟩⟩) (.product (.result 56250 .summary) (.transfer 56254) (⟨false, false, none, none, none⟩))

def event56256 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27664⟩⟩, .operator (⟨56250, 0⟩, ⟨55973, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6696⟩⟩, ⟨.program ⟨214⟩, ⟨27662⟩⟩]⟩, (1)⟩)

def event56257 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27664⟩⟩, .operator (⟨56250, 1⟩, ⟨55973, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨15825⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨27662⟩⟩]⟩, (-1)⟩)

def event56258 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨27664⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨15825⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨27662⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨27662⟩⟩) ⟨24102⟩ 55970)

def event56259 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27664⟩⟩, .relation 56258 0, ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨15825⟩⟩], [⟨.program ⟨214⟩, ⟨24102⟩⟩]⟩, (-1)⟩)

def exact56260RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6696⟩⟩, ⟨.program ⟨214⟩, ⟨27662⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨15825⟩⟩], [⟨.program ⟨214⟩, ⟨24102⟩⟩]⟩, (-1)⟩]

theorem exact56260RawTermsValid :
    exact56260RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event56260 : Event := .resultExact (⟨.program ⟨214⟩, ⟨27664⟩⟩) exact56260RawTerms .large 56253 (.finite 1292046059683262234624) (some (56255))

def event56261 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21260⟩⟩) 0 ⟨15826⟩ 2608

def event56262 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21260⟩⟩) (.authority (.relationPreimageSource ⟨41⟩))

def exact56263RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨21260⟩⟩]⟩, (1)⟩]

theorem exact56263RawTermsValid :
    exact56263RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event56263 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21260⟩⟩) exact56263RawTerms (.finite 136065468) 56262 .exactZero (none)

def event56264 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21262⟩⟩) 0 ⟨21260⟩ 56263

def event56265 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21262⟩⟩) 1 ⟨2348⟩ 4

def event56266 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21262⟩⟩) (.scale (.predecessor 0 56264 .coefficient) (.value (.predecessor 1 56265 .coefficient)))

def exact56267RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨21260⟩⟩]⟩, (1)⟩]

theorem exact56267RawTermsValid :
    exact56267RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event56267 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21262⟩⟩) exact56267RawTerms (.finite 136065468) 56266 .exactZero (none)

def event56268 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21263⟩⟩) 0 ⟨5547⟩ 50762

def event56269 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21263⟩⟩) 1 ⟨21262⟩ 56267

def event56270 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21263⟩⟩) (.product (.predecessor 0 56268 .coefficient) (.predecessor 1 56269 .coefficient) (⟨false, false, none, none, none⟩))

def event56271 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21263⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨21260⟩⟩]⟩) [⟨.result 56263 .coefficient, false, none⟩])

def event56272 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21263⟩⟩) (.product (.result 50762 .summary) (.transfer 56271) (⟨false, false, none, none, none⟩))

def event56273 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21263⟩⟩, .operator (⟨50762, 0⟩, ⟨56267, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21260⟩⟩]⟩, (1)⟩)

def event56274 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨21261⟩⟩)

def event56275 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event56276 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event56277 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨2875⟩⟩) (.authority (.operator))

def event56278 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨2875⟩⟩) (.finite 3)

def event56279 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event56280 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event56281 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event56282 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event56283 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 56282

def event56284 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 56280

def event56285 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 56283 .coefficient) (.value (.predecessor 1 56284 .coefficient)))

def event56286 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event56287 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5510⟩⟩) 0 ⟨5503⟩ 56286

def event56288 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5510⟩⟩) 1 ⟨2875⟩ 56278

def event56289 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5510⟩⟩) (.sum [.predecessor 0 56287 .coefficient, .predecessor 1 56288 .coefficient])

def event56290 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5510⟩⟩) (.finite 220)

def event56291 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5542⟩⟩) 0 ⟨5510⟩ 56290

def event56292 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5542⟩⟩) 1 ⟨961⟩ 56276

def event56293 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5542⟩⟩) (.identity (.predecessor 1 56292 .coefficient))

def event56294 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5542⟩⟩) (.finite 224)

def event56295 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11389⟩⟩) 0 ⟨5542⟩ 56294

def event56296 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11389⟩⟩) (.authority (.programFamilyFact))

def exact56297RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11389⟩⟩], []⟩, (1)⟩]

theorem exact56297RawTermsValid :
    exact56297RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event56297 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11389⟩⟩) exact56297RawTerms (.finite 16) 56296 .exactZero (none)

def event56298 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13999⟩⟩) 0 ⟨5542⟩ 56294

def event56299 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13999⟩⟩) (.authority (.programFamilyFact))

def exact56300RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨13999⟩⟩], []⟩, (1)⟩]

theorem exact56300RawTermsValid :
    exact56300RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event56300 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13999⟩⟩) exact56300RawTerms (.finite 16) 56299 .exactZero (none)

def event56301 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14000⟩⟩) 0 ⟨13999⟩ 56300

def event56302 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14000⟩⟩) 1 ⟨11389⟩ 56297

def event56303 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14000⟩⟩) (.product (.predecessor 0 56301 .coefficient) (.predecessor 1 56302 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event56304 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14000⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨11389⟩⟩, ⟨.program ⟨214⟩, ⟨13999⟩⟩], []⟩) [⟨.result 56300 .coefficient, true, some 1⟩, ⟨.result 56297 .coefficient, true, some 1⟩])

def event56305 : Event := .survivorFold (1) 56304

def exact56306RawTerms : List Term := []

theorem exact56306RawTermsValid :
    exact56306RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event56306 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14000⟩⟩) exact56306RawTerms (.finite 256) 56303 (.finite 256) (some (56304))

def event56307 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14001⟩⟩) 0 ⟨14000⟩ 56306

def event56308 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14001⟩⟩) (.identity (.predecessor 0 56307 .coefficient))

def event56309 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨14001⟩⟩) (.finite 256)

def event56310 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15825⟩⟩) 0 ⟨14001⟩ 56309

def event56311 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15825⟩⟩) (.authority (.programFamilyFact))

def exact56312RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15825⟩⟩], []⟩, (1)⟩]

theorem exact56312RawTermsValid :
    exact56312RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event56312 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15825⟩⟩) exact56312RawTerms (.finite 16) 56311 .exactZero (none)

def event56313 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15826⟩⟩) 0 ⟨15825⟩ 56312

def event56314 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15826⟩⟩) (.identity (.predecessor 0 56313 .coefficient))

def event56315 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨15826⟩⟩) (.finite 16)

def event56316 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21260⟩⟩) 0 ⟨15826⟩ 56315

def event56317 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21260⟩⟩) (.authority (.relationPreimageSource ⟨41⟩))

def exact56318RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨21260⟩⟩]⟩, (1)⟩]

theorem exact56318RawTermsValid :
    exact56318RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event56318 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21260⟩⟩) exact56318RawTerms (.finite 136065468) 56317 .exactZero (none)

def event56319 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6⟩⟩) (.authority (.operator))

def eventLeaf3504 : Array AnnotatedEvent := #[
  { event := event56064
    frameStart := 0 },
  { event := event56065
    frameStart := 0 },
  { event := event56066
    frameStart := 0 },
  { event := event56067
    frameStart := 0 },
  { event := event56068
    frameStart := 0 },
  { event := event56069
    frameStart := 0 },
  { event := event56070
    frameStart := 0 },
  { event := event56071
    frameStart := 56071 },
  { event := event56072
    frameStart := 56071 },
  { event := event56073
    frameStart := 56071 },
  { event := event56074
    frameStart := 56071 },
  { event := event56075
    frameStart := 56071 },
  { event := event56076
    frameStart := 56071 },
  { event := event56077
    frameStart := 56071 },
  { event := event56078
    frameStart := 56071 },
  { event := event56079
    frameStart := 56071 }
]

def eventLeaf3505 : Array AnnotatedEvent := #[
  { event := event56080
    frameStart := 56071 },
  { event := event56081
    frameStart := 56071 },
  { event := event56082
    frameStart := 56071 },
  { event := event56083
    frameStart := 56071 },
  { event := event56084
    frameStart := 56071 },
  { event := event56085
    frameStart := 56071 },
  { event := event56086
    frameStart := 56071 },
  { event := event56087
    frameStart := 56071 },
  { event := event56088
    frameStart := 56071 },
  { event := event56089
    frameStart := 56071 },
  { event := event56090
    frameStart := 56071 },
  { event := event56091
    frameStart := 56071 },
  { event := event56092
    frameStart := 56071 },
  { event := event56093
    frameStart := 56071 },
  { event := event56094
    frameStart := 56071 },
  { event := event56095
    frameStart := 56071 }
]

def eventLeaf3506 : Array AnnotatedEvent := #[
  { event := event56096
    frameStart := 56071 },
  { event := event56097
    frameStart := 56071 },
  { event := event56098
    frameStart := 56071 },
  { event := event56099
    frameStart := 56071 },
  { event := event56100
    frameStart := 56071 },
  { event := event56101
    frameStart := 56071 },
  { event := event56102
    frameStart := 56071 },
  { event := event56103
    frameStart := 56071 },
  { event := event56104
    frameStart := 56071 },
  { event := event56105
    frameStart := 56071 },
  { event := event56106
    frameStart := 56071 },
  { event := event56107
    frameStart := 56071 },
  { event := event56108
    frameStart := 56071 },
  { event := event56109
    frameStart := 56071 },
  { event := event56110
    frameStart := 56071 },
  { event := event56111
    frameStart := 56071 }
]

def eventLeaf3507 : Array AnnotatedEvent := #[
  { event := event56112
    frameStart := 56071 },
  { event := event56113
    frameStart := 56071 },
  { event := event56114
    frameStart := 56071 },
  { event := event56115
    frameStart := 56071 },
  { event := event56116
    frameStart := 56071 },
  { event := event56117
    frameStart := 56071 },
  { event := event56118
    frameStart := 56071 },
  { event := event56119
    frameStart := 56119 },
  { event := event56120
    frameStart := 56119 },
  { event := event56121
    frameStart := 56119 },
  { event := event56122
    frameStart := 56119 },
  { event := event56123
    frameStart := 56119 },
  { event := event56124
    frameStart := 56119 },
  { event := event56125
    frameStart := 56119 },
  { event := event56126
    frameStart := 56119 },
  { event := event56127
    frameStart := 56119 }
]

def eventLeaf3508 : Array AnnotatedEvent := #[
  { event := event56128
    frameStart := 56119 },
  { event := event56129
    frameStart := 56119 },
  { event := event56130
    frameStart := 56119 },
  { event := event56131
    frameStart := 56119 },
  { event := event56132
    frameStart := 56119 },
  { event := event56133
    frameStart := 56119 },
  { event := event56134
    frameStart := 56119 },
  { event := event56135
    frameStart := 56119 },
  { event := event56136
    frameStart := 56119 },
  { event := event56137
    frameStart := 56119 },
  { event := event56138
    frameStart := 56119 },
  { event := event56139
    frameStart := 56119 },
  { event := event56140
    frameStart := 56119 },
  { event := event56141
    frameStart := 56119 },
  { event := event56142
    frameStart := 56119 },
  { event := event56143
    frameStart := 56119 }
]

def eventLeaf3509 : Array AnnotatedEvent := #[
  { event := event56144
    frameStart := 56119 },
  { event := event56145
    frameStart := 56119 },
  { event := event56146
    frameStart := 56119 },
  { event := event56147
    frameStart := 56119 },
  { event := event56148
    frameStart := 56119 },
  { event := event56149
    frameStart := 56119 },
  { event := event56150
    frameStart := 56119 },
  { event := event56151
    frameStart := 56119 },
  { event := event56152
    frameStart := 56119 },
  { event := event56153
    frameStart := 56119 },
  { event := event56154
    frameStart := 56119 },
  { event := event56155
    frameStart := 56119 },
  { event := event56156
    frameStart := 56119 },
  { event := event56157
    frameStart := 56119 },
  { event := event56158
    frameStart := 56119 },
  { event := event56159
    frameStart := 56119 }
]

def eventLeaf3510 : Array AnnotatedEvent := #[
  { event := event56160
    frameStart := 56119 },
  { event := event56161
    frameStart := 56119 },
  { event := event56162
    frameStart := 56119 },
  { event := event56163
    frameStart := 56119 },
  { event := event56164
    frameStart := 56119 },
  { event := event56165
    frameStart := 56119 },
  { event := event56166
    frameStart := 56119 },
  { event := event56167
    frameStart := 56119 },
  { event := event56168
    frameStart := 56119 },
  { event := event56169
    frameStart := 56119 },
  { event := event56170
    frameStart := 56119 },
  { event := event56171
    frameStart := 56119 },
  { event := event56172
    frameStart := 56119 },
  { event := event56173
    frameStart := 56119 },
  { event := event56174
    frameStart := 56119 },
  { event := event56175
    frameStart := 56119 }
]

def eventLeaf3511 : Array AnnotatedEvent := #[
  { event := event56176
    frameStart := 56119 },
  { event := event56177
    frameStart := 56119 },
  { event := event56178
    frameStart := 56119 },
  { event := event56179
    frameStart := 56119 },
  { event := event56180
    frameStart := 56119 },
  { event := event56181
    frameStart := 56119 },
  { event := event56182
    frameStart := 56119 },
  { event := event56183
    frameStart := 56119 },
  { event := event56184
    frameStart := 56119 },
  { event := event56185
    frameStart := 56119 },
  { event := event56186
    frameStart := 56119 },
  { event := event56187
    frameStart := 56119 },
  { event := event56188
    frameStart := 56119 },
  { event := event56189
    frameStart := 56119 },
  { event := event56190
    frameStart := 56119 },
  { event := event56191
    frameStart := 56119 }
]

def eventLeaf3512 : Array AnnotatedEvent := #[
  { event := event56192
    frameStart := 56119 },
  { event := event56193
    frameStart := 56119 },
  { event := event56194
    frameStart := 56119 },
  { event := event56195
    frameStart := 56119 },
  { event := event56196
    frameStart := 56119 },
  { event := event56197
    frameStart := 56119 },
  { event := event56198
    frameStart := 56119 },
  { event := event56199
    frameStart := 56119 },
  { event := event56200
    frameStart := 56119 },
  { event := event56201
    frameStart := 56119 },
  { event := event56202
    frameStart := 56119 },
  { event := event56203
    frameStart := 56119 },
  { event := event56204
    frameStart := 56119 },
  { event := event56205
    frameStart := 56119 },
  { event := event56206
    frameStart := 56119 },
  { event := event56207
    frameStart := 56119 }
]

def eventLeaf3513 : Array AnnotatedEvent := #[
  { event := event56208
    frameStart := 56119 },
  { event := event56209
    frameStart := 56119 },
  { event := event56210
    frameStart := 56119 },
  { event := event56211
    frameStart := 56119 },
  { event := event56212
    frameStart := 56119 },
  { event := event56213
    frameStart := 56119 },
  { event := event56214
    frameStart := 56119 },
  { event := event56215
    frameStart := 56119 },
  { event := event56216
    frameStart := 56119 },
  { event := event56217
    frameStart := 56119 },
  { event := event56218
    frameStart := 56119 },
  { event := event56219
    frameStart := 56119 },
  { event := event56220
    frameStart := 56119 },
  { event := event56221
    frameStart := 56119 },
  { event := event56222
    frameStart := 56119 },
  { event := event56223
    frameStart := 56119 }
]

def eventLeaf3514 : Array AnnotatedEvent := #[
  { event := event56224
    frameStart := 56119 },
  { event := event56225
    frameStart := 56119 },
  { event := event56226
    frameStart := 56119 },
  { event := event56227
    frameStart := 56119 },
  { event := event56228
    frameStart := 56119 },
  { event := event56229
    frameStart := 56119 },
  { event := event56230
    frameStart := 56119 },
  { event := event56231
    frameStart := 56119 },
  { event := event56232
    frameStart := 56119 },
  { event := event56233
    frameStart := 56119 },
  { event := event56234
    frameStart := 56119 },
  { event := event56235
    frameStart := 56119 },
  { event := event56236
    frameStart := 56119 },
  { event := event56237
    frameStart := 0 },
  { event := event56238
    frameStart := 0 },
  { event := event56239
    frameStart := 0 }
]

def eventLeaf3515 : Array AnnotatedEvent := #[
  { event := event56240
    frameStart := 0 },
  { event := event56241
    frameStart := 0 },
  { event := event56242
    frameStart := 0 },
  { event := event56243
    frameStart := 0 },
  { event := event56244
    frameStart := 0 },
  { event := event56245
    frameStart := 0 },
  { event := event56246
    frameStart := 0 },
  { event := event56247
    frameStart := 0 },
  { event := event56248
    frameStart := 0 },
  { event := event56249
    frameStart := 0 },
  { event := event56250
    frameStart := 0 },
  { event := event56251
    frameStart := 0 },
  { event := event56252
    frameStart := 0 },
  { event := event56253
    frameStart := 0 },
  { event := event56254
    frameStart := 0 },
  { event := event56255
    frameStart := 0 }
]

def eventLeaf3516 : Array AnnotatedEvent := #[
  { event := event56256
    frameStart := 0 },
  { event := event56257
    frameStart := 0 },
  { event := event56258
    frameStart := 0 },
  { event := event56259
    frameStart := 0 },
  { event := event56260
    frameStart := 0 },
  { event := event56261
    frameStart := 0 },
  { event := event56262
    frameStart := 0 },
  { event := event56263
    frameStart := 0 },
  { event := event56264
    frameStart := 0 },
  { event := event56265
    frameStart := 0 },
  { event := event56266
    frameStart := 0 },
  { event := event56267
    frameStart := 0 },
  { event := event56268
    frameStart := 0 },
  { event := event56269
    frameStart := 0 },
  { event := event56270
    frameStart := 0 },
  { event := event56271
    frameStart := 0 }
]

def eventLeaf3517 : Array AnnotatedEvent := #[
  { event := event56272
    frameStart := 0 },
  { event := event56273
    frameStart := 0 },
  { event := event56274
    frameStart := 56274 },
  { event := event56275
    frameStart := 56274 },
  { event := event56276
    frameStart := 56274 },
  { event := event56277
    frameStart := 56274 },
  { event := event56278
    frameStart := 56274 },
  { event := event56279
    frameStart := 56274 },
  { event := event56280
    frameStart := 56274 },
  { event := event56281
    frameStart := 56274 },
  { event := event56282
    frameStart := 56274 },
  { event := event56283
    frameStart := 56274 },
  { event := event56284
    frameStart := 56274 },
  { event := event56285
    frameStart := 56274 },
  { event := event56286
    frameStart := 56274 },
  { event := event56287
    frameStart := 56274 }
]

def eventLeaf3518 : Array AnnotatedEvent := #[
  { event := event56288
    frameStart := 56274 },
  { event := event56289
    frameStart := 56274 },
  { event := event56290
    frameStart := 56274 },
  { event := event56291
    frameStart := 56274 },
  { event := event56292
    frameStart := 56274 },
  { event := event56293
    frameStart := 56274 },
  { event := event56294
    frameStart := 56274 },
  { event := event56295
    frameStart := 56274 },
  { event := event56296
    frameStart := 56274 },
  { event := event56297
    frameStart := 56274 },
  { event := event56298
    frameStart := 56274 },
  { event := event56299
    frameStart := 56274 },
  { event := event56300
    frameStart := 56274 },
  { event := event56301
    frameStart := 56274 },
  { event := event56302
    frameStart := 56274 },
  { event := event56303
    frameStart := 56274 }
]

def eventLeaf3519 : Array AnnotatedEvent := #[
  { event := event56304
    frameStart := 56274 },
  { event := event56305
    frameStart := 56274 },
  { event := event56306
    frameStart := 56274 },
  { event := event56307
    frameStart := 56274 },
  { event := event56308
    frameStart := 56274 },
  { event := event56309
    frameStart := 56274 },
  { event := event56310
    frameStart := 56274 },
  { event := event56311
    frameStart := 56274 },
  { event := event56312
    frameStart := 56274 },
  { event := event56313
    frameStart := 56274 },
  { event := event56314
    frameStart := 56274 },
  { event := event56315
    frameStart := 56274 },
  { event := event56316
    frameStart := 56274 },
  { event := event56317
    frameStart := 56274 },
  { event := event56318
    frameStart := 56274 },
  { event := event56319
    frameStart := 56274 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events219
