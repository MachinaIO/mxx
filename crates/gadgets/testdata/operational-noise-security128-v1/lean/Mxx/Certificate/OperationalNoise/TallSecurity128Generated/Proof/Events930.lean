import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events930

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event238080 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42772⟩⟩) 0 ⟨42428⟩ 238079

def event238081 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42772⟩⟩) (.authority (.programFamilyFact))

def exact238082RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨42772⟩⟩], []⟩, (1)⟩]

theorem exact238082RawTermsValid :
    exact238082RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event238082 : Event := .resultExact (⟨.program ⟨257⟩, ⟨42772⟩⟩) exact238082RawTerms (.finite 52) 238081 .exactZero (none)

def event238083 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42773⟩⟩) 0 ⟨42772⟩ 238082

def event238084 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42773⟩⟩) (.identity (.predecessor 0 238083 .coefficient))

def event238085 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨42773⟩⟩) (.finite 52)

def event238086 : Event := .predecessor (⟨.program ⟨257⟩, ⟨43496⟩⟩) 0 ⟨42773⟩ 238085

def event238087 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨43496⟩⟩) (.authority (.relationPreimageSource ⟨90⟩))

def exact238088RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨43496⟩⟩]⟩, (1)⟩]

theorem exact238088RawTermsValid :
    exact238088RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event238088 : Event := .resultExact (⟨.program ⟨257⟩, ⟨43496⟩⟩) exact238088RawTerms (.finite 5647228698) 238087 .exactZero (none)

def event238089 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35⟩⟩) (.authority (.operator))

def exact238090RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩]⟩, (1)⟩]

theorem exact238090RawTermsValid :
    exact238090RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event238090 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35⟩⟩) exact238090RawTerms .large 238089 .exactZero (none)

def event238091 : Event := .predecessor (⟨.program ⟨257⟩, ⟨43497⟩⟩) 0 ⟨35⟩ 238090

def event238092 : Event := .predecessor (⟨.program ⟨257⟩, ⟨43497⟩⟩) 1 ⟨43496⟩ 238088

def event238093 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨43497⟩⟩) (.product (.predecessor 0 238091 .coefficient) (.predecessor 1 238092 .coefficient) (⟨false, false, none, none, none⟩))

def event238094 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨43497⟩⟩, .operator (⟨238090, 0⟩, ⟨238088, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43496⟩⟩]⟩, (1)⟩)

def exact238095RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43496⟩⟩]⟩, (1)⟩]

theorem exact238095RawTermsValid :
    exact238095RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event238095 : Event := .resultExact (⟨.program ⟨257⟩, ⟨43497⟩⟩) exact238095RawTerms .large 238093 .exactZero (none)

def event238096 : Event := .preFoldPolynomial 238095 [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43496⟩⟩]⟩, (1)⟩] .exactZero none

def exact238097RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43496⟩⟩]⟩, (1)⟩]

def event238097 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨43497⟩⟩) 238096 exact238097RawTerms .large 238093 .exactZero (none)

def event238098 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨44623⟩⟩)

def event238099 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event238100 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event238101 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨4740⟩⟩) (.authority (.operator))

def event238102 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨4740⟩⟩) (.finite 4)

def event238103 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event238104 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event238105 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event238106 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event238107 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 238106

def event238108 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 238104

def event238109 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 238107 .coefficient) (.value (.predecessor 1 238108 .coefficient)))

def event238110 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event238111 : Event := .predecessor (⟨.program ⟨257⟩, ⟨4742⟩⟩) 0 ⟨392⟩ 238110

def event238112 : Event := .predecessor (⟨.program ⟨257⟩, ⟨4742⟩⟩) 1 ⟨4740⟩ 238102

def event238113 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨4742⟩⟩) (.sum [.predecessor 0 238111 .coefficient, .predecessor 1 238112 .coefficient])

def event238114 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨4742⟩⟩) (.finite 655344)

def event238115 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5559⟩⟩) 0 ⟨4742⟩ 238114

def event238116 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5559⟩⟩) 1 ⟨5426⟩ 238100

def event238117 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5559⟩⟩) (.identity (.predecessor 1 238116 .coefficient))

def event238118 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5559⟩⟩) (.finite 655360)

def event238119 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42426⟩⟩) 0 ⟨5559⟩ 238118

def event238120 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42426⟩⟩) (.authority (.programFamilyFact))

def exact238121RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨42426⟩⟩], []⟩, (1)⟩]

theorem exact238121RawTermsValid :
    exact238121RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event238121 : Event := .resultExact (⟨.program ⟨257⟩, ⟨42426⟩⟩) exact238121RawTerms (.finite 52) 238120 .exactZero (none)

def event238122 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14451⟩⟩) 0 ⟨5559⟩ 238118

def event238123 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14451⟩⟩) (.authority (.programFamilyFact))

def exact238124RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14451⟩⟩], []⟩, (1)⟩]

theorem exact238124RawTermsValid :
    exact238124RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event238124 : Event := .resultExact (⟨.program ⟨257⟩, ⟨14451⟩⟩) exact238124RawTerms (.finite 52) 238123 .exactZero (none)

def event238125 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42427⟩⟩) 0 ⟨14451⟩ 238124

def event238126 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42427⟩⟩) 1 ⟨42426⟩ 238121

def event238127 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42427⟩⟩) (.product (.predecessor 0 238125 .coefficient) (.predecessor 1 238126 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event238128 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨42427⟩⟩, .operator (⟨238124, 0⟩, ⟨238121, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨14451⟩⟩, ⟨.program ⟨257⟩, ⟨42426⟩⟩], []⟩, (1)⟩)

def exact238129RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14451⟩⟩, ⟨.program ⟨257⟩, ⟨42426⟩⟩], []⟩, (1)⟩]

theorem exact238129RawTermsValid :
    exact238129RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event238129 : Event := .resultExact (⟨.program ⟨257⟩, ⟨42427⟩⟩) exact238129RawTerms (.finite 2704) 238127 .exactZero (none)

def event238130 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42428⟩⟩) 0 ⟨42427⟩ 238129

def event238131 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42428⟩⟩) (.identity (.predecessor 0 238130 .coefficient))

def event238132 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨42428⟩⟩) (.finite 2704)

def event238133 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42772⟩⟩) 0 ⟨42428⟩ 238132

def event238134 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42772⟩⟩) (.authority (.programFamilyFact))

def exact238135RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨42772⟩⟩], []⟩, (1)⟩]

theorem exact238135RawTermsValid :
    exact238135RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event238135 : Event := .resultExact (⟨.program ⟨257⟩, ⟨42772⟩⟩) exact238135RawTerms (.finite 52) 238134 .exactZero (none)

def event238136 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42773⟩⟩) 0 ⟨42772⟩ 238135

def event238137 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42773⟩⟩) (.identity (.predecessor 0 238136 .coefficient))

def event238138 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨42773⟩⟩) (.finite 52)

def event238139 : Event := .predecessor (⟨.program ⟨257⟩, ⟨43921⟩⟩) 0 ⟨42773⟩ 238138

def event238140 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨43921⟩⟩) (.authority (.programFamilyFact))

def event238141 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨43921⟩⟩) (.finite 3720)

def event238142 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨7177⟩⟩) .missing

def event238143 : Event := .predecessor (⟨.program ⟨257⟩, ⟨43923⟩⟩) 0 ⟨7177⟩ 238142

def event238144 : Event := .predecessor (⟨.program ⟨257⟩, ⟨43923⟩⟩) 1 ⟨43921⟩ 238141

def event238145 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨43923⟩⟩) (.authority (.operator))

def exact238146RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨43923⟩⟩]⟩, (1)⟩]

theorem exact238146RawTermsValid :
    exact238146RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event238146 : Event := .resultExact (⟨.program ⟨257⟩, ⟨43923⟩⟩) exact238146RawTerms .large 238145 .exactZero (none)

def event238147 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44619⟩⟩) 0 ⟨43923⟩ 238146

def event238148 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44619⟩⟩) (.authority (.operator))

def exact238149RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨44619⟩⟩]⟩, (1)⟩]

theorem exact238149RawTermsValid :
    exact238149RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event238149 : Event := .resultExact (⟨.program ⟨257⟩, ⟨44619⟩⟩) exact238149RawTerms (.finite 8192) 238148 .exactZero (none)

def event238150 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨136⟩⟩) (.authority (.operator))

def event238151 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨136⟩⟩) .exactZero

def event238152 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44138⟩⟩) 0 ⟨42773⟩ 238138

def event238153 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44138⟩⟩) 1 ⟨136⟩ 238151

def event238154 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44138⟩⟩) (.sum [.predecessor 0 238152 .coefficient, .predecessor 1 238153 .coefficient])

def event238155 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨44138⟩⟩) (.finite 52)

def event238156 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44139⟩⟩) 0 ⟨44138⟩ 238155

def event238157 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44139⟩⟩) (.identity (.predecessor 0 238156 .coefficient))

def exact238158RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨42772⟩⟩], []⟩, (1)⟩]

theorem exact238158RawTermsValid :
    exact238158RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event238158 : Event := .resultExact (⟨.program ⟨257⟩, ⟨44139⟩⟩) exact238158RawTerms (.finite 52) 238157 .exactZero (none)

def event238159 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6908⟩⟩) (.authority (.factStore))

def exact238160RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact238160RawTermsValid :
    exact238160RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event238160 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6908⟩⟩) exact238160RawTerms .large 238159 .exactZero (none)

def event238161 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44140⟩⟩) 0 ⟨6908⟩ 238160

def event238162 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44140⟩⟩) 1 ⟨44139⟩ 238158

def event238163 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44140⟩⟩) (.product (.predecessor 0 238161 .coefficient) (.predecessor 1 238162 .coefficient) (⟨false, false, none, none, none⟩))

def event238164 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨44140⟩⟩, .operator (⟨238160, 0⟩, ⟨238158, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨42772⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact238165RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨42772⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact238165RawTermsValid :
    exact238165RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event238165 : Event := .resultExact (⟨.program ⟨257⟩, ⟨44140⟩⟩) exact238165RawTerms .large 238163 .exactZero (none)

def event238166 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7194⟩⟩) 0 ⟨7177⟩ 238142

def event238167 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7194⟩⟩) (.authority (.operator))

def exact238168RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7194⟩⟩]⟩, (1)⟩]

theorem exact238168RawTermsValid :
    exact238168RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event238168 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7194⟩⟩) exact238168RawTerms .large 238167 .exactZero (none)

def event238169 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44141⟩⟩) 0 ⟨7194⟩ 238168

def event238170 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44141⟩⟩) 1 ⟨44140⟩ 238165

def event238171 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44141⟩⟩) (.sum [.predecessor 0 238169 .coefficient, .predecessor 1 238170 .coefficient])

def exact238172RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7194⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨42772⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact238172RawTermsValid :
    exact238172RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event238172 : Event := .resultExact (⟨.program ⟨257⟩, ⟨44141⟩⟩) exact238172RawTerms .large 238171 .exactZero (none)

def event238173 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44620⟩⟩) 0 ⟨44141⟩ 238172

def event238174 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44620⟩⟩) 1 ⟨44619⟩ 238149

def event238175 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44620⟩⟩) (.product (.predecessor 0 238173 .coefficient) (.predecessor 1 238174 .coefficient) (⟨false, false, none, none, none⟩))

def event238176 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨44620⟩⟩, .operator (⟨238172, 0⟩, ⟨238149, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7194⟩⟩, ⟨.program ⟨257⟩, ⟨44619⟩⟩]⟩, (1)⟩)

def event238177 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨44620⟩⟩, .operator (⟨238172, 1⟩, ⟨238149, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨42772⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨44619⟩⟩]⟩, (-1)⟩)

def event238178 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨44620⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨42772⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨44619⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨44619⟩⟩) ⟨43923⟩ 238146)

def event238179 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨44620⟩⟩, .relation 238178 0, ⟨[⟨.program ⟨257⟩, ⟨42772⟩⟩], [⟨.program ⟨257⟩, ⟨43923⟩⟩]⟩, (-1)⟩)

def exact238180RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7194⟩⟩, ⟨.program ⟨257⟩, ⟨44619⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨42772⟩⟩], [⟨.program ⟨257⟩, ⟨43923⟩⟩]⟩, (-1)⟩]

theorem exact238180RawTermsValid :
    exact238180RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event238180 : Event := .resultExact (⟨.program ⟨257⟩, ⟨44620⟩⟩) exact238180RawTerms .large 238175 .exactZero (none)

def event238181 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42973⟩⟩) 0 ⟨42773⟩ 238138

def event238182 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42973⟩⟩) (.authority (.programFamilyFact))

def exact238183RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨42973⟩⟩], []⟩, (1)⟩]

theorem exact238183RawTermsValid :
    exact238183RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event238183 : Event := .resultExact (⟨.program ⟨257⟩, ⟨42973⟩⟩) exact238183RawTerms (.finite 63) 238182 .exactZero (none)

def event238184 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42974⟩⟩) 0 ⟨6908⟩ 238160

def event238185 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42974⟩⟩) 1 ⟨42973⟩ 238183

def event238186 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42974⟩⟩) (.product (.predecessor 0 238184 .coefficient) (.predecessor 1 238185 .coefficient) (⟨false, true, none, none, some 1⟩))

def event238187 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨42974⟩⟩, .operator (⟨238160, 0⟩, ⟨238183, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨42973⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact238188RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨42973⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact238188RawTermsValid :
    exact238188RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event238188 : Event := .resultExact (⟨.program ⟨257⟩, ⟨42974⟩⟩) exact238188RawTerms .large 238186 .exactZero (none)

def event238189 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7228⟩⟩) 0 ⟨7177⟩ 238142

def event238190 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7228⟩⟩) (.authority (.operator))

def exact238191RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7228⟩⟩]⟩, (1)⟩]

theorem exact238191RawTermsValid :
    exact238191RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event238191 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7228⟩⟩) exact238191RawTerms .large 238190 .exactZero (none)

def event238192 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42975⟩⟩) 0 ⟨7228⟩ 238191

def event238193 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42975⟩⟩) 1 ⟨42974⟩ 238188

def event238194 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42975⟩⟩) (.sum [.predecessor 0 238192 .coefficient, .predecessor 1 238193 .coefficient])

def exact238195RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7228⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨42973⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact238195RawTermsValid :
    exact238195RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event238195 : Event := .resultExact (⟨.program ⟨257⟩, ⟨42975⟩⟩) exact238195RawTerms .large 238194 .exactZero (none)

def event238196 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44623⟩⟩) 0 ⟨42975⟩ 238195

def event238197 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44623⟩⟩) 1 ⟨44620⟩ 238180

def event238198 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44623⟩⟩) (.sum [.predecessor 0 238196 .coefficient, .predecessor 1 238197 .coefficient])

def exact238199RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7194⟩⟩, ⟨.program ⟨257⟩, ⟨44619⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7228⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨42772⟩⟩], [⟨.program ⟨257⟩, ⟨43923⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨42973⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact238199RawTermsValid :
    exact238199RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event238199 : Event := .resultExact (⟨.program ⟨257⟩, ⟨44623⟩⟩) exact238199RawTerms .large 238198 .exactZero (none)

def event238200 : Event := .preFoldPolynomial 238199 [⟨⟨[], [⟨.program ⟨257⟩, ⟨7194⟩⟩, ⟨.program ⟨257⟩, ⟨44619⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7228⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨42772⟩⟩], [⟨.program ⟨257⟩, ⟨43923⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨42973⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩] .exactZero none

def exact238201RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7194⟩⟩, ⟨.program ⟨257⟩, ⟨44619⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7228⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨42772⟩⟩], [⟨.program ⟨257⟩, ⟨43923⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨42973⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

def event238201 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨44623⟩⟩) 238200 exact238201RawTerms .large 238198 .exactZero (none)

def event238202 : Event := .specializationComputed (⟨.program ⟨257⟩, ⟨42773⟩⟩) ⟨⟨107⟩, ⟨90⟩, ⟨135⟩⟩ ⟨238044, 238202⟩

def event238203 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨43499⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43496⟩⟩]⟩) (1) 0 2 (.universal 238202 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43496⟩⟩]⟩) (none) 238201)

def event238204 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨43499⟩⟩, .relation 238203 1, ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7228⟩⟩]⟩, (1)⟩)

def event238205 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨43499⟩⟩, .relation 238203 0, ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7194⟩⟩, ⟨.program ⟨257⟩, ⟨44619⟩⟩]⟩, (-1)⟩)

def event238206 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨43499⟩⟩, .relation 238203 2, ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨42772⟩⟩], [⟨.program ⟨257⟩, ⟨43923⟩⟩]⟩, (1)⟩)

def event238207 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨43499⟩⟩, .relation 238203 3, ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨42973⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact238208RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7194⟩⟩, ⟨.program ⟨257⟩, ⟨44619⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7228⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨42772⟩⟩], [⟨.program ⟨257⟩, ⟨43923⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨42973⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact238208RawTermsValid :
    exact238208RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event238208 : Event := .resultExact (⟨.program ⟨257⟩, ⟨43499⟩⟩) exact238208RawTerms .large 238040 (.finite 202072841853861888) (some (238042))

def event238209 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44622⟩⟩) 0 ⟨43499⟩ 238208

def event238210 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44622⟩⟩) 1 ⟨44621⟩ 238030

def event238211 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44622⟩⟩) (.sum [.predecessor 0 238209 .coefficient, .predecessor 1 238210 .coefficient])

def event238212 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨44622⟩⟩, .operator (⟨238208, 0⟩, ⟨238030, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7194⟩⟩, ⟨.program ⟨257⟩, ⟨44619⟩⟩]⟩, (1)⟩)

def event238213 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨44622⟩⟩, .operator (⟨238208, 2⟩, ⟨238030, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨42772⟩⟩], [⟨.program ⟨257⟩, ⟨43923⟩⟩]⟩, (-1)⟩)

def event238214 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44622⟩⟩) (.sum [.result 238208 .summary, .result 238030 .summary])

def exact238215RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7228⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨42973⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact238215RawTermsValid :
    exact238215RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event238215 : Event := .resultExact (⟨.program ⟨257⟩, ⟨44622⟩⟩) exact238215RawTerms .large 238211 (.finite 32193718473625891320532869316608) (some (238214))

def event238216 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41241⟩⟩) 0 ⟨40093⟩ 11400

def event238217 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41241⟩⟩) (.authority (.programFamilyFact))

def event238218 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨41241⟩⟩) (.finite 3720)

def event238219 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41243⟩⟩) 0 ⟨7177⟩ 15500

def event238220 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41243⟩⟩) 1 ⟨41241⟩ 238218

def event238221 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41243⟩⟩) (.authority (.operator))

def exact238222RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨41243⟩⟩]⟩, (1)⟩]

theorem exact238222RawTermsValid :
    exact238222RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event238222 : Event := .resultExact (⟨.program ⟨257⟩, ⟨41243⟩⟩) exact238222RawTerms .large 238221 .exactZero (none)

def event238223 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41939⟩⟩) 0 ⟨41243⟩ 238222

def event238224 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41939⟩⟩) (.authority (.operator))

def exact238225RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨41939⟩⟩]⟩, (1)⟩]

theorem exact238225RawTermsValid :
    exact238225RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event238225 : Event := .resultExact (⟨.program ⟨257⟩, ⟨41939⟩⟩) exact238225RawTerms (.finite 8192) 238224 .exactZero (none)

def event238226 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41096⟩⟩) 0 ⟨39748⟩ 11394

def event238227 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41096⟩⟩) (.authority (.programFamilyFact))

def event238228 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨41096⟩⟩) (.finite 3720)

def event238229 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41097⟩⟩) 0 ⟨7177⟩ 15500

def event238230 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41097⟩⟩) 1 ⟨41096⟩ 238228

def event238231 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41097⟩⟩) (.authority (.operator))

def exact238232RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨41097⟩⟩]⟩, (1)⟩]

theorem exact238232RawTermsValid :
    exact238232RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event238232 : Event := .resultExact (⟨.program ⟨257⟩, ⟨41097⟩⟩) exact238232RawTerms .large 238231 .exactZero (none)

def event238233 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41597⟩⟩) 0 ⟨41097⟩ 238232

def event238234 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41597⟩⟩) (.authority (.operator))

def exact238235RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨41597⟩⟩]⟩, (1)⟩]

theorem exact238235RawTermsValid :
    exact238235RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event238235 : Event := .resultExact (⟨.program ⟨257⟩, ⟨41597⟩⟩) exact238235RawTerms (.finite 8192) 238234 .exactZero (none)

def event238236 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39749⟩⟩) 0 ⟨39746⟩ 11383

def event238237 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39749⟩⟩) 1 ⟨6934⟩ 236778

def event238238 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39749⟩⟩) (.tensor (.predecessor 0 238236 .coefficient) (.predecessor 1 238237 .coefficient) true false)

def event238239 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨39749⟩⟩, .operator (⟨11383, 0⟩, ⟨236778, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨39746⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact238240RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨39746⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact238240RawTermsValid :
    exact238240RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event238240 : Event := .resultExact (⟨.program ⟨257⟩, ⟨39749⟩⟩) exact238240RawTerms .large 238238 .exactZero (none)

def event238241 : Event := .predecessor (⟨.program ⟨257⟩, ⟨8360⟩⟩) 0 ⟨5561⟩ 236648

def event238242 : Event := .predecessor (⟨.program ⟨257⟩, ⟨8360⟩⟩) 1 ⟨7282⟩ 18583

def event238243 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨8360⟩⟩) (.product (.predecessor 0 238241 .coefficient) (.predecessor 1 238242 .coefficient) (⟨false, false, none, none, none⟩))

def event238244 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨8360⟩⟩, .operator (⟨236648, 0⟩, ⟨18583, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7282⟩⟩]⟩, (1)⟩)

def exact238245RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7282⟩⟩]⟩, (1)⟩]

theorem exact238245RawTermsValid :
    exact238245RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event238245 : Event := .resultExact (⟨.program ⟨257⟩, ⟨8360⟩⟩) exact238245RawTerms .large 238243 .exactZero (none)

def event238246 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39750⟩⟩) 0 ⟨8360⟩ 238245

def event238247 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39750⟩⟩) 1 ⟨39749⟩ 238240

def event238248 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39750⟩⟩) (.sum [.predecessor 0 238246 .coefficient, .predecessor 1 238247 .coefficient])

def exact238249RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7282⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨39746⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact238249RawTermsValid :
    exact238249RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event238249 : Event := .resultExact (⟨.program ⟨257⟩, ⟨39750⟩⟩) exact238249RawTerms .large 238248 .exactZero (none)

def event238250 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39751⟩⟩) 0 ⟨39750⟩ 238249

def event238251 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39751⟩⟩) 1 ⟨108⟩ 18575

def event238252 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39751⟩⟩) (.sum [.predecessor 0 238250 .coefficient, .predecessor 1 238251 .coefficient])

def event238253 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39751⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨108⟩⟩]⟩) [⟨.result 18575 .coefficient, false, none⟩])

def event238254 : Event := .survivorFold (1) 238253

def exact238255RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7282⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨39746⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact238255RawTermsValid :
    exact238255RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event238255 : Event := .resultExact (⟨.program ⟨257⟩, ⟨39751⟩⟩) exact238255RawTerms .large 238252 (.finite 26) (some (238253))

def event238256 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39752⟩⟩) 0 ⟨39751⟩ 238255

def event238257 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39752⟩⟩) 1 ⟨14151⟩ 11386

def event238258 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39752⟩⟩) (.product (.predecessor 0 238256 .coefficient) (.predecessor 1 238257 .coefficient) (⟨false, true, none, none, some 1⟩))

def event238259 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39752⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨14151⟩⟩], []⟩) [⟨.result 11386 .coefficient, true, some 1⟩])

def event238260 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39752⟩⟩) (.product (.result 238255 .summary) (.transfer 238259) (⟨false, false, none, none, none⟩))

def event238261 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨39752⟩⟩, .operator (⟨238255, 1⟩, ⟨11386, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨14151⟩⟩, ⟨.program ⟨257⟩, ⟨39746⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def event238262 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨39752⟩⟩, .operator (⟨238255, 0⟩, ⟨11386, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨14151⟩⟩], [⟨.program ⟨257⟩, ⟨7282⟩⟩]⟩, (1)⟩)

def exact238263RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨14151⟩⟩], [⟨.program ⟨257⟩, ⟨7282⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨14151⟩⟩, ⟨.program ⟨257⟩, ⟨39746⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact238263RawTermsValid :
    exact238263RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event238263 : Event := .resultExact (⟨.program ⟨257⟩, ⟨39752⟩⟩) exact238263RawTerms .large 238258 (.finite 39190528) (some (238260))

def event238264 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14152⟩⟩) 0 ⟨14151⟩ 11386

def event238265 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14152⟩⟩) 1 ⟨6934⟩ 236778

def event238266 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14152⟩⟩) (.tensor (.predecessor 0 238264 .coefficient) (.predecessor 1 238265 .coefficient) true false)

def event238267 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨14152⟩⟩, .operator (⟨11386, 0⟩, ⟨236778, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨14151⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact238268RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨14151⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact238268RawTermsValid :
    exact238268RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event238268 : Event := .resultExact (⟨.program ⟨257⟩, ⟨14152⟩⟩) exact238268RawTerms .large 238266 .exactZero (none)

def event238269 : Event := .predecessor (⟨.program ⟨257⟩, ⟨8377⟩⟩) 0 ⟨5561⟩ 236648

def event238270 : Event := .predecessor (⟨.program ⟨257⟩, ⟨8377⟩⟩) 1 ⟨7299⟩ 18624

def event238271 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨8377⟩⟩) (.product (.predecessor 0 238269 .coefficient) (.predecessor 1 238270 .coefficient) (⟨false, false, none, none, none⟩))

def event238272 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨8377⟩⟩, .operator (⟨236648, 0⟩, ⟨18624, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7299⟩⟩]⟩, (1)⟩)

def exact238273RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7299⟩⟩]⟩, (1)⟩]

theorem exact238273RawTermsValid :
    exact238273RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event238273 : Event := .resultExact (⟨.program ⟨257⟩, ⟨8377⟩⟩) exact238273RawTerms .large 238271 .exactZero (none)

def event238274 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14153⟩⟩) 0 ⟨8377⟩ 238273

def event238275 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14153⟩⟩) 1 ⟨14152⟩ 238268

def event238276 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14153⟩⟩) (.sum [.predecessor 0 238274 .coefficient, .predecessor 1 238275 .coefficient])

def exact238277RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7299⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨14151⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact238277RawTermsValid :
    exact238277RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event238277 : Event := .resultExact (⟨.program ⟨257⟩, ⟨14153⟩⟩) exact238277RawTerms .large 238276 .exactZero (none)

def event238278 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14154⟩⟩) 0 ⟨14153⟩ 238277

def event238279 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14154⟩⟩) 1 ⟨125⟩ 18616

def event238280 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14154⟩⟩) (.sum [.predecessor 0 238278 .coefficient, .predecessor 1 238279 .coefficient])

def event238281 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14154⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨125⟩⟩]⟩) [⟨.result 18616 .coefficient, false, none⟩])

def event238282 : Event := .survivorFold (1) 238281

def exact238283RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7299⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨14151⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact238283RawTermsValid :
    exact238283RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event238283 : Event := .resultExact (⟨.program ⟨257⟩, ⟨14154⟩⟩) exact238283RawTerms .large 238280 (.finite 26) (some (238281))

def event238284 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14155⟩⟩) 0 ⟨14154⟩ 238283

def event238285 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14155⟩⟩) 1 ⟨9557⟩ 18613

def event238286 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14155⟩⟩) (.product (.predecessor 0 238284 .coefficient) (.predecessor 1 238285 .coefficient) (⟨false, false, none, none, none⟩))

def event238287 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14155⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨9556⟩⟩]⟩) [⟨.result 18609 .coefficient, false, none⟩])

def event238288 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14155⟩⟩) (.product (.result 238283 .summary) (.transfer 238287) (⟨false, false, none, none, none⟩))

def event238289 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨14155⟩⟩, .operator (⟨238283, 1⟩, ⟨18613, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨14151⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩]⟩, (-1)⟩)

def event238290 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨14155⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨14151⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨9556⟩⟩) ⟨7282⟩ 18583)

def event238291 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨14155⟩⟩, .relation 238290 0, ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨14151⟩⟩], [⟨.program ⟨257⟩, ⟨7282⟩⟩]⟩, (-1)⟩)

def event238292 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨14155⟩⟩, .operator (⟨238283, 0⟩, ⟨18613, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩]⟩, (1)⟩)

def exact238293RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨14151⟩⟩], [⟨.program ⟨257⟩, ⟨7282⟩⟩]⟩, (-1)⟩]

theorem exact238293RawTermsValid :
    exact238293RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event238293 : Event := .resultExact (⟨.program ⟨257⟩, ⟨14155⟩⟩) exact238293RawTerms .large 238286 (.finite 279172874240) (some (238288))

def event238294 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39753⟩⟩) 0 ⟨14155⟩ 238293

def event238295 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39753⟩⟩) 1 ⟨39752⟩ 238263

def event238296 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39753⟩⟩) (.sum [.predecessor 0 238294 .coefficient, .predecessor 1 238295 .coefficient])

def event238297 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨39753⟩⟩, .operator (⟨238293, 1⟩, ⟨238263, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨14151⟩⟩], [⟨.program ⟨257⟩, ⟨7282⟩⟩]⟩, (1)⟩)

def event238298 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39753⟩⟩) (.sum [.result 238293 .summary, .result 238263 .summary])

def exact238299RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨14151⟩⟩, ⟨.program ⟨257⟩, ⟨39746⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact238299RawTermsValid :
    exact238299RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event238299 : Event := .resultExact (⟨.program ⟨257⟩, ⟨39753⟩⟩) exact238299RawTerms .large 238296 (.finite 279212064768) (some (238298))

def event238300 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41598⟩⟩) 0 ⟨39753⟩ 238299

def event238301 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41598⟩⟩) 1 ⟨41597⟩ 238235

def event238302 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41598⟩⟩) (.product (.predecessor 0 238300 .coefficient) (.predecessor 1 238301 .coefficient) (⟨false, false, none, none, none⟩))

def event238303 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41598⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨41597⟩⟩]⟩) [⟨.result 238235 .coefficient, false, none⟩])

def event238304 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41598⟩⟩) (.product (.result 238299 .summary) (.transfer 238303) (⟨false, false, none, none, none⟩))

def event238305 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨41598⟩⟩, .operator (⟨238299, 1⟩, ⟨238235, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨14151⟩⟩, ⟨.program ⟨257⟩, ⟨39746⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨41597⟩⟩]⟩, (-1)⟩)

def event238306 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨41598⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨14151⟩⟩, ⟨.program ⟨257⟩, ⟨39746⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨41597⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨41597⟩⟩) ⟨41097⟩ 238232)

def event238307 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨41598⟩⟩, .relation 238306 0, ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨14151⟩⟩, ⟨.program ⟨257⟩, ⟨39746⟩⟩], [⟨.program ⟨257⟩, ⟨41097⟩⟩]⟩, (-1)⟩)

def event238308 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨41598⟩⟩, .operator (⟨238299, 0⟩, ⟨238235, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41597⟩⟩]⟩, (1)⟩)

def exact238309RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41597⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨14151⟩⟩, ⟨.program ⟨257⟩, ⟨39746⟩⟩], [⟨.program ⟨257⟩, ⟨41097⟩⟩]⟩, (-1)⟩]

theorem exact238309RawTermsValid :
    exact238309RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event238309 : Event := .resultExact (⟨.program ⟨257⟩, ⟨41598⟩⟩) exact238309RawTerms .large 238302 (.finite 2998016717067984568320) (some (238304))

def event238310 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40529⟩⟩) 0 ⟨39748⟩ 11394

def event238311 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨40529⟩⟩) (.authority (.relationPreimageSource ⟨51⟩))

def exact238312RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨40529⟩⟩]⟩, (1)⟩]

theorem exact238312RawTermsValid :
    exact238312RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event238312 : Event := .resultExact (⟨.program ⟨257⟩, ⟨40529⟩⟩) exact238312RawTerms (.finite 5647228698) 238311 .exactZero (none)

def event238313 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40531⟩⟩) 0 ⟨40529⟩ 238312

def event238314 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40531⟩⟩) 1 ⟨2370⟩ 4

def event238315 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨40531⟩⟩) (.scale (.predecessor 0 238313 .coefficient) (.value (.predecessor 1 238314 .coefficient)))

def exact238316RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨40529⟩⟩]⟩, (1)⟩]

theorem exact238316RawTermsValid :
    exact238316RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event238316 : Event := .resultExact (⟨.program ⟨257⟩, ⟨40531⟩⟩) exact238316RawTerms (.finite 5647228698) 238315 .exactZero (none)

def event238317 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40532⟩⟩) 0 ⟨5563⟩ 236870

def event238318 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40532⟩⟩) 1 ⟨40531⟩ 238316

def event238319 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨40532⟩⟩) (.product (.predecessor 0 238317 .coefficient) (.predecessor 1 238318 .coefficient) (⟨false, false, none, none, none⟩))

def event238320 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨40532⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨40529⟩⟩]⟩) [⟨.result 238312 .coefficient, false, none⟩])

def event238321 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨40532⟩⟩) (.product (.result 236870 .summary) (.transfer 238320) (⟨false, false, none, none, none⟩))

def event238322 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨40532⟩⟩, .operator (⟨236870, 0⟩, ⟨238316, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40529⟩⟩]⟩, (1)⟩)

def event238323 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨40530⟩⟩)

def event238324 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event238325 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event238326 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨4740⟩⟩) (.authority (.operator))

def event238327 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨4740⟩⟩) (.finite 4)

def event238328 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event238329 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event238330 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event238331 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event238332 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 238331

def event238333 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 238329

def event238334 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 238332 .coefficient) (.value (.predecessor 1 238333 .coefficient)))

def event238335 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def eventLeaf14880 : Array AnnotatedEvent := #[
  { event := event238080
    frameStart := 238044 },
  { event := event238081
    frameStart := 238044 },
  { event := event238082
    frameStart := 238044 },
  { event := event238083
    frameStart := 238044 },
  { event := event238084
    frameStart := 238044 },
  { event := event238085
    frameStart := 238044 },
  { event := event238086
    frameStart := 238044 },
  { event := event238087
    frameStart := 238044 },
  { event := event238088
    frameStart := 238044 },
  { event := event238089
    frameStart := 238044 },
  { event := event238090
    frameStart := 238044 },
  { event := event238091
    frameStart := 238044 },
  { event := event238092
    frameStart := 238044 },
  { event := event238093
    frameStart := 238044 },
  { event := event238094
    frameStart := 238044 },
  { event := event238095
    frameStart := 238044 }
]

def eventLeaf14881 : Array AnnotatedEvent := #[
  { event := event238096
    frameStart := 238044 },
  { event := event238097
    frameStart := 238044 },
  { event := event238098
    frameStart := 238098 },
  { event := event238099
    frameStart := 238098 },
  { event := event238100
    frameStart := 238098 },
  { event := event238101
    frameStart := 238098 },
  { event := event238102
    frameStart := 238098 },
  { event := event238103
    frameStart := 238098 },
  { event := event238104
    frameStart := 238098 },
  { event := event238105
    frameStart := 238098 },
  { event := event238106
    frameStart := 238098 },
  { event := event238107
    frameStart := 238098 },
  { event := event238108
    frameStart := 238098 },
  { event := event238109
    frameStart := 238098 },
  { event := event238110
    frameStart := 238098 },
  { event := event238111
    frameStart := 238098 }
]

def eventLeaf14882 : Array AnnotatedEvent := #[
  { event := event238112
    frameStart := 238098 },
  { event := event238113
    frameStart := 238098 },
  { event := event238114
    frameStart := 238098 },
  { event := event238115
    frameStart := 238098 },
  { event := event238116
    frameStart := 238098 },
  { event := event238117
    frameStart := 238098 },
  { event := event238118
    frameStart := 238098 },
  { event := event238119
    frameStart := 238098 },
  { event := event238120
    frameStart := 238098 },
  { event := event238121
    frameStart := 238098 },
  { event := event238122
    frameStart := 238098 },
  { event := event238123
    frameStart := 238098 },
  { event := event238124
    frameStart := 238098 },
  { event := event238125
    frameStart := 238098 },
  { event := event238126
    frameStart := 238098 },
  { event := event238127
    frameStart := 238098 }
]

def eventLeaf14883 : Array AnnotatedEvent := #[
  { event := event238128
    frameStart := 238098 },
  { event := event238129
    frameStart := 238098 },
  { event := event238130
    frameStart := 238098 },
  { event := event238131
    frameStart := 238098 },
  { event := event238132
    frameStart := 238098 },
  { event := event238133
    frameStart := 238098 },
  { event := event238134
    frameStart := 238098 },
  { event := event238135
    frameStart := 238098 },
  { event := event238136
    frameStart := 238098 },
  { event := event238137
    frameStart := 238098 },
  { event := event238138
    frameStart := 238098 },
  { event := event238139
    frameStart := 238098 },
  { event := event238140
    frameStart := 238098 },
  { event := event238141
    frameStart := 238098 },
  { event := event238142
    frameStart := 238098 },
  { event := event238143
    frameStart := 238098 }
]

def eventLeaf14884 : Array AnnotatedEvent := #[
  { event := event238144
    frameStart := 238098 },
  { event := event238145
    frameStart := 238098 },
  { event := event238146
    frameStart := 238098 },
  { event := event238147
    frameStart := 238098 },
  { event := event238148
    frameStart := 238098 },
  { event := event238149
    frameStart := 238098 },
  { event := event238150
    frameStart := 238098 },
  { event := event238151
    frameStart := 238098 },
  { event := event238152
    frameStart := 238098 },
  { event := event238153
    frameStart := 238098 },
  { event := event238154
    frameStart := 238098 },
  { event := event238155
    frameStart := 238098 },
  { event := event238156
    frameStart := 238098 },
  { event := event238157
    frameStart := 238098 },
  { event := event238158
    frameStart := 238098 },
  { event := event238159
    frameStart := 238098 }
]

def eventLeaf14885 : Array AnnotatedEvent := #[
  { event := event238160
    frameStart := 238098 },
  { event := event238161
    frameStart := 238098 },
  { event := event238162
    frameStart := 238098 },
  { event := event238163
    frameStart := 238098 },
  { event := event238164
    frameStart := 238098 },
  { event := event238165
    frameStart := 238098 },
  { event := event238166
    frameStart := 238098 },
  { event := event238167
    frameStart := 238098 },
  { event := event238168
    frameStart := 238098 },
  { event := event238169
    frameStart := 238098 },
  { event := event238170
    frameStart := 238098 },
  { event := event238171
    frameStart := 238098 },
  { event := event238172
    frameStart := 238098 },
  { event := event238173
    frameStart := 238098 },
  { event := event238174
    frameStart := 238098 },
  { event := event238175
    frameStart := 238098 }
]

def eventLeaf14886 : Array AnnotatedEvent := #[
  { event := event238176
    frameStart := 238098 },
  { event := event238177
    frameStart := 238098 },
  { event := event238178
    frameStart := 238098 },
  { event := event238179
    frameStart := 238098 },
  { event := event238180
    frameStart := 238098 },
  { event := event238181
    frameStart := 238098 },
  { event := event238182
    frameStart := 238098 },
  { event := event238183
    frameStart := 238098 },
  { event := event238184
    frameStart := 238098 },
  { event := event238185
    frameStart := 238098 },
  { event := event238186
    frameStart := 238098 },
  { event := event238187
    frameStart := 238098 },
  { event := event238188
    frameStart := 238098 },
  { event := event238189
    frameStart := 238098 },
  { event := event238190
    frameStart := 238098 },
  { event := event238191
    frameStart := 238098 }
]

def eventLeaf14887 : Array AnnotatedEvent := #[
  { event := event238192
    frameStart := 238098 },
  { event := event238193
    frameStart := 238098 },
  { event := event238194
    frameStart := 238098 },
  { event := event238195
    frameStart := 238098 },
  { event := event238196
    frameStart := 238098 },
  { event := event238197
    frameStart := 238098 },
  { event := event238198
    frameStart := 238098 },
  { event := event238199
    frameStart := 238098 },
  { event := event238200
    frameStart := 238098 },
  { event := event238201
    frameStart := 238098 },
  { event := event238202
    frameStart := 0 },
  { event := event238203
    frameStart := 0 },
  { event := event238204
    frameStart := 0 },
  { event := event238205
    frameStart := 0 },
  { event := event238206
    frameStart := 0 },
  { event := event238207
    frameStart := 0 }
]

def eventLeaf14888 : Array AnnotatedEvent := #[
  { event := event238208
    frameStart := 0 },
  { event := event238209
    frameStart := 0 },
  { event := event238210
    frameStart := 0 },
  { event := event238211
    frameStart := 0 },
  { event := event238212
    frameStart := 0 },
  { event := event238213
    frameStart := 0 },
  { event := event238214
    frameStart := 0 },
  { event := event238215
    frameStart := 0 },
  { event := event238216
    frameStart := 0 },
  { event := event238217
    frameStart := 0 },
  { event := event238218
    frameStart := 0 },
  { event := event238219
    frameStart := 0 },
  { event := event238220
    frameStart := 0 },
  { event := event238221
    frameStart := 0 },
  { event := event238222
    frameStart := 0 },
  { event := event238223
    frameStart := 0 }
]

def eventLeaf14889 : Array AnnotatedEvent := #[
  { event := event238224
    frameStart := 0 },
  { event := event238225
    frameStart := 0 },
  { event := event238226
    frameStart := 0 },
  { event := event238227
    frameStart := 0 },
  { event := event238228
    frameStart := 0 },
  { event := event238229
    frameStart := 0 },
  { event := event238230
    frameStart := 0 },
  { event := event238231
    frameStart := 0 },
  { event := event238232
    frameStart := 0 },
  { event := event238233
    frameStart := 0 },
  { event := event238234
    frameStart := 0 },
  { event := event238235
    frameStart := 0 },
  { event := event238236
    frameStart := 0 },
  { event := event238237
    frameStart := 0 },
  { event := event238238
    frameStart := 0 },
  { event := event238239
    frameStart := 0 }
]

def eventLeaf14890 : Array AnnotatedEvent := #[
  { event := event238240
    frameStart := 0 },
  { event := event238241
    frameStart := 0 },
  { event := event238242
    frameStart := 0 },
  { event := event238243
    frameStart := 0 },
  { event := event238244
    frameStart := 0 },
  { event := event238245
    frameStart := 0 },
  { event := event238246
    frameStart := 0 },
  { event := event238247
    frameStart := 0 },
  { event := event238248
    frameStart := 0 },
  { event := event238249
    frameStart := 0 },
  { event := event238250
    frameStart := 0 },
  { event := event238251
    frameStart := 0 },
  { event := event238252
    frameStart := 0 },
  { event := event238253
    frameStart := 0 },
  { event := event238254
    frameStart := 0 },
  { event := event238255
    frameStart := 0 }
]

def eventLeaf14891 : Array AnnotatedEvent := #[
  { event := event238256
    frameStart := 0 },
  { event := event238257
    frameStart := 0 },
  { event := event238258
    frameStart := 0 },
  { event := event238259
    frameStart := 0 },
  { event := event238260
    frameStart := 0 },
  { event := event238261
    frameStart := 0 },
  { event := event238262
    frameStart := 0 },
  { event := event238263
    frameStart := 0 },
  { event := event238264
    frameStart := 0 },
  { event := event238265
    frameStart := 0 },
  { event := event238266
    frameStart := 0 },
  { event := event238267
    frameStart := 0 },
  { event := event238268
    frameStart := 0 },
  { event := event238269
    frameStart := 0 },
  { event := event238270
    frameStart := 0 },
  { event := event238271
    frameStart := 0 }
]

def eventLeaf14892 : Array AnnotatedEvent := #[
  { event := event238272
    frameStart := 0 },
  { event := event238273
    frameStart := 0 },
  { event := event238274
    frameStart := 0 },
  { event := event238275
    frameStart := 0 },
  { event := event238276
    frameStart := 0 },
  { event := event238277
    frameStart := 0 },
  { event := event238278
    frameStart := 0 },
  { event := event238279
    frameStart := 0 },
  { event := event238280
    frameStart := 0 },
  { event := event238281
    frameStart := 0 },
  { event := event238282
    frameStart := 0 },
  { event := event238283
    frameStart := 0 },
  { event := event238284
    frameStart := 0 },
  { event := event238285
    frameStart := 0 },
  { event := event238286
    frameStart := 0 },
  { event := event238287
    frameStart := 0 }
]

def eventLeaf14893 : Array AnnotatedEvent := #[
  { event := event238288
    frameStart := 0 },
  { event := event238289
    frameStart := 0 },
  { event := event238290
    frameStart := 0 },
  { event := event238291
    frameStart := 0 },
  { event := event238292
    frameStart := 0 },
  { event := event238293
    frameStart := 0 },
  { event := event238294
    frameStart := 0 },
  { event := event238295
    frameStart := 0 },
  { event := event238296
    frameStart := 0 },
  { event := event238297
    frameStart := 0 },
  { event := event238298
    frameStart := 0 },
  { event := event238299
    frameStart := 0 },
  { event := event238300
    frameStart := 0 },
  { event := event238301
    frameStart := 0 },
  { event := event238302
    frameStart := 0 },
  { event := event238303
    frameStart := 0 }
]

def eventLeaf14894 : Array AnnotatedEvent := #[
  { event := event238304
    frameStart := 0 },
  { event := event238305
    frameStart := 0 },
  { event := event238306
    frameStart := 0 },
  { event := event238307
    frameStart := 0 },
  { event := event238308
    frameStart := 0 },
  { event := event238309
    frameStart := 0 },
  { event := event238310
    frameStart := 0 },
  { event := event238311
    frameStart := 0 },
  { event := event238312
    frameStart := 0 },
  { event := event238313
    frameStart := 0 },
  { event := event238314
    frameStart := 0 },
  { event := event238315
    frameStart := 0 },
  { event := event238316
    frameStart := 0 },
  { event := event238317
    frameStart := 0 },
  { event := event238318
    frameStart := 0 },
  { event := event238319
    frameStart := 0 }
]

def eventLeaf14895 : Array AnnotatedEvent := #[
  { event := event238320
    frameStart := 0 },
  { event := event238321
    frameStart := 0 },
  { event := event238322
    frameStart := 0 },
  { event := event238323
    frameStart := 238323 },
  { event := event238324
    frameStart := 238323 },
  { event := event238325
    frameStart := 238323 },
  { event := event238326
    frameStart := 238323 },
  { event := event238327
    frameStart := 238323 },
  { event := event238328
    frameStart := 238323 },
  { event := event238329
    frameStart := 238323 },
  { event := event238330
    frameStart := 238323 },
  { event := event238331
    frameStart := 238323 },
  { event := event238332
    frameStart := 238323 },
  { event := event238333
    frameStart := 238323 },
  { event := event238334
    frameStart := 238323 },
  { event := event238335
    frameStart := 238323 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events930
