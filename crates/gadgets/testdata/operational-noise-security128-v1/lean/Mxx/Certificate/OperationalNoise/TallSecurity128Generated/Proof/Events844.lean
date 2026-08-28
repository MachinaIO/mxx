import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events844

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event216064 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15789⟩⟩) (.identity (.predecessor 0 216063 .coefficient))

def event216065 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨15789⟩⟩) (.finite 2)

def event216066 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16596⟩⟩) 0 ⟨15789⟩ 216065

def event216067 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨16596⟩⟩) (.authority (.relationPreimageSource ⟨57⟩))

def exact216068RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨16596⟩⟩]⟩, (1)⟩]

theorem exact216068RawTermsValid :
    exact216068RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event216068 : Event := .resultExact (⟨.program ⟨257⟩, ⟨16596⟩⟩) exact216068RawTerms (.finite 5647228698) 216067 .exactZero (none)

def event216069 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35⟩⟩) (.authority (.operator))

def exact216070RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩]⟩, (1)⟩]

theorem exact216070RawTermsValid :
    exact216070RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event216070 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35⟩⟩) exact216070RawTerms .large 216069 .exactZero (none)

def event216071 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16597⟩⟩) 0 ⟨35⟩ 216070

def event216072 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16597⟩⟩) 1 ⟨16596⟩ 216068

def event216073 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨16597⟩⟩) (.product (.predecessor 0 216071 .coefficient) (.predecessor 1 216072 .coefficient) (⟨false, false, none, none, none⟩))

def event216074 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨16597⟩⟩, .operator (⟨216070, 0⟩, ⟨216068, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16596⟩⟩]⟩, (1)⟩)

def exact216075RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16596⟩⟩]⟩, (1)⟩]

theorem exact216075RawTermsValid :
    exact216075RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event216075 : Event := .resultExact (⟨.program ⟨257⟩, ⟨16597⟩⟩) exact216075RawTerms .large 216073 .exactZero (none)

def event216076 : Event := .preFoldPolynomial 216075 [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16596⟩⟩]⟩, (1)⟩] .exactZero none

def exact216077RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16596⟩⟩]⟩, (1)⟩]

def event216077 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨16597⟩⟩) 216076 exact216077RawTerms .large 216073 .exactZero (none)

def event216078 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨17765⟩⟩)

def event216079 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event216080 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event216081 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5240⟩⟩) (.authority (.operator))

def event216082 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5240⟩⟩) (.finite 6)

def event216083 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event216084 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event216085 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event216086 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event216087 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 216086

def event216088 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 216084

def event216089 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 216087 .coefficient) (.value (.predecessor 1 216088 .coefficient)))

def event216090 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event216091 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5242⟩⟩) 0 ⟨392⟩ 216090

def event216092 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5242⟩⟩) 1 ⟨5240⟩ 216082

def event216093 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5242⟩⟩) (.sum [.predecessor 0 216091 .coefficient, .predecessor 1 216092 .coefficient])

def event216094 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5242⟩⟩) (.finite 655346)

def event216095 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5595⟩⟩) 0 ⟨5242⟩ 216094

def event216096 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5595⟩⟩) 1 ⟨5426⟩ 216080

def event216097 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5595⟩⟩) (.identity (.predecessor 1 216096 .coefficient))

def event216098 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5595⟩⟩) (.finite 655360)

def event216099 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15474⟩⟩) 0 ⟨5595⟩ 216098

def event216100 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15474⟩⟩) (.authority (.programFamilyFact))

def exact216101RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15474⟩⟩], []⟩, (1)⟩]

theorem exact216101RawTermsValid :
    exact216101RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event216101 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15474⟩⟩) exact216101RawTerms (.finite 2) 216100 .exactZero (none)

def event216102 : Event := .predecessor (⟨.program ⟨257⟩, ⟨12381⟩⟩) 0 ⟨5595⟩ 216098

def event216103 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨12381⟩⟩) (.authority (.programFamilyFact))

def exact216104RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨12381⟩⟩], []⟩, (1)⟩]

theorem exact216104RawTermsValid :
    exact216104RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event216104 : Event := .resultExact (⟨.program ⟨257⟩, ⟨12381⟩⟩) exact216104RawTerms (.finite 2) 216103 .exactZero (none)

def event216105 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15475⟩⟩) 0 ⟨12381⟩ 216104

def event216106 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15475⟩⟩) 1 ⟨15474⟩ 216101

def event216107 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15475⟩⟩) (.product (.predecessor 0 216105 .coefficient) (.predecessor 1 216106 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event216108 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨15475⟩⟩, .operator (⟨216104, 0⟩, ⟨216101, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨12381⟩⟩, ⟨.program ⟨257⟩, ⟨15474⟩⟩], []⟩, (1)⟩)

def exact216109RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨12381⟩⟩, ⟨.program ⟨257⟩, ⟨15474⟩⟩], []⟩, (1)⟩]

theorem exact216109RawTermsValid :
    exact216109RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event216109 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15475⟩⟩) exact216109RawTerms (.finite 4) 216107 .exactZero (none)

def event216110 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15476⟩⟩) 0 ⟨15475⟩ 216109

def event216111 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15476⟩⟩) (.identity (.predecessor 0 216110 .coefficient))

def event216112 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨15476⟩⟩) (.finite 4)

def event216113 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15788⟩⟩) 0 ⟨15476⟩ 216112

def event216114 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15788⟩⟩) (.authority (.programFamilyFact))

def exact216115RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15788⟩⟩], []⟩, (1)⟩]

theorem exact216115RawTermsValid :
    exact216115RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event216115 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15788⟩⟩) exact216115RawTerms (.finite 2) 216114 .exactZero (none)

def event216116 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15789⟩⟩) 0 ⟨15788⟩ 216115

def event216117 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15789⟩⟩) (.identity (.predecessor 0 216116 .coefficient))

def event216118 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨15789⟩⟩) (.finite 2)

def event216119 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16999⟩⟩) 0 ⟨15789⟩ 216118

def event216120 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨16999⟩⟩) (.authority (.programFamilyFact))

def event216121 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨16999⟩⟩) (.finite 3720)

def event216122 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨7177⟩⟩) .missing

def event216123 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17001⟩⟩) 0 ⟨7177⟩ 216122

def event216124 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17001⟩⟩) 1 ⟨16999⟩ 216121

def event216125 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17001⟩⟩) (.authority (.operator))

def exact216126RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨17001⟩⟩]⟩, (1)⟩]

theorem exact216126RawTermsValid :
    exact216126RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event216126 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17001⟩⟩) exact216126RawTerms .large 216125 .exactZero (none)

def event216127 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17761⟩⟩) 0 ⟨17001⟩ 216126

def event216128 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17761⟩⟩) (.authority (.operator))

def exact216129RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨17761⟩⟩]⟩, (1)⟩]

theorem exact216129RawTermsValid :
    exact216129RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event216129 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17761⟩⟩) exact216129RawTerms (.finite 8192) 216128 .exactZero (none)

def event216130 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨136⟩⟩) (.authority (.operator))

def event216131 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨136⟩⟩) .exactZero

def event216132 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17206⟩⟩) 0 ⟨15789⟩ 216118

def event216133 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17206⟩⟩) 1 ⟨136⟩ 216131

def event216134 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17206⟩⟩) (.sum [.predecessor 0 216132 .coefficient, .predecessor 1 216133 .coefficient])

def event216135 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨17206⟩⟩) (.finite 2)

def event216136 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17207⟩⟩) 0 ⟨17206⟩ 216135

def event216137 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17207⟩⟩) (.identity (.predecessor 0 216136 .coefficient))

def exact216138RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15788⟩⟩], []⟩, (1)⟩]

theorem exact216138RawTermsValid :
    exact216138RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event216138 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17207⟩⟩) exact216138RawTerms (.finite 2) 216137 .exactZero (none)

def event216139 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6908⟩⟩) (.authority (.factStore))

def exact216140RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact216140RawTermsValid :
    exact216140RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event216140 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6908⟩⟩) exact216140RawTerms .large 216139 .exactZero (none)

def event216141 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17208⟩⟩) 0 ⟨6908⟩ 216140

def event216142 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17208⟩⟩) 1 ⟨17207⟩ 216138

def event216143 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17208⟩⟩) (.product (.predecessor 0 216141 .coefficient) (.predecessor 1 216142 .coefficient) (⟨false, false, none, none, none⟩))

def event216144 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17208⟩⟩, .operator (⟨216140, 0⟩, ⟨216138, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨15788⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact216145RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15788⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact216145RawTermsValid :
    exact216145RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event216145 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17208⟩⟩) exact216145RawTerms .large 216143 .exactZero (none)

def event216146 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7179⟩⟩) 0 ⟨7177⟩ 216122

def event216147 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7179⟩⟩) (.authority (.operator))

def exact216148RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩]⟩, (1)⟩]

theorem exact216148RawTermsValid :
    exact216148RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event216148 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7179⟩⟩) exact216148RawTerms .large 216147 .exactZero (none)

def event216149 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17209⟩⟩) 0 ⟨7179⟩ 216148

def event216150 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17209⟩⟩) 1 ⟨17208⟩ 216145

def event216151 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17209⟩⟩) (.sum [.predecessor 0 216149 .coefficient, .predecessor 1 216150 .coefficient])

def exact216152RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15788⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact216152RawTermsValid :
    exact216152RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event216152 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17209⟩⟩) exact216152RawTerms .large 216151 .exactZero (none)

def event216153 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17762⟩⟩) 0 ⟨17209⟩ 216152

def event216154 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17762⟩⟩) 1 ⟨17761⟩ 216129

def event216155 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17762⟩⟩) (.product (.predecessor 0 216153 .coefficient) (.predecessor 1 216154 .coefficient) (⟨false, false, none, none, none⟩))

def event216156 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17762⟩⟩, .operator (⟨216152, 0⟩, ⟨216129, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17761⟩⟩]⟩, (1)⟩)

def event216157 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17762⟩⟩, .operator (⟨216152, 1⟩, ⟨216129, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨15788⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17761⟩⟩]⟩, (-1)⟩)

def event216158 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨17762⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨15788⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17761⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨17761⟩⟩) ⟨17001⟩ 216126)

def event216159 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17762⟩⟩, .relation 216158 0, ⟨[⟨.program ⟨257⟩, ⟨15788⟩⟩], [⟨.program ⟨257⟩, ⟨17001⟩⟩]⟩, (-1)⟩)

def exact216160RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17761⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15788⟩⟩], [⟨.program ⟨257⟩, ⟨17001⟩⟩]⟩, (-1)⟩]

theorem exact216160RawTermsValid :
    exact216160RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event216160 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17762⟩⟩) exact216160RawTerms .large 216155 .exactZero (none)

def event216161 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16035⟩⟩) 0 ⟨15789⟩ 216118

def event216162 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨16035⟩⟩) (.authority (.programFamilyFact))

def exact216163RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨16035⟩⟩], []⟩, (1)⟩]

theorem exact216163RawTermsValid :
    exact216163RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event216163 : Event := .resultExact (⟨.program ⟨257⟩, ⟨16035⟩⟩) exact216163RawTerms (.finite 43) 216162 .exactZero (none)

def event216164 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16036⟩⟩) 0 ⟨6908⟩ 216140

def event216165 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16036⟩⟩) 1 ⟨16035⟩ 216163

def event216166 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨16036⟩⟩) (.product (.predecessor 0 216164 .coefficient) (.predecessor 1 216165 .coefficient) (⟨false, true, none, none, some 1⟩))

def event216167 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨16036⟩⟩, .operator (⟨216140, 0⟩, ⟨216163, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨16035⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact216168RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨16035⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact216168RawTermsValid :
    exact216168RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event216168 : Event := .resultExact (⟨.program ⟨257⟩, ⟨16036⟩⟩) exact216168RawTerms .large 216166 .exactZero (none)

def event216169 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7198⟩⟩) 0 ⟨7177⟩ 216122

def event216170 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7198⟩⟩) (.authority (.operator))

def exact216171RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩]

theorem exact216171RawTermsValid :
    exact216171RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event216171 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7198⟩⟩) exact216171RawTerms .large 216170 .exactZero (none)

def event216172 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16037⟩⟩) 0 ⟨7198⟩ 216171

def event216173 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16037⟩⟩) 1 ⟨16036⟩ 216168

def event216174 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨16037⟩⟩) (.sum [.predecessor 0 216172 .coefficient, .predecessor 1 216173 .coefficient])

def exact216175RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨16035⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact216175RawTermsValid :
    exact216175RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event216175 : Event := .resultExact (⟨.program ⟨257⟩, ⟨16037⟩⟩) exact216175RawTerms .large 216174 .exactZero (none)

def event216176 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17765⟩⟩) 0 ⟨16037⟩ 216175

def event216177 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17765⟩⟩) 1 ⟨17762⟩ 216160

def event216178 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17765⟩⟩) (.sum [.predecessor 0 216176 .coefficient, .predecessor 1 216177 .coefficient])

def exact216179RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17761⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15788⟩⟩], [⟨.program ⟨257⟩, ⟨17001⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨16035⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact216179RawTermsValid :
    exact216179RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event216179 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17765⟩⟩) exact216179RawTerms .large 216178 .exactZero (none)

def event216180 : Event := .preFoldPolynomial 216179 [⟨⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17761⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15788⟩⟩], [⟨.program ⟨257⟩, ⟨17001⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨16035⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩] .exactZero none

def exact216181RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17761⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15788⟩⟩], [⟨.program ⟨257⟩, ⟨17001⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨16035⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

def event216181 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨17765⟩⟩) 216180 exact216181RawTerms .large 216178 .exactZero (none)

def event216182 : Event := .specializationComputed (⟨.program ⟨257⟩, ⟨15789⟩⟩) ⟨⟨77⟩, ⟨57⟩, ⟨135⟩⟩ ⟨216024, 216182⟩

def event216183 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨16599⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16596⟩⟩]⟩) (1) 0 2 (.universal 216182 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16596⟩⟩]⟩) (none) 216181)

def event216184 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨16599⟩⟩, .relation 216183 1, ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩)

def event216185 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨16599⟩⟩, .relation 216183 0, ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17761⟩⟩]⟩, (-1)⟩)

def event216186 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨16599⟩⟩, .relation 216183 2, ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨15788⟩⟩], [⟨.program ⟨257⟩, ⟨17001⟩⟩]⟩, (1)⟩)

def event216187 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨16599⟩⟩, .relation 216183 3, ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨16035⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact216188RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17761⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨15788⟩⟩], [⟨.program ⟨257⟩, ⟨17001⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨16035⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact216188RawTermsValid :
    exact216188RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event216188 : Event := .resultExact (⟨.program ⟨257⟩, ⟨16599⟩⟩) exact216188RawTerms .large 216020 (.finite 202072841853861888) (some (216022))

def event216189 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17764⟩⟩) 0 ⟨16599⟩ 216188

def event216190 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17764⟩⟩) 1 ⟨17763⟩ 216010

def event216191 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17764⟩⟩) (.sum [.predecessor 0 216189 .coefficient, .predecessor 1 216190 .coefficient])

def event216192 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17764⟩⟩, .operator (⟨216188, 0⟩, ⟨216010, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17761⟩⟩]⟩, (1)⟩)

def event216193 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17764⟩⟩, .operator (⟨216188, 2⟩, ⟨216010, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨15788⟩⟩], [⟨.program ⟨257⟩, ⟨17001⟩⟩]⟩, (-1)⟩)

def event216194 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17764⟩⟩) (.sum [.result 216188 .summary, .result 216010 .summary])

def exact216195RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨16035⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact216195RawTermsValid :
    exact216195RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event216195 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17764⟩⟩) exact216195RawTerms .large 216191 (.finite 32188807212483706889510625476608) (some (216194))

def event216196 : Event := .predecessor (⟨.program ⟨257⟩, ⟨20656⟩⟩) 0 ⟨17764⟩ 216195

def event216197 : Event := .predecessor (⟨.program ⟨257⟩, ⟨20656⟩⟩) 1 ⟨20655⟩ 215713

def event216198 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨20656⟩⟩) (.sum [.predecessor 0 216196 .coefficient, .predecessor 1 216197 .coefficient])

def event216199 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨20656⟩⟩) (.sum [.result 216195 .summary, .result 215713 .summary])

def exact216200RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨16035⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨18866⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact216200RawTermsValid :
    exact216200RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event216200 : Event := .resultExact (⟨.program ⟨257⟩, ⟨20656⟩⟩) exact216200RawTerms .large 216198 (.finite 64377712650190257467641695830016) (some (216199))

def event216201 : Event := .predecessor (⟨.program ⟨257⟩, ⟨23876⟩⟩) 0 ⟨20656⟩ 216200

def event216202 : Event := .predecessor (⟨.program ⟨257⟩, ⟨23876⟩⟩) 1 ⟨23875⟩ 215231

def event216203 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨23876⟩⟩) (.sum [.predecessor 0 216201 .coefficient, .predecessor 1 216202 .coefficient])

def event216204 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨23876⟩⟩) (.sum [.result 216200 .summary, .result 215231 .summary])

def exact216205RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨16035⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨18866⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨22086⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact216205RawTermsValid :
    exact216205RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event216205 : Event := .resultExact (⟨.program ⟨257⟩, ⟨23876⟩⟩) exact216205RawTerms .large 216203 (.finite 96566716313119651734393211060224) (some (216204))

def event216206 : Event := .predecessor (⟨.program ⟨257⟩, ⟨33896⟩⟩) 0 ⟨23876⟩ 216205

def event216207 : Event := .predecessor (⟨.program ⟨257⟩, ⟨33896⟩⟩) 1 ⟨33895⟩ 214749

def event216208 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨33896⟩⟩) (.sum [.predecessor 0 216206 .coefficient, .predecessor 1 216207 .coefficient])

def event216209 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨33896⟩⟩) (.sum [.result 216205 .summary, .result 214749 .summary])

def exact216210RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨16035⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨18866⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨22086⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨32106⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact216210RawTermsValid :
    exact216210RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event216210 : Event := .resultExact (⟨.program ⟨257⟩, ⟨33896⟩⟩) exact216210RawTerms .large 216208 (.finite 128755916426494733378385616044032) (some (216209))

def event216211 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52956⟩⟩) 0 ⟨33896⟩ 216210

def event216212 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52956⟩⟩) 1 ⟨52955⟩ 214267

def event216213 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨52956⟩⟩) (.sum [.predecessor 0 216211 .coefficient, .predecessor 1 216212 .coefficient])

def event216214 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨52956⟩⟩) (.sum [.result 216210 .summary, .result 214267 .summary])

def exact216215RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨16035⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨18866⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨22086⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨32106⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨51161⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact216215RawTermsValid :
    exact216215RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event216215 : Event := .resultExact (⟨.program ⟨257⟩, ⟨52956⟩⟩) exact216215RawTerms .large 216213 (.finite 160945509440761189776859800535040) (some (216214))

def event216216 : Event := .predecessor (⟨.program ⟨257⟩, ⟨55936⟩⟩) 0 ⟨52956⟩ 216215

def event216217 : Event := .predecessor (⟨.program ⟨257⟩, ⟨55936⟩⟩) 1 ⟨55935⟩ 213785

def event216218 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨55936⟩⟩) (.sum [.predecessor 0 216216 .coefficient, .predecessor 1 216217 .coefficient])

def event216219 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨55936⟩⟩) (.sum [.result 216215 .summary, .result 213785 .summary])

def exact216220RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨16035⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨18866⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨22086⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨32106⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨51161⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨54141⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact216220RawTermsValid :
    exact216220RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event216220 : Event := .resultExact (⟨.program ⟨257⟩, ⟨55936⟩⟩) exact216220RawTerms .large 216218 (.finite 193135298905473333552574874779648) (some (216219))

def event216221 : Event := .predecessor (⟨.program ⟨257⟩, ⟨58916⟩⟩) 0 ⟨55936⟩ 216220

def event216222 : Event := .predecessor (⟨.program ⟨257⟩, ⟨58916⟩⟩) 1 ⟨58915⟩ 213303

def event216223 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨58916⟩⟩) (.sum [.predecessor 0 216221 .coefficient, .predecessor 1 216222 .coefficient])

def event216224 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨58916⟩⟩) (.sum [.result 216220 .summary, .result 213303 .summary])

def exact216225RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨16035⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨18866⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨22086⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨32106⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨51161⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨54141⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨57121⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact216225RawTermsValid :
    exact216225RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event216225 : Event := .resultExact (⟨.program ⟨257⟩, ⟨58916⟩⟩) exact216225RawTerms .large 216223 (.finite 225325481271076852082771728531456) (some (216224))

def event216226 : Event := .predecessor (⟨.program ⟨257⟩, ⟨61896⟩⟩) 0 ⟨58916⟩ 216225

def event216227 : Event := .predecessor (⟨.program ⟨257⟩, ⟨61896⟩⟩) 1 ⟨61895⟩ 212821

def event216228 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨61896⟩⟩) (.sum [.predecessor 0 216226 .coefficient, .predecessor 1 216227 .coefficient])

def event216229 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨61896⟩⟩) (.sum [.result 216225 .summary, .result 212821 .summary])

def exact216230RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨16035⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨18866⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨22086⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨32106⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨51161⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨54141⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨57121⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨60101⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact216230RawTermsValid :
    exact216230RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event216230 : Event := .resultExact (⟨.program ⟨257⟩, ⟨61896⟩⟩) exact216230RawTerms .large 216228 (.finite 257515860087126057990209472036864) (some (216229))

def event216231 : Event := .predecessor (⟨.program ⟨257⟩, ⟨64876⟩⟩) 0 ⟨61896⟩ 216230

def event216232 : Event := .predecessor (⟨.program ⟨257⟩, ⟨64876⟩⟩) 1 ⟨64875⟩ 212339

def event216233 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨64876⟩⟩) (.sum [.predecessor 0 216231 .coefficient, .predecessor 1 216232 .coefficient])

def event216234 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨64876⟩⟩) (.sum [.result 216230 .summary, .result 212339 .summary])

def exact216235RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨16035⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨18866⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨22086⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨32106⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨51161⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨54141⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨57121⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨60101⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨63081⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact216235RawTermsValid :
    exact216235RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event216235 : Event := .resultExact (⟨.program ⟨257⟩, ⟨64876⟩⟩) exact216235RawTerms .large 216233 (.finite 289706631804066638652128995049472) (some (216234))

def event216236 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70181⟩⟩) 0 ⟨64876⟩ 216235

def event216237 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70181⟩⟩) 1 ⟨70180⟩ 211857

def event216238 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70181⟩⟩) (.sum [.predecessor 0 216236 .coefficient, .predecessor 1 216237 .coefficient])

def event216239 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70181⟩⟩) (.sum [.result 216235 .summary, .result 211857 .summary])

def exact216240RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨16035⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨18866⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨22086⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨32106⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨51161⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨54141⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨57121⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨60101⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨63081⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨66601⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact216240RawTermsValid :
    exact216240RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event216240 : Event := .resultExact (⟨.program ⟨257⟩, ⟨70181⟩⟩) exact216240RawTerms .large 216238 (.finite 321897992872344281445771187322880) (some (216239))

def event216241 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70182⟩⟩) 0 ⟨70181⟩ 216240

def event216242 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70182⟩⟩) 1 ⟨28292⟩ 211375

def event216243 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70182⟩⟩) (.sum [.predecessor 0 216241 .coefficient, .predecessor 1 216242 .coefficient])

def event216244 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70182⟩⟩) (.sum [.result 216240 .summary, .result 211375 .summary])

def exact216245RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨16035⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨18866⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨22086⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨26619⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨32106⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨51161⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨54141⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨57121⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨60101⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨63081⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨66601⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact216245RawTermsValid :
    exact216245RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event216245 : Event := .resultExact (⟨.program ⟨257⟩, ⟨70182⟩⟩) exact216245RawTerms .large 216243 (.finite 354089550391067611616654269349888) (some (216244))

def event216246 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70183⟩⟩) 0 ⟨70182⟩ 216245

def event216247 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70183⟩⟩) 1 ⟨30972⟩ 210893

def event216248 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70183⟩⟩) (.sum [.predecessor 0 216246 .coefficient, .predecessor 1 216247 .coefficient])

def event216249 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70183⟩⟩) (.sum [.result 216245 .summary, .result 210893 .summary])

def exact216250RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨16035⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨18866⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨22086⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨26619⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨29299⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨32106⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨51161⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨54141⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨57121⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨60101⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨63081⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨66601⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact216250RawTermsValid :
    exact216250RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event216250 : Event := .resultExact (⟨.program ⟨257⟩, ⟨70183⟩⟩) exact216250RawTerms .large 216248 (.finite 386281697261128003919260020637696) (some (216249))

def event216251 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70184⟩⟩) 0 ⟨70183⟩ 216250

def event216252 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70184⟩⟩) 1 ⟨36632⟩ 210411

def event216253 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70184⟩⟩) (.sum [.predecessor 0 216251 .coefficient, .predecessor 1 216252 .coefficient])

def event216254 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70184⟩⟩) (.sum [.result 216250 .summary, .result 210411 .summary])

def exact216255RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7222⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨16035⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨18866⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨22086⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨26619⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨29299⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨32106⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨34963⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨51161⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨54141⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨57121⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨60101⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨63081⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨66601⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact216255RawTermsValid :
    exact216255RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event216255 : Event := .resultExact (⟨.program ⟨257⟩, ⟨70184⟩⟩) exact216255RawTerms .large 216253 (.finite 418474237032079770976347551432704) (some (216254))

def event216256 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70185⟩⟩) 0 ⟨70184⟩ 216255

def event216257 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70185⟩⟩) 1 ⟨39312⟩ 209929

def event216258 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70185⟩⟩) (.sum [.predecessor 0 216256 .coefficient, .predecessor 1 216257 .coefficient])

def event216259 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70185⟩⟩) (.sum [.result 216255 .summary, .result 209929 .summary])

def exact216260RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7222⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7224⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨16035⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨18866⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨22086⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨26619⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨29299⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨32106⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨34963⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨37643⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨51161⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨54141⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨57121⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨60101⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨63081⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨66601⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact216260RawTermsValid :
    exact216260RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event216260 : Event := .resultExact (⟨.program ⟨257⟩, ⟨70185⟩⟩) exact216260RawTerms .large 216258 (.finite 450666973253477225410675971981312) (some (216259))

def event216261 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70186⟩⟩) 0 ⟨70185⟩ 216260

def event216262 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70186⟩⟩) 1 ⟨41992⟩ 209447

def event216263 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70186⟩⟩) (.sum [.predecessor 0 216261 .coefficient, .predecessor 1 216262 .coefficient])

def event216264 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70186⟩⟩) (.sum [.result 216260 .summary, .result 209447 .summary])

def exact216265RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7222⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7224⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7226⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨16035⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨18866⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨22086⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨26619⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨29299⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨32106⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨34963⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨37643⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨40319⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨51161⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨54141⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨57121⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨60101⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨63081⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨66601⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact216265RawTermsValid :
    exact216265RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event216265 : Event := .resultExact (⟨.program ⟨257⟩, ⟨70186⟩⟩) exact216265RawTerms .large 216263 (.finite 482860102375766054599486172037120) (some (216264))

def event216266 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70187⟩⟩) 0 ⟨70186⟩ 216265

def event216267 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70187⟩⟩) 1 ⟨44672⟩ 208965

def event216268 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70187⟩⟩) (.sum [.predecessor 0 216266 .coefficient, .predecessor 1 216267 .coefficient])

def event216269 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70187⟩⟩) (.sum [.result 216265 .summary, .result 208965 .summary])

def exact216270RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7222⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7224⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7226⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7228⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨16035⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨18866⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨22086⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨26619⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨29299⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨32106⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨34963⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨37643⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨40319⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨42999⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨51161⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨54141⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨57121⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨60101⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨63081⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨66601⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact216270RawTermsValid :
    exact216270RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event216270 : Event := .resultExact (⟨.program ⟨257⟩, ⟨70187⟩⟩) exact216270RawTerms .large 216268 (.finite 515053820849391945920019041353728) (some (216269))

def event216271 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70188⟩⟩) 0 ⟨70187⟩ 216270

def event216272 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70188⟩⟩) 1 ⟨47352⟩ 208483

def event216273 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70188⟩⟩) (.sum [.predecessor 0 216271 .coefficient, .predecessor 1 216272 .coefficient])

def event216274 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70188⟩⟩) (.sum [.result 216270 .summary, .result 208483 .summary])

def exact216275RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7222⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7224⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7226⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7228⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7230⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨16035⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨18866⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨22086⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨26619⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨29299⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨32106⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨34963⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨37643⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨40319⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨42999⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨45683⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨51161⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨54141⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨57121⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨60101⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨63081⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨66601⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact216275RawTermsValid :
    exact216275RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event216275 : Event := .resultExact (⟨.program ⟨257⟩, ⟨70188⟩⟩) exact216275RawTerms .large 216273 (.finite 547248128674354899372274579931136) (some (216274))

def event216276 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70189⟩⟩) 0 ⟨70188⟩ 216275

def event216277 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70189⟩⟩) 1 ⟨50032⟩ 208001

def event216278 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70189⟩⟩) (.sum [.predecessor 0 216276 .coefficient, .predecessor 1 216277 .coefficient])

def event216279 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70189⟩⟩) (.sum [.result 216275 .summary, .result 208001 .summary])

def exact216280RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7222⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7224⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7226⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7228⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7230⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7232⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨16035⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨18866⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨22086⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨26619⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨29299⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨32106⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨34963⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨37643⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨40319⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨42999⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨45683⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨48363⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨51161⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨54141⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨57121⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨60101⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨63081⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨66601⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact216280RawTermsValid :
    exact216280RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event216280 : Event := .resultExact (⟨.program ⟨257⟩, ⟨70189⟩⟩) exact216280RawTerms .large 216278 (.finite 579442632949763540201771008262144) (some (216279))

def event216281 : Event := .predecessor (⟨.program ⟨257⟩, ⟨71238⟩⟩) 0 ⟨70189⟩ 216280

def event216282 : Event := .predecessor (⟨.program ⟨257⟩, ⟨71238⟩⟩) 1 ⟨71236⟩ 207503

def event216283 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨71238⟩⟩) (.product (.predecessor 0 216281 .coefficient) (.predecessor 1 216282 .coefficient) (⟨false, false, none, none, none⟩))

def event216284 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨71238⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩) [⟨.result 207503 .coefficient, false, none⟩])

def event216285 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨71238⟩⟩) (.product (.result 216280 .summary) (.transfer 216284) (⟨false, false, none, none, none⟩))

def event216286 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71238⟩⟩, .operator (⟨216280, 17⟩, ⟨207503, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7232⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (1)⟩)

def event216287 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71238⟩⟩, .operator (⟨216280, 29⟩, ⟨207503, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨48363⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (-1)⟩)

def event216288 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71238⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨48363⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71236⟩⟩) ⟨68830⟩ 207500)

def event216289 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71238⟩⟩, .relation 216288 0, ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨48363⟩⟩], [⟨.program ⟨257⟩, ⟨68830⟩⟩]⟩, (-1)⟩)

def event216290 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71238⟩⟩, .operator (⟨216280, 16⟩, ⟨207503, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7230⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (1)⟩)

def event216291 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71238⟩⟩, .operator (⟨216280, 28⟩, ⟨207503, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨45683⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (-1)⟩)

def event216292 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71238⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨45683⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71236⟩⟩) ⟨68830⟩ 207500)

def event216293 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71238⟩⟩, .relation 216292 0, ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨45683⟩⟩], [⟨.program ⟨257⟩, ⟨68830⟩⟩]⟩, (-1)⟩)

def event216294 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71238⟩⟩, .operator (⟨216280, 15⟩, ⟨207503, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7228⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (1)⟩)

def event216295 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71238⟩⟩, .operator (⟨216280, 27⟩, ⟨207503, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨42999⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (-1)⟩)

def event216296 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71238⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨42999⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71236⟩⟩) ⟨68830⟩ 207500)

def event216297 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71238⟩⟩, .relation 216296 0, ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨42999⟩⟩], [⟨.program ⟨257⟩, ⟨68830⟩⟩]⟩, (-1)⟩)

def event216298 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71238⟩⟩, .operator (⟨216280, 14⟩, ⟨207503, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7226⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (1)⟩)

def event216299 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71238⟩⟩, .operator (⟨216280, 26⟩, ⟨207503, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨40319⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (-1)⟩)

def event216300 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71238⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨40319⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71236⟩⟩) ⟨68830⟩ 207500)

def event216301 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71238⟩⟩, .relation 216300 0, ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨40319⟩⟩], [⟨.program ⟨257⟩, ⟨68830⟩⟩]⟩, (-1)⟩)

def event216302 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71238⟩⟩, .operator (⟨216280, 13⟩, ⟨207503, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7224⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (1)⟩)

def event216303 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71238⟩⟩, .operator (⟨216280, 25⟩, ⟨207503, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨37643⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (-1)⟩)

def event216304 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71238⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨37643⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71236⟩⟩) ⟨68830⟩ 207500)

def event216305 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71238⟩⟩, .relation 216304 0, ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨37643⟩⟩], [⟨.program ⟨257⟩, ⟨68830⟩⟩]⟩, (-1)⟩)

def event216306 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71238⟩⟩, .operator (⟨216280, 12⟩, ⟨207503, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7222⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (1)⟩)

def event216307 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71238⟩⟩, .operator (⟨216280, 24⟩, ⟨207503, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨34963⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (-1)⟩)

def event216308 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71238⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨34963⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71236⟩⟩) ⟨68830⟩ 207500)

def event216309 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71238⟩⟩, .relation 216308 0, ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨34963⟩⟩], [⟨.program ⟨257⟩, ⟨68830⟩⟩]⟩, (-1)⟩)

def event216310 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71238⟩⟩, .operator (⟨216280, 11⟩, ⟨207503, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7220⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (1)⟩)

def event216311 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71238⟩⟩, .operator (⟨216280, 22⟩, ⟨207503, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨29299⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (-1)⟩)

def event216312 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71238⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨29299⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71236⟩⟩) ⟨68830⟩ 207500)

def event216313 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71238⟩⟩, .relation 216312 0, ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨29299⟩⟩], [⟨.program ⟨257⟩, ⟨68830⟩⟩]⟩, (-1)⟩)

def event216314 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71238⟩⟩, .operator (⟨216280, 10⟩, ⟨207503, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (1)⟩)

def event216315 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71238⟩⟩, .operator (⟨216280, 21⟩, ⟨207503, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨26619⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (-1)⟩)

def event216316 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71238⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨26619⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71236⟩⟩) ⟨68830⟩ 207500)

def event216317 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71238⟩⟩, .relation 216316 0, ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨26619⟩⟩], [⟨.program ⟨257⟩, ⟨68830⟩⟩]⟩, (-1)⟩)

def event216318 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71238⟩⟩, .operator (⟨216280, 9⟩, ⟨207503, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (1)⟩)

def event216319 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71238⟩⟩, .operator (⟨216280, 35⟩, ⟨207503, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨66601⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (-1)⟩)

def eventLeaf13504 : Array AnnotatedEvent := #[
  { event := event216064
    frameStart := 216024 },
  { event := event216065
    frameStart := 216024 },
  { event := event216066
    frameStart := 216024 },
  { event := event216067
    frameStart := 216024 },
  { event := event216068
    frameStart := 216024 },
  { event := event216069
    frameStart := 216024 },
  { event := event216070
    frameStart := 216024 },
  { event := event216071
    frameStart := 216024 },
  { event := event216072
    frameStart := 216024 },
  { event := event216073
    frameStart := 216024 },
  { event := event216074
    frameStart := 216024 },
  { event := event216075
    frameStart := 216024 },
  { event := event216076
    frameStart := 216024 },
  { event := event216077
    frameStart := 216024 },
  { event := event216078
    frameStart := 216078 },
  { event := event216079
    frameStart := 216078 }
]

def eventLeaf13505 : Array AnnotatedEvent := #[
  { event := event216080
    frameStart := 216078 },
  { event := event216081
    frameStart := 216078 },
  { event := event216082
    frameStart := 216078 },
  { event := event216083
    frameStart := 216078 },
  { event := event216084
    frameStart := 216078 },
  { event := event216085
    frameStart := 216078 },
  { event := event216086
    frameStart := 216078 },
  { event := event216087
    frameStart := 216078 },
  { event := event216088
    frameStart := 216078 },
  { event := event216089
    frameStart := 216078 },
  { event := event216090
    frameStart := 216078 },
  { event := event216091
    frameStart := 216078 },
  { event := event216092
    frameStart := 216078 },
  { event := event216093
    frameStart := 216078 },
  { event := event216094
    frameStart := 216078 },
  { event := event216095
    frameStart := 216078 }
]

def eventLeaf13506 : Array AnnotatedEvent := #[
  { event := event216096
    frameStart := 216078 },
  { event := event216097
    frameStart := 216078 },
  { event := event216098
    frameStart := 216078 },
  { event := event216099
    frameStart := 216078 },
  { event := event216100
    frameStart := 216078 },
  { event := event216101
    frameStart := 216078 },
  { event := event216102
    frameStart := 216078 },
  { event := event216103
    frameStart := 216078 },
  { event := event216104
    frameStart := 216078 },
  { event := event216105
    frameStart := 216078 },
  { event := event216106
    frameStart := 216078 },
  { event := event216107
    frameStart := 216078 },
  { event := event216108
    frameStart := 216078 },
  { event := event216109
    frameStart := 216078 },
  { event := event216110
    frameStart := 216078 },
  { event := event216111
    frameStart := 216078 }
]

def eventLeaf13507 : Array AnnotatedEvent := #[
  { event := event216112
    frameStart := 216078 },
  { event := event216113
    frameStart := 216078 },
  { event := event216114
    frameStart := 216078 },
  { event := event216115
    frameStart := 216078 },
  { event := event216116
    frameStart := 216078 },
  { event := event216117
    frameStart := 216078 },
  { event := event216118
    frameStart := 216078 },
  { event := event216119
    frameStart := 216078 },
  { event := event216120
    frameStart := 216078 },
  { event := event216121
    frameStart := 216078 },
  { event := event216122
    frameStart := 216078 },
  { event := event216123
    frameStart := 216078 },
  { event := event216124
    frameStart := 216078 },
  { event := event216125
    frameStart := 216078 },
  { event := event216126
    frameStart := 216078 },
  { event := event216127
    frameStart := 216078 }
]

def eventLeaf13508 : Array AnnotatedEvent := #[
  { event := event216128
    frameStart := 216078 },
  { event := event216129
    frameStart := 216078 },
  { event := event216130
    frameStart := 216078 },
  { event := event216131
    frameStart := 216078 },
  { event := event216132
    frameStart := 216078 },
  { event := event216133
    frameStart := 216078 },
  { event := event216134
    frameStart := 216078 },
  { event := event216135
    frameStart := 216078 },
  { event := event216136
    frameStart := 216078 },
  { event := event216137
    frameStart := 216078 },
  { event := event216138
    frameStart := 216078 },
  { event := event216139
    frameStart := 216078 },
  { event := event216140
    frameStart := 216078 },
  { event := event216141
    frameStart := 216078 },
  { event := event216142
    frameStart := 216078 },
  { event := event216143
    frameStart := 216078 }
]

def eventLeaf13509 : Array AnnotatedEvent := #[
  { event := event216144
    frameStart := 216078 },
  { event := event216145
    frameStart := 216078 },
  { event := event216146
    frameStart := 216078 },
  { event := event216147
    frameStart := 216078 },
  { event := event216148
    frameStart := 216078 },
  { event := event216149
    frameStart := 216078 },
  { event := event216150
    frameStart := 216078 },
  { event := event216151
    frameStart := 216078 },
  { event := event216152
    frameStart := 216078 },
  { event := event216153
    frameStart := 216078 },
  { event := event216154
    frameStart := 216078 },
  { event := event216155
    frameStart := 216078 },
  { event := event216156
    frameStart := 216078 },
  { event := event216157
    frameStart := 216078 },
  { event := event216158
    frameStart := 216078 },
  { event := event216159
    frameStart := 216078 }
]

def eventLeaf13510 : Array AnnotatedEvent := #[
  { event := event216160
    frameStart := 216078 },
  { event := event216161
    frameStart := 216078 },
  { event := event216162
    frameStart := 216078 },
  { event := event216163
    frameStart := 216078 },
  { event := event216164
    frameStart := 216078 },
  { event := event216165
    frameStart := 216078 },
  { event := event216166
    frameStart := 216078 },
  { event := event216167
    frameStart := 216078 },
  { event := event216168
    frameStart := 216078 },
  { event := event216169
    frameStart := 216078 },
  { event := event216170
    frameStart := 216078 },
  { event := event216171
    frameStart := 216078 },
  { event := event216172
    frameStart := 216078 },
  { event := event216173
    frameStart := 216078 },
  { event := event216174
    frameStart := 216078 },
  { event := event216175
    frameStart := 216078 }
]

def eventLeaf13511 : Array AnnotatedEvent := #[
  { event := event216176
    frameStart := 216078 },
  { event := event216177
    frameStart := 216078 },
  { event := event216178
    frameStart := 216078 },
  { event := event216179
    frameStart := 216078 },
  { event := event216180
    frameStart := 216078 },
  { event := event216181
    frameStart := 216078 },
  { event := event216182
    frameStart := 0 },
  { event := event216183
    frameStart := 0 },
  { event := event216184
    frameStart := 0 },
  { event := event216185
    frameStart := 0 },
  { event := event216186
    frameStart := 0 },
  { event := event216187
    frameStart := 0 },
  { event := event216188
    frameStart := 0 },
  { event := event216189
    frameStart := 0 },
  { event := event216190
    frameStart := 0 },
  { event := event216191
    frameStart := 0 }
]

def eventLeaf13512 : Array AnnotatedEvent := #[
  { event := event216192
    frameStart := 0 },
  { event := event216193
    frameStart := 0 },
  { event := event216194
    frameStart := 0 },
  { event := event216195
    frameStart := 0 },
  { event := event216196
    frameStart := 0 },
  { event := event216197
    frameStart := 0 },
  { event := event216198
    frameStart := 0 },
  { event := event216199
    frameStart := 0 },
  { event := event216200
    frameStart := 0 },
  { event := event216201
    frameStart := 0 },
  { event := event216202
    frameStart := 0 },
  { event := event216203
    frameStart := 0 },
  { event := event216204
    frameStart := 0 },
  { event := event216205
    frameStart := 0 },
  { event := event216206
    frameStart := 0 },
  { event := event216207
    frameStart := 0 }
]

def eventLeaf13513 : Array AnnotatedEvent := #[
  { event := event216208
    frameStart := 0 },
  { event := event216209
    frameStart := 0 },
  { event := event216210
    frameStart := 0 },
  { event := event216211
    frameStart := 0 },
  { event := event216212
    frameStart := 0 },
  { event := event216213
    frameStart := 0 },
  { event := event216214
    frameStart := 0 },
  { event := event216215
    frameStart := 0 },
  { event := event216216
    frameStart := 0 },
  { event := event216217
    frameStart := 0 },
  { event := event216218
    frameStart := 0 },
  { event := event216219
    frameStart := 0 },
  { event := event216220
    frameStart := 0 },
  { event := event216221
    frameStart := 0 },
  { event := event216222
    frameStart := 0 },
  { event := event216223
    frameStart := 0 }
]

def eventLeaf13514 : Array AnnotatedEvent := #[
  { event := event216224
    frameStart := 0 },
  { event := event216225
    frameStart := 0 },
  { event := event216226
    frameStart := 0 },
  { event := event216227
    frameStart := 0 },
  { event := event216228
    frameStart := 0 },
  { event := event216229
    frameStart := 0 },
  { event := event216230
    frameStart := 0 },
  { event := event216231
    frameStart := 0 },
  { event := event216232
    frameStart := 0 },
  { event := event216233
    frameStart := 0 },
  { event := event216234
    frameStart := 0 },
  { event := event216235
    frameStart := 0 },
  { event := event216236
    frameStart := 0 },
  { event := event216237
    frameStart := 0 },
  { event := event216238
    frameStart := 0 },
  { event := event216239
    frameStart := 0 }
]

def eventLeaf13515 : Array AnnotatedEvent := #[
  { event := event216240
    frameStart := 0 },
  { event := event216241
    frameStart := 0 },
  { event := event216242
    frameStart := 0 },
  { event := event216243
    frameStart := 0 },
  { event := event216244
    frameStart := 0 },
  { event := event216245
    frameStart := 0 },
  { event := event216246
    frameStart := 0 },
  { event := event216247
    frameStart := 0 },
  { event := event216248
    frameStart := 0 },
  { event := event216249
    frameStart := 0 },
  { event := event216250
    frameStart := 0 },
  { event := event216251
    frameStart := 0 },
  { event := event216252
    frameStart := 0 },
  { event := event216253
    frameStart := 0 },
  { event := event216254
    frameStart := 0 },
  { event := event216255
    frameStart := 0 }
]

def eventLeaf13516 : Array AnnotatedEvent := #[
  { event := event216256
    frameStart := 0 },
  { event := event216257
    frameStart := 0 },
  { event := event216258
    frameStart := 0 },
  { event := event216259
    frameStart := 0 },
  { event := event216260
    frameStart := 0 },
  { event := event216261
    frameStart := 0 },
  { event := event216262
    frameStart := 0 },
  { event := event216263
    frameStart := 0 },
  { event := event216264
    frameStart := 0 },
  { event := event216265
    frameStart := 0 },
  { event := event216266
    frameStart := 0 },
  { event := event216267
    frameStart := 0 },
  { event := event216268
    frameStart := 0 },
  { event := event216269
    frameStart := 0 },
  { event := event216270
    frameStart := 0 },
  { event := event216271
    frameStart := 0 }
]

def eventLeaf13517 : Array AnnotatedEvent := #[
  { event := event216272
    frameStart := 0 },
  { event := event216273
    frameStart := 0 },
  { event := event216274
    frameStart := 0 },
  { event := event216275
    frameStart := 0 },
  { event := event216276
    frameStart := 0 },
  { event := event216277
    frameStart := 0 },
  { event := event216278
    frameStart := 0 },
  { event := event216279
    frameStart := 0 },
  { event := event216280
    frameStart := 0 },
  { event := event216281
    frameStart := 0 },
  { event := event216282
    frameStart := 0 },
  { event := event216283
    frameStart := 0 },
  { event := event216284
    frameStart := 0 },
  { event := event216285
    frameStart := 0 },
  { event := event216286
    frameStart := 0 },
  { event := event216287
    frameStart := 0 }
]

def eventLeaf13518 : Array AnnotatedEvent := #[
  { event := event216288
    frameStart := 0 },
  { event := event216289
    frameStart := 0 },
  { event := event216290
    frameStart := 0 },
  { event := event216291
    frameStart := 0 },
  { event := event216292
    frameStart := 0 },
  { event := event216293
    frameStart := 0 },
  { event := event216294
    frameStart := 0 },
  { event := event216295
    frameStart := 0 },
  { event := event216296
    frameStart := 0 },
  { event := event216297
    frameStart := 0 },
  { event := event216298
    frameStart := 0 },
  { event := event216299
    frameStart := 0 },
  { event := event216300
    frameStart := 0 },
  { event := event216301
    frameStart := 0 },
  { event := event216302
    frameStart := 0 },
  { event := event216303
    frameStart := 0 }
]

def eventLeaf13519 : Array AnnotatedEvent := #[
  { event := event216304
    frameStart := 0 },
  { event := event216305
    frameStart := 0 },
  { event := event216306
    frameStart := 0 },
  { event := event216307
    frameStart := 0 },
  { event := event216308
    frameStart := 0 },
  { event := event216309
    frameStart := 0 },
  { event := event216310
    frameStart := 0 },
  { event := event216311
    frameStart := 0 },
  { event := event216312
    frameStart := 0 },
  { event := event216313
    frameStart := 0 },
  { event := event216314
    frameStart := 0 },
  { event := event216315
    frameStart := 0 },
  { event := event216316
    frameStart := 0 },
  { event := event216317
    frameStart := 0 },
  { event := event216318
    frameStart := 0 },
  { event := event216319
    frameStart := 0 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events844
