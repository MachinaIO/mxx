import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events313

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event80128 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10269⟩⟩) 0 ⟨392⟩ 80127

def event80129 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10269⟩⟩) 1 ⟨10267⟩ 80119

def event80130 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10269⟩⟩) (.sum [.predecessor 0 80128 .coefficient, .predecessor 1 80129 .coefficient])

def event80131 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10269⟩⟩) (.finite 655355)

def event80132 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10325⟩⟩) 0 ⟨10269⟩ 80131

def event80133 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10325⟩⟩) 1 ⟨5426⟩ 80117

def event80134 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10325⟩⟩) (.identity (.predecessor 1 80133 .coefficient))

def event80135 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10325⟩⟩) (.finite 655360)

def event80136 : Event := .predecessor (⟨.program ⟨257⟩, ⟨25802⟩⟩) 0 ⟨10325⟩ 80135

def event80137 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨25802⟩⟩) (.authority (.programFamilyFact))

def exact80138RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨25802⟩⟩], []⟩, (1)⟩]

theorem exact80138RawTermsValid :
    exact80138RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80138 : Event := .resultExact (⟨.program ⟨257⟩, ⟨25802⟩⟩) exact80138RawTerms (.finite 28) 80137 .exactZero (none)

def event80139 : Event := .predecessor (⟨.program ⟨257⟩, ⟨65607⟩⟩) 0 ⟨10325⟩ 80135

def event80140 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨65607⟩⟩) (.authority (.programFamilyFact))

def exact80141RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨65607⟩⟩], []⟩, (1)⟩]

theorem exact80141RawTermsValid :
    exact80141RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80141 : Event := .resultExact (⟨.program ⟨257⟩, ⟨65607⟩⟩) exact80141RawTerms (.finite 28) 80140 .exactZero (none)

def event80142 : Event := .predecessor (⟨.program ⟨257⟩, ⟨65608⟩⟩) 0 ⟨65607⟩ 80141

def event80143 : Event := .predecessor (⟨.program ⟨257⟩, ⟨65608⟩⟩) 1 ⟨25802⟩ 80138

def event80144 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨65608⟩⟩) (.product (.predecessor 0 80142 .coefficient) (.predecessor 1 80143 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event80145 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨65608⟩⟩, .operator (⟨80141, 0⟩, ⟨80138, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨25802⟩⟩, ⟨.program ⟨257⟩, ⟨65607⟩⟩], []⟩, (1)⟩)

def exact80146RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨25802⟩⟩, ⟨.program ⟨257⟩, ⟨65607⟩⟩], []⟩, (1)⟩]

theorem exact80146RawTermsValid :
    exact80146RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80146 : Event := .resultExact (⟨.program ⟨257⟩, ⟨65608⟩⟩) exact80146RawTerms (.finite 784) 80144 .exactZero (none)

def event80147 : Event := .predecessor (⟨.program ⟨257⟩, ⟨65609⟩⟩) 0 ⟨65608⟩ 80146

def event80148 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨65609⟩⟩) (.identity (.predecessor 0 80147 .coefficient))

def event80149 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨65609⟩⟩) (.finite 784)

def event80150 : Event := .predecessor (⟨.program ⟨257⟩, ⟨65836⟩⟩) 0 ⟨65609⟩ 80149

def event80151 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨65836⟩⟩) (.authority (.programFamilyFact))

def exact80152RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨65836⟩⟩], []⟩, (1)⟩]

theorem exact80152RawTermsValid :
    exact80152RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80152 : Event := .resultExact (⟨.program ⟨257⟩, ⟨65836⟩⟩) exact80152RawTerms (.finite 28) 80151 .exactZero (none)

def event80153 : Event := .predecessor (⟨.program ⟨257⟩, ⟨65837⟩⟩) 0 ⟨65836⟩ 80152

def event80154 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨65837⟩⟩) (.identity (.predecessor 0 80153 .coefficient))

def event80155 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨65837⟩⟩) (.finite 28)

def event80156 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68734⟩⟩) 0 ⟨65837⟩ 80155

def event80157 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨68734⟩⟩) (.authority (.programFamilyFact))

def event80158 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨68734⟩⟩) (.finite 3720)

def event80159 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨7177⟩⟩) .missing

def event80160 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68736⟩⟩) 0 ⟨7177⟩ 80159

def event80161 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68736⟩⟩) 1 ⟨68734⟩ 80158

def event80162 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨68736⟩⟩) (.authority (.operator))

def exact80163RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨68736⟩⟩]⟩, (1)⟩]

theorem exact80163RawTermsValid :
    exact80163RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80163 : Event := .resultExact (⟨.program ⟨257⟩, ⟨68736⟩⟩) exact80163RawTerms .large 80162 .exactZero (none)

def event80164 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70651⟩⟩) 0 ⟨68736⟩ 80163

def event80165 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70651⟩⟩) (.authority (.operator))

def exact80166RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨70651⟩⟩]⟩, (1)⟩]

theorem exact80166RawTermsValid :
    exact80166RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80166 : Event := .resultExact (⟨.program ⟨257⟩, ⟨70651⟩⟩) exact80166RawTerms (.finite 8192) 80165 .exactZero (none)

def event80167 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨136⟩⟩) (.authority (.operator))

def event80168 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨136⟩⟩) .exactZero

def event80169 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69031⟩⟩) 0 ⟨65837⟩ 80155

def event80170 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69031⟩⟩) 1 ⟨136⟩ 80168

def event80171 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69031⟩⟩) (.sum [.predecessor 0 80169 .coefficient, .predecessor 1 80170 .coefficient])

def event80172 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨69031⟩⟩) (.finite 28)

def event80173 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69032⟩⟩) 0 ⟨69031⟩ 80172

def event80174 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69032⟩⟩) (.identity (.predecessor 0 80173 .coefficient))

def exact80175RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨65836⟩⟩], []⟩, (1)⟩]

theorem exact80175RawTermsValid :
    exact80175RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80175 : Event := .resultExact (⟨.program ⟨257⟩, ⟨69032⟩⟩) exact80175RawTerms (.finite 28) 80174 .exactZero (none)

def event80176 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6908⟩⟩) (.authority (.factStore))

def exact80177RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact80177RawTermsValid :
    exact80177RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80177 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6908⟩⟩) exact80177RawTerms .large 80176 .exactZero (none)

def event80178 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69033⟩⟩) 0 ⟨6908⟩ 80177

def event80179 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69033⟩⟩) 1 ⟨69032⟩ 80175

def event80180 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69033⟩⟩) (.product (.predecessor 0 80178 .coefficient) (.predecessor 1 80179 .coefficient) (⟨false, false, none, none, none⟩))

def event80181 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨69033⟩⟩, .operator (⟨80177, 0⟩, ⟨80175, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨65836⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact80182RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨65836⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact80182RawTermsValid :
    exact80182RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80182 : Event := .resultExact (⟨.program ⟨257⟩, ⟨69033⟩⟩) exact80182RawTerms .large 80180 .exactZero (none)

def event80183 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7188⟩⟩) 0 ⟨7177⟩ 80159

def event80184 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7188⟩⟩) (.authority (.operator))

def exact80185RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7188⟩⟩]⟩, (1)⟩]

theorem exact80185RawTermsValid :
    exact80185RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80185 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7188⟩⟩) exact80185RawTerms .large 80184 .exactZero (none)

def event80186 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69034⟩⟩) 0 ⟨7188⟩ 80185

def event80187 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69034⟩⟩) 1 ⟨69033⟩ 80182

def event80188 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69034⟩⟩) (.sum [.predecessor 0 80186 .coefficient, .predecessor 1 80187 .coefficient])

def exact80189RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7188⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨65836⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact80189RawTermsValid :
    exact80189RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80189 : Event := .resultExact (⟨.program ⟨257⟩, ⟨69034⟩⟩) exact80189RawTerms .large 80188 .exactZero (none)

def event80190 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70652⟩⟩) 0 ⟨69034⟩ 80189

def event80191 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70652⟩⟩) 1 ⟨70651⟩ 80166

def event80192 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70652⟩⟩) (.product (.predecessor 0 80190 .coefficient) (.predecessor 1 80191 .coefficient) (⟨false, false, none, none, none⟩))

def event80193 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70652⟩⟩, .operator (⟨80189, 0⟩, ⟨80166, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7188⟩⟩, ⟨.program ⟨257⟩, ⟨70651⟩⟩]⟩, (1)⟩)

def event80194 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70652⟩⟩, .operator (⟨80189, 1⟩, ⟨80166, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨65836⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70651⟩⟩]⟩, (-1)⟩)

def event80195 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨70652⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨65836⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70651⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨70651⟩⟩) ⟨68736⟩ 80163)

def event80196 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70652⟩⟩, .relation 80195 0, ⟨[⟨.program ⟨257⟩, ⟨65836⟩⟩], [⟨.program ⟨257⟩, ⟨68736⟩⟩]⟩, (-1)⟩)

def exact80197RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7188⟩⟩, ⟨.program ⟨257⟩, ⟨70651⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨65836⟩⟩], [⟨.program ⟨257⟩, ⟨68736⟩⟩]⟩, (-1)⟩]

theorem exact80197RawTermsValid :
    exact80197RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80197 : Event := .resultExact (⟨.program ⟨257⟩, ⟨70652⟩⟩) exact80197RawTerms .large 80192 .exactZero (none)

def event80198 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67021⟩⟩) 0 ⟨65837⟩ 80155

def event80199 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67021⟩⟩) (.authority (.programFamilyFact))

def exact80200RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨67021⟩⟩], []⟩, (1)⟩]

theorem exact80200RawTermsValid :
    exact80200RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80200 : Event := .resultExact (⟨.program ⟨257⟩, ⟨67021⟩⟩) exact80200RawTerms (.finite 62) 80199 .exactZero (none)

def event80201 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67032⟩⟩) 0 ⟨6908⟩ 80177

def event80202 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67032⟩⟩) 1 ⟨67021⟩ 80200

def event80203 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67032⟩⟩) (.product (.predecessor 0 80201 .coefficient) (.predecessor 1 80202 .coefficient) (⟨false, true, none, none, some 1⟩))

def event80204 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67032⟩⟩, .operator (⟨80177, 0⟩, ⟨80200, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨67021⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact80205RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨67021⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact80205RawTermsValid :
    exact80205RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80205 : Event := .resultExact (⟨.program ⟨257⟩, ⟨67032⟩⟩) exact80205RawTerms .large 80203 .exactZero (none)

def event80206 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7216⟩⟩) 0 ⟨7177⟩ 80159

def event80207 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7216⟩⟩) (.authority (.operator))

def exact80208RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩]

theorem exact80208RawTermsValid :
    exact80208RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80208 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7216⟩⟩) exact80208RawTerms .large 80207 .exactZero (none)

def event80209 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67033⟩⟩) 0 ⟨7216⟩ 80208

def event80210 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67033⟩⟩) 1 ⟨67032⟩ 80205

def event80211 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67033⟩⟩) (.sum [.predecessor 0 80209 .coefficient, .predecessor 1 80210 .coefficient])

def exact80212RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨67021⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact80212RawTermsValid :
    exact80212RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80212 : Event := .resultExact (⟨.program ⟨257⟩, ⟨67033⟩⟩) exact80212RawTerms .large 80211 .exactZero (none)

def event80213 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70664⟩⟩) 0 ⟨67033⟩ 80212

def event80214 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70664⟩⟩) 1 ⟨70652⟩ 80197

def event80215 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70664⟩⟩) (.sum [.predecessor 0 80213 .coefficient, .predecessor 1 80214 .coefficient])

def exact80216RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7188⟩⟩, ⟨.program ⟨257⟩, ⟨70651⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨65836⟩⟩], [⟨.program ⟨257⟩, ⟨68736⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨67021⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact80216RawTermsValid :
    exact80216RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80216 : Event := .resultExact (⟨.program ⟨257⟩, ⟨70664⟩⟩) exact80216RawTerms .large 80215 .exactZero (none)

def event80217 : Event := .preFoldPolynomial 80216 [⟨⟨[], [⟨.program ⟨257⟩, ⟨7188⟩⟩, ⟨.program ⟨257⟩, ⟨70651⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨65836⟩⟩], [⟨.program ⟨257⟩, ⟨68736⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨67021⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩] .exactZero none

def exact80218RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7188⟩⟩, ⟨.program ⟨257⟩, ⟨70651⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨65836⟩⟩], [⟨.program ⟨257⟩, ⟨68736⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨67021⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

def event80218 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨70664⟩⟩) 80217 exact80218RawTerms .large 80215 .exactZero (none)

def event80219 : Event := .specializationComputed (⟨.program ⟨257⟩, ⟨65837⟩⟩) ⟨⟨95⟩, ⟨76⟩, ⟨135⟩⟩ ⟨80061, 80219⟩

def event80220 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨68200⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68197⟩⟩]⟩) (1) 0 2 (.universal 80219 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68197⟩⟩]⟩) (none) 80218)

def event80221 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨68200⟩⟩, .relation 80220 1, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩)

def event80222 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨68200⟩⟩, .relation 80220 0, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7188⟩⟩, ⟨.program ⟨257⟩, ⟨70651⟩⟩]⟩, (-1)⟩)

def event80223 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨68200⟩⟩, .relation 80220 2, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨65836⟩⟩], [⟨.program ⟨257⟩, ⟨68736⟩⟩]⟩, (1)⟩)

def event80224 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨68200⟩⟩, .relation 80220 3, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨67021⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact80225RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7188⟩⟩, ⟨.program ⟨257⟩, ⟨70651⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨65836⟩⟩], [⟨.program ⟨257⟩, ⟨68736⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨67021⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact80225RawTermsValid :
    exact80225RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80225 : Event := .resultExact (⟨.program ⟨257⟩, ⟨68200⟩⟩) exact80225RawTerms .large 80057 (.finite 202072841853861888) (some (80059))

def event80226 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70654⟩⟩) 0 ⟨68200⟩ 80225

def event80227 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70654⟩⟩) 1 ⟨70653⟩ 80047

def event80228 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70654⟩⟩) (.sum [.predecessor 0 80226 .coefficient, .predecessor 1 80227 .coefficient])

def event80229 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70654⟩⟩, .operator (⟨80225, 0⟩, ⟨80047, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7188⟩⟩, ⟨.program ⟨257⟩, ⟨70651⟩⟩]⟩, (1)⟩)

def event80230 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70654⟩⟩, .operator (⟨80225, 2⟩, ⟨80047, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨65836⟩⟩], [⟨.program ⟨257⟩, ⟨68736⟩⟩]⟩, (-1)⟩)

def event80231 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70654⟩⟩) (.sum [.result 80225 .summary, .result 80047 .summary])

def exact80232RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨67021⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact80232RawTermsValid :
    exact80232RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80232 : Event := .resultExact (⟨.program ⟨257⟩, ⟨70654⟩⟩) exact80232RawTerms .large 80228 (.finite 32191361068277642793642192273408) (some (80231))

def event80233 : Event := .predecessor (⟨.program ⟨257⟩, ⟨64133⟩⟩) 0 ⟨62857⟩ 3310

def event80234 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨64133⟩⟩) (.authority (.programFamilyFact))

def event80235 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨64133⟩⟩) (.finite 3720)

def event80236 : Event := .predecessor (⟨.program ⟨257⟩, ⟨64135⟩⟩) 0 ⟨7177⟩ 15500

def event80237 : Event := .predecessor (⟨.program ⟨257⟩, ⟨64135⟩⟩) 1 ⟨64133⟩ 80235

def event80238 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨64135⟩⟩) (.authority (.operator))

def exact80239RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨64135⟩⟩]⟩, (1)⟩]

theorem exact80239RawTermsValid :
    exact80239RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80239 : Event := .resultExact (⟨.program ⟨257⟩, ⟨64135⟩⟩) exact80239RawTerms .large 80238 .exactZero (none)

def event80240 : Event := .predecessor (⟨.program ⟨257⟩, ⟨65058⟩⟩) 0 ⟨64135⟩ 80239

def event80241 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨65058⟩⟩) (.authority (.operator))

def exact80242RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨65058⟩⟩]⟩, (1)⟩]

theorem exact80242RawTermsValid :
    exact80242RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80242 : Event := .resultExact (⟨.program ⟨257⟩, ⟨65058⟩⟩) exact80242RawTerms (.finite 8192) 80241 .exactZero (none)

def event80243 : Event := .predecessor (⟨.program ⟨257⟩, ⟨63964⟩⟩) 0 ⟨62629⟩ 3304

def event80244 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨63964⟩⟩) (.authority (.programFamilyFact))

def event80245 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨63964⟩⟩) (.finite 3720)

def event80246 : Event := .predecessor (⟨.program ⟨257⟩, ⟨63965⟩⟩) 0 ⟨7177⟩ 15500

def event80247 : Event := .predecessor (⟨.program ⟨257⟩, ⟨63965⟩⟩) 1 ⟨63964⟩ 80245

def event80248 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨63965⟩⟩) (.authority (.operator))

def exact80249RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨63965⟩⟩]⟩, (1)⟩]

theorem exact80249RawTermsValid :
    exact80249RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80249 : Event := .resultExact (⟨.program ⟨257⟩, ⟨63965⟩⟩) exact80249RawTerms .large 80248 .exactZero (none)

def event80250 : Event := .predecessor (⟨.program ⟨257⟩, ⟨64505⟩⟩) 0 ⟨63965⟩ 80249

def event80251 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨64505⟩⟩) (.authority (.operator))

def exact80252RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨64505⟩⟩]⟩, (1)⟩]

theorem exact80252RawTermsValid :
    exact80252RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80252 : Event := .resultExact (⟨.program ⟨257⟩, ⟨64505⟩⟩) exact80252RawTerms (.finite 8192) 80251 .exactZero (none)

def event80253 : Event := .predecessor (⟨.program ⟨257⟩, ⟨25563⟩⟩) 0 ⟨25562⟩ 3293

def event80254 : Event := .predecessor (⟨.program ⟨257⟩, ⟨25563⟩⟩) 1 ⟨10328⟩ 75903

def event80255 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨25563⟩⟩) (.tensor (.predecessor 0 80253 .coefficient) (.predecessor 1 80254 .coefficient) true false)

def event80256 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨25563⟩⟩, .operator (⟨3293, 0⟩, ⟨75903, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨25562⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact80257RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨25562⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact80257RawTermsValid :
    exact80257RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80257 : Event := .resultExact (⟨.program ⟨257⟩, ⟨25563⟩⟩) exact80257RawTerms .large 80255 .exactZero (none)

def event80258 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10333⟩⟩) 0 ⟨10327⟩ 75773

def event80259 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10333⟩⟩) 1 ⟨7275⟩ 21589

def event80260 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10333⟩⟩) (.product (.predecessor 0 80258 .coefficient) (.predecessor 1 80259 .coefficient) (⟨false, false, none, none, none⟩))

def event80261 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨10333⟩⟩, .operator (⟨75773, 0⟩, ⟨21589, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7275⟩⟩]⟩, (1)⟩)

def exact80262RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7275⟩⟩]⟩, (1)⟩]

theorem exact80262RawTermsValid :
    exact80262RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80262 : Event := .resultExact (⟨.program ⟨257⟩, ⟨10333⟩⟩) exact80262RawTerms .large 80260 .exactZero (none)

def event80263 : Event := .predecessor (⟨.program ⟨257⟩, ⟨25564⟩⟩) 0 ⟨10333⟩ 80262

def event80264 : Event := .predecessor (⟨.program ⟨257⟩, ⟨25564⟩⟩) 1 ⟨25563⟩ 80257

def event80265 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨25564⟩⟩) (.sum [.predecessor 0 80263 .coefficient, .predecessor 1 80264 .coefficient])

def exact80266RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7275⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨25562⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact80266RawTermsValid :
    exact80266RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80266 : Event := .resultExact (⟨.program ⟨257⟩, ⟨25564⟩⟩) exact80266RawTerms .large 80265 .exactZero (none)

def event80267 : Event := .predecessor (⟨.program ⟨257⟩, ⟨25565⟩⟩) 0 ⟨25564⟩ 80266

def event80268 : Event := .predecessor (⟨.program ⟨257⟩, ⟨25565⟩⟩) 1 ⟨101⟩ 21581

def event80269 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨25565⟩⟩) (.sum [.predecessor 0 80267 .coefficient, .predecessor 1 80268 .coefficient])

def event80270 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨25565⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨101⟩⟩]⟩) [⟨.result 21581 .coefficient, false, none⟩])

def event80271 : Event := .survivorFold (1) 80270

def exact80272RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7275⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨25562⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact80272RawTermsValid :
    exact80272RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80272 : Event := .resultExact (⟨.program ⟨257⟩, ⟨25565⟩⟩) exact80272RawTerms .large 80269 (.finite 26) (some (80270))

def event80273 : Event := .predecessor (⟨.program ⟨257⟩, ⟨62630⟩⟩) 0 ⟨25565⟩ 80272

def event80274 : Event := .predecessor (⟨.program ⟨257⟩, ⟨62630⟩⟩) 1 ⟨62627⟩ 3296

def event80275 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨62630⟩⟩) (.product (.predecessor 0 80273 .coefficient) (.predecessor 1 80274 .coefficient) (⟨false, true, none, none, some 1⟩))

def event80276 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨62630⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨62627⟩⟩], []⟩) [⟨.result 3296 .coefficient, true, some 1⟩])

def event80277 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨62630⟩⟩) (.product (.result 80272 .summary) (.transfer 80276) (⟨false, false, none, none, none⟩))

def event80278 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨62630⟩⟩, .operator (⟨80272, 1⟩, ⟨3296, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨25562⟩⟩, ⟨.program ⟨257⟩, ⟨62627⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def event80279 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨62630⟩⟩, .operator (⟨80272, 0⟩, ⟨3296, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨62627⟩⟩], [⟨.program ⟨257⟩, ⟨7275⟩⟩]⟩, (1)⟩)

def exact80280RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨25562⟩⟩, ⟨.program ⟨257⟩, ⟨62627⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨62627⟩⟩], [⟨.program ⟨257⟩, ⟨7275⟩⟩]⟩, (1)⟩]

theorem exact80280RawTermsValid :
    exact80280RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80280 : Event := .resultExact (⟨.program ⟨257⟩, ⟨62630⟩⟩) exact80280RawTerms .large 80275 (.finite 18743296) (some (80277))

def event80281 : Event := .predecessor (⟨.program ⟨257⟩, ⟨62631⟩⟩) 0 ⟨62627⟩ 3296

def event80282 : Event := .predecessor (⟨.program ⟨257⟩, ⟨62631⟩⟩) 1 ⟨10328⟩ 75903

def event80283 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨62631⟩⟩) (.tensor (.predecessor 0 80281 .coefficient) (.predecessor 1 80282 .coefficient) true false)

def event80284 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨62631⟩⟩, .operator (⟨3296, 0⟩, ⟨75903, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨62627⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact80285RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨62627⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact80285RawTermsValid :
    exact80285RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80285 : Event := .resultExact (⟨.program ⟨257⟩, ⟨62631⟩⟩) exact80285RawTerms .large 80283 .exactZero (none)

def event80286 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10351⟩⟩) 0 ⟨10327⟩ 75773

def event80287 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10351⟩⟩) 1 ⟨7293⟩ 21630

def event80288 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10351⟩⟩) (.product (.predecessor 0 80286 .coefficient) (.predecessor 1 80287 .coefficient) (⟨false, false, none, none, none⟩))

def event80289 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨10351⟩⟩, .operator (⟨75773, 0⟩, ⟨21630, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7293⟩⟩]⟩, (1)⟩)

def exact80290RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7293⟩⟩]⟩, (1)⟩]

theorem exact80290RawTermsValid :
    exact80290RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80290 : Event := .resultExact (⟨.program ⟨257⟩, ⟨10351⟩⟩) exact80290RawTerms .large 80288 .exactZero (none)

def event80291 : Event := .predecessor (⟨.program ⟨257⟩, ⟨62632⟩⟩) 0 ⟨10351⟩ 80290

def event80292 : Event := .predecessor (⟨.program ⟨257⟩, ⟨62632⟩⟩) 1 ⟨62631⟩ 80285

def event80293 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨62632⟩⟩) (.sum [.predecessor 0 80291 .coefficient, .predecessor 1 80292 .coefficient])

def exact80294RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7293⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨62627⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact80294RawTermsValid :
    exact80294RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80294 : Event := .resultExact (⟨.program ⟨257⟩, ⟨62632⟩⟩) exact80294RawTerms .large 80293 .exactZero (none)

def event80295 : Event := .predecessor (⟨.program ⟨257⟩, ⟨62633⟩⟩) 0 ⟨62632⟩ 80294

def event80296 : Event := .predecessor (⟨.program ⟨257⟩, ⟨62633⟩⟩) 1 ⟨119⟩ 21622

def event80297 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨62633⟩⟩) (.sum [.predecessor 0 80295 .coefficient, .predecessor 1 80296 .coefficient])

def event80298 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨62633⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨119⟩⟩]⟩) [⟨.result 21622 .coefficient, false, none⟩])

def event80299 : Event := .survivorFold (1) 80298

def exact80300RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7293⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨62627⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact80300RawTermsValid :
    exact80300RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80300 : Event := .resultExact (⟨.program ⟨257⟩, ⟨62633⟩⟩) exact80300RawTerms .large 80297 (.finite 26) (some (80298))

def event80301 : Event := .predecessor (⟨.program ⟨257⟩, ⟨62634⟩⟩) 0 ⟨62633⟩ 80300

def event80302 : Event := .predecessor (⟨.program ⟨257⟩, ⟨62634⟩⟩) 1 ⟨9539⟩ 21619

def event80303 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨62634⟩⟩) (.product (.predecessor 0 80301 .coefficient) (.predecessor 1 80302 .coefficient) (⟨false, false, none, none, none⟩))

def event80304 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨62634⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨9538⟩⟩]⟩) [⟨.result 21615 .coefficient, false, none⟩])

def event80305 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨62634⟩⟩) (.product (.result 80300 .summary) (.transfer 80304) (⟨false, false, none, none, none⟩))

def event80306 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨62634⟩⟩, .operator (⟨80300, 1⟩, ⟨21619, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨62627⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩]⟩, (-1)⟩)

def event80307 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨62634⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨62627⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨9538⟩⟩) ⟨7275⟩ 21589)

def event80308 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨62634⟩⟩, .relation 80307 0, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨62627⟩⟩], [⟨.program ⟨257⟩, ⟨7275⟩⟩]⟩, (-1)⟩)

def event80309 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨62634⟩⟩, .operator (⟨80300, 0⟩, ⟨21619, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩]⟩, (1)⟩)

def exact80310RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨62627⟩⟩], [⟨.program ⟨257⟩, ⟨7275⟩⟩]⟩, (-1)⟩]

theorem exact80310RawTermsValid :
    exact80310RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80310 : Event := .resultExact (⟨.program ⟨257⟩, ⟨62634⟩⟩) exact80310RawTerms .large 80303 (.finite 279172874240) (some (80305))

def event80311 : Event := .predecessor (⟨.program ⟨257⟩, ⟨62635⟩⟩) 0 ⟨62634⟩ 80310

def event80312 : Event := .predecessor (⟨.program ⟨257⟩, ⟨62635⟩⟩) 1 ⟨62630⟩ 80280

def event80313 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨62635⟩⟩) (.sum [.predecessor 0 80311 .coefficient, .predecessor 1 80312 .coefficient])

def event80314 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨62635⟩⟩, .operator (⟨80310, 1⟩, ⟨80280, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨62627⟩⟩], [⟨.program ⟨257⟩, ⟨7275⟩⟩]⟩, (1)⟩)

def event80315 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨62635⟩⟩) (.sum [.result 80310 .summary, .result 80280 .summary])

def exact80316RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨25562⟩⟩, ⟨.program ⟨257⟩, ⟨62627⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact80316RawTermsValid :
    exact80316RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80316 : Event := .resultExact (⟨.program ⟨257⟩, ⟨62635⟩⟩) exact80316RawTerms .large 80313 (.finite 279191617536) (some (80315))

def event80317 : Event := .predecessor (⟨.program ⟨257⟩, ⟨64506⟩⟩) 0 ⟨62635⟩ 80316

def event80318 : Event := .predecessor (⟨.program ⟨257⟩, ⟨64506⟩⟩) 1 ⟨64505⟩ 80252

def event80319 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨64506⟩⟩) (.product (.predecessor 0 80317 .coefficient) (.predecessor 1 80318 .coefficient) (⟨false, false, none, none, none⟩))

def event80320 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨64506⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨64505⟩⟩]⟩) [⟨.result 80252 .coefficient, false, none⟩])

def event80321 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨64506⟩⟩) (.product (.result 80316 .summary) (.transfer 80320) (⟨false, false, none, none, none⟩))

def event80322 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨64506⟩⟩, .operator (⟨80316, 1⟩, ⟨80252, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨25562⟩⟩, ⟨.program ⟨257⟩, ⟨62627⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨64505⟩⟩]⟩, (-1)⟩)

def event80323 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨64506⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨25562⟩⟩, ⟨.program ⟨257⟩, ⟨62627⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨64505⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨64505⟩⟩) ⟨63965⟩ 80249)

def event80324 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨64506⟩⟩, .relation 80323 0, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨25562⟩⟩, ⟨.program ⟨257⟩, ⟨62627⟩⟩], [⟨.program ⟨257⟩, ⟨63965⟩⟩]⟩, (-1)⟩)

def event80325 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨64506⟩⟩, .operator (⟨80316, 0⟩, ⟨80252, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩, ⟨.program ⟨257⟩, ⟨64505⟩⟩]⟩, (1)⟩)

def exact80326RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩, ⟨.program ⟨257⟩, ⟨64505⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨25562⟩⟩, ⟨.program ⟨257⟩, ⟨62627⟩⟩], [⟨.program ⟨257⟩, ⟨63965⟩⟩]⟩, (-1)⟩]

theorem exact80326RawTermsValid :
    exact80326RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80326 : Event := .resultExact (⟨.program ⟨257⟩, ⟨64506⟩⟩) exact80326RawTerms .large 80319 (.finite 2997797166586150256640) (some (80321))

def event80327 : Event := .predecessor (⟨.program ⟨257⟩, ⟨63429⟩⟩) 0 ⟨62629⟩ 3304

def event80328 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨63429⟩⟩) (.authority (.relationPreimageSource ⟨45⟩))

def exact80329RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨63429⟩⟩]⟩, (1)⟩]

theorem exact80329RawTermsValid :
    exact80329RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80329 : Event := .resultExact (⟨.program ⟨257⟩, ⟨63429⟩⟩) exact80329RawTerms (.finite 5647228698) 80328 .exactZero (none)

def event80330 : Event := .predecessor (⟨.program ⟨257⟩, ⟨63431⟩⟩) 0 ⟨63429⟩ 80329

def event80331 : Event := .predecessor (⟨.program ⟨257⟩, ⟨63431⟩⟩) 1 ⟨2370⟩ 4

def event80332 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨63431⟩⟩) (.scale (.predecessor 0 80330 .coefficient) (.value (.predecessor 1 80331 .coefficient)))

def exact80333RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨63429⟩⟩]⟩, (1)⟩]

theorem exact80333RawTermsValid :
    exact80333RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80333 : Event := .resultExact (⟨.program ⟨257⟩, ⟨63431⟩⟩) exact80333RawTerms (.finite 5647228698) 80332 .exactZero (none)

def event80334 : Event := .predecessor (⟨.program ⟨257⟩, ⟨63432⟩⟩) 0 ⟨10368⟩ 75995

def event80335 : Event := .predecessor (⟨.program ⟨257⟩, ⟨63432⟩⟩) 1 ⟨63431⟩ 80333

def event80336 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨63432⟩⟩) (.product (.predecessor 0 80334 .coefficient) (.predecessor 1 80335 .coefficient) (⟨false, false, none, none, none⟩))

def event80337 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨63432⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨63429⟩⟩]⟩) [⟨.result 80329 .coefficient, false, none⟩])

def event80338 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨63432⟩⟩) (.product (.result 75995 .summary) (.transfer 80337) (⟨false, false, none, none, none⟩))

def event80339 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨63432⟩⟩, .operator (⟨75995, 0⟩, ⟨80333, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63429⟩⟩]⟩, (1)⟩)

def event80340 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨63430⟩⟩)

def event80341 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event80342 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event80343 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10267⟩⟩) (.authority (.operator))

def event80344 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10267⟩⟩) (.finite 15)

def event80345 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event80346 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event80347 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event80348 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event80349 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 80348

def event80350 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 80346

def event80351 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 80349 .coefficient) (.value (.predecessor 1 80350 .coefficient)))

def event80352 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event80353 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10269⟩⟩) 0 ⟨392⟩ 80352

def event80354 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10269⟩⟩) 1 ⟨10267⟩ 80344

def event80355 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10269⟩⟩) (.sum [.predecessor 0 80353 .coefficient, .predecessor 1 80354 .coefficient])

def event80356 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10269⟩⟩) (.finite 655355)

def event80357 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10325⟩⟩) 0 ⟨10269⟩ 80356

def event80358 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10325⟩⟩) 1 ⟨5426⟩ 80342

def event80359 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10325⟩⟩) (.identity (.predecessor 1 80358 .coefficient))

def event80360 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10325⟩⟩) (.finite 655360)

def event80361 : Event := .predecessor (⟨.program ⟨257⟩, ⟨25562⟩⟩) 0 ⟨10325⟩ 80360

def event80362 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨25562⟩⟩) (.authority (.programFamilyFact))

def exact80363RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨25562⟩⟩], []⟩, (1)⟩]

theorem exact80363RawTermsValid :
    exact80363RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80363 : Event := .resultExact (⟨.program ⟨257⟩, ⟨25562⟩⟩) exact80363RawTerms (.finite 22) 80362 .exactZero (none)

def event80364 : Event := .predecessor (⟨.program ⟨257⟩, ⟨62627⟩⟩) 0 ⟨10325⟩ 80360

def event80365 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨62627⟩⟩) (.authority (.programFamilyFact))

def exact80366RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨62627⟩⟩], []⟩, (1)⟩]

theorem exact80366RawTermsValid :
    exact80366RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80366 : Event := .resultExact (⟨.program ⟨257⟩, ⟨62627⟩⟩) exact80366RawTerms (.finite 22) 80365 .exactZero (none)

def event80367 : Event := .predecessor (⟨.program ⟨257⟩, ⟨62628⟩⟩) 0 ⟨62627⟩ 80366

def event80368 : Event := .predecessor (⟨.program ⟨257⟩, ⟨62628⟩⟩) 1 ⟨25562⟩ 80363

def event80369 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨62628⟩⟩) (.product (.predecessor 0 80367 .coefficient) (.predecessor 1 80368 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event80370 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨62628⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨25562⟩⟩, ⟨.program ⟨257⟩, ⟨62627⟩⟩], []⟩) [⟨.result 80366 .coefficient, true, some 1⟩, ⟨.result 80363 .coefficient, true, some 1⟩])

def event80371 : Event := .survivorFold (1) 80370

def exact80372RawTerms : List Term := []

theorem exact80372RawTermsValid :
    exact80372RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80372 : Event := .resultExact (⟨.program ⟨257⟩, ⟨62628⟩⟩) exact80372RawTerms (.finite 484) 80369 (.finite 484) (some (80370))

def event80373 : Event := .predecessor (⟨.program ⟨257⟩, ⟨62629⟩⟩) 0 ⟨62628⟩ 80372

def event80374 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨62629⟩⟩) (.identity (.predecessor 0 80373 .coefficient))

def event80375 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨62629⟩⟩) (.finite 484)

def event80376 : Event := .predecessor (⟨.program ⟨257⟩, ⟨63429⟩⟩) 0 ⟨62629⟩ 80375

def event80377 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨63429⟩⟩) (.authority (.relationPreimageSource ⟨45⟩))

def exact80378RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨63429⟩⟩]⟩, (1)⟩]

theorem exact80378RawTermsValid :
    exact80378RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80378 : Event := .resultExact (⟨.program ⟨257⟩, ⟨63429⟩⟩) exact80378RawTerms (.finite 5647228698) 80377 .exactZero (none)

def event80379 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35⟩⟩) (.authority (.operator))

def exact80380RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩]⟩, (1)⟩]

theorem exact80380RawTermsValid :
    exact80380RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80380 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35⟩⟩) exact80380RawTerms .large 80379 .exactZero (none)

def event80381 : Event := .predecessor (⟨.program ⟨257⟩, ⟨63430⟩⟩) 0 ⟨35⟩ 80380

def event80382 : Event := .predecessor (⟨.program ⟨257⟩, ⟨63430⟩⟩) 1 ⟨63429⟩ 80378

def event80383 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨63430⟩⟩) (.product (.predecessor 0 80381 .coefficient) (.predecessor 1 80382 .coefficient) (⟨false, false, none, none, none⟩))

def eventLeaf5008 : Array AnnotatedEvent := #[
  { event := event80128
    frameStart := 80115 },
  { event := event80129
    frameStart := 80115 },
  { event := event80130
    frameStart := 80115 },
  { event := event80131
    frameStart := 80115 },
  { event := event80132
    frameStart := 80115 },
  { event := event80133
    frameStart := 80115 },
  { event := event80134
    frameStart := 80115 },
  { event := event80135
    frameStart := 80115 },
  { event := event80136
    frameStart := 80115 },
  { event := event80137
    frameStart := 80115 },
  { event := event80138
    frameStart := 80115 },
  { event := event80139
    frameStart := 80115 },
  { event := event80140
    frameStart := 80115 },
  { event := event80141
    frameStart := 80115 },
  { event := event80142
    frameStart := 80115 },
  { event := event80143
    frameStart := 80115 }
]

def eventLeaf5009 : Array AnnotatedEvent := #[
  { event := event80144
    frameStart := 80115 },
  { event := event80145
    frameStart := 80115 },
  { event := event80146
    frameStart := 80115 },
  { event := event80147
    frameStart := 80115 },
  { event := event80148
    frameStart := 80115 },
  { event := event80149
    frameStart := 80115 },
  { event := event80150
    frameStart := 80115 },
  { event := event80151
    frameStart := 80115 },
  { event := event80152
    frameStart := 80115 },
  { event := event80153
    frameStart := 80115 },
  { event := event80154
    frameStart := 80115 },
  { event := event80155
    frameStart := 80115 },
  { event := event80156
    frameStart := 80115 },
  { event := event80157
    frameStart := 80115 },
  { event := event80158
    frameStart := 80115 },
  { event := event80159
    frameStart := 80115 }
]

def eventLeaf5010 : Array AnnotatedEvent := #[
  { event := event80160
    frameStart := 80115 },
  { event := event80161
    frameStart := 80115 },
  { event := event80162
    frameStart := 80115 },
  { event := event80163
    frameStart := 80115 },
  { event := event80164
    frameStart := 80115 },
  { event := event80165
    frameStart := 80115 },
  { event := event80166
    frameStart := 80115 },
  { event := event80167
    frameStart := 80115 },
  { event := event80168
    frameStart := 80115 },
  { event := event80169
    frameStart := 80115 },
  { event := event80170
    frameStart := 80115 },
  { event := event80171
    frameStart := 80115 },
  { event := event80172
    frameStart := 80115 },
  { event := event80173
    frameStart := 80115 },
  { event := event80174
    frameStart := 80115 },
  { event := event80175
    frameStart := 80115 }
]

def eventLeaf5011 : Array AnnotatedEvent := #[
  { event := event80176
    frameStart := 80115 },
  { event := event80177
    frameStart := 80115 },
  { event := event80178
    frameStart := 80115 },
  { event := event80179
    frameStart := 80115 },
  { event := event80180
    frameStart := 80115 },
  { event := event80181
    frameStart := 80115 },
  { event := event80182
    frameStart := 80115 },
  { event := event80183
    frameStart := 80115 },
  { event := event80184
    frameStart := 80115 },
  { event := event80185
    frameStart := 80115 },
  { event := event80186
    frameStart := 80115 },
  { event := event80187
    frameStart := 80115 },
  { event := event80188
    frameStart := 80115 },
  { event := event80189
    frameStart := 80115 },
  { event := event80190
    frameStart := 80115 },
  { event := event80191
    frameStart := 80115 }
]

def eventLeaf5012 : Array AnnotatedEvent := #[
  { event := event80192
    frameStart := 80115 },
  { event := event80193
    frameStart := 80115 },
  { event := event80194
    frameStart := 80115 },
  { event := event80195
    frameStart := 80115 },
  { event := event80196
    frameStart := 80115 },
  { event := event80197
    frameStart := 80115 },
  { event := event80198
    frameStart := 80115 },
  { event := event80199
    frameStart := 80115 },
  { event := event80200
    frameStart := 80115 },
  { event := event80201
    frameStart := 80115 },
  { event := event80202
    frameStart := 80115 },
  { event := event80203
    frameStart := 80115 },
  { event := event80204
    frameStart := 80115 },
  { event := event80205
    frameStart := 80115 },
  { event := event80206
    frameStart := 80115 },
  { event := event80207
    frameStart := 80115 }
]

def eventLeaf5013 : Array AnnotatedEvent := #[
  { event := event80208
    frameStart := 80115 },
  { event := event80209
    frameStart := 80115 },
  { event := event80210
    frameStart := 80115 },
  { event := event80211
    frameStart := 80115 },
  { event := event80212
    frameStart := 80115 },
  { event := event80213
    frameStart := 80115 },
  { event := event80214
    frameStart := 80115 },
  { event := event80215
    frameStart := 80115 },
  { event := event80216
    frameStart := 80115 },
  { event := event80217
    frameStart := 80115 },
  { event := event80218
    frameStart := 80115 },
  { event := event80219
    frameStart := 0 },
  { event := event80220
    frameStart := 0 },
  { event := event80221
    frameStart := 0 },
  { event := event80222
    frameStart := 0 },
  { event := event80223
    frameStart := 0 }
]

def eventLeaf5014 : Array AnnotatedEvent := #[
  { event := event80224
    frameStart := 0 },
  { event := event80225
    frameStart := 0 },
  { event := event80226
    frameStart := 0 },
  { event := event80227
    frameStart := 0 },
  { event := event80228
    frameStart := 0 },
  { event := event80229
    frameStart := 0 },
  { event := event80230
    frameStart := 0 },
  { event := event80231
    frameStart := 0 },
  { event := event80232
    frameStart := 0 },
  { event := event80233
    frameStart := 0 },
  { event := event80234
    frameStart := 0 },
  { event := event80235
    frameStart := 0 },
  { event := event80236
    frameStart := 0 },
  { event := event80237
    frameStart := 0 },
  { event := event80238
    frameStart := 0 },
  { event := event80239
    frameStart := 0 }
]

def eventLeaf5015 : Array AnnotatedEvent := #[
  { event := event80240
    frameStart := 0 },
  { event := event80241
    frameStart := 0 },
  { event := event80242
    frameStart := 0 },
  { event := event80243
    frameStart := 0 },
  { event := event80244
    frameStart := 0 },
  { event := event80245
    frameStart := 0 },
  { event := event80246
    frameStart := 0 },
  { event := event80247
    frameStart := 0 },
  { event := event80248
    frameStart := 0 },
  { event := event80249
    frameStart := 0 },
  { event := event80250
    frameStart := 0 },
  { event := event80251
    frameStart := 0 },
  { event := event80252
    frameStart := 0 },
  { event := event80253
    frameStart := 0 },
  { event := event80254
    frameStart := 0 },
  { event := event80255
    frameStart := 0 }
]

def eventLeaf5016 : Array AnnotatedEvent := #[
  { event := event80256
    frameStart := 0 },
  { event := event80257
    frameStart := 0 },
  { event := event80258
    frameStart := 0 },
  { event := event80259
    frameStart := 0 },
  { event := event80260
    frameStart := 0 },
  { event := event80261
    frameStart := 0 },
  { event := event80262
    frameStart := 0 },
  { event := event80263
    frameStart := 0 },
  { event := event80264
    frameStart := 0 },
  { event := event80265
    frameStart := 0 },
  { event := event80266
    frameStart := 0 },
  { event := event80267
    frameStart := 0 },
  { event := event80268
    frameStart := 0 },
  { event := event80269
    frameStart := 0 },
  { event := event80270
    frameStart := 0 },
  { event := event80271
    frameStart := 0 }
]

def eventLeaf5017 : Array AnnotatedEvent := #[
  { event := event80272
    frameStart := 0 },
  { event := event80273
    frameStart := 0 },
  { event := event80274
    frameStart := 0 },
  { event := event80275
    frameStart := 0 },
  { event := event80276
    frameStart := 0 },
  { event := event80277
    frameStart := 0 },
  { event := event80278
    frameStart := 0 },
  { event := event80279
    frameStart := 0 },
  { event := event80280
    frameStart := 0 },
  { event := event80281
    frameStart := 0 },
  { event := event80282
    frameStart := 0 },
  { event := event80283
    frameStart := 0 },
  { event := event80284
    frameStart := 0 },
  { event := event80285
    frameStart := 0 },
  { event := event80286
    frameStart := 0 },
  { event := event80287
    frameStart := 0 }
]

def eventLeaf5018 : Array AnnotatedEvent := #[
  { event := event80288
    frameStart := 0 },
  { event := event80289
    frameStart := 0 },
  { event := event80290
    frameStart := 0 },
  { event := event80291
    frameStart := 0 },
  { event := event80292
    frameStart := 0 },
  { event := event80293
    frameStart := 0 },
  { event := event80294
    frameStart := 0 },
  { event := event80295
    frameStart := 0 },
  { event := event80296
    frameStart := 0 },
  { event := event80297
    frameStart := 0 },
  { event := event80298
    frameStart := 0 },
  { event := event80299
    frameStart := 0 },
  { event := event80300
    frameStart := 0 },
  { event := event80301
    frameStart := 0 },
  { event := event80302
    frameStart := 0 },
  { event := event80303
    frameStart := 0 }
]

def eventLeaf5019 : Array AnnotatedEvent := #[
  { event := event80304
    frameStart := 0 },
  { event := event80305
    frameStart := 0 },
  { event := event80306
    frameStart := 0 },
  { event := event80307
    frameStart := 0 },
  { event := event80308
    frameStart := 0 },
  { event := event80309
    frameStart := 0 },
  { event := event80310
    frameStart := 0 },
  { event := event80311
    frameStart := 0 },
  { event := event80312
    frameStart := 0 },
  { event := event80313
    frameStart := 0 },
  { event := event80314
    frameStart := 0 },
  { event := event80315
    frameStart := 0 },
  { event := event80316
    frameStart := 0 },
  { event := event80317
    frameStart := 0 },
  { event := event80318
    frameStart := 0 },
  { event := event80319
    frameStart := 0 }
]

def eventLeaf5020 : Array AnnotatedEvent := #[
  { event := event80320
    frameStart := 0 },
  { event := event80321
    frameStart := 0 },
  { event := event80322
    frameStart := 0 },
  { event := event80323
    frameStart := 0 },
  { event := event80324
    frameStart := 0 },
  { event := event80325
    frameStart := 0 },
  { event := event80326
    frameStart := 0 },
  { event := event80327
    frameStart := 0 },
  { event := event80328
    frameStart := 0 },
  { event := event80329
    frameStart := 0 },
  { event := event80330
    frameStart := 0 },
  { event := event80331
    frameStart := 0 },
  { event := event80332
    frameStart := 0 },
  { event := event80333
    frameStart := 0 },
  { event := event80334
    frameStart := 0 },
  { event := event80335
    frameStart := 0 }
]

def eventLeaf5021 : Array AnnotatedEvent := #[
  { event := event80336
    frameStart := 0 },
  { event := event80337
    frameStart := 0 },
  { event := event80338
    frameStart := 0 },
  { event := event80339
    frameStart := 0 },
  { event := event80340
    frameStart := 80340 },
  { event := event80341
    frameStart := 80340 },
  { event := event80342
    frameStart := 80340 },
  { event := event80343
    frameStart := 80340 },
  { event := event80344
    frameStart := 80340 },
  { event := event80345
    frameStart := 80340 },
  { event := event80346
    frameStart := 80340 },
  { event := event80347
    frameStart := 80340 },
  { event := event80348
    frameStart := 80340 },
  { event := event80349
    frameStart := 80340 },
  { event := event80350
    frameStart := 80340 },
  { event := event80351
    frameStart := 80340 }
]

def eventLeaf5022 : Array AnnotatedEvent := #[
  { event := event80352
    frameStart := 80340 },
  { event := event80353
    frameStart := 80340 },
  { event := event80354
    frameStart := 80340 },
  { event := event80355
    frameStart := 80340 },
  { event := event80356
    frameStart := 80340 },
  { event := event80357
    frameStart := 80340 },
  { event := event80358
    frameStart := 80340 },
  { event := event80359
    frameStart := 80340 },
  { event := event80360
    frameStart := 80340 },
  { event := event80361
    frameStart := 80340 },
  { event := event80362
    frameStart := 80340 },
  { event := event80363
    frameStart := 80340 },
  { event := event80364
    frameStart := 80340 },
  { event := event80365
    frameStart := 80340 },
  { event := event80366
    frameStart := 80340 },
  { event := event80367
    frameStart := 80340 }
]

def eventLeaf5023 : Array AnnotatedEvent := #[
  { event := event80368
    frameStart := 80340 },
  { event := event80369
    frameStart := 80340 },
  { event := event80370
    frameStart := 80340 },
  { event := event80371
    frameStart := 80340 },
  { event := event80372
    frameStart := 80340 },
  { event := event80373
    frameStart := 80340 },
  { event := event80374
    frameStart := 80340 },
  { event := event80375
    frameStart := 80340 },
  { event := event80376
    frameStart := 80340 },
  { event := event80377
    frameStart := 80340 },
  { event := event80378
    frameStart := 80340 },
  { event := event80379
    frameStart := 80340 },
  { event := event80380
    frameStart := 80340 },
  { event := event80381
    frameStart := 80340 },
  { event := event80382
    frameStart := 80340 },
  { event := event80383
    frameStart := 80340 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events313
