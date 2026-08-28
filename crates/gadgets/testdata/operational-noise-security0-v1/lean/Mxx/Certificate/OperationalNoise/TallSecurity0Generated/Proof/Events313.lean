import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events313

open Mxx.Certificate.OperationalNoise
open CertificateABI

def exact80128RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨10345⟩⟩, ⟨.program ⟨214⟩, ⟨13350⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact80128RawTermsValid :
    exact80128RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80128 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13448⟩⟩) exact80128RawTerms .large 80126 .exactZero (none)

def event80129 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6757⟩⟩) 0 ⟨6689⟩ 80105

def event80130 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6757⟩⟩) (.authority (.operator))

def exact80131RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6757⟩⟩]⟩, (1)⟩]

theorem exact80131RawTermsValid :
    exact80131RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80131 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6757⟩⟩) exact80131RawTerms .large 80130 .exactZero (none)

def event80132 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6790⟩⟩) 0 ⟨6757⟩ 80131

def event80133 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6790⟩⟩) (.identity (.predecessor 0 80132 .coefficient))

def exact80134RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6790⟩⟩]⟩, (1)⟩]

theorem exact80134RawTermsValid :
    exact80134RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80134 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6790⟩⟩) exact80134RawTerms .large 80133 .exactZero (none)

def event80135 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7882⟩⟩) 0 ⟨6790⟩ 80134

def event80136 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7882⟩⟩) (.authority (.operator))

def exact80137RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨7882⟩⟩]⟩, (1)⟩]

theorem exact80137RawTermsValid :
    exact80137RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80137 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7882⟩⟩) exact80137RawTerms (.finite 8192) 80136 .exactZero (none)

def event80138 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7883⟩⟩) 0 ⟨7882⟩ 80137

def event80139 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7883⟩⟩) 1 ⟨2348⟩ 80071

def event80140 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7883⟩⟩) (.scale (.predecessor 0 80138 .coefficient) (.value (.predecessor 1 80139 .coefficient)))

def exact80141RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨7882⟩⟩]⟩, (1)⟩]

theorem exact80141RawTermsValid :
    exact80141RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80141 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7883⟩⟩) exact80141RawTerms (.finite 8192) 80140 .exactZero (none)

def event80142 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6770⟩⟩) 0 ⟨6757⟩ 80131

def event80143 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6770⟩⟩) (.identity (.predecessor 0 80142 .coefficient))

def exact80144RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6770⟩⟩]⟩, (1)⟩]

theorem exact80144RawTermsValid :
    exact80144RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80144 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6770⟩⟩) exact80144RawTerms .large 80143 .exactZero (none)

def event80145 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7884⟩⟩) 0 ⟨6770⟩ 80144

def event80146 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7884⟩⟩) 1 ⟨7883⟩ 80141

def event80147 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7884⟩⟩) (.product (.predecessor 0 80145 .coefficient) (.predecessor 1 80146 .coefficient) (⟨false, false, none, none, none⟩))

def event80148 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7884⟩⟩, .operator (⟨80144, 0⟩, ⟨80141, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6770⟩⟩, ⟨.program ⟨214⟩, ⟨7882⟩⟩]⟩, (1)⟩)

def exact80149RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6770⟩⟩, ⟨.program ⟨214⟩, ⟨7882⟩⟩]⟩, (1)⟩]

theorem exact80149RawTermsValid :
    exact80149RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80149 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7884⟩⟩) exact80149RawTerms .large 80147 .exactZero (none)

def event80150 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13449⟩⟩) 0 ⟨7884⟩ 80149

def event80151 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13449⟩⟩) 1 ⟨13448⟩ 80128

def event80152 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13449⟩⟩) (.sum [.predecessor 0 80150 .coefficient, .predecessor 1 80151 .coefficient])

def exact80153RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6770⟩⟩, ⟨.program ⟨214⟩, ⟨7882⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨10345⟩⟩, ⟨.program ⟨214⟩, ⟨13350⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact80153RawTermsValid :
    exact80153RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80153 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13449⟩⟩) exact80153RawTerms .large 80152 .exactZero (none)

def event80154 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25761⟩⟩) 0 ⟨13449⟩ 80153

def event80155 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25761⟩⟩) 1 ⟨25758⟩ 80112

def event80156 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25761⟩⟩) (.product (.predecessor 0 80154 .coefficient) (.predecessor 1 80155 .coefficient) (⟨false, false, none, none, none⟩))

def event80157 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25761⟩⟩, .operator (⟨80153, 0⟩, ⟨80112, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6770⟩⟩, ⟨.program ⟨214⟩, ⟨7882⟩⟩, ⟨.program ⟨214⟩, ⟨25758⟩⟩]⟩, (1)⟩)

def event80158 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25761⟩⟩, .operator (⟨80153, 1⟩, ⟨80112, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨10345⟩⟩, ⟨.program ⟨214⟩, ⟨13350⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25758⟩⟩]⟩, (-1)⟩)

def event80159 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨25761⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨10345⟩⟩, ⟨.program ⟨214⟩, ⟨13350⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25758⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨25758⟩⟩) ⟨23416⟩ 80109)

def event80160 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25761⟩⟩, .relation 80159 0, ⟨[⟨.program ⟨214⟩, ⟨10345⟩⟩, ⟨.program ⟨214⟩, ⟨13350⟩⟩], [⟨.program ⟨214⟩, ⟨23416⟩⟩]⟩, (-1)⟩)

def exact80161RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6770⟩⟩, ⟨.program ⟨214⟩, ⟨7882⟩⟩, ⟨.program ⟨214⟩, ⟨25758⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨10345⟩⟩, ⟨.program ⟨214⟩, ⟨13350⟩⟩], [⟨.program ⟨214⟩, ⟨23416⟩⟩]⟩, (-1)⟩]

theorem exact80161RawTermsValid :
    exact80161RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80161 : Event := .resultExact (⟨.program ⟨214⟩, ⟨25761⟩⟩) exact80161RawTerms .large 80156 .exactZero (none)

def event80162 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17011⟩⟩) 0 ⟨13352⟩ 80101

def event80163 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨17011⟩⟩) (.authority (.programFamilyFact))

def exact80164RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨17011⟩⟩], []⟩, (1)⟩]

theorem exact80164RawTermsValid :
    exact80164RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80164 : Event := .resultExact (⟨.program ⟨214⟩, ⟨17011⟩⟩) exact80164RawTerms (.finite 60) 80163 .exactZero (none)

def event80165 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17013⟩⟩) 0 ⟨6544⟩ 80123

def event80166 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17013⟩⟩) 1 ⟨17011⟩ 80164

def event80167 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨17013⟩⟩) (.product (.predecessor 0 80165 .coefficient) (.predecessor 1 80166 .coefficient) (⟨false, true, none, none, some 1⟩))

def event80168 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨17013⟩⟩, .operator (⟨80123, 0⟩, ⟨80164, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨17011⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact80169RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨17011⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact80169RawTermsValid :
    exact80169RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80169 : Event := .resultExact (⟨.program ⟨214⟩, ⟨17013⟩⟩) exact80169RawTerms .large 80167 .exactZero (none)

def event80170 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6707⟩⟩) 0 ⟨6689⟩ 80105

def event80171 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6707⟩⟩) (.authority (.operator))

def exact80172RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6707⟩⟩]⟩, (1)⟩]

theorem exact80172RawTermsValid :
    exact80172RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80172 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6707⟩⟩) exact80172RawTerms .large 80171 .exactZero (none)

def event80173 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17014⟩⟩) 0 ⟨6707⟩ 80172

def event80174 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17014⟩⟩) 1 ⟨17013⟩ 80169

def event80175 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨17014⟩⟩) (.sum [.predecessor 0 80173 .coefficient, .predecessor 1 80174 .coefficient])

def exact80176RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6707⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17011⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact80176RawTermsValid :
    exact80176RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80176 : Event := .resultExact (⟨.program ⟨214⟩, ⟨17014⟩⟩) exact80176RawTerms .large 80175 .exactZero (none)

def event80177 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25762⟩⟩) 0 ⟨17014⟩ 80176

def event80178 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25762⟩⟩) 1 ⟨25761⟩ 80161

def event80179 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25762⟩⟩) (.sum [.predecessor 0 80177 .coefficient, .predecessor 1 80178 .coefficient])

def exact80180RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6707⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6770⟩⟩, ⟨.program ⟨214⟩, ⟨7882⟩⟩, ⟨.program ⟨214⟩, ⟨25758⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨10345⟩⟩, ⟨.program ⟨214⟩, ⟨13350⟩⟩], [⟨.program ⟨214⟩, ⟨23416⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17011⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact80180RawTermsValid :
    exact80180RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80180 : Event := .resultExact (⟨.program ⟨214⟩, ⟨25762⟩⟩) exact80180RawTerms .large 80179 .exactZero (none)

def event80181 : Event := .preFoldPolynomial 80180 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6707⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6770⟩⟩, ⟨.program ⟨214⟩, ⟨7882⟩⟩, ⟨.program ⟨214⟩, ⟨25758⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨10345⟩⟩, ⟨.program ⟨214⟩, ⟨13350⟩⟩], [⟨.program ⟨214⟩, ⟨23416⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17011⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩] .exactZero none

def exact80182RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6707⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6770⟩⟩, ⟨.program ⟨214⟩, ⟨7882⟩⟩, ⟨.program ⟨214⟩, ⟨25758⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨10345⟩⟩, ⟨.program ⟨214⟩, ⟨13350⟩⟩], [⟨.program ⟨214⟩, ⟨23416⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17011⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

def event80182 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨25762⟩⟩) 80181 exact80182RawTerms .large 80179 .exactZero (none)

def event80183 : Event := .specializationComputed (⟨.program ⟨214⟩, ⟨13352⟩⟩) ⟨⟨120⟩, ⟨26⟩, ⟨109⟩⟩ ⟨80019, 80183⟩

def event80184 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨20251⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20248⟩⟩]⟩) (1) 0 2 (.universal 80183 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20248⟩⟩]⟩) (none) 80182)

def event80185 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20251⟩⟩, .relation 80184 0, ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6707⟩⟩]⟩, (1)⟩)

def event80186 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20251⟩⟩, .relation 80184 1, ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6770⟩⟩, ⟨.program ⟨214⟩, ⟨7882⟩⟩, ⟨.program ⟨214⟩, ⟨25758⟩⟩]⟩, (-1)⟩)

def event80187 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20251⟩⟩, .relation 80184 2, ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨10345⟩⟩, ⟨.program ⟨214⟩, ⟨13350⟩⟩], [⟨.program ⟨214⟩, ⟨23416⟩⟩]⟩, (1)⟩)

def event80188 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20251⟩⟩, .relation 80184 3, ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨17011⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def exact80189RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6707⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6770⟩⟩, ⟨.program ⟨214⟩, ⟨7882⟩⟩, ⟨.program ⟨214⟩, ⟨25758⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨10345⟩⟩, ⟨.program ⟨214⟩, ⟨13350⟩⟩], [⟨.program ⟨214⟩, ⟨23416⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨17011⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact80189RawTermsValid :
    exact80189RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80189 : Event := .resultExact (⟨.program ⟨214⟩, ⟨20251⟩⟩) exact80189RawTerms .large 80015 (.finite 1811303510016) (some (80017))

def event80190 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25760⟩⟩) 0 ⟨20251⟩ 80189

def event80191 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25760⟩⟩) 1 ⟨25759⟩ 79994

def event80192 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25760⟩⟩) (.sum [.predecessor 0 80190 .coefficient, .predecessor 1 80191 .coefficient])

def event80193 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25760⟩⟩, .operator (⟨80189, 2⟩, ⟨79994, 1⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨10345⟩⟩, ⟨.program ⟨214⟩, ⟨13350⟩⟩], [⟨.program ⟨214⟩, ⟨23416⟩⟩]⟩, (-1)⟩)

def event80194 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25760⟩⟩, .operator (⟨80189, 1⟩, ⟨79994, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6770⟩⟩, ⟨.program ⟨214⟩, ⟨7882⟩⟩, ⟨.program ⟨214⟩, ⟨25758⟩⟩]⟩, (1)⟩)

def event80195 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25760⟩⟩) (.sum [.result 80189 .summary, .result 79994 .summary])

def exact80196RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6707⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨17011⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact80196RawTermsValid :
    exact80196RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80196 : Event := .resultExact (⟨.program ⟨214⟩, ⟨25760⟩⟩) exact80196RawTerms .large 80192 (.finite 352188964155392) (some (80195))

def event80197 : Event := .predecessor (⟨.program ⟨214⟩, ⟨30118⟩⟩) 0 ⟨25760⟩ 80196

def event80198 : Event := .predecessor (⟨.program ⟨214⟩, ⟨30118⟩⟩) 1 ⟨30116⟩ 79905

def event80199 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨30118⟩⟩) (.product (.predecessor 0 80197 .coefficient) (.predecessor 1 80198 .coefficient) (⟨false, false, none, none, none⟩))

def event80200 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨30118⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨30116⟩⟩]⟩) [⟨.result 79905 .coefficient, false, none⟩])

def event80201 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨30118⟩⟩) (.product (.result 80196 .summary) (.transfer 80200) (⟨false, false, none, none, none⟩))

def event80202 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30118⟩⟩, .operator (⟨80196, 0⟩, ⟨79905, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6707⟩⟩, ⟨.program ⟨214⟩, ⟨30116⟩⟩]⟩, (1)⟩)

def event80203 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30118⟩⟩, .operator (⟨80196, 1⟩, ⟨79905, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨17011⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨30116⟩⟩]⟩, (-1)⟩)

def event80204 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨30118⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨17011⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨30116⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨30116⟩⟩) ⟨24792⟩ 79902)

def event80205 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30118⟩⟩, .relation 80204 0, ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨17011⟩⟩], [⟨.program ⟨214⟩, ⟨24792⟩⟩]⟩, (-1)⟩)

def exact80206RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6707⟩⟩, ⟨.program ⟨214⟩, ⟨30116⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨17011⟩⟩], [⟨.program ⟨214⟩, ⟨24792⟩⟩]⟩, (-1)⟩]

theorem exact80206RawTermsValid :
    exact80206RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80206 : Event := .resultExact (⟨.program ⟨214⟩, ⟨30118⟩⟩) exact80206RawTerms .large 80199 (.finite 1292539133473715126272) (some (80201))

def event80207 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22840⟩⟩) 0 ⟨17012⟩ 3845

def event80208 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨22840⟩⟩) (.authority (.relationPreimageSource ⟨65⟩))

def exact80209RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨22840⟩⟩]⟩, (1)⟩]

theorem exact80209RawTermsValid :
    exact80209RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80209 : Event := .resultExact (⟨.program ⟨214⟩, ⟨22840⟩⟩) exact80209RawTerms (.finite 136065468) 80208 .exactZero (none)

def event80210 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22842⟩⟩) 0 ⟨22840⟩ 80209

def event80211 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22842⟩⟩) 1 ⟨2348⟩ 4

def event80212 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨22842⟩⟩) (.scale (.predecessor 0 80210 .coefficient) (.value (.predecessor 1 80211 .coefficient)))

def exact80213RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨22840⟩⟩]⟩, (1)⟩]

theorem exact80213RawTermsValid :
    exact80213RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80213 : Event := .resultExact (⟨.program ⟨214⟩, ⟨22842⟩⟩) exact80213RawTerms (.finite 136065468) 80212 .exactZero (none)

def event80214 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22843⟩⟩) 0 ⟨5541⟩ 80012

def event80215 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22843⟩⟩) 1 ⟨22842⟩ 80213

def event80216 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨22843⟩⟩) (.product (.predecessor 0 80214 .coefficient) (.predecessor 1 80215 .coefficient) (⟨false, false, none, none, none⟩))

def event80217 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨22843⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨22840⟩⟩]⟩) [⟨.result 80209 .coefficient, false, none⟩])

def event80218 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨22843⟩⟩) (.product (.result 80012 .summary) (.transfer 80217) (⟨false, false, none, none, none⟩))

def event80219 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨22843⟩⟩, .operator (⟨80012, 0⟩, ⟨80213, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22840⟩⟩]⟩, (1)⟩)

def event80220 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨22841⟩⟩)

def event80221 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event80222 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event80223 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨2348⟩⟩) (.authority (.operator))

def event80224 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨2348⟩⟩) (.finite 1)

def event80225 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event80226 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event80227 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event80228 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event80229 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 80228

def event80230 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 80226

def event80231 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 80229 .coefficient) (.value (.predecessor 1 80230 .coefficient)))

def event80232 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event80233 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5505⟩⟩) 0 ⟨5503⟩ 80232

def event80234 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5505⟩⟩) 1 ⟨2348⟩ 80224

def event80235 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5505⟩⟩) (.sum [.predecessor 0 80233 .coefficient, .predecessor 1 80234 .coefficient])

def event80236 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5505⟩⟩) (.finite 218)

def event80237 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5536⟩⟩) 0 ⟨5505⟩ 80236

def event80238 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5536⟩⟩) 1 ⟨961⟩ 80222

def event80239 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5536⟩⟩) (.identity (.predecessor 1 80238 .coefficient))

def event80240 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5536⟩⟩) (.finite 224)

def event80241 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13350⟩⟩) 0 ⟨5536⟩ 80240

def event80242 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13350⟩⟩) (.authority (.programFamilyFact))

def exact80243RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨13350⟩⟩], []⟩, (1)⟩]

theorem exact80243RawTermsValid :
    exact80243RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80243 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13350⟩⟩) exact80243RawTerms (.finite 60) 80242 .exactZero (none)

def event80244 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10345⟩⟩) 0 ⟨5536⟩ 80240

def event80245 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10345⟩⟩) (.authority (.programFamilyFact))

def exact80246RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨10345⟩⟩], []⟩, (1)⟩]

theorem exact80246RawTermsValid :
    exact80246RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80246 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10345⟩⟩) exact80246RawTerms (.finite 60) 80245 .exactZero (none)

def event80247 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13351⟩⟩) 0 ⟨10345⟩ 80246

def event80248 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13351⟩⟩) 1 ⟨13350⟩ 80243

def event80249 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13351⟩⟩) (.product (.predecessor 0 80247 .coefficient) (.predecessor 1 80248 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event80250 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13351⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨10345⟩⟩, ⟨.program ⟨214⟩, ⟨13350⟩⟩], []⟩) [⟨.result 80246 .coefficient, true, some 1⟩, ⟨.result 80243 .coefficient, true, some 1⟩])

def event80251 : Event := .survivorFold (1) 80250

def exact80252RawTerms : List Term := []

theorem exact80252RawTermsValid :
    exact80252RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80252 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13351⟩⟩) exact80252RawTerms (.finite 3600) 80249 (.finite 3600) (some (80250))

def event80253 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13352⟩⟩) 0 ⟨13351⟩ 80252

def event80254 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13352⟩⟩) (.identity (.predecessor 0 80253 .coefficient))

def event80255 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨13352⟩⟩) (.finite 3600)

def event80256 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17011⟩⟩) 0 ⟨13352⟩ 80255

def event80257 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨17011⟩⟩) (.authority (.programFamilyFact))

def exact80258RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨17011⟩⟩], []⟩, (1)⟩]

theorem exact80258RawTermsValid :
    exact80258RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80258 : Event := .resultExact (⟨.program ⟨214⟩, ⟨17011⟩⟩) exact80258RawTerms (.finite 60) 80257 .exactZero (none)

def event80259 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17012⟩⟩) 0 ⟨17011⟩ 80258

def event80260 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨17012⟩⟩) (.identity (.predecessor 0 80259 .coefficient))

def event80261 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨17012⟩⟩) (.finite 60)

def event80262 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22840⟩⟩) 0 ⟨17012⟩ 80261

def event80263 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨22840⟩⟩) (.authority (.relationPreimageSource ⟨65⟩))

def exact80264RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨22840⟩⟩]⟩, (1)⟩]

theorem exact80264RawTermsValid :
    exact80264RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80264 : Event := .resultExact (⟨.program ⟨214⟩, ⟨22840⟩⟩) exact80264RawTerms (.finite 136065468) 80263 .exactZero (none)

def event80265 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6⟩⟩) (.authority (.operator))

def exact80266RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩]⟩, (1)⟩]

theorem exact80266RawTermsValid :
    exact80266RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80266 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6⟩⟩) exact80266RawTerms .large 80265 .exactZero (none)

def event80267 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22841⟩⟩) 0 ⟨6⟩ 80266

def event80268 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22841⟩⟩) 1 ⟨22840⟩ 80264

def event80269 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨22841⟩⟩) (.product (.predecessor 0 80267 .coefficient) (.predecessor 1 80268 .coefficient) (⟨false, false, none, none, none⟩))

def event80270 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨22841⟩⟩, .operator (⟨80266, 0⟩, ⟨80264, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22840⟩⟩]⟩, (1)⟩)

def exact80271RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22840⟩⟩]⟩, (1)⟩]

theorem exact80271RawTermsValid :
    exact80271RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80271 : Event := .resultExact (⟨.program ⟨214⟩, ⟨22841⟩⟩) exact80271RawTerms .large 80269 .exactZero (none)

def event80272 : Event := .preFoldPolynomial 80271 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22840⟩⟩]⟩, (1)⟩] .exactZero none

def exact80273RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22840⟩⟩]⟩, (1)⟩]

def event80273 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨22841⟩⟩) 80272 exact80273RawTerms .large 80269 .exactZero (none)

def event80274 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨30124⟩⟩)

def event80275 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event80276 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event80277 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨2348⟩⟩) (.authority (.operator))

def event80278 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨2348⟩⟩) (.finite 1)

def event80279 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event80280 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event80281 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event80282 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event80283 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 80282

def event80284 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 80280

def event80285 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 80283 .coefficient) (.value (.predecessor 1 80284 .coefficient)))

def event80286 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event80287 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5505⟩⟩) 0 ⟨5503⟩ 80286

def event80288 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5505⟩⟩) 1 ⟨2348⟩ 80278

def event80289 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5505⟩⟩) (.sum [.predecessor 0 80287 .coefficient, .predecessor 1 80288 .coefficient])

def event80290 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5505⟩⟩) (.finite 218)

def event80291 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5536⟩⟩) 0 ⟨5505⟩ 80290

def event80292 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5536⟩⟩) 1 ⟨961⟩ 80276

def event80293 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5536⟩⟩) (.identity (.predecessor 1 80292 .coefficient))

def event80294 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5536⟩⟩) (.finite 224)

def event80295 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13350⟩⟩) 0 ⟨5536⟩ 80294

def event80296 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13350⟩⟩) (.authority (.programFamilyFact))

def exact80297RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨13350⟩⟩], []⟩, (1)⟩]

theorem exact80297RawTermsValid :
    exact80297RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80297 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13350⟩⟩) exact80297RawTerms (.finite 60) 80296 .exactZero (none)

def event80298 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10345⟩⟩) 0 ⟨5536⟩ 80294

def event80299 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10345⟩⟩) (.authority (.programFamilyFact))

def exact80300RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨10345⟩⟩], []⟩, (1)⟩]

theorem exact80300RawTermsValid :
    exact80300RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80300 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10345⟩⟩) exact80300RawTerms (.finite 60) 80299 .exactZero (none)

def event80301 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13351⟩⟩) 0 ⟨10345⟩ 80300

def event80302 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13351⟩⟩) 1 ⟨13350⟩ 80297

def event80303 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13351⟩⟩) (.product (.predecessor 0 80301 .coefficient) (.predecessor 1 80302 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event80304 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨13351⟩⟩, .operator (⟨80300, 0⟩, ⟨80297, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨10345⟩⟩, ⟨.program ⟨214⟩, ⟨13350⟩⟩], []⟩, (1)⟩)

def exact80305RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨10345⟩⟩, ⟨.program ⟨214⟩, ⟨13350⟩⟩], []⟩, (1)⟩]

theorem exact80305RawTermsValid :
    exact80305RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80305 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13351⟩⟩) exact80305RawTerms (.finite 3600) 80303 .exactZero (none)

def event80306 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13352⟩⟩) 0 ⟨13351⟩ 80305

def event80307 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13352⟩⟩) (.identity (.predecessor 0 80306 .coefficient))

def event80308 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨13352⟩⟩) (.finite 3600)

def event80309 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17011⟩⟩) 0 ⟨13352⟩ 80308

def event80310 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨17011⟩⟩) (.authority (.programFamilyFact))

def exact80311RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨17011⟩⟩], []⟩, (1)⟩]

theorem exact80311RawTermsValid :
    exact80311RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80311 : Event := .resultExact (⟨.program ⟨214⟩, ⟨17011⟩⟩) exact80311RawTerms (.finite 60) 80310 .exactZero (none)

def event80312 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17012⟩⟩) 0 ⟨17011⟩ 80311

def event80313 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨17012⟩⟩) (.identity (.predecessor 0 80312 .coefficient))

def event80314 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨17012⟩⟩) (.finite 60)

def event80315 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24790⟩⟩) 0 ⟨17012⟩ 80314

def event80316 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24790⟩⟩) (.authority (.programFamilyFact))

def event80317 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨24790⟩⟩) (.finite 3720)

def event80318 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨6689⟩⟩) .missing

def event80319 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24792⟩⟩) 0 ⟨6689⟩ 80318

def event80320 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24792⟩⟩) 1 ⟨24790⟩ 80317

def event80321 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24792⟩⟩) (.authority (.operator))

def exact80322RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨24792⟩⟩]⟩, (1)⟩]

theorem exact80322RawTermsValid :
    exact80322RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80322 : Event := .resultExact (⟨.program ⟨214⟩, ⟨24792⟩⟩) exact80322RawTerms .large 80321 .exactZero (none)

def event80323 : Event := .predecessor (⟨.program ⟨214⟩, ⟨30116⟩⟩) 0 ⟨24792⟩ 80322

def event80324 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨30116⟩⟩) (.authority (.operator))

def exact80325RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨30116⟩⟩]⟩, (1)⟩]

theorem exact80325RawTermsValid :
    exact80325RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80325 : Event := .resultExact (⟨.program ⟨214⟩, ⟨30116⟩⟩) exact80325RawTerms (.finite 8192) 80324 .exactZero (none)

def event80326 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨110⟩⟩) (.authority (.operator))

def event80327 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨110⟩⟩) .exactZero

def event80328 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17051⟩⟩) 0 ⟨17012⟩ 80314

def event80329 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17051⟩⟩) 1 ⟨110⟩ 80327

def event80330 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨17051⟩⟩) (.sum [.predecessor 0 80328 .coefficient, .predecessor 1 80329 .coefficient])

def event80331 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨17051⟩⟩) (.finite 60)

def event80332 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17052⟩⟩) 0 ⟨17051⟩ 80331

def event80333 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨17052⟩⟩) (.identity (.predecessor 0 80332 .coefficient))

def exact80334RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨17011⟩⟩], []⟩, (1)⟩]

theorem exact80334RawTermsValid :
    exact80334RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80334 : Event := .resultExact (⟨.program ⟨214⟩, ⟨17052⟩⟩) exact80334RawTerms (.finite 60) 80333 .exactZero (none)

def event80335 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6544⟩⟩) (.authority (.factStore))

def exact80336RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact80336RawTermsValid :
    exact80336RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80336 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6544⟩⟩) exact80336RawTerms .large 80335 .exactZero (none)

def event80337 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17053⟩⟩) 0 ⟨6544⟩ 80336

def event80338 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17053⟩⟩) 1 ⟨17052⟩ 80334

def event80339 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨17053⟩⟩) (.product (.predecessor 0 80337 .coefficient) (.predecessor 1 80338 .coefficient) (⟨false, false, none, none, none⟩))

def event80340 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨17053⟩⟩, .operator (⟨80336, 0⟩, ⟨80334, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨17011⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact80341RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨17011⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact80341RawTermsValid :
    exact80341RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80341 : Event := .resultExact (⟨.program ⟨214⟩, ⟨17053⟩⟩) exact80341RawTerms .large 80339 .exactZero (none)

def event80342 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6707⟩⟩) 0 ⟨6689⟩ 80318

def event80343 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6707⟩⟩) (.authority (.operator))

def exact80344RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6707⟩⟩]⟩, (1)⟩]

theorem exact80344RawTermsValid :
    exact80344RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80344 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6707⟩⟩) exact80344RawTerms .large 80343 .exactZero (none)

def event80345 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17054⟩⟩) 0 ⟨6707⟩ 80344

def event80346 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17054⟩⟩) 1 ⟨17053⟩ 80341

def event80347 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨17054⟩⟩) (.sum [.predecessor 0 80345 .coefficient, .predecessor 1 80346 .coefficient])

def exact80348RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6707⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17011⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact80348RawTermsValid :
    exact80348RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80348 : Event := .resultExact (⟨.program ⟨214⟩, ⟨17054⟩⟩) exact80348RawTerms .large 80347 .exactZero (none)

def event80349 : Event := .predecessor (⟨.program ⟨214⟩, ⟨30117⟩⟩) 0 ⟨17054⟩ 80348

def event80350 : Event := .predecessor (⟨.program ⟨214⟩, ⟨30117⟩⟩) 1 ⟨30116⟩ 80325

def event80351 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨30117⟩⟩) (.product (.predecessor 0 80349 .coefficient) (.predecessor 1 80350 .coefficient) (⟨false, false, none, none, none⟩))

def event80352 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30117⟩⟩, .operator (⟨80348, 0⟩, ⟨80325, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6707⟩⟩, ⟨.program ⟨214⟩, ⟨30116⟩⟩]⟩, (1)⟩)

def event80353 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30117⟩⟩, .operator (⟨80348, 1⟩, ⟨80325, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨17011⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨30116⟩⟩]⟩, (-1)⟩)

def event80354 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨30117⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨17011⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨30116⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨30116⟩⟩) ⟨24792⟩ 80322)

def event80355 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30117⟩⟩, .relation 80354 0, ⟨[⟨.program ⟨214⟩, ⟨17011⟩⟩], [⟨.program ⟨214⟩, ⟨24792⟩⟩]⟩, (-1)⟩)

def exact80356RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6707⟩⟩, ⟨.program ⟨214⟩, ⟨30116⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17011⟩⟩], [⟨.program ⟨214⟩, ⟨24792⟩⟩]⟩, (-1)⟩]

theorem exact80356RawTermsValid :
    exact80356RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80356 : Event := .resultExact (⟨.program ⟨214⟩, ⟨30117⟩⟩) exact80356RawTerms .large 80351 .exactZero (none)

def event80357 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18170⟩⟩) 0 ⟨17012⟩ 80314

def event80358 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨18170⟩⟩) (.authority (.programFamilyFact))

def exact80359RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨18170⟩⟩], []⟩, (1)⟩]

theorem exact80359RawTermsValid :
    exact80359RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80359 : Event := .resultExact (⟨.program ⟨214⟩, ⟨18170⟩⟩) exact80359RawTerms (.finite 63) 80358 .exactZero (none)

def event80360 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18171⟩⟩) 0 ⟨6544⟩ 80336

def event80361 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18171⟩⟩) 1 ⟨18170⟩ 80359

def event80362 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨18171⟩⟩) (.product (.predecessor 0 80360 .coefficient) (.predecessor 1 80361 .coefficient) (⟨false, true, none, none, some 1⟩))

def event80363 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18171⟩⟩, .operator (⟨80336, 0⟩, ⟨80359, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨18170⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact80364RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨18170⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact80364RawTermsValid :
    exact80364RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80364 : Event := .resultExact (⟨.program ⟨214⟩, ⟨18171⟩⟩) exact80364RawTerms .large 80362 .exactZero (none)

def event80365 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6743⟩⟩) 0 ⟨6689⟩ 80318

def event80366 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6743⟩⟩) (.authority (.operator))

def exact80367RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6743⟩⟩]⟩, (1)⟩]

theorem exact80367RawTermsValid :
    exact80367RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80367 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6743⟩⟩) exact80367RawTerms .large 80366 .exactZero (none)

def event80368 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18172⟩⟩) 0 ⟨6743⟩ 80367

def event80369 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18172⟩⟩) 1 ⟨18171⟩ 80364

def event80370 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨18172⟩⟩) (.sum [.predecessor 0 80368 .coefficient, .predecessor 1 80369 .coefficient])

def exact80371RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6743⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨18170⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact80371RawTermsValid :
    exact80371RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80371 : Event := .resultExact (⟨.program ⟨214⟩, ⟨18172⟩⟩) exact80371RawTerms .large 80370 .exactZero (none)

def event80372 : Event := .predecessor (⟨.program ⟨214⟩, ⟨30124⟩⟩) 0 ⟨18172⟩ 80371

def event80373 : Event := .predecessor (⟨.program ⟨214⟩, ⟨30124⟩⟩) 1 ⟨30117⟩ 80356

def event80374 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨30124⟩⟩) (.sum [.predecessor 0 80372 .coefficient, .predecessor 1 80373 .coefficient])

def exact80375RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6707⟩⟩, ⟨.program ⟨214⟩, ⟨30116⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6743⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17011⟩⟩], [⟨.program ⟨214⟩, ⟨24792⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨18170⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact80375RawTermsValid :
    exact80375RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80375 : Event := .resultExact (⟨.program ⟨214⟩, ⟨30124⟩⟩) exact80375RawTerms .large 80374 .exactZero (none)

def event80376 : Event := .preFoldPolynomial 80375 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6707⟩⟩, ⟨.program ⟨214⟩, ⟨30116⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6743⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17011⟩⟩], [⟨.program ⟨214⟩, ⟨24792⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨18170⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩] .exactZero none

def exact80377RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6707⟩⟩, ⟨.program ⟨214⟩, ⟨30116⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6743⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17011⟩⟩], [⟨.program ⟨214⟩, ⟨24792⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨18170⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

def event80377 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨30124⟩⟩) 80376 exact80377RawTerms .large 80374 .exactZero (none)

def event80378 : Event := .specializationComputed (⟨.program ⟨214⟩, ⟨17012⟩⟩) ⟨⟨156⟩, ⟨65⟩, ⟨109⟩⟩ ⟨80220, 80378⟩

def event80379 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨22843⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22840⟩⟩]⟩) (1) 0 2 (.universal 80378 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22840⟩⟩]⟩) (none) 80377)

def event80380 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨22843⟩⟩, .relation 80379 1, ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6743⟩⟩]⟩, (1)⟩)

def event80381 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨22843⟩⟩, .relation 80379 0, ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6707⟩⟩, ⟨.program ⟨214⟩, ⟨30116⟩⟩]⟩, (-1)⟩)

def event80382 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨22843⟩⟩, .relation 80379 2, ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨17011⟩⟩], [⟨.program ⟨214⟩, ⟨24792⟩⟩]⟩, (1)⟩)

def event80383 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨22843⟩⟩, .relation 80379 3, ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨18170⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def eventLeaf5008 : Array AnnotatedEvent := #[
  { event := event80128
    frameStart := 80067 },
  { event := event80129
    frameStart := 80067 },
  { event := event80130
    frameStart := 80067 },
  { event := event80131
    frameStart := 80067 },
  { event := event80132
    frameStart := 80067 },
  { event := event80133
    frameStart := 80067 },
  { event := event80134
    frameStart := 80067 },
  { event := event80135
    frameStart := 80067 },
  { event := event80136
    frameStart := 80067 },
  { event := event80137
    frameStart := 80067 },
  { event := event80138
    frameStart := 80067 },
  { event := event80139
    frameStart := 80067 },
  { event := event80140
    frameStart := 80067 },
  { event := event80141
    frameStart := 80067 },
  { event := event80142
    frameStart := 80067 },
  { event := event80143
    frameStart := 80067 }
]

def eventLeaf5009 : Array AnnotatedEvent := #[
  { event := event80144
    frameStart := 80067 },
  { event := event80145
    frameStart := 80067 },
  { event := event80146
    frameStart := 80067 },
  { event := event80147
    frameStart := 80067 },
  { event := event80148
    frameStart := 80067 },
  { event := event80149
    frameStart := 80067 },
  { event := event80150
    frameStart := 80067 },
  { event := event80151
    frameStart := 80067 },
  { event := event80152
    frameStart := 80067 },
  { event := event80153
    frameStart := 80067 },
  { event := event80154
    frameStart := 80067 },
  { event := event80155
    frameStart := 80067 },
  { event := event80156
    frameStart := 80067 },
  { event := event80157
    frameStart := 80067 },
  { event := event80158
    frameStart := 80067 },
  { event := event80159
    frameStart := 80067 }
]

def eventLeaf5010 : Array AnnotatedEvent := #[
  { event := event80160
    frameStart := 80067 },
  { event := event80161
    frameStart := 80067 },
  { event := event80162
    frameStart := 80067 },
  { event := event80163
    frameStart := 80067 },
  { event := event80164
    frameStart := 80067 },
  { event := event80165
    frameStart := 80067 },
  { event := event80166
    frameStart := 80067 },
  { event := event80167
    frameStart := 80067 },
  { event := event80168
    frameStart := 80067 },
  { event := event80169
    frameStart := 80067 },
  { event := event80170
    frameStart := 80067 },
  { event := event80171
    frameStart := 80067 },
  { event := event80172
    frameStart := 80067 },
  { event := event80173
    frameStart := 80067 },
  { event := event80174
    frameStart := 80067 },
  { event := event80175
    frameStart := 80067 }
]

def eventLeaf5011 : Array AnnotatedEvent := #[
  { event := event80176
    frameStart := 80067 },
  { event := event80177
    frameStart := 80067 },
  { event := event80178
    frameStart := 80067 },
  { event := event80179
    frameStart := 80067 },
  { event := event80180
    frameStart := 80067 },
  { event := event80181
    frameStart := 80067 },
  { event := event80182
    frameStart := 80067 },
  { event := event80183
    frameStart := 0 },
  { event := event80184
    frameStart := 0 },
  { event := event80185
    frameStart := 0 },
  { event := event80186
    frameStart := 0 },
  { event := event80187
    frameStart := 0 },
  { event := event80188
    frameStart := 0 },
  { event := event80189
    frameStart := 0 },
  { event := event80190
    frameStart := 0 },
  { event := event80191
    frameStart := 0 }
]

def eventLeaf5012 : Array AnnotatedEvent := #[
  { event := event80192
    frameStart := 0 },
  { event := event80193
    frameStart := 0 },
  { event := event80194
    frameStart := 0 },
  { event := event80195
    frameStart := 0 },
  { event := event80196
    frameStart := 0 },
  { event := event80197
    frameStart := 0 },
  { event := event80198
    frameStart := 0 },
  { event := event80199
    frameStart := 0 },
  { event := event80200
    frameStart := 0 },
  { event := event80201
    frameStart := 0 },
  { event := event80202
    frameStart := 0 },
  { event := event80203
    frameStart := 0 },
  { event := event80204
    frameStart := 0 },
  { event := event80205
    frameStart := 0 },
  { event := event80206
    frameStart := 0 },
  { event := event80207
    frameStart := 0 }
]

def eventLeaf5013 : Array AnnotatedEvent := #[
  { event := event80208
    frameStart := 0 },
  { event := event80209
    frameStart := 0 },
  { event := event80210
    frameStart := 0 },
  { event := event80211
    frameStart := 0 },
  { event := event80212
    frameStart := 0 },
  { event := event80213
    frameStart := 0 },
  { event := event80214
    frameStart := 0 },
  { event := event80215
    frameStart := 0 },
  { event := event80216
    frameStart := 0 },
  { event := event80217
    frameStart := 0 },
  { event := event80218
    frameStart := 0 },
  { event := event80219
    frameStart := 0 },
  { event := event80220
    frameStart := 80220 },
  { event := event80221
    frameStart := 80220 },
  { event := event80222
    frameStart := 80220 },
  { event := event80223
    frameStart := 80220 }
]

def eventLeaf5014 : Array AnnotatedEvent := #[
  { event := event80224
    frameStart := 80220 },
  { event := event80225
    frameStart := 80220 },
  { event := event80226
    frameStart := 80220 },
  { event := event80227
    frameStart := 80220 },
  { event := event80228
    frameStart := 80220 },
  { event := event80229
    frameStart := 80220 },
  { event := event80230
    frameStart := 80220 },
  { event := event80231
    frameStart := 80220 },
  { event := event80232
    frameStart := 80220 },
  { event := event80233
    frameStart := 80220 },
  { event := event80234
    frameStart := 80220 },
  { event := event80235
    frameStart := 80220 },
  { event := event80236
    frameStart := 80220 },
  { event := event80237
    frameStart := 80220 },
  { event := event80238
    frameStart := 80220 },
  { event := event80239
    frameStart := 80220 }
]

def eventLeaf5015 : Array AnnotatedEvent := #[
  { event := event80240
    frameStart := 80220 },
  { event := event80241
    frameStart := 80220 },
  { event := event80242
    frameStart := 80220 },
  { event := event80243
    frameStart := 80220 },
  { event := event80244
    frameStart := 80220 },
  { event := event80245
    frameStart := 80220 },
  { event := event80246
    frameStart := 80220 },
  { event := event80247
    frameStart := 80220 },
  { event := event80248
    frameStart := 80220 },
  { event := event80249
    frameStart := 80220 },
  { event := event80250
    frameStart := 80220 },
  { event := event80251
    frameStart := 80220 },
  { event := event80252
    frameStart := 80220 },
  { event := event80253
    frameStart := 80220 },
  { event := event80254
    frameStart := 80220 },
  { event := event80255
    frameStart := 80220 }
]

def eventLeaf5016 : Array AnnotatedEvent := #[
  { event := event80256
    frameStart := 80220 },
  { event := event80257
    frameStart := 80220 },
  { event := event80258
    frameStart := 80220 },
  { event := event80259
    frameStart := 80220 },
  { event := event80260
    frameStart := 80220 },
  { event := event80261
    frameStart := 80220 },
  { event := event80262
    frameStart := 80220 },
  { event := event80263
    frameStart := 80220 },
  { event := event80264
    frameStart := 80220 },
  { event := event80265
    frameStart := 80220 },
  { event := event80266
    frameStart := 80220 },
  { event := event80267
    frameStart := 80220 },
  { event := event80268
    frameStart := 80220 },
  { event := event80269
    frameStart := 80220 },
  { event := event80270
    frameStart := 80220 },
  { event := event80271
    frameStart := 80220 }
]

def eventLeaf5017 : Array AnnotatedEvent := #[
  { event := event80272
    frameStart := 80220 },
  { event := event80273
    frameStart := 80220 },
  { event := event80274
    frameStart := 80274 },
  { event := event80275
    frameStart := 80274 },
  { event := event80276
    frameStart := 80274 },
  { event := event80277
    frameStart := 80274 },
  { event := event80278
    frameStart := 80274 },
  { event := event80279
    frameStart := 80274 },
  { event := event80280
    frameStart := 80274 },
  { event := event80281
    frameStart := 80274 },
  { event := event80282
    frameStart := 80274 },
  { event := event80283
    frameStart := 80274 },
  { event := event80284
    frameStart := 80274 },
  { event := event80285
    frameStart := 80274 },
  { event := event80286
    frameStart := 80274 },
  { event := event80287
    frameStart := 80274 }
]

def eventLeaf5018 : Array AnnotatedEvent := #[
  { event := event80288
    frameStart := 80274 },
  { event := event80289
    frameStart := 80274 },
  { event := event80290
    frameStart := 80274 },
  { event := event80291
    frameStart := 80274 },
  { event := event80292
    frameStart := 80274 },
  { event := event80293
    frameStart := 80274 },
  { event := event80294
    frameStart := 80274 },
  { event := event80295
    frameStart := 80274 },
  { event := event80296
    frameStart := 80274 },
  { event := event80297
    frameStart := 80274 },
  { event := event80298
    frameStart := 80274 },
  { event := event80299
    frameStart := 80274 },
  { event := event80300
    frameStart := 80274 },
  { event := event80301
    frameStart := 80274 },
  { event := event80302
    frameStart := 80274 },
  { event := event80303
    frameStart := 80274 }
]

def eventLeaf5019 : Array AnnotatedEvent := #[
  { event := event80304
    frameStart := 80274 },
  { event := event80305
    frameStart := 80274 },
  { event := event80306
    frameStart := 80274 },
  { event := event80307
    frameStart := 80274 },
  { event := event80308
    frameStart := 80274 },
  { event := event80309
    frameStart := 80274 },
  { event := event80310
    frameStart := 80274 },
  { event := event80311
    frameStart := 80274 },
  { event := event80312
    frameStart := 80274 },
  { event := event80313
    frameStart := 80274 },
  { event := event80314
    frameStart := 80274 },
  { event := event80315
    frameStart := 80274 },
  { event := event80316
    frameStart := 80274 },
  { event := event80317
    frameStart := 80274 },
  { event := event80318
    frameStart := 80274 },
  { event := event80319
    frameStart := 80274 }
]

def eventLeaf5020 : Array AnnotatedEvent := #[
  { event := event80320
    frameStart := 80274 },
  { event := event80321
    frameStart := 80274 },
  { event := event80322
    frameStart := 80274 },
  { event := event80323
    frameStart := 80274 },
  { event := event80324
    frameStart := 80274 },
  { event := event80325
    frameStart := 80274 },
  { event := event80326
    frameStart := 80274 },
  { event := event80327
    frameStart := 80274 },
  { event := event80328
    frameStart := 80274 },
  { event := event80329
    frameStart := 80274 },
  { event := event80330
    frameStart := 80274 },
  { event := event80331
    frameStart := 80274 },
  { event := event80332
    frameStart := 80274 },
  { event := event80333
    frameStart := 80274 },
  { event := event80334
    frameStart := 80274 },
  { event := event80335
    frameStart := 80274 }
]

def eventLeaf5021 : Array AnnotatedEvent := #[
  { event := event80336
    frameStart := 80274 },
  { event := event80337
    frameStart := 80274 },
  { event := event80338
    frameStart := 80274 },
  { event := event80339
    frameStart := 80274 },
  { event := event80340
    frameStart := 80274 },
  { event := event80341
    frameStart := 80274 },
  { event := event80342
    frameStart := 80274 },
  { event := event80343
    frameStart := 80274 },
  { event := event80344
    frameStart := 80274 },
  { event := event80345
    frameStart := 80274 },
  { event := event80346
    frameStart := 80274 },
  { event := event80347
    frameStart := 80274 },
  { event := event80348
    frameStart := 80274 },
  { event := event80349
    frameStart := 80274 },
  { event := event80350
    frameStart := 80274 },
  { event := event80351
    frameStart := 80274 }
]

def eventLeaf5022 : Array AnnotatedEvent := #[
  { event := event80352
    frameStart := 80274 },
  { event := event80353
    frameStart := 80274 },
  { event := event80354
    frameStart := 80274 },
  { event := event80355
    frameStart := 80274 },
  { event := event80356
    frameStart := 80274 },
  { event := event80357
    frameStart := 80274 },
  { event := event80358
    frameStart := 80274 },
  { event := event80359
    frameStart := 80274 },
  { event := event80360
    frameStart := 80274 },
  { event := event80361
    frameStart := 80274 },
  { event := event80362
    frameStart := 80274 },
  { event := event80363
    frameStart := 80274 },
  { event := event80364
    frameStart := 80274 },
  { event := event80365
    frameStart := 80274 },
  { event := event80366
    frameStart := 80274 },
  { event := event80367
    frameStart := 80274 }
]

def eventLeaf5023 : Array AnnotatedEvent := #[
  { event := event80368
    frameStart := 80274 },
  { event := event80369
    frameStart := 80274 },
  { event := event80370
    frameStart := 80274 },
  { event := event80371
    frameStart := 80274 },
  { event := event80372
    frameStart := 80274 },
  { event := event80373
    frameStart := 80274 },
  { event := event80374
    frameStart := 80274 },
  { event := event80375
    frameStart := 80274 },
  { event := event80376
    frameStart := 80274 },
  { event := event80377
    frameStart := 80274 },
  { event := event80378
    frameStart := 0 },
  { event := event80379
    frameStart := 0 },
  { event := event80380
    frameStart := 0 },
  { event := event80381
    frameStart := 0 },
  { event := event80382
    frameStart := 0 },
  { event := event80383
    frameStart := 0 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events313
