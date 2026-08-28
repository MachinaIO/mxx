import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events321

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event82176 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16549⟩⟩) 0 ⟨12568⟩ 82175

def event82177 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16549⟩⟩) (.authority (.programFamilyFact))

def exact82178RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16549⟩⟩], []⟩, (1)⟩]

theorem exact82178RawTermsValid :
    exact82178RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82178 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16549⟩⟩) exact82178RawTerms (.finite 42) 82177 .exactZero (none)

def event82179 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16550⟩⟩) 0 ⟨16549⟩ 82178

def event82180 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16550⟩⟩) (.identity (.predecessor 0 82179 .coefficient))

def event82181 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨16550⟩⟩) (.finite 42)

def event82182 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22264⟩⟩) 0 ⟨16550⟩ 82181

def event82183 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨22264⟩⟩) (.authority (.relationPreimageSource ⟨57⟩))

def exact82184RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨22264⟩⟩]⟩, (1)⟩]

theorem exact82184RawTermsValid :
    exact82184RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82184 : Event := .resultExact (⟨.program ⟨214⟩, ⟨22264⟩⟩) exact82184RawTerms (.finite 136065468) 82183 .exactZero (none)

def event82185 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6⟩⟩) (.authority (.operator))

def exact82186RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩]⟩, (1)⟩]

theorem exact82186RawTermsValid :
    exact82186RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82186 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6⟩⟩) exact82186RawTerms .large 82185 .exactZero (none)

def event82187 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22265⟩⟩) 0 ⟨6⟩ 82186

def event82188 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22265⟩⟩) 1 ⟨22264⟩ 82184

def event82189 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨22265⟩⟩) (.product (.predecessor 0 82187 .coefficient) (.predecessor 1 82188 .coefficient) (⟨false, false, none, none, none⟩))

def event82190 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨22265⟩⟩, .operator (⟨82186, 0⟩, ⟨82184, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22264⟩⟩]⟩, (1)⟩)

def exact82191RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22264⟩⟩]⟩, (1)⟩]

theorem exact82191RawTermsValid :
    exact82191RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82191 : Event := .resultExact (⟨.program ⟨214⟩, ⟨22265⟩⟩) exact82191RawTerms .large 82189 .exactZero (none)

def event82192 : Event := .preFoldPolynomial 82191 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22264⟩⟩]⟩, (1)⟩] .exactZero none

def exact82193RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22264⟩⟩]⟩, (1)⟩]

def event82193 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨22265⟩⟩) 82192 exact82193RawTerms .large 82189 .exactZero (none)

def event82194 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨29173⟩⟩)

def event82195 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event82196 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event82197 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨2348⟩⟩) (.authority (.operator))

def event82198 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨2348⟩⟩) (.finite 1)

def event82199 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event82200 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event82201 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event82202 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event82203 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 82202

def event82204 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 82200

def event82205 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 82203 .coefficient) (.value (.predecessor 1 82204 .coefficient)))

def event82206 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event82207 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5505⟩⟩) 0 ⟨5503⟩ 82206

def event82208 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5505⟩⟩) 1 ⟨2348⟩ 82198

def event82209 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5505⟩⟩) (.sum [.predecessor 0 82207 .coefficient, .predecessor 1 82208 .coefficient])

def event82210 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5505⟩⟩) (.finite 218)

def event82211 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5536⟩⟩) 0 ⟨5505⟩ 82210

def event82212 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5536⟩⟩) 1 ⟨961⟩ 82196

def event82213 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5536⟩⟩) (.identity (.predecessor 1 82212 .coefficient))

def event82214 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5536⟩⟩) (.finite 224)

def event82215 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12566⟩⟩) 0 ⟨5536⟩ 82214

def event82216 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12566⟩⟩) (.authority (.programFamilyFact))

def exact82217RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨12566⟩⟩], []⟩, (1)⟩]

theorem exact82217RawTermsValid :
    exact82217RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82217 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12566⟩⟩) exact82217RawTerms (.finite 42) 82216 .exactZero (none)

def event82218 : Event := .predecessor (⟨.program ⟨214⟩, ⟨9925⟩⟩) 0 ⟨5536⟩ 82214

def event82219 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨9925⟩⟩) (.authority (.programFamilyFact))

def exact82220RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨9925⟩⟩], []⟩, (1)⟩]

theorem exact82220RawTermsValid :
    exact82220RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82220 : Event := .resultExact (⟨.program ⟨214⟩, ⟨9925⟩⟩) exact82220RawTerms (.finite 42) 82219 .exactZero (none)

def event82221 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12567⟩⟩) 0 ⟨9925⟩ 82220

def event82222 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12567⟩⟩) 1 ⟨12566⟩ 82217

def event82223 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12567⟩⟩) (.product (.predecessor 0 82221 .coefficient) (.predecessor 1 82222 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event82224 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨12567⟩⟩, .operator (⟨82220, 0⟩, ⟨82217, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨9925⟩⟩, ⟨.program ⟨214⟩, ⟨12566⟩⟩], []⟩, (1)⟩)

def exact82225RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨9925⟩⟩, ⟨.program ⟨214⟩, ⟨12566⟩⟩], []⟩, (1)⟩]

theorem exact82225RawTermsValid :
    exact82225RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82225 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12567⟩⟩) exact82225RawTerms (.finite 1764) 82223 .exactZero (none)

def event82226 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12568⟩⟩) 0 ⟨12567⟩ 82225

def event82227 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12568⟩⟩) (.identity (.predecessor 0 82226 .coefficient))

def event82228 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨12568⟩⟩) (.finite 1764)

def event82229 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16549⟩⟩) 0 ⟨12568⟩ 82228

def event82230 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16549⟩⟩) (.authority (.programFamilyFact))

def exact82231RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16549⟩⟩], []⟩, (1)⟩]

theorem exact82231RawTermsValid :
    exact82231RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82231 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16549⟩⟩) exact82231RawTerms (.finite 42) 82230 .exactZero (none)

def event82232 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16550⟩⟩) 0 ⟨16549⟩ 82231

def event82233 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16550⟩⟩) (.identity (.predecessor 0 82232 .coefficient))

def event82234 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨16550⟩⟩) (.finite 42)

def event82235 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24538⟩⟩) 0 ⟨16550⟩ 82234

def event82236 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24538⟩⟩) (.authority (.programFamilyFact))

def event82237 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨24538⟩⟩) (.finite 3720)

def event82238 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨6689⟩⟩) .missing

def event82239 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24540⟩⟩) 0 ⟨6689⟩ 82238

def event82240 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24540⟩⟩) 1 ⟨24538⟩ 82237

def event82241 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24540⟩⟩) (.authority (.operator))

def exact82242RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨24540⟩⟩]⟩, (1)⟩]

theorem exact82242RawTermsValid :
    exact82242RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82242 : Event := .resultExact (⟨.program ⟨214⟩, ⟨24540⟩⟩) exact82242RawTerms .large 82241 .exactZero (none)

def event82243 : Event := .predecessor (⟨.program ⟨214⟩, ⟨29168⟩⟩) 0 ⟨24540⟩ 82242

def event82244 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨29168⟩⟩) (.authority (.operator))

def exact82245RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨29168⟩⟩]⟩, (1)⟩]

theorem exact82245RawTermsValid :
    exact82245RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82245 : Event := .resultExact (⟨.program ⟨214⟩, ⟨29168⟩⟩) exact82245RawTerms (.finite 8192) 82244 .exactZero (none)

def event82246 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨110⟩⟩) (.authority (.operator))

def event82247 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨110⟩⟩) .exactZero

def event82248 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16589⟩⟩) 0 ⟨16550⟩ 82234

def event82249 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16589⟩⟩) 1 ⟨110⟩ 82247

def event82250 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16589⟩⟩) (.sum [.predecessor 0 82248 .coefficient, .predecessor 1 82249 .coefficient])

def event82251 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨16589⟩⟩) (.finite 42)

def event82252 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16590⟩⟩) 0 ⟨16589⟩ 82251

def event82253 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16590⟩⟩) (.identity (.predecessor 0 82252 .coefficient))

def exact82254RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16549⟩⟩], []⟩, (1)⟩]

theorem exact82254RawTermsValid :
    exact82254RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82254 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16590⟩⟩) exact82254RawTerms (.finite 42) 82253 .exactZero (none)

def event82255 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6544⟩⟩) (.authority (.factStore))

def exact82256RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact82256RawTermsValid :
    exact82256RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82256 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6544⟩⟩) exact82256RawTerms .large 82255 .exactZero (none)

def event82257 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16591⟩⟩) 0 ⟨6544⟩ 82256

def event82258 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16591⟩⟩) 1 ⟨16590⟩ 82254

def event82259 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16591⟩⟩) (.product (.predecessor 0 82257 .coefficient) (.predecessor 1 82258 .coefficient) (⟨false, false, none, none, none⟩))

def event82260 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨16591⟩⟩, .operator (⟨82256, 0⟩, ⟨82254, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨16549⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact82261RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16549⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact82261RawTermsValid :
    exact82261RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82261 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16591⟩⟩) exact82261RawTerms .large 82259 .exactZero (none)

def event82262 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6703⟩⟩) 0 ⟨6689⟩ 82238

def event82263 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6703⟩⟩) (.authority (.operator))

def exact82264RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6703⟩⟩]⟩, (1)⟩]

theorem exact82264RawTermsValid :
    exact82264RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82264 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6703⟩⟩) exact82264RawTerms .large 82263 .exactZero (none)

def event82265 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16592⟩⟩) 0 ⟨6703⟩ 82264

def event82266 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16592⟩⟩) 1 ⟨16591⟩ 82261

def event82267 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16592⟩⟩) (.sum [.predecessor 0 82265 .coefficient, .predecessor 1 82266 .coefficient])

def exact82268RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6703⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16549⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact82268RawTermsValid :
    exact82268RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82268 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16592⟩⟩) exact82268RawTerms .large 82267 .exactZero (none)

def event82269 : Event := .predecessor (⟨.program ⟨214⟩, ⟨29169⟩⟩) 0 ⟨16592⟩ 82268

def event82270 : Event := .predecessor (⟨.program ⟨214⟩, ⟨29169⟩⟩) 1 ⟨29168⟩ 82245

def event82271 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨29169⟩⟩) (.product (.predecessor 0 82269 .coefficient) (.predecessor 1 82270 .coefficient) (⟨false, false, none, none, none⟩))

def event82272 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨29169⟩⟩, .operator (⟨82268, 0⟩, ⟨82245, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6703⟩⟩, ⟨.program ⟨214⟩, ⟨29168⟩⟩]⟩, (1)⟩)

def event82273 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨29169⟩⟩, .operator (⟨82268, 1⟩, ⟨82245, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨16549⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29168⟩⟩]⟩, (-1)⟩)

def event82274 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨29169⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨16549⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29168⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨29168⟩⟩) ⟨24540⟩ 82242)

def event82275 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨29169⟩⟩, .relation 82274 0, ⟨[⟨.program ⟨214⟩, ⟨16549⟩⟩], [⟨.program ⟨214⟩, ⟨24540⟩⟩]⟩, (-1)⟩)

def exact82276RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6703⟩⟩, ⟨.program ⟨214⟩, ⟨29168⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16549⟩⟩], [⟨.program ⟨214⟩, ⟨24540⟩⟩]⟩, (-1)⟩]

theorem exact82276RawTermsValid :
    exact82276RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82276 : Event := .resultExact (⟨.program ⟨214⟩, ⟨29169⟩⟩) exact82276RawTerms .large 82271 .exactZero (none)

def event82277 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18205⟩⟩) 0 ⟨16550⟩ 82234

def event82278 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨18205⟩⟩) (.authority (.programFamilyFact))

def exact82279RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨18205⟩⟩], []⟩, (1)⟩]

theorem exact82279RawTermsValid :
    exact82279RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82279 : Event := .resultExact (⟨.program ⟨214⟩, ⟨18205⟩⟩) exact82279RawTerms (.finite 63) 82278 .exactZero (none)

def event82280 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18206⟩⟩) 0 ⟨6544⟩ 82256

def event82281 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18206⟩⟩) 1 ⟨18205⟩ 82279

def event82282 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨18206⟩⟩) (.product (.predecessor 0 82280 .coefficient) (.predecessor 1 82281 .coefficient) (⟨false, true, none, none, some 1⟩))

def event82283 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18206⟩⟩, .operator (⟨82256, 0⟩, ⟨82279, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨18205⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact82284RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨18205⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact82284RawTermsValid :
    exact82284RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82284 : Event := .resultExact (⟨.program ⟨214⟩, ⟨18206⟩⟩) exact82284RawTerms .large 82282 .exactZero (none)

def event82285 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6735⟩⟩) 0 ⟨6689⟩ 82238

def event82286 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6735⟩⟩) (.authority (.operator))

def exact82287RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6735⟩⟩]⟩, (1)⟩]

theorem exact82287RawTermsValid :
    exact82287RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82287 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6735⟩⟩) exact82287RawTerms .large 82286 .exactZero (none)

def event82288 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18207⟩⟩) 0 ⟨6735⟩ 82287

def event82289 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18207⟩⟩) 1 ⟨18206⟩ 82284

def event82290 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨18207⟩⟩) (.sum [.predecessor 0 82288 .coefficient, .predecessor 1 82289 .coefficient])

def exact82291RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6735⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨18205⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact82291RawTermsValid :
    exact82291RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82291 : Event := .resultExact (⟨.program ⟨214⟩, ⟨18207⟩⟩) exact82291RawTerms .large 82290 .exactZero (none)

def event82292 : Event := .predecessor (⟨.program ⟨214⟩, ⟨29173⟩⟩) 0 ⟨18207⟩ 82291

def event82293 : Event := .predecessor (⟨.program ⟨214⟩, ⟨29173⟩⟩) 1 ⟨29169⟩ 82276

def event82294 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨29173⟩⟩) (.sum [.predecessor 0 82292 .coefficient, .predecessor 1 82293 .coefficient])

def exact82295RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6703⟩⟩, ⟨.program ⟨214⟩, ⟨29168⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6735⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16549⟩⟩], [⟨.program ⟨214⟩, ⟨24540⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨18205⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact82295RawTermsValid :
    exact82295RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82295 : Event := .resultExact (⟨.program ⟨214⟩, ⟨29173⟩⟩) exact82295RawTerms .large 82294 .exactZero (none)

def event82296 : Event := .preFoldPolynomial 82295 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6703⟩⟩, ⟨.program ⟨214⟩, ⟨29168⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6735⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16549⟩⟩], [⟨.program ⟨214⟩, ⟨24540⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨18205⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩] .exactZero none

def exact82297RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6703⟩⟩, ⟨.program ⟨214⟩, ⟨29168⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6735⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16549⟩⟩], [⟨.program ⟨214⟩, ⟨24540⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨18205⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

def event82297 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨29173⟩⟩) 82296 exact82297RawTerms .large 82294 .exactZero (none)

def event82298 : Event := .specializationComputed (⟨.program ⟨214⟩, ⟨16550⟩⟩) ⟨⟨148⟩, ⟨57⟩, ⟨109⟩⟩ ⟨82140, 82298⟩

def event82299 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨22267⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22264⟩⟩]⟩) (1) 0 2 (.universal 82298 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22264⟩⟩]⟩) (none) 82297)

def event82300 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨22267⟩⟩, .relation 82299 1, ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6735⟩⟩]⟩, (1)⟩)

def event82301 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨22267⟩⟩, .relation 82299 0, ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6703⟩⟩, ⟨.program ⟨214⟩, ⟨29168⟩⟩]⟩, (-1)⟩)

def event82302 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨22267⟩⟩, .relation 82299 2, ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16549⟩⟩], [⟨.program ⟨214⟩, ⟨24540⟩⟩]⟩, (1)⟩)

def event82303 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨22267⟩⟩, .relation 82299 3, ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨18205⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def exact82304RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6703⟩⟩, ⟨.program ⟨214⟩, ⟨29168⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6735⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16549⟩⟩], [⟨.program ⟨214⟩, ⟨24540⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨18205⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact82304RawTermsValid :
    exact82304RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82304 : Event := .resultExact (⟨.program ⟨214⟩, ⟨22267⟩⟩) exact82304RawTerms .large 82136 (.finite 1811303510016) (some (82138))

def event82305 : Event := .predecessor (⟨.program ⟨214⟩, ⟨29171⟩⟩) 0 ⟨22267⟩ 82304

def event82306 : Event := .predecessor (⟨.program ⟨214⟩, ⟨29171⟩⟩) 1 ⟨29170⟩ 82126

def event82307 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨29171⟩⟩) (.sum [.predecessor 0 82305 .coefficient, .predecessor 1 82306 .coefficient])

def event82308 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨29171⟩⟩, .operator (⟨82304, 0⟩, ⟨82126, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6703⟩⟩, ⟨.program ⟨214⟩, ⟨29168⟩⟩]⟩, (1)⟩)

def event82309 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨29171⟩⟩, .operator (⟨82304, 2⟩, ⟨82126, 1⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16549⟩⟩], [⟨.program ⟨214⟩, ⟨24540⟩⟩]⟩, (-1)⟩)

def event82310 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨29171⟩⟩) (.sum [.result 82304 .summary, .result 82126 .summary])

def exact82311RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6735⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨18205⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact82311RawTermsValid :
    exact82311RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82311 : Event := .resultExact (⟨.program ⟨214⟩, ⟨29171⟩⟩) exact82311RawTerms .large 82307 (.finite 1292337423279833362432) (some (82310))

def event82312 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24475⟩⟩) 0 ⟨16466⟩ 3960

def event82313 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24475⟩⟩) (.authority (.programFamilyFact))

def event82314 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨24475⟩⟩) (.finite 3720)

def event82315 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24477⟩⟩) 0 ⟨6689⟩ 5477

def event82316 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24477⟩⟩) 1 ⟨24475⟩ 82314

def event82317 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24477⟩⟩) (.authority (.operator))

def exact82318RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨24477⟩⟩]⟩, (1)⟩]

theorem exact82318RawTermsValid :
    exact82318RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82318 : Event := .resultExact (⟨.program ⟨214⟩, ⟨24477⟩⟩) exact82318RawTerms .large 82317 .exactZero (none)

def event82319 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28951⟩⟩) 0 ⟨24477⟩ 82318

def event82320 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28951⟩⟩) (.authority (.operator))

def exact82321RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨28951⟩⟩]⟩, (1)⟩]

theorem exact82321RawTermsValid :
    exact82321RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82321 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28951⟩⟩) exact82321RawTerms (.finite 8192) 82320 .exactZero (none)

def event82322 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23205⟩⟩) 0 ⟨12372⟩ 3954

def event82323 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23205⟩⟩) (.authority (.programFamilyFact))

def event82324 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨23205⟩⟩) (.finite 3720)

def event82325 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23206⟩⟩) 0 ⟨6689⟩ 5477

def event82326 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23206⟩⟩) 1 ⟨23205⟩ 82324

def event82327 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23206⟩⟩) (.authority (.operator))

def exact82328RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨23206⟩⟩]⟩, (1)⟩]

theorem exact82328RawTermsValid :
    exact82328RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82328 : Event := .resultExact (⟨.program ⟨214⟩, ⟨23206⟩⟩) exact82328RawTerms .large 82327 .exactZero (none)

def event82329 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25373⟩⟩) 0 ⟨23206⟩ 82328

def event82330 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25373⟩⟩) (.authority (.operator))

def exact82331RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨25373⟩⟩]⟩, (1)⟩]

theorem exact82331RawTermsValid :
    exact82331RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82331 : Event := .resultExact (⟨.program ⟨214⟩, ⟨25373⟩⟩) exact82331RawTerms (.finite 8192) 82330 .exactZero (none)

def event82332 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12373⟩⟩) 0 ⟨12370⟩ 3943

def event82333 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12373⟩⟩) 1 ⟨6567⟩ 79920

def event82334 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12373⟩⟩) (.tensor (.predecessor 0 82332 .coefficient) (.predecessor 1 82333 .coefficient) true false)

def event82335 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨12373⟩⟩, .operator (⟨3943, 0⟩, ⟨79920, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨12370⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact82336RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨12370⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact82336RawTermsValid :
    exact82336RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82336 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12373⟩⟩) exact82336RawTerms .large 82334 .exactZero (none)

def event82337 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7241⟩⟩) 0 ⟨5539⟩ 79790

def event82338 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7241⟩⟩) 1 ⟨6785⟩ 8977

def event82339 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7241⟩⟩) (.product (.predecessor 0 82337 .coefficient) (.predecessor 1 82338 .coefficient) (⟨false, false, none, none, none⟩))

def event82340 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7241⟩⟩, .operator (⟨79790, 0⟩, ⟨8977, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6785⟩⟩]⟩, (1)⟩)

def exact82341RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6785⟩⟩]⟩, (1)⟩]

theorem exact82341RawTermsValid :
    exact82341RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82341 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7241⟩⟩) exact82341RawTerms .large 82339 .exactZero (none)

def event82342 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12374⟩⟩) 0 ⟨7241⟩ 82341

def event82343 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12374⟩⟩) 1 ⟨12373⟩ 82336

def event82344 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12374⟩⟩) (.sum [.predecessor 0 82342 .coefficient, .predecessor 1 82343 .coefficient])

def exact82345RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6785⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨12370⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact82345RawTermsValid :
    exact82345RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82345 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12374⟩⟩) exact82345RawTerms .large 82344 .exactZero (none)

def event82346 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12375⟩⟩) 0 ⟨12374⟩ 82345

def event82347 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12375⟩⟩) 1 ⟨99⟩ 8969

def event82348 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12375⟩⟩) (.sum [.predecessor 0 82346 .coefficient, .predecessor 1 82347 .coefficient])

def event82349 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12375⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨99⟩⟩]⟩) [⟨.result 8969 .coefficient, false, none⟩])

def event82350 : Event := .survivorFold (1) 82349

def exact82351RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6785⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨12370⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact82351RawTermsValid :
    exact82351RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82351 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12375⟩⟩) exact82351RawTerms .large 82348 (.finite 26) (some (82349))

def event82352 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12376⟩⟩) 0 ⟨12375⟩ 82351

def event82353 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12376⟩⟩) 1 ⟨9820⟩ 3946

def event82354 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12376⟩⟩) (.product (.predecessor 0 82352 .coefficient) (.predecessor 1 82353 .coefficient) (⟨false, true, none, none, some 1⟩))

def event82355 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12376⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨9820⟩⟩], []⟩) [⟨.result 3946 .coefficient, true, some 1⟩])

def event82356 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12376⟩⟩) (.product (.result 82351 .summary) (.transfer 82355) (⟨false, false, none, none, none⟩))

def event82357 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨12376⟩⟩, .operator (⟨82351, 1⟩, ⟨3946, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨9820⟩⟩, ⟨.program ⟨214⟩, ⟨12370⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def event82358 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨12376⟩⟩, .operator (⟨82351, 0⟩, ⟨3946, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨9820⟩⟩], [⟨.program ⟨214⟩, ⟨6785⟩⟩]⟩, (1)⟩)

def exact82359RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨9820⟩⟩], [⟨.program ⟨214⟩, ⟨6785⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨9820⟩⟩, ⟨.program ⟨214⟩, ⟨12370⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact82359RawTermsValid :
    exact82359RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82359 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12376⟩⟩) exact82359RawTerms .large 82354 (.finite 33280) (some (82356))

def event82360 : Event := .predecessor (⟨.program ⟨214⟩, ⟨9821⟩⟩) 0 ⟨9820⟩ 3946

def event82361 : Event := .predecessor (⟨.program ⟨214⟩, ⟨9821⟩⟩) 1 ⟨6567⟩ 79920

def event82362 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨9821⟩⟩) (.tensor (.predecessor 0 82360 .coefficient) (.predecessor 1 82361 .coefficient) true false)

def event82363 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨9821⟩⟩, .operator (⟨3946, 0⟩, ⟨79920, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨9820⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact82364RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨9820⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact82364RawTermsValid :
    exact82364RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82364 : Event := .resultExact (⟨.program ⟨214⟩, ⟨9821⟩⟩) exact82364RawTerms .large 82362 .exactZero (none)

def event82365 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7221⟩⟩) 0 ⟨5539⟩ 79790

def event82366 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7221⟩⟩) 1 ⟨6765⟩ 9018

def event82367 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7221⟩⟩) (.product (.predecessor 0 82365 .coefficient) (.predecessor 1 82366 .coefficient) (⟨false, false, none, none, none⟩))

def event82368 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7221⟩⟩, .operator (⟨79790, 0⟩, ⟨9018, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6765⟩⟩]⟩, (1)⟩)

def exact82369RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6765⟩⟩]⟩, (1)⟩]

theorem exact82369RawTermsValid :
    exact82369RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82369 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7221⟩⟩) exact82369RawTerms .large 82367 .exactZero (none)

def event82370 : Event := .predecessor (⟨.program ⟨214⟩, ⟨9822⟩⟩) 0 ⟨7221⟩ 82369

def event82371 : Event := .predecessor (⟨.program ⟨214⟩, ⟨9822⟩⟩) 1 ⟨9821⟩ 82364

def event82372 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨9822⟩⟩) (.sum [.predecessor 0 82370 .coefficient, .predecessor 1 82371 .coefficient])

def exact82373RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6765⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨9820⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact82373RawTermsValid :
    exact82373RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82373 : Event := .resultExact (⟨.program ⟨214⟩, ⟨9822⟩⟩) exact82373RawTerms .large 82372 .exactZero (none)

def event82374 : Event := .predecessor (⟨.program ⟨214⟩, ⟨9823⟩⟩) 0 ⟨9822⟩ 82373

def event82375 : Event := .predecessor (⟨.program ⟨214⟩, ⟨9823⟩⟩) 1 ⟨79⟩ 9010

def event82376 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨9823⟩⟩) (.sum [.predecessor 0 82374 .coefficient, .predecessor 1 82375 .coefficient])

def event82377 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨9823⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨79⟩⟩]⟩) [⟨.result 9010 .coefficient, false, none⟩])

def event82378 : Event := .survivorFold (1) 82377

def exact82379RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6765⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨9820⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact82379RawTermsValid :
    exact82379RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82379 : Event := .resultExact (⟨.program ⟨214⟩, ⟨9823⟩⟩) exact82379RawTerms .large 82376 (.finite 26) (some (82377))

def event82380 : Event := .predecessor (⟨.program ⟨214⟩, ⟨9824⟩⟩) 0 ⟨9823⟩ 82379

def event82381 : Event := .predecessor (⟨.program ⟨214⟩, ⟨9824⟩⟩) 1 ⟨7868⟩ 9007

def event82382 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨9824⟩⟩) (.product (.predecessor 0 82380 .coefficient) (.predecessor 1 82381 .coefficient) (⟨false, false, none, none, none⟩))

def event82383 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨9824⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨7867⟩⟩]⟩) [⟨.result 9003 .coefficient, false, none⟩])

def event82384 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨9824⟩⟩) (.product (.result 82379 .summary) (.transfer 82383) (⟨false, false, none, none, none⟩))

def event82385 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨9824⟩⟩, .operator (⟨82379, 1⟩, ⟨9007, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨9820⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7867⟩⟩]⟩, (-1)⟩)

def event82386 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨9824⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨9820⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7867⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨7867⟩⟩) ⟨6785⟩ 8977)

def event82387 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨9824⟩⟩, .relation 82386 0, ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨9820⟩⟩], [⟨.program ⟨214⟩, ⟨6785⟩⟩]⟩, (-1)⟩)

def event82388 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨9824⟩⟩, .operator (⟨82379, 0⟩, ⟨9007, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6765⟩⟩, ⟨.program ⟨214⟩, ⟨7867⟩⟩]⟩, (1)⟩)

def exact82389RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6765⟩⟩, ⟨.program ⟨214⟩, ⟨7867⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨9820⟩⟩], [⟨.program ⟨214⟩, ⟨6785⟩⟩]⟩, (-1)⟩]

theorem exact82389RawTermsValid :
    exact82389RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82389 : Event := .resultExact (⟨.program ⟨214⟩, ⟨9824⟩⟩) exact82389RawTerms .large 82382 (.finite 95420416) (some (82384))

def event82390 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12377⟩⟩) 0 ⟨9824⟩ 82389

def event82391 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12377⟩⟩) 1 ⟨12376⟩ 82359

def event82392 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12377⟩⟩) (.sum [.predecessor 0 82390 .coefficient, .predecessor 1 82391 .coefficient])

def event82393 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨12377⟩⟩, .operator (⟨82389, 1⟩, ⟨82359, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨9820⟩⟩], [⟨.program ⟨214⟩, ⟨6785⟩⟩]⟩, (1)⟩)

def event82394 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12377⟩⟩) (.sum [.result 82389 .summary, .result 82359 .summary])

def exact82395RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6765⟩⟩, ⟨.program ⟨214⟩, ⟨7867⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨9820⟩⟩, ⟨.program ⟨214⟩, ⟨12370⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact82395RawTermsValid :
    exact82395RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82395 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12377⟩⟩) exact82395RawTerms .large 82392 (.finite 95453696) (some (82394))

def event82396 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25374⟩⟩) 0 ⟨12377⟩ 82395

def event82397 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25374⟩⟩) 1 ⟨25373⟩ 82331

def event82398 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25374⟩⟩) (.product (.predecessor 0 82396 .coefficient) (.predecessor 1 82397 .coefficient) (⟨false, false, none, none, none⟩))

def event82399 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25374⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨25373⟩⟩]⟩) [⟨.result 82331 .coefficient, false, none⟩])

def event82400 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25374⟩⟩) (.product (.result 82395 .summary) (.transfer 82399) (⟨false, false, none, none, none⟩))

def event82401 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25374⟩⟩, .operator (⟨82395, 1⟩, ⟨82331, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨9820⟩⟩, ⟨.program ⟨214⟩, ⟨12370⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25373⟩⟩]⟩, (-1)⟩)

def event82402 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨25374⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨9820⟩⟩, ⟨.program ⟨214⟩, ⟨12370⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25373⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨25373⟩⟩) ⟨23206⟩ 82328)

def event82403 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25374⟩⟩, .relation 82402 0, ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨9820⟩⟩, ⟨.program ⟨214⟩, ⟨12370⟩⟩], [⟨.program ⟨214⟩, ⟨23206⟩⟩]⟩, (-1)⟩)

def event82404 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25374⟩⟩, .operator (⟨82395, 0⟩, ⟨82331, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6765⟩⟩, ⟨.program ⟨214⟩, ⟨7867⟩⟩, ⟨.program ⟨214⟩, ⟨25373⟩⟩]⟩, (1)⟩)

def exact82405RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6765⟩⟩, ⟨.program ⟨214⟩, ⟨7867⟩⟩, ⟨.program ⟨214⟩, ⟨25373⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨9820⟩⟩, ⟨.program ⟨214⟩, ⟨12370⟩⟩], [⟨.program ⟨214⟩, ⟨23206⟩⟩]⟩, (-1)⟩]

theorem exact82405RawTermsValid :
    exact82405RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82405 : Event := .resultExact (⟨.program ⟨214⟩, ⟨25374⟩⟩) exact82405RawTerms .large 82398 (.finite 350316591579136) (some (82400))

def event82406 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19888⟩⟩) 0 ⟨12372⟩ 3954

def event82407 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19888⟩⟩) (.authority (.relationPreimageSource ⟨20⟩))

def exact82408RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨19888⟩⟩]⟩, (1)⟩]

theorem exact82408RawTermsValid :
    exact82408RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82408 : Event := .resultExact (⟨.program ⟨214⟩, ⟨19888⟩⟩) exact82408RawTerms (.finite 136065468) 82407 .exactZero (none)

def event82409 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19890⟩⟩) 0 ⟨19888⟩ 82408

def event82410 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19890⟩⟩) 1 ⟨2348⟩ 4

def event82411 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19890⟩⟩) (.scale (.predecessor 0 82409 .coefficient) (.value (.predecessor 1 82410 .coefficient)))

def exact82412RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨19888⟩⟩]⟩, (1)⟩]

theorem exact82412RawTermsValid :
    exact82412RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82412 : Event := .resultExact (⟨.program ⟨214⟩, ⟨19890⟩⟩) exact82412RawTerms (.finite 136065468) 82411 .exactZero (none)

def event82413 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19891⟩⟩) 0 ⟨5541⟩ 80012

def event82414 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19891⟩⟩) 1 ⟨19890⟩ 82412

def event82415 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19891⟩⟩) (.product (.predecessor 0 82413 .coefficient) (.predecessor 1 82414 .coefficient) (⟨false, false, none, none, none⟩))

def event82416 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19891⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨19888⟩⟩]⟩) [⟨.result 82408 .coefficient, false, none⟩])

def event82417 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19891⟩⟩) (.product (.result 80012 .summary) (.transfer 82416) (⟨false, false, none, none, none⟩))

def event82418 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19891⟩⟩, .operator (⟨80012, 0⟩, ⟨82412, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19888⟩⟩]⟩, (1)⟩)

def event82419 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨19889⟩⟩)

def event82420 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event82421 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event82422 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨2348⟩⟩) (.authority (.operator))

def event82423 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨2348⟩⟩) (.finite 1)

def event82424 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event82425 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event82426 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event82427 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event82428 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 82427

def event82429 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 82425

def event82430 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 82428 .coefficient) (.value (.predecessor 1 82429 .coefficient)))

def event82431 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def eventLeaf5136 : Array AnnotatedEvent := #[
  { event := event82176
    frameStart := 82140 },
  { event := event82177
    frameStart := 82140 },
  { event := event82178
    frameStart := 82140 },
  { event := event82179
    frameStart := 82140 },
  { event := event82180
    frameStart := 82140 },
  { event := event82181
    frameStart := 82140 },
  { event := event82182
    frameStart := 82140 },
  { event := event82183
    frameStart := 82140 },
  { event := event82184
    frameStart := 82140 },
  { event := event82185
    frameStart := 82140 },
  { event := event82186
    frameStart := 82140 },
  { event := event82187
    frameStart := 82140 },
  { event := event82188
    frameStart := 82140 },
  { event := event82189
    frameStart := 82140 },
  { event := event82190
    frameStart := 82140 },
  { event := event82191
    frameStart := 82140 }
]

def eventLeaf5137 : Array AnnotatedEvent := #[
  { event := event82192
    frameStart := 82140 },
  { event := event82193
    frameStart := 82140 },
  { event := event82194
    frameStart := 82194 },
  { event := event82195
    frameStart := 82194 },
  { event := event82196
    frameStart := 82194 },
  { event := event82197
    frameStart := 82194 },
  { event := event82198
    frameStart := 82194 },
  { event := event82199
    frameStart := 82194 },
  { event := event82200
    frameStart := 82194 },
  { event := event82201
    frameStart := 82194 },
  { event := event82202
    frameStart := 82194 },
  { event := event82203
    frameStart := 82194 },
  { event := event82204
    frameStart := 82194 },
  { event := event82205
    frameStart := 82194 },
  { event := event82206
    frameStart := 82194 },
  { event := event82207
    frameStart := 82194 }
]

def eventLeaf5138 : Array AnnotatedEvent := #[
  { event := event82208
    frameStart := 82194 },
  { event := event82209
    frameStart := 82194 },
  { event := event82210
    frameStart := 82194 },
  { event := event82211
    frameStart := 82194 },
  { event := event82212
    frameStart := 82194 },
  { event := event82213
    frameStart := 82194 },
  { event := event82214
    frameStart := 82194 },
  { event := event82215
    frameStart := 82194 },
  { event := event82216
    frameStart := 82194 },
  { event := event82217
    frameStart := 82194 },
  { event := event82218
    frameStart := 82194 },
  { event := event82219
    frameStart := 82194 },
  { event := event82220
    frameStart := 82194 },
  { event := event82221
    frameStart := 82194 },
  { event := event82222
    frameStart := 82194 },
  { event := event82223
    frameStart := 82194 }
]

def eventLeaf5139 : Array AnnotatedEvent := #[
  { event := event82224
    frameStart := 82194 },
  { event := event82225
    frameStart := 82194 },
  { event := event82226
    frameStart := 82194 },
  { event := event82227
    frameStart := 82194 },
  { event := event82228
    frameStart := 82194 },
  { event := event82229
    frameStart := 82194 },
  { event := event82230
    frameStart := 82194 },
  { event := event82231
    frameStart := 82194 },
  { event := event82232
    frameStart := 82194 },
  { event := event82233
    frameStart := 82194 },
  { event := event82234
    frameStart := 82194 },
  { event := event82235
    frameStart := 82194 },
  { event := event82236
    frameStart := 82194 },
  { event := event82237
    frameStart := 82194 },
  { event := event82238
    frameStart := 82194 },
  { event := event82239
    frameStart := 82194 }
]

def eventLeaf5140 : Array AnnotatedEvent := #[
  { event := event82240
    frameStart := 82194 },
  { event := event82241
    frameStart := 82194 },
  { event := event82242
    frameStart := 82194 },
  { event := event82243
    frameStart := 82194 },
  { event := event82244
    frameStart := 82194 },
  { event := event82245
    frameStart := 82194 },
  { event := event82246
    frameStart := 82194 },
  { event := event82247
    frameStart := 82194 },
  { event := event82248
    frameStart := 82194 },
  { event := event82249
    frameStart := 82194 },
  { event := event82250
    frameStart := 82194 },
  { event := event82251
    frameStart := 82194 },
  { event := event82252
    frameStart := 82194 },
  { event := event82253
    frameStart := 82194 },
  { event := event82254
    frameStart := 82194 },
  { event := event82255
    frameStart := 82194 }
]

def eventLeaf5141 : Array AnnotatedEvent := #[
  { event := event82256
    frameStart := 82194 },
  { event := event82257
    frameStart := 82194 },
  { event := event82258
    frameStart := 82194 },
  { event := event82259
    frameStart := 82194 },
  { event := event82260
    frameStart := 82194 },
  { event := event82261
    frameStart := 82194 },
  { event := event82262
    frameStart := 82194 },
  { event := event82263
    frameStart := 82194 },
  { event := event82264
    frameStart := 82194 },
  { event := event82265
    frameStart := 82194 },
  { event := event82266
    frameStart := 82194 },
  { event := event82267
    frameStart := 82194 },
  { event := event82268
    frameStart := 82194 },
  { event := event82269
    frameStart := 82194 },
  { event := event82270
    frameStart := 82194 },
  { event := event82271
    frameStart := 82194 }
]

def eventLeaf5142 : Array AnnotatedEvent := #[
  { event := event82272
    frameStart := 82194 },
  { event := event82273
    frameStart := 82194 },
  { event := event82274
    frameStart := 82194 },
  { event := event82275
    frameStart := 82194 },
  { event := event82276
    frameStart := 82194 },
  { event := event82277
    frameStart := 82194 },
  { event := event82278
    frameStart := 82194 },
  { event := event82279
    frameStart := 82194 },
  { event := event82280
    frameStart := 82194 },
  { event := event82281
    frameStart := 82194 },
  { event := event82282
    frameStart := 82194 },
  { event := event82283
    frameStart := 82194 },
  { event := event82284
    frameStart := 82194 },
  { event := event82285
    frameStart := 82194 },
  { event := event82286
    frameStart := 82194 },
  { event := event82287
    frameStart := 82194 }
]

def eventLeaf5143 : Array AnnotatedEvent := #[
  { event := event82288
    frameStart := 82194 },
  { event := event82289
    frameStart := 82194 },
  { event := event82290
    frameStart := 82194 },
  { event := event82291
    frameStart := 82194 },
  { event := event82292
    frameStart := 82194 },
  { event := event82293
    frameStart := 82194 },
  { event := event82294
    frameStart := 82194 },
  { event := event82295
    frameStart := 82194 },
  { event := event82296
    frameStart := 82194 },
  { event := event82297
    frameStart := 82194 },
  { event := event82298
    frameStart := 0 },
  { event := event82299
    frameStart := 0 },
  { event := event82300
    frameStart := 0 },
  { event := event82301
    frameStart := 0 },
  { event := event82302
    frameStart := 0 },
  { event := event82303
    frameStart := 0 }
]

def eventLeaf5144 : Array AnnotatedEvent := #[
  { event := event82304
    frameStart := 0 },
  { event := event82305
    frameStart := 0 },
  { event := event82306
    frameStart := 0 },
  { event := event82307
    frameStart := 0 },
  { event := event82308
    frameStart := 0 },
  { event := event82309
    frameStart := 0 },
  { event := event82310
    frameStart := 0 },
  { event := event82311
    frameStart := 0 },
  { event := event82312
    frameStart := 0 },
  { event := event82313
    frameStart := 0 },
  { event := event82314
    frameStart := 0 },
  { event := event82315
    frameStart := 0 },
  { event := event82316
    frameStart := 0 },
  { event := event82317
    frameStart := 0 },
  { event := event82318
    frameStart := 0 },
  { event := event82319
    frameStart := 0 }
]

def eventLeaf5145 : Array AnnotatedEvent := #[
  { event := event82320
    frameStart := 0 },
  { event := event82321
    frameStart := 0 },
  { event := event82322
    frameStart := 0 },
  { event := event82323
    frameStart := 0 },
  { event := event82324
    frameStart := 0 },
  { event := event82325
    frameStart := 0 },
  { event := event82326
    frameStart := 0 },
  { event := event82327
    frameStart := 0 },
  { event := event82328
    frameStart := 0 },
  { event := event82329
    frameStart := 0 },
  { event := event82330
    frameStart := 0 },
  { event := event82331
    frameStart := 0 },
  { event := event82332
    frameStart := 0 },
  { event := event82333
    frameStart := 0 },
  { event := event82334
    frameStart := 0 },
  { event := event82335
    frameStart := 0 }
]

def eventLeaf5146 : Array AnnotatedEvent := #[
  { event := event82336
    frameStart := 0 },
  { event := event82337
    frameStart := 0 },
  { event := event82338
    frameStart := 0 },
  { event := event82339
    frameStart := 0 },
  { event := event82340
    frameStart := 0 },
  { event := event82341
    frameStart := 0 },
  { event := event82342
    frameStart := 0 },
  { event := event82343
    frameStart := 0 },
  { event := event82344
    frameStart := 0 },
  { event := event82345
    frameStart := 0 },
  { event := event82346
    frameStart := 0 },
  { event := event82347
    frameStart := 0 },
  { event := event82348
    frameStart := 0 },
  { event := event82349
    frameStart := 0 },
  { event := event82350
    frameStart := 0 },
  { event := event82351
    frameStart := 0 }
]

def eventLeaf5147 : Array AnnotatedEvent := #[
  { event := event82352
    frameStart := 0 },
  { event := event82353
    frameStart := 0 },
  { event := event82354
    frameStart := 0 },
  { event := event82355
    frameStart := 0 },
  { event := event82356
    frameStart := 0 },
  { event := event82357
    frameStart := 0 },
  { event := event82358
    frameStart := 0 },
  { event := event82359
    frameStart := 0 },
  { event := event82360
    frameStart := 0 },
  { event := event82361
    frameStart := 0 },
  { event := event82362
    frameStart := 0 },
  { event := event82363
    frameStart := 0 },
  { event := event82364
    frameStart := 0 },
  { event := event82365
    frameStart := 0 },
  { event := event82366
    frameStart := 0 },
  { event := event82367
    frameStart := 0 }
]

def eventLeaf5148 : Array AnnotatedEvent := #[
  { event := event82368
    frameStart := 0 },
  { event := event82369
    frameStart := 0 },
  { event := event82370
    frameStart := 0 },
  { event := event82371
    frameStart := 0 },
  { event := event82372
    frameStart := 0 },
  { event := event82373
    frameStart := 0 },
  { event := event82374
    frameStart := 0 },
  { event := event82375
    frameStart := 0 },
  { event := event82376
    frameStart := 0 },
  { event := event82377
    frameStart := 0 },
  { event := event82378
    frameStart := 0 },
  { event := event82379
    frameStart := 0 },
  { event := event82380
    frameStart := 0 },
  { event := event82381
    frameStart := 0 },
  { event := event82382
    frameStart := 0 },
  { event := event82383
    frameStart := 0 }
]

def eventLeaf5149 : Array AnnotatedEvent := #[
  { event := event82384
    frameStart := 0 },
  { event := event82385
    frameStart := 0 },
  { event := event82386
    frameStart := 0 },
  { event := event82387
    frameStart := 0 },
  { event := event82388
    frameStart := 0 },
  { event := event82389
    frameStart := 0 },
  { event := event82390
    frameStart := 0 },
  { event := event82391
    frameStart := 0 },
  { event := event82392
    frameStart := 0 },
  { event := event82393
    frameStart := 0 },
  { event := event82394
    frameStart := 0 },
  { event := event82395
    frameStart := 0 },
  { event := event82396
    frameStart := 0 },
  { event := event82397
    frameStart := 0 },
  { event := event82398
    frameStart := 0 },
  { event := event82399
    frameStart := 0 }
]

def eventLeaf5150 : Array AnnotatedEvent := #[
  { event := event82400
    frameStart := 0 },
  { event := event82401
    frameStart := 0 },
  { event := event82402
    frameStart := 0 },
  { event := event82403
    frameStart := 0 },
  { event := event82404
    frameStart := 0 },
  { event := event82405
    frameStart := 0 },
  { event := event82406
    frameStart := 0 },
  { event := event82407
    frameStart := 0 },
  { event := event82408
    frameStart := 0 },
  { event := event82409
    frameStart := 0 },
  { event := event82410
    frameStart := 0 },
  { event := event82411
    frameStart := 0 },
  { event := event82412
    frameStart := 0 },
  { event := event82413
    frameStart := 0 },
  { event := event82414
    frameStart := 0 },
  { event := event82415
    frameStart := 0 }
]

def eventLeaf5151 : Array AnnotatedEvent := #[
  { event := event82416
    frameStart := 0 },
  { event := event82417
    frameStart := 0 },
  { event := event82418
    frameStart := 0 },
  { event := event82419
    frameStart := 82419 },
  { event := event82420
    frameStart := 82419 },
  { event := event82421
    frameStart := 82419 },
  { event := event82422
    frameStart := 82419 },
  { event := event82423
    frameStart := 82419 },
  { event := event82424
    frameStart := 82419 },
  { event := event82425
    frameStart := 82419 },
  { event := event82426
    frameStart := 82419 },
  { event := event82427
    frameStart := 82419 },
  { event := event82428
    frameStart := 82419 },
  { event := event82429
    frameStart := 82419 },
  { event := event82430
    frameStart := 82419 },
  { event := event82431
    frameStart := 82419 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events321
