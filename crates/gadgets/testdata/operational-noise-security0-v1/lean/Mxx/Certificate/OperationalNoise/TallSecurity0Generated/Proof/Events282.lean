import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events282

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event72192 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event72193 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨360⟩⟩) (.authority (.operator))

def event72194 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨360⟩⟩) (.finite 2)

def event72195 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event72196 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event72197 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event72198 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event72199 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 72198

def event72200 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 72196

def event72201 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 72199 .coefficient) (.value (.predecessor 1 72200 .coefficient)))

def event72202 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event72203 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5504⟩⟩) 0 ⟨5503⟩ 72202

def event72204 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5504⟩⟩) 1 ⟨360⟩ 72194

def event72205 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5504⟩⟩) (.sum [.predecessor 0 72203 .coefficient, .predecessor 1 72204 .coefficient])

def event72206 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5504⟩⟩) (.finite 219)

def event72207 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5530⟩⟩) 0 ⟨5504⟩ 72206

def event72208 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5530⟩⟩) 1 ⟨961⟩ 72192

def event72209 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5530⟩⟩) (.identity (.predecessor 1 72208 .coefficient))

def event72210 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5530⟩⟩) (.finite 224)

def event72211 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11129⟩⟩) 0 ⟨5530⟩ 72210

def event72212 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11129⟩⟩) (.authority (.programFamilyFact))

def exact72213RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11129⟩⟩], []⟩, (1)⟩]

theorem exact72213RawTermsValid :
    exact72213RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event72213 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11129⟩⟩) exact72213RawTerms (.finite 6) 72212 .exactZero (none)

def event72214 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12154⟩⟩) 0 ⟨5530⟩ 72210

def event72215 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12154⟩⟩) (.authority (.programFamilyFact))

def exact72216RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨12154⟩⟩], []⟩, (1)⟩]

theorem exact72216RawTermsValid :
    exact72216RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event72216 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12154⟩⟩) exact72216RawTerms (.finite 6) 72215 .exactZero (none)

def event72217 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12155⟩⟩) 0 ⟨12154⟩ 72216

def event72218 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12155⟩⟩) 1 ⟨11129⟩ 72213

def event72219 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12155⟩⟩) (.product (.predecessor 0 72217 .coefficient) (.predecessor 1 72218 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event72220 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨12155⟩⟩, .operator (⟨72216, 0⟩, ⟨72213, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨11129⟩⟩, ⟨.program ⟨214⟩, ⟨12154⟩⟩], []⟩, (1)⟩)

def exact72221RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11129⟩⟩, ⟨.program ⟨214⟩, ⟨12154⟩⟩], []⟩, (1)⟩]

theorem exact72221RawTermsValid :
    exact72221RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event72221 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12155⟩⟩) exact72221RawTerms (.finite 36) 72219 .exactZero (none)

def event72222 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12156⟩⟩) 0 ⟨12155⟩ 72221

def event72223 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12156⟩⟩) (.identity (.predecessor 0 72222 .coefficient))

def event72224 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨12156⟩⟩) (.finite 36)

def event72225 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23161⟩⟩) 0 ⟨12156⟩ 72224

def event72226 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23161⟩⟩) (.authority (.programFamilyFact))

def event72227 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨23161⟩⟩) (.finite 3720)

def event72228 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨6689⟩⟩) .missing

def event72229 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23162⟩⟩) 0 ⟨6689⟩ 72228

def event72230 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23162⟩⟩) 1 ⟨23161⟩ 72227

def event72231 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23162⟩⟩) (.authority (.operator))

def exact72232RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨23162⟩⟩]⟩, (1)⟩]

theorem exact72232RawTermsValid :
    exact72232RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event72232 : Event := .resultExact (⟨.program ⟨214⟩, ⟨23162⟩⟩) exact72232RawTerms .large 72231 .exactZero (none)

def event72233 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25291⟩⟩) 0 ⟨23162⟩ 72232

def event72234 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25291⟩⟩) (.authority (.operator))

def exact72235RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨25291⟩⟩]⟩, (1)⟩]

theorem exact72235RawTermsValid :
    exact72235RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event72235 : Event := .resultExact (⟨.program ⟨214⟩, ⟨25291⟩⟩) exact72235RawTerms (.finite 8192) 72234 .exactZero (none)

def event72236 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨110⟩⟩) (.authority (.operator))

def event72237 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨110⟩⟩) .exactZero

def event72238 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12266⟩⟩) 0 ⟨12156⟩ 72224

def event72239 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12266⟩⟩) 1 ⟨110⟩ 72237

def event72240 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12266⟩⟩) (.sum [.predecessor 0 72238 .coefficient, .predecessor 1 72239 .coefficient])

def event72241 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨12266⟩⟩) (.finite 36)

def event72242 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12267⟩⟩) 0 ⟨12266⟩ 72241

def event72243 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12267⟩⟩) (.identity (.predecessor 0 72242 .coefficient))

def exact72244RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11129⟩⟩, ⟨.program ⟨214⟩, ⟨12154⟩⟩], []⟩, (1)⟩]

theorem exact72244RawTermsValid :
    exact72244RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event72244 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12267⟩⟩) exact72244RawTerms (.finite 36) 72243 .exactZero (none)

def event72245 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6544⟩⟩) (.authority (.factStore))

def exact72246RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact72246RawTermsValid :
    exact72246RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event72246 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6544⟩⟩) exact72246RawTerms .large 72245 .exactZero (none)

def event72247 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12268⟩⟩) 0 ⟨6544⟩ 72246

def event72248 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12268⟩⟩) 1 ⟨12267⟩ 72244

def event72249 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12268⟩⟩) (.product (.predecessor 0 72247 .coefficient) (.predecessor 1 72248 .coefficient) (⟨false, false, none, none, none⟩))

def event72250 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨12268⟩⟩, .operator (⟨72246, 0⟩, ⟨72244, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨11129⟩⟩, ⟨.program ⟨214⟩, ⟨12154⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact72251RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11129⟩⟩, ⟨.program ⟨214⟩, ⟨12154⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact72251RawTermsValid :
    exact72251RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event72251 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12268⟩⟩) exact72251RawTerms .large 72249 .exactZero (none)

def event72252 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨2348⟩⟩) (.authority (.operator))

def event72253 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨2348⟩⟩) (.finite 1)

def event72254 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6757⟩⟩) 0 ⟨6689⟩ 72228

def event72255 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6757⟩⟩) (.authority (.operator))

def exact72256RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6757⟩⟩]⟩, (1)⟩]

theorem exact72256RawTermsValid :
    exact72256RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event72256 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6757⟩⟩) exact72256RawTerms .large 72255 .exactZero (none)

def event72257 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6775⟩⟩) 0 ⟨6757⟩ 72256

def event72258 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6775⟩⟩) (.identity (.predecessor 0 72257 .coefficient))

def exact72259RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6775⟩⟩]⟩, (1)⟩]

theorem exact72259RawTermsValid :
    exact72259RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event72259 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6775⟩⟩) exact72259RawTerms .large 72258 .exactZero (none)

def event72260 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7840⟩⟩) 0 ⟨6775⟩ 72259

def event72261 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7840⟩⟩) (.authority (.operator))

def exact72262RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨7840⟩⟩]⟩, (1)⟩]

theorem exact72262RawTermsValid :
    exact72262RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event72262 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7840⟩⟩) exact72262RawTerms (.finite 8192) 72261 .exactZero (none)

def event72263 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7841⟩⟩) 0 ⟨7840⟩ 72262

def event72264 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7841⟩⟩) 1 ⟨2348⟩ 72253

def event72265 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7841⟩⟩) (.scale (.predecessor 0 72263 .coefficient) (.value (.predecessor 1 72264 .coefficient)))

def exact72266RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨7840⟩⟩]⟩, (1)⟩]

theorem exact72266RawTermsValid :
    exact72266RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event72266 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7841⟩⟩) exact72266RawTerms (.finite 8192) 72265 .exactZero (none)

def event72267 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6792⟩⟩) 0 ⟨6757⟩ 72256

def event72268 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6792⟩⟩) (.identity (.predecessor 0 72267 .coefficient))

def exact72269RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6792⟩⟩]⟩, (1)⟩]

theorem exact72269RawTermsValid :
    exact72269RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event72269 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6792⟩⟩) exact72269RawTerms .large 72268 .exactZero (none)

def event72270 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7842⟩⟩) 0 ⟨6792⟩ 72269

def event72271 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7842⟩⟩) 1 ⟨7841⟩ 72266

def event72272 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7842⟩⟩) (.product (.predecessor 0 72270 .coefficient) (.predecessor 1 72271 .coefficient) (⟨false, false, none, none, none⟩))

def event72273 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7842⟩⟩, .operator (⟨72269, 0⟩, ⟨72266, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6792⟩⟩, ⟨.program ⟨214⟩, ⟨7840⟩⟩]⟩, (1)⟩)

def exact72274RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6792⟩⟩, ⟨.program ⟨214⟩, ⟨7840⟩⟩]⟩, (1)⟩]

theorem exact72274RawTermsValid :
    exact72274RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event72274 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7842⟩⟩) exact72274RawTerms .large 72272 .exactZero (none)

def event72275 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12269⟩⟩) 0 ⟨7842⟩ 72274

def event72276 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12269⟩⟩) 1 ⟨12268⟩ 72251

def event72277 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12269⟩⟩) (.sum [.predecessor 0 72275 .coefficient, .predecessor 1 72276 .coefficient])

def exact72278RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6792⟩⟩, ⟨.program ⟨214⟩, ⟨7840⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨11129⟩⟩, ⟨.program ⟨214⟩, ⟨12154⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact72278RawTermsValid :
    exact72278RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event72278 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12269⟩⟩) exact72278RawTerms .large 72277 .exactZero (none)

def event72279 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25294⟩⟩) 0 ⟨12269⟩ 72278

def event72280 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25294⟩⟩) 1 ⟨25291⟩ 72235

def event72281 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25294⟩⟩) (.product (.predecessor 0 72279 .coefficient) (.predecessor 1 72280 .coefficient) (⟨false, false, none, none, none⟩))

def event72282 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25294⟩⟩, .operator (⟨72278, 0⟩, ⟨72235, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6792⟩⟩, ⟨.program ⟨214⟩, ⟨7840⟩⟩, ⟨.program ⟨214⟩, ⟨25291⟩⟩]⟩, (1)⟩)

def event72283 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25294⟩⟩, .operator (⟨72278, 1⟩, ⟨72235, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨11129⟩⟩, ⟨.program ⟨214⟩, ⟨12154⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25291⟩⟩]⟩, (-1)⟩)

def event72284 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨25294⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨11129⟩⟩, ⟨.program ⟨214⟩, ⟨12154⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25291⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨25291⟩⟩) ⟨23162⟩ 72232)

def event72285 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25294⟩⟩, .relation 72284 0, ⟨[⟨.program ⟨214⟩, ⟨11129⟩⟩, ⟨.program ⟨214⟩, ⟨12154⟩⟩], [⟨.program ⟨214⟩, ⟨23162⟩⟩]⟩, (-1)⟩)

def exact72286RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6792⟩⟩, ⟨.program ⟨214⟩, ⟨7840⟩⟩, ⟨.program ⟨214⟩, ⟨25291⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨11129⟩⟩, ⟨.program ⟨214⟩, ⟨12154⟩⟩], [⟨.program ⟨214⟩, ⟨23162⟩⟩]⟩, (-1)⟩]

theorem exact72286RawTermsValid :
    exact72286RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event72286 : Event := .resultExact (⟨.program ⟨214⟩, ⟨25294⟩⟩) exact72286RawTerms .large 72281 .exactZero (none)

def event72287 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15418⟩⟩) 0 ⟨12156⟩ 72224

def event72288 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15418⟩⟩) (.authority (.programFamilyFact))

def exact72289RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15418⟩⟩], []⟩, (1)⟩]

theorem exact72289RawTermsValid :
    exact72289RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event72289 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15418⟩⟩) exact72289RawTerms (.finite 6) 72288 .exactZero (none)

def event72290 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15420⟩⟩) 0 ⟨6544⟩ 72246

def event72291 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15420⟩⟩) 1 ⟨15418⟩ 72289

def event72292 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15420⟩⟩) (.product (.predecessor 0 72290 .coefficient) (.predecessor 1 72291 .coefficient) (⟨false, true, none, none, some 1⟩))

def event72293 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨15420⟩⟩, .operator (⟨72246, 0⟩, ⟨72289, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨15418⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact72294RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15418⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact72294RawTermsValid :
    exact72294RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event72294 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15420⟩⟩) exact72294RawTerms .large 72292 .exactZero (none)

def event72295 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6693⟩⟩) 0 ⟨6689⟩ 72228

def event72296 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6693⟩⟩) (.authority (.operator))

def exact72297RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6693⟩⟩]⟩, (1)⟩]

theorem exact72297RawTermsValid :
    exact72297RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event72297 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6693⟩⟩) exact72297RawTerms .large 72296 .exactZero (none)

def event72298 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15421⟩⟩) 0 ⟨6693⟩ 72297

def event72299 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15421⟩⟩) 1 ⟨15420⟩ 72294

def event72300 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15421⟩⟩) (.sum [.predecessor 0 72298 .coefficient, .predecessor 1 72299 .coefficient])

def exact72301RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6693⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15418⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact72301RawTermsValid :
    exact72301RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event72301 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15421⟩⟩) exact72301RawTerms .large 72300 .exactZero (none)

def event72302 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25295⟩⟩) 0 ⟨15421⟩ 72301

def event72303 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25295⟩⟩) 1 ⟨25294⟩ 72286

def event72304 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25295⟩⟩) (.sum [.predecessor 0 72302 .coefficient, .predecessor 1 72303 .coefficient])

def exact72305RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6693⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6792⟩⟩, ⟨.program ⟨214⟩, ⟨7840⟩⟩, ⟨.program ⟨214⟩, ⟨25291⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨11129⟩⟩, ⟨.program ⟨214⟩, ⟨12154⟩⟩], [⟨.program ⟨214⟩, ⟨23162⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15418⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact72305RawTermsValid :
    exact72305RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event72305 : Event := .resultExact (⟨.program ⟨214⟩, ⟨25295⟩⟩) exact72305RawTerms .large 72304 .exactZero (none)

def event72306 : Event := .preFoldPolynomial 72305 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6693⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6792⟩⟩, ⟨.program ⟨214⟩, ⟨7840⟩⟩, ⟨.program ⟨214⟩, ⟨25291⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨11129⟩⟩, ⟨.program ⟨214⟩, ⟨12154⟩⟩], [⟨.program ⟨214⟩, ⟨23162⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15418⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩] .exactZero none

def exact72307RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6693⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6792⟩⟩, ⟨.program ⟨214⟩, ⟨7840⟩⟩, ⟨.program ⟨214⟩, ⟨25291⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨11129⟩⟩, ⟨.program ⟨214⟩, ⟨12154⟩⟩], [⟨.program ⟨214⟩, ⟨23162⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15418⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

def event72307 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨25295⟩⟩) 72306 exact72307RawTerms .large 72304 .exactZero (none)

def event72308 : Event := .specializationComputed (⟨.program ⟨214⟩, ⟨12156⟩⟩) ⟨⟨106⟩, ⟨10⟩, ⟨109⟩⟩ ⟨72142, 72308⟩

def event72309 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨19239⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19236⟩⟩]⟩) (1) 0 2 (.universal 72308 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19236⟩⟩]⟩) (none) 72307)

def event72310 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19239⟩⟩, .relation 72309 0, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6693⟩⟩]⟩, (1)⟩)

def event72311 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19239⟩⟩, .relation 72309 1, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6792⟩⟩, ⟨.program ⟨214⟩, ⟨7840⟩⟩, ⟨.program ⟨214⟩, ⟨25291⟩⟩]⟩, (-1)⟩)

def event72312 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19239⟩⟩, .relation 72309 2, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨11129⟩⟩, ⟨.program ⟨214⟩, ⟨12154⟩⟩], [⟨.program ⟨214⟩, ⟨23162⟩⟩]⟩, (1)⟩)

def event72313 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19239⟩⟩, .relation 72309 3, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15418⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def exact72314RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6693⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6792⟩⟩, ⟨.program ⟨214⟩, ⟨7840⟩⟩, ⟨.program ⟨214⟩, ⟨25291⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨11129⟩⟩, ⟨.program ⟨214⟩, ⟨12154⟩⟩], [⟨.program ⟨214⟩, ⟨23162⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15418⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact72314RawTermsValid :
    exact72314RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event72314 : Event := .resultExact (⟨.program ⟨214⟩, ⟨19239⟩⟩) exact72314RawTerms .large 72138 (.finite 1811303510016) (some (72140))

def event72315 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25293⟩⟩) 0 ⟨19239⟩ 72314

def event72316 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25293⟩⟩) 1 ⟨25292⟩ 72128

def event72317 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25293⟩⟩) (.sum [.predecessor 0 72315 .coefficient, .predecessor 1 72316 .coefficient])

def event72318 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25293⟩⟩, .operator (⟨72314, 2⟩, ⟨72128, 1⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨11129⟩⟩, ⟨.program ⟨214⟩, ⟨12154⟩⟩], [⟨.program ⟨214⟩, ⟨23162⟩⟩]⟩, (-1)⟩)

def event72319 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25293⟩⟩, .operator (⟨72314, 1⟩, ⟨72128, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6792⟩⟩, ⟨.program ⟨214⟩, ⟨7840⟩⟩, ⟨.program ⟨214⟩, ⟨25291⟩⟩]⟩, (1)⟩)

def event72320 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25293⟩⟩) (.sum [.result 72314 .summary, .result 72128 .summary])

def exact72321RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6693⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15418⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact72321RawTermsValid :
    exact72321RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event72321 : Event := .resultExact (⟨.program ⟨214⟩, ⟨25293⟩⟩) exact72321RawTerms .large 72317 (.finite 352024077676544) (some (72320))

def event72322 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26987⟩⟩) 0 ⟨25293⟩ 72321

def event72323 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26987⟩⟩) 1 ⟨26985⟩ 72044

def event72324 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26987⟩⟩) (.product (.predecessor 0 72322 .coefficient) (.predecessor 1 72323 .coefficient) (⟨false, false, none, none, none⟩))

def event72325 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26987⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨26985⟩⟩]⟩) [⟨.result 72044 .coefficient, false, none⟩])

def event72326 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26987⟩⟩) (.product (.result 72321 .summary) (.transfer 72325) (⟨false, false, none, none, none⟩))

def event72327 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26987⟩⟩, .operator (⟨72321, 0⟩, ⟨72044, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6693⟩⟩, ⟨.program ⟨214⟩, ⟨26985⟩⟩]⟩, (1)⟩)

def event72328 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26987⟩⟩, .operator (⟨72321, 1⟩, ⟨72044, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15418⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26985⟩⟩]⟩, (-1)⟩)

def event72329 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨26987⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15418⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26985⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨26985⟩⟩) ⟨23907⟩ 72041)

def event72330 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26987⟩⟩, .relation 72329 0, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15418⟩⟩], [⟨.program ⟨214⟩, ⟨23907⟩⟩]⟩, (-1)⟩)

def exact72331RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6693⟩⟩, ⟨.program ⟨214⟩, ⟨26985⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15418⟩⟩], [⟨.program ⟨214⟩, ⟨23907⟩⟩]⟩, (-1)⟩]

theorem exact72331RawTermsValid :
    exact72331RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event72331 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26987⟩⟩) exact72331RawTerms .large 72324 (.finite 1291933997458159304704) (some (72326))

def event72332 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20820⟩⟩) 0 ⟨15419⟩ 3425

def event72333 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20820⟩⟩) (.authority (.relationPreimageSource ⟨35⟩))

def exact72334RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨20820⟩⟩]⟩, (1)⟩]

theorem exact72334RawTermsValid :
    exact72334RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event72334 : Event := .resultExact (⟨.program ⟨214⟩, ⟨20820⟩⟩) exact72334RawTerms (.finite 136065468) 72333 .exactZero (none)

def event72335 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20822⟩⟩) 0 ⟨20820⟩ 72334

def event72336 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20822⟩⟩) 1 ⟨2348⟩ 4

def event72337 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20822⟩⟩) (.scale (.predecessor 0 72335 .coefficient) (.value (.predecessor 1 72336 .coefficient)))

def exact72338RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨20820⟩⟩]⟩, (1)⟩]

theorem exact72338RawTermsValid :
    exact72338RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event72338 : Event := .resultExact (⟨.program ⟨214⟩, ⟨20822⟩⟩) exact72338RawTerms (.finite 136065468) 72337 .exactZero (none)

def event72339 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20823⟩⟩) 0 ⟨5535⟩ 65387

def event72340 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20823⟩⟩) 1 ⟨20822⟩ 72338

def event72341 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20823⟩⟩) (.product (.predecessor 0 72339 .coefficient) (.predecessor 1 72340 .coefficient) (⟨false, false, none, none, none⟩))

def event72342 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20823⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨20820⟩⟩]⟩) [⟨.result 72334 .coefficient, false, none⟩])

def event72343 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20823⟩⟩) (.product (.result 65387 .summary) (.transfer 72342) (⟨false, false, none, none, none⟩))

def event72344 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20823⟩⟩, .operator (⟨65387, 0⟩, ⟨72338, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20820⟩⟩]⟩, (1)⟩)

def event72345 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨20821⟩⟩)

def event72346 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event72347 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event72348 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨360⟩⟩) (.authority (.operator))

def event72349 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨360⟩⟩) (.finite 2)

def event72350 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event72351 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event72352 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event72353 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event72354 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 72353

def event72355 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 72351

def event72356 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 72354 .coefficient) (.value (.predecessor 1 72355 .coefficient)))

def event72357 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event72358 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5504⟩⟩) 0 ⟨5503⟩ 72357

def event72359 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5504⟩⟩) 1 ⟨360⟩ 72349

def event72360 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5504⟩⟩) (.sum [.predecessor 0 72358 .coefficient, .predecessor 1 72359 .coefficient])

def event72361 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5504⟩⟩) (.finite 219)

def event72362 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5530⟩⟩) 0 ⟨5504⟩ 72361

def event72363 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5530⟩⟩) 1 ⟨961⟩ 72347

def event72364 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5530⟩⟩) (.identity (.predecessor 1 72363 .coefficient))

def event72365 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5530⟩⟩) (.finite 224)

def event72366 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11129⟩⟩) 0 ⟨5530⟩ 72365

def event72367 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11129⟩⟩) (.authority (.programFamilyFact))

def exact72368RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11129⟩⟩], []⟩, (1)⟩]

theorem exact72368RawTermsValid :
    exact72368RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event72368 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11129⟩⟩) exact72368RawTerms (.finite 6) 72367 .exactZero (none)

def event72369 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12154⟩⟩) 0 ⟨5530⟩ 72365

def event72370 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12154⟩⟩) (.authority (.programFamilyFact))

def exact72371RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨12154⟩⟩], []⟩, (1)⟩]

theorem exact72371RawTermsValid :
    exact72371RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event72371 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12154⟩⟩) exact72371RawTerms (.finite 6) 72370 .exactZero (none)

def event72372 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12155⟩⟩) 0 ⟨12154⟩ 72371

def event72373 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12155⟩⟩) 1 ⟨11129⟩ 72368

def event72374 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12155⟩⟩) (.product (.predecessor 0 72372 .coefficient) (.predecessor 1 72373 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event72375 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12155⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨11129⟩⟩, ⟨.program ⟨214⟩, ⟨12154⟩⟩], []⟩) [⟨.result 72371 .coefficient, true, some 1⟩, ⟨.result 72368 .coefficient, true, some 1⟩])

def event72376 : Event := .survivorFold (1) 72375

def exact72377RawTerms : List Term := []

theorem exact72377RawTermsValid :
    exact72377RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event72377 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12155⟩⟩) exact72377RawTerms (.finite 36) 72374 (.finite 36) (some (72375))

def event72378 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12156⟩⟩) 0 ⟨12155⟩ 72377

def event72379 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12156⟩⟩) (.identity (.predecessor 0 72378 .coefficient))

def event72380 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨12156⟩⟩) (.finite 36)

def event72381 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15418⟩⟩) 0 ⟨12156⟩ 72380

def event72382 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15418⟩⟩) (.authority (.programFamilyFact))

def exact72383RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15418⟩⟩], []⟩, (1)⟩]

theorem exact72383RawTermsValid :
    exact72383RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event72383 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15418⟩⟩) exact72383RawTerms (.finite 6) 72382 .exactZero (none)

def event72384 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15419⟩⟩) 0 ⟨15418⟩ 72383

def event72385 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15419⟩⟩) (.identity (.predecessor 0 72384 .coefficient))

def event72386 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨15419⟩⟩) (.finite 6)

def event72387 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20820⟩⟩) 0 ⟨15419⟩ 72386

def event72388 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20820⟩⟩) (.authority (.relationPreimageSource ⟨35⟩))

def exact72389RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨20820⟩⟩]⟩, (1)⟩]

theorem exact72389RawTermsValid :
    exact72389RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event72389 : Event := .resultExact (⟨.program ⟨214⟩, ⟨20820⟩⟩) exact72389RawTerms (.finite 136065468) 72388 .exactZero (none)

def event72390 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6⟩⟩) (.authority (.operator))

def exact72391RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩]⟩, (1)⟩]

theorem exact72391RawTermsValid :
    exact72391RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event72391 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6⟩⟩) exact72391RawTerms .large 72390 .exactZero (none)

def event72392 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20821⟩⟩) 0 ⟨6⟩ 72391

def event72393 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20821⟩⟩) 1 ⟨20820⟩ 72389

def event72394 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20821⟩⟩) (.product (.predecessor 0 72392 .coefficient) (.predecessor 1 72393 .coefficient) (⟨false, false, none, none, none⟩))

def event72395 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20821⟩⟩, .operator (⟨72391, 0⟩, ⟨72389, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20820⟩⟩]⟩, (1)⟩)

def exact72396RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20820⟩⟩]⟩, (1)⟩]

theorem exact72396RawTermsValid :
    exact72396RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event72396 : Event := .resultExact (⟨.program ⟨214⟩, ⟨20821⟩⟩) exact72396RawTerms .large 72394 .exactZero (none)

def event72397 : Event := .preFoldPolynomial 72396 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20820⟩⟩]⟩, (1)⟩] .exactZero none

def exact72398RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20820⟩⟩]⟩, (1)⟩]

def event72398 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨20821⟩⟩) 72397 exact72398RawTerms .large 72394 .exactZero (none)

def event72399 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨26990⟩⟩)

def event72400 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event72401 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event72402 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨360⟩⟩) (.authority (.operator))

def event72403 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨360⟩⟩) (.finite 2)

def event72404 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event72405 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event72406 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event72407 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event72408 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 72407

def event72409 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 72405

def event72410 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 72408 .coefficient) (.value (.predecessor 1 72409 .coefficient)))

def event72411 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event72412 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5504⟩⟩) 0 ⟨5503⟩ 72411

def event72413 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5504⟩⟩) 1 ⟨360⟩ 72403

def event72414 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5504⟩⟩) (.sum [.predecessor 0 72412 .coefficient, .predecessor 1 72413 .coefficient])

def event72415 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5504⟩⟩) (.finite 219)

def event72416 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5530⟩⟩) 0 ⟨5504⟩ 72415

def event72417 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5530⟩⟩) 1 ⟨961⟩ 72401

def event72418 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5530⟩⟩) (.identity (.predecessor 1 72417 .coefficient))

def event72419 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5530⟩⟩) (.finite 224)

def event72420 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11129⟩⟩) 0 ⟨5530⟩ 72419

def event72421 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11129⟩⟩) (.authority (.programFamilyFact))

def exact72422RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11129⟩⟩], []⟩, (1)⟩]

theorem exact72422RawTermsValid :
    exact72422RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event72422 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11129⟩⟩) exact72422RawTerms (.finite 6) 72421 .exactZero (none)

def event72423 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12154⟩⟩) 0 ⟨5530⟩ 72419

def event72424 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12154⟩⟩) (.authority (.programFamilyFact))

def exact72425RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨12154⟩⟩], []⟩, (1)⟩]

theorem exact72425RawTermsValid :
    exact72425RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event72425 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12154⟩⟩) exact72425RawTerms (.finite 6) 72424 .exactZero (none)

def event72426 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12155⟩⟩) 0 ⟨12154⟩ 72425

def event72427 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12155⟩⟩) 1 ⟨11129⟩ 72422

def event72428 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12155⟩⟩) (.product (.predecessor 0 72426 .coefficient) (.predecessor 1 72427 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event72429 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨12155⟩⟩, .operator (⟨72425, 0⟩, ⟨72422, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨11129⟩⟩, ⟨.program ⟨214⟩, ⟨12154⟩⟩], []⟩, (1)⟩)

def exact72430RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11129⟩⟩, ⟨.program ⟨214⟩, ⟨12154⟩⟩], []⟩, (1)⟩]

theorem exact72430RawTermsValid :
    exact72430RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event72430 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12155⟩⟩) exact72430RawTerms (.finite 36) 72428 .exactZero (none)

def event72431 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12156⟩⟩) 0 ⟨12155⟩ 72430

def event72432 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12156⟩⟩) (.identity (.predecessor 0 72431 .coefficient))

def event72433 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨12156⟩⟩) (.finite 36)

def event72434 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15418⟩⟩) 0 ⟨12156⟩ 72433

def event72435 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15418⟩⟩) (.authority (.programFamilyFact))

def exact72436RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15418⟩⟩], []⟩, (1)⟩]

theorem exact72436RawTermsValid :
    exact72436RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event72436 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15418⟩⟩) exact72436RawTerms (.finite 6) 72435 .exactZero (none)

def event72437 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15419⟩⟩) 0 ⟨15418⟩ 72436

def event72438 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15419⟩⟩) (.identity (.predecessor 0 72437 .coefficient))

def event72439 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨15419⟩⟩) (.finite 6)

def event72440 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23905⟩⟩) 0 ⟨15419⟩ 72439

def event72441 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23905⟩⟩) (.authority (.programFamilyFact))

def event72442 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨23905⟩⟩) (.finite 3720)

def event72443 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨6689⟩⟩) .missing

def event72444 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23907⟩⟩) 0 ⟨6689⟩ 72443

def event72445 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23907⟩⟩) 1 ⟨23905⟩ 72442

def event72446 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23907⟩⟩) (.authority (.operator))

def exact72447RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨23907⟩⟩]⟩, (1)⟩]

theorem exact72447RawTermsValid :
    exact72447RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event72447 : Event := .resultExact (⟨.program ⟨214⟩, ⟨23907⟩⟩) exact72447RawTerms .large 72446 .exactZero (none)

def eventLeaf4512 : Array AnnotatedEvent := #[
  { event := event72192
    frameStart := 72190 },
  { event := event72193
    frameStart := 72190 },
  { event := event72194
    frameStart := 72190 },
  { event := event72195
    frameStart := 72190 },
  { event := event72196
    frameStart := 72190 },
  { event := event72197
    frameStart := 72190 },
  { event := event72198
    frameStart := 72190 },
  { event := event72199
    frameStart := 72190 },
  { event := event72200
    frameStart := 72190 },
  { event := event72201
    frameStart := 72190 },
  { event := event72202
    frameStart := 72190 },
  { event := event72203
    frameStart := 72190 },
  { event := event72204
    frameStart := 72190 },
  { event := event72205
    frameStart := 72190 },
  { event := event72206
    frameStart := 72190 },
  { event := event72207
    frameStart := 72190 }
]

def eventLeaf4513 : Array AnnotatedEvent := #[
  { event := event72208
    frameStart := 72190 },
  { event := event72209
    frameStart := 72190 },
  { event := event72210
    frameStart := 72190 },
  { event := event72211
    frameStart := 72190 },
  { event := event72212
    frameStart := 72190 },
  { event := event72213
    frameStart := 72190 },
  { event := event72214
    frameStart := 72190 },
  { event := event72215
    frameStart := 72190 },
  { event := event72216
    frameStart := 72190 },
  { event := event72217
    frameStart := 72190 },
  { event := event72218
    frameStart := 72190 },
  { event := event72219
    frameStart := 72190 },
  { event := event72220
    frameStart := 72190 },
  { event := event72221
    frameStart := 72190 },
  { event := event72222
    frameStart := 72190 },
  { event := event72223
    frameStart := 72190 }
]

def eventLeaf4514 : Array AnnotatedEvent := #[
  { event := event72224
    frameStart := 72190 },
  { event := event72225
    frameStart := 72190 },
  { event := event72226
    frameStart := 72190 },
  { event := event72227
    frameStart := 72190 },
  { event := event72228
    frameStart := 72190 },
  { event := event72229
    frameStart := 72190 },
  { event := event72230
    frameStart := 72190 },
  { event := event72231
    frameStart := 72190 },
  { event := event72232
    frameStart := 72190 },
  { event := event72233
    frameStart := 72190 },
  { event := event72234
    frameStart := 72190 },
  { event := event72235
    frameStart := 72190 },
  { event := event72236
    frameStart := 72190 },
  { event := event72237
    frameStart := 72190 },
  { event := event72238
    frameStart := 72190 },
  { event := event72239
    frameStart := 72190 }
]

def eventLeaf4515 : Array AnnotatedEvent := #[
  { event := event72240
    frameStart := 72190 },
  { event := event72241
    frameStart := 72190 },
  { event := event72242
    frameStart := 72190 },
  { event := event72243
    frameStart := 72190 },
  { event := event72244
    frameStart := 72190 },
  { event := event72245
    frameStart := 72190 },
  { event := event72246
    frameStart := 72190 },
  { event := event72247
    frameStart := 72190 },
  { event := event72248
    frameStart := 72190 },
  { event := event72249
    frameStart := 72190 },
  { event := event72250
    frameStart := 72190 },
  { event := event72251
    frameStart := 72190 },
  { event := event72252
    frameStart := 72190 },
  { event := event72253
    frameStart := 72190 },
  { event := event72254
    frameStart := 72190 },
  { event := event72255
    frameStart := 72190 }
]

def eventLeaf4516 : Array AnnotatedEvent := #[
  { event := event72256
    frameStart := 72190 },
  { event := event72257
    frameStart := 72190 },
  { event := event72258
    frameStart := 72190 },
  { event := event72259
    frameStart := 72190 },
  { event := event72260
    frameStart := 72190 },
  { event := event72261
    frameStart := 72190 },
  { event := event72262
    frameStart := 72190 },
  { event := event72263
    frameStart := 72190 },
  { event := event72264
    frameStart := 72190 },
  { event := event72265
    frameStart := 72190 },
  { event := event72266
    frameStart := 72190 },
  { event := event72267
    frameStart := 72190 },
  { event := event72268
    frameStart := 72190 },
  { event := event72269
    frameStart := 72190 },
  { event := event72270
    frameStart := 72190 },
  { event := event72271
    frameStart := 72190 }
]

def eventLeaf4517 : Array AnnotatedEvent := #[
  { event := event72272
    frameStart := 72190 },
  { event := event72273
    frameStart := 72190 },
  { event := event72274
    frameStart := 72190 },
  { event := event72275
    frameStart := 72190 },
  { event := event72276
    frameStart := 72190 },
  { event := event72277
    frameStart := 72190 },
  { event := event72278
    frameStart := 72190 },
  { event := event72279
    frameStart := 72190 },
  { event := event72280
    frameStart := 72190 },
  { event := event72281
    frameStart := 72190 },
  { event := event72282
    frameStart := 72190 },
  { event := event72283
    frameStart := 72190 },
  { event := event72284
    frameStart := 72190 },
  { event := event72285
    frameStart := 72190 },
  { event := event72286
    frameStart := 72190 },
  { event := event72287
    frameStart := 72190 }
]

def eventLeaf4518 : Array AnnotatedEvent := #[
  { event := event72288
    frameStart := 72190 },
  { event := event72289
    frameStart := 72190 },
  { event := event72290
    frameStart := 72190 },
  { event := event72291
    frameStart := 72190 },
  { event := event72292
    frameStart := 72190 },
  { event := event72293
    frameStart := 72190 },
  { event := event72294
    frameStart := 72190 },
  { event := event72295
    frameStart := 72190 },
  { event := event72296
    frameStart := 72190 },
  { event := event72297
    frameStart := 72190 },
  { event := event72298
    frameStart := 72190 },
  { event := event72299
    frameStart := 72190 },
  { event := event72300
    frameStart := 72190 },
  { event := event72301
    frameStart := 72190 },
  { event := event72302
    frameStart := 72190 },
  { event := event72303
    frameStart := 72190 }
]

def eventLeaf4519 : Array AnnotatedEvent := #[
  { event := event72304
    frameStart := 72190 },
  { event := event72305
    frameStart := 72190 },
  { event := event72306
    frameStart := 72190 },
  { event := event72307
    frameStart := 72190 },
  { event := event72308
    frameStart := 0 },
  { event := event72309
    frameStart := 0 },
  { event := event72310
    frameStart := 0 },
  { event := event72311
    frameStart := 0 },
  { event := event72312
    frameStart := 0 },
  { event := event72313
    frameStart := 0 },
  { event := event72314
    frameStart := 0 },
  { event := event72315
    frameStart := 0 },
  { event := event72316
    frameStart := 0 },
  { event := event72317
    frameStart := 0 },
  { event := event72318
    frameStart := 0 },
  { event := event72319
    frameStart := 0 }
]

def eventLeaf4520 : Array AnnotatedEvent := #[
  { event := event72320
    frameStart := 0 },
  { event := event72321
    frameStart := 0 },
  { event := event72322
    frameStart := 0 },
  { event := event72323
    frameStart := 0 },
  { event := event72324
    frameStart := 0 },
  { event := event72325
    frameStart := 0 },
  { event := event72326
    frameStart := 0 },
  { event := event72327
    frameStart := 0 },
  { event := event72328
    frameStart := 0 },
  { event := event72329
    frameStart := 0 },
  { event := event72330
    frameStart := 0 },
  { event := event72331
    frameStart := 0 },
  { event := event72332
    frameStart := 0 },
  { event := event72333
    frameStart := 0 },
  { event := event72334
    frameStart := 0 },
  { event := event72335
    frameStart := 0 }
]

def eventLeaf4521 : Array AnnotatedEvent := #[
  { event := event72336
    frameStart := 0 },
  { event := event72337
    frameStart := 0 },
  { event := event72338
    frameStart := 0 },
  { event := event72339
    frameStart := 0 },
  { event := event72340
    frameStart := 0 },
  { event := event72341
    frameStart := 0 },
  { event := event72342
    frameStart := 0 },
  { event := event72343
    frameStart := 0 },
  { event := event72344
    frameStart := 0 },
  { event := event72345
    frameStart := 72345 },
  { event := event72346
    frameStart := 72345 },
  { event := event72347
    frameStart := 72345 },
  { event := event72348
    frameStart := 72345 },
  { event := event72349
    frameStart := 72345 },
  { event := event72350
    frameStart := 72345 },
  { event := event72351
    frameStart := 72345 }
]

def eventLeaf4522 : Array AnnotatedEvent := #[
  { event := event72352
    frameStart := 72345 },
  { event := event72353
    frameStart := 72345 },
  { event := event72354
    frameStart := 72345 },
  { event := event72355
    frameStart := 72345 },
  { event := event72356
    frameStart := 72345 },
  { event := event72357
    frameStart := 72345 },
  { event := event72358
    frameStart := 72345 },
  { event := event72359
    frameStart := 72345 },
  { event := event72360
    frameStart := 72345 },
  { event := event72361
    frameStart := 72345 },
  { event := event72362
    frameStart := 72345 },
  { event := event72363
    frameStart := 72345 },
  { event := event72364
    frameStart := 72345 },
  { event := event72365
    frameStart := 72345 },
  { event := event72366
    frameStart := 72345 },
  { event := event72367
    frameStart := 72345 }
]

def eventLeaf4523 : Array AnnotatedEvent := #[
  { event := event72368
    frameStart := 72345 },
  { event := event72369
    frameStart := 72345 },
  { event := event72370
    frameStart := 72345 },
  { event := event72371
    frameStart := 72345 },
  { event := event72372
    frameStart := 72345 },
  { event := event72373
    frameStart := 72345 },
  { event := event72374
    frameStart := 72345 },
  { event := event72375
    frameStart := 72345 },
  { event := event72376
    frameStart := 72345 },
  { event := event72377
    frameStart := 72345 },
  { event := event72378
    frameStart := 72345 },
  { event := event72379
    frameStart := 72345 },
  { event := event72380
    frameStart := 72345 },
  { event := event72381
    frameStart := 72345 },
  { event := event72382
    frameStart := 72345 },
  { event := event72383
    frameStart := 72345 }
]

def eventLeaf4524 : Array AnnotatedEvent := #[
  { event := event72384
    frameStart := 72345 },
  { event := event72385
    frameStart := 72345 },
  { event := event72386
    frameStart := 72345 },
  { event := event72387
    frameStart := 72345 },
  { event := event72388
    frameStart := 72345 },
  { event := event72389
    frameStart := 72345 },
  { event := event72390
    frameStart := 72345 },
  { event := event72391
    frameStart := 72345 },
  { event := event72392
    frameStart := 72345 },
  { event := event72393
    frameStart := 72345 },
  { event := event72394
    frameStart := 72345 },
  { event := event72395
    frameStart := 72345 },
  { event := event72396
    frameStart := 72345 },
  { event := event72397
    frameStart := 72345 },
  { event := event72398
    frameStart := 72345 },
  { event := event72399
    frameStart := 72399 }
]

def eventLeaf4525 : Array AnnotatedEvent := #[
  { event := event72400
    frameStart := 72399 },
  { event := event72401
    frameStart := 72399 },
  { event := event72402
    frameStart := 72399 },
  { event := event72403
    frameStart := 72399 },
  { event := event72404
    frameStart := 72399 },
  { event := event72405
    frameStart := 72399 },
  { event := event72406
    frameStart := 72399 },
  { event := event72407
    frameStart := 72399 },
  { event := event72408
    frameStart := 72399 },
  { event := event72409
    frameStart := 72399 },
  { event := event72410
    frameStart := 72399 },
  { event := event72411
    frameStart := 72399 },
  { event := event72412
    frameStart := 72399 },
  { event := event72413
    frameStart := 72399 },
  { event := event72414
    frameStart := 72399 },
  { event := event72415
    frameStart := 72399 }
]

def eventLeaf4526 : Array AnnotatedEvent := #[
  { event := event72416
    frameStart := 72399 },
  { event := event72417
    frameStart := 72399 },
  { event := event72418
    frameStart := 72399 },
  { event := event72419
    frameStart := 72399 },
  { event := event72420
    frameStart := 72399 },
  { event := event72421
    frameStart := 72399 },
  { event := event72422
    frameStart := 72399 },
  { event := event72423
    frameStart := 72399 },
  { event := event72424
    frameStart := 72399 },
  { event := event72425
    frameStart := 72399 },
  { event := event72426
    frameStart := 72399 },
  { event := event72427
    frameStart := 72399 },
  { event := event72428
    frameStart := 72399 },
  { event := event72429
    frameStart := 72399 },
  { event := event72430
    frameStart := 72399 },
  { event := event72431
    frameStart := 72399 }
]

def eventLeaf4527 : Array AnnotatedEvent := #[
  { event := event72432
    frameStart := 72399 },
  { event := event72433
    frameStart := 72399 },
  { event := event72434
    frameStart := 72399 },
  { event := event72435
    frameStart := 72399 },
  { event := event72436
    frameStart := 72399 },
  { event := event72437
    frameStart := 72399 },
  { event := event72438
    frameStart := 72399 },
  { event := event72439
    frameStart := 72399 },
  { event := event72440
    frameStart := 72399 },
  { event := event72441
    frameStart := 72399 },
  { event := event72442
    frameStart := 72399 },
  { event := event72443
    frameStart := 72399 },
  { event := event72444
    frameStart := 72399 },
  { event := event72445
    frameStart := 72399 },
  { event := event72446
    frameStart := 72399 },
  { event := event72447
    frameStart := 72399 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events282
