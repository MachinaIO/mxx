import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events110

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event28160 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event28161 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 28160

def event28162 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 28158

def event28163 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 28161 .coefficient) (.value (.predecessor 1 28162 .coefficient)))

def event28164 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event28165 : Event := .predecessor (⟨.program ⟨257⟩, ⟨393⟩⟩) 0 ⟨392⟩ 28164

def event28166 : Event := .predecessor (⟨.program ⟨257⟩, ⟨393⟩⟩) 1 ⟨140⟩ 28156

def event28167 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨393⟩⟩) (.sum [.predecessor 0 28165 .coefficient, .predecessor 1 28166 .coefficient])

def event28168 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨393⟩⟩) (.finite 655359)

def event28169 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5439⟩⟩) 0 ⟨393⟩ 28168

def event28170 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5439⟩⟩) 1 ⟨5426⟩ 28154

def event28171 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5439⟩⟩) (.identity (.predecessor 1 28170 .coefficient))

def event28172 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5439⟩⟩) (.finite 655360)

def event28173 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42266⟩⟩) 0 ⟨5439⟩ 28172

def event28174 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42266⟩⟩) (.authority (.programFamilyFact))

def exact28175RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨42266⟩⟩], []⟩, (1)⟩]

theorem exact28175RawTermsValid :
    exact28175RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28175 : Event := .resultExact (⟨.program ⟨257⟩, ⟨42266⟩⟩) exact28175RawTerms (.finite 52) 28174 .exactZero (none)

def event28176 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14351⟩⟩) 0 ⟨5439⟩ 28172

def event28177 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14351⟩⟩) (.authority (.programFamilyFact))

def exact28178RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14351⟩⟩], []⟩, (1)⟩]

theorem exact28178RawTermsValid :
    exact28178RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28178 : Event := .resultExact (⟨.program ⟨257⟩, ⟨14351⟩⟩) exact28178RawTerms (.finite 52) 28177 .exactZero (none)

def event28179 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42267⟩⟩) 0 ⟨14351⟩ 28178

def event28180 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42267⟩⟩) 1 ⟨42266⟩ 28175

def event28181 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42267⟩⟩) (.product (.predecessor 0 28179 .coefficient) (.predecessor 1 28180 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event28182 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42267⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨14351⟩⟩, ⟨.program ⟨257⟩, ⟨42266⟩⟩], []⟩) [⟨.result 28178 .coefficient, true, some 1⟩, ⟨.result 28175 .coefficient, true, some 1⟩])

def event28183 : Event := .survivorFold (1) 28182

def exact28184RawTerms : List Term := []

theorem exact28184RawTermsValid :
    exact28184RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28184 : Event := .resultExact (⟨.program ⟨257⟩, ⟨42267⟩⟩) exact28184RawTerms (.finite 2704) 28181 (.finite 2704) (some (28182))

def event28185 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42268⟩⟩) 0 ⟨42267⟩ 28184

def event28186 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42268⟩⟩) (.identity (.predecessor 0 28185 .coefficient))

def event28187 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨42268⟩⟩) (.finite 2704)

def event28188 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42718⟩⟩) 0 ⟨42268⟩ 28187

def event28189 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42718⟩⟩) (.authority (.programFamilyFact))

def exact28190RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨42718⟩⟩], []⟩, (1)⟩]

theorem exact28190RawTermsValid :
    exact28190RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28190 : Event := .resultExact (⟨.program ⟨257⟩, ⟨42718⟩⟩) exact28190RawTerms (.finite 52) 28189 .exactZero (none)

def event28191 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42719⟩⟩) 0 ⟨42718⟩ 28190

def event28192 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42719⟩⟩) (.identity (.predecessor 0 28191 .coefficient))

def event28193 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨42719⟩⟩) (.finite 52)

def event28194 : Event := .predecessor (⟨.program ⟨257⟩, ⟨43358⟩⟩) 0 ⟨42719⟩ 28193

def event28195 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨43358⟩⟩) (.authority (.relationPreimageSource ⟨89⟩))

def exact28196RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨43358⟩⟩]⟩, (1)⟩]

theorem exact28196RawTermsValid :
    exact28196RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28196 : Event := .resultExact (⟨.program ⟨257⟩, ⟨43358⟩⟩) exact28196RawTerms (.finite 5647228698) 28195 .exactZero (none)

def event28197 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35⟩⟩) (.authority (.operator))

def exact28198RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩]⟩, (1)⟩]

theorem exact28198RawTermsValid :
    exact28198RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28198 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35⟩⟩) exact28198RawTerms .large 28197 .exactZero (none)

def event28199 : Event := .predecessor (⟨.program ⟨257⟩, ⟨43359⟩⟩) 0 ⟨35⟩ 28198

def event28200 : Event := .predecessor (⟨.program ⟨257⟩, ⟨43359⟩⟩) 1 ⟨43358⟩ 28196

def event28201 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨43359⟩⟩) (.product (.predecessor 0 28199 .coefficient) (.predecessor 1 28200 .coefficient) (⟨false, false, none, none, none⟩))

def event28202 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨43359⟩⟩, .operator (⟨28198, 0⟩, ⟨28196, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43358⟩⟩]⟩, (1)⟩)

def exact28203RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43358⟩⟩]⟩, (1)⟩]

theorem exact28203RawTermsValid :
    exact28203RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28203 : Event := .resultExact (⟨.program ⟨257⟩, ⟨43359⟩⟩) exact28203RawTerms .large 28201 .exactZero (none)

def event28204 : Event := .preFoldPolynomial 28203 [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43358⟩⟩]⟩, (1)⟩] .exactZero none

def exact28205RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43358⟩⟩]⟩, (1)⟩]

def event28205 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨43359⟩⟩) 28204 exact28205RawTerms .large 28201 .exactZero (none)

def event28206 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨44450⟩⟩)

def event28207 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event28208 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event28209 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨140⟩⟩) (.authority (.operator))

def event28210 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨140⟩⟩) (.finite 19)

def event28211 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event28212 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event28213 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event28214 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event28215 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 28214

def event28216 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 28212

def event28217 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 28215 .coefficient) (.value (.predecessor 1 28216 .coefficient)))

def event28218 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event28219 : Event := .predecessor (⟨.program ⟨257⟩, ⟨393⟩⟩) 0 ⟨392⟩ 28218

def event28220 : Event := .predecessor (⟨.program ⟨257⟩, ⟨393⟩⟩) 1 ⟨140⟩ 28210

def event28221 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨393⟩⟩) (.sum [.predecessor 0 28219 .coefficient, .predecessor 1 28220 .coefficient])

def event28222 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨393⟩⟩) (.finite 655359)

def event28223 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5439⟩⟩) 0 ⟨393⟩ 28222

def event28224 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5439⟩⟩) 1 ⟨5426⟩ 28208

def event28225 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5439⟩⟩) (.identity (.predecessor 1 28224 .coefficient))

def event28226 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5439⟩⟩) (.finite 655360)

def event28227 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42266⟩⟩) 0 ⟨5439⟩ 28226

def event28228 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42266⟩⟩) (.authority (.programFamilyFact))

def exact28229RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨42266⟩⟩], []⟩, (1)⟩]

theorem exact28229RawTermsValid :
    exact28229RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28229 : Event := .resultExact (⟨.program ⟨257⟩, ⟨42266⟩⟩) exact28229RawTerms (.finite 52) 28228 .exactZero (none)

def event28230 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14351⟩⟩) 0 ⟨5439⟩ 28226

def event28231 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14351⟩⟩) (.authority (.programFamilyFact))

def exact28232RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14351⟩⟩], []⟩, (1)⟩]

theorem exact28232RawTermsValid :
    exact28232RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28232 : Event := .resultExact (⟨.program ⟨257⟩, ⟨14351⟩⟩) exact28232RawTerms (.finite 52) 28231 .exactZero (none)

def event28233 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42267⟩⟩) 0 ⟨14351⟩ 28232

def event28234 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42267⟩⟩) 1 ⟨42266⟩ 28229

def event28235 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42267⟩⟩) (.product (.predecessor 0 28233 .coefficient) (.predecessor 1 28234 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event28236 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨42267⟩⟩, .operator (⟨28232, 0⟩, ⟨28229, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨14351⟩⟩, ⟨.program ⟨257⟩, ⟨42266⟩⟩], []⟩, (1)⟩)

def exact28237RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14351⟩⟩, ⟨.program ⟨257⟩, ⟨42266⟩⟩], []⟩, (1)⟩]

theorem exact28237RawTermsValid :
    exact28237RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28237 : Event := .resultExact (⟨.program ⟨257⟩, ⟨42267⟩⟩) exact28237RawTerms (.finite 2704) 28235 .exactZero (none)

def event28238 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42268⟩⟩) 0 ⟨42267⟩ 28237

def event28239 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42268⟩⟩) (.identity (.predecessor 0 28238 .coefficient))

def event28240 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨42268⟩⟩) (.finite 2704)

def event28241 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42718⟩⟩) 0 ⟨42268⟩ 28240

def event28242 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42718⟩⟩) (.authority (.programFamilyFact))

def exact28243RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨42718⟩⟩], []⟩, (1)⟩]

theorem exact28243RawTermsValid :
    exact28243RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28243 : Event := .resultExact (⟨.program ⟨257⟩, ⟨42718⟩⟩) exact28243RawTerms (.finite 52) 28242 .exactZero (none)

def event28244 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42719⟩⟩) 0 ⟨42718⟩ 28243

def event28245 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42719⟩⟩) (.identity (.predecessor 0 28244 .coefficient))

def event28246 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨42719⟩⟩) (.finite 52)

def event28247 : Event := .predecessor (⟨.program ⟨257⟩, ⟨43861⟩⟩) 0 ⟨42719⟩ 28246

def event28248 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨43861⟩⟩) (.authority (.programFamilyFact))

def event28249 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨43861⟩⟩) (.finite 3720)

def event28250 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨7177⟩⟩) .missing

def event28251 : Event := .predecessor (⟨.program ⟨257⟩, ⟨43862⟩⟩) 0 ⟨7177⟩ 28250

def event28252 : Event := .predecessor (⟨.program ⟨257⟩, ⟨43862⟩⟩) 1 ⟨43861⟩ 28249

def event28253 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨43862⟩⟩) (.authority (.operator))

def exact28254RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨43862⟩⟩]⟩, (1)⟩]

theorem exact28254RawTermsValid :
    exact28254RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28254 : Event := .resultExact (⟨.program ⟨257⟩, ⟨43862⟩⟩) exact28254RawTerms .large 28253 .exactZero (none)

def event28255 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44445⟩⟩) 0 ⟨43862⟩ 28254

def event28256 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44445⟩⟩) (.authority (.operator))

def exact28257RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨44445⟩⟩]⟩, (1)⟩]

theorem exact28257RawTermsValid :
    exact28257RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28257 : Event := .resultExact (⟨.program ⟨257⟩, ⟨44445⟩⟩) exact28257RawTerms (.finite 8192) 28256 .exactZero (none)

def event28258 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨136⟩⟩) (.authority (.operator))

def event28259 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨136⟩⟩) .exactZero

def event28260 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44110⟩⟩) 0 ⟨42719⟩ 28246

def event28261 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44110⟩⟩) 1 ⟨136⟩ 28259

def event28262 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44110⟩⟩) (.sum [.predecessor 0 28260 .coefficient, .predecessor 1 28261 .coefficient])

def event28263 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨44110⟩⟩) (.finite 52)

def event28264 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44111⟩⟩) 0 ⟨44110⟩ 28263

def event28265 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44111⟩⟩) (.identity (.predecessor 0 28264 .coefficient))

def exact28266RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨42718⟩⟩], []⟩, (1)⟩]

theorem exact28266RawTermsValid :
    exact28266RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28266 : Event := .resultExact (⟨.program ⟨257⟩, ⟨44111⟩⟩) exact28266RawTerms (.finite 52) 28265 .exactZero (none)

def event28267 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6908⟩⟩) (.authority (.factStore))

def exact28268RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact28268RawTermsValid :
    exact28268RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28268 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6908⟩⟩) exact28268RawTerms .large 28267 .exactZero (none)

def event28269 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44112⟩⟩) 0 ⟨6908⟩ 28268

def event28270 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44112⟩⟩) 1 ⟨44111⟩ 28266

def event28271 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44112⟩⟩) (.product (.predecessor 0 28269 .coefficient) (.predecessor 1 28270 .coefficient) (⟨false, false, none, none, none⟩))

def event28272 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨44112⟩⟩, .operator (⟨28268, 0⟩, ⟨28266, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨42718⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact28273RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨42718⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact28273RawTermsValid :
    exact28273RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28273 : Event := .resultExact (⟨.program ⟨257⟩, ⟨44112⟩⟩) exact28273RawTerms .large 28271 .exactZero (none)

def event28274 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7194⟩⟩) 0 ⟨7177⟩ 28250

def event28275 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7194⟩⟩) (.authority (.operator))

def exact28276RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7194⟩⟩]⟩, (1)⟩]

theorem exact28276RawTermsValid :
    exact28276RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28276 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7194⟩⟩) exact28276RawTerms .large 28275 .exactZero (none)

def event28277 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44113⟩⟩) 0 ⟨7194⟩ 28276

def event28278 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44113⟩⟩) 1 ⟨44112⟩ 28273

def event28279 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44113⟩⟩) (.sum [.predecessor 0 28277 .coefficient, .predecessor 1 28278 .coefficient])

def exact28280RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7194⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨42718⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact28280RawTermsValid :
    exact28280RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28280 : Event := .resultExact (⟨.program ⟨257⟩, ⟨44113⟩⟩) exact28280RawTerms .large 28279 .exactZero (none)

def event28281 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44446⟩⟩) 0 ⟨44113⟩ 28280

def event28282 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44446⟩⟩) 1 ⟨44445⟩ 28257

def event28283 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44446⟩⟩) (.product (.predecessor 0 28281 .coefficient) (.predecessor 1 28282 .coefficient) (⟨false, false, none, none, none⟩))

def event28284 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨44446⟩⟩, .operator (⟨28280, 1⟩, ⟨28257, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨42718⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨44445⟩⟩]⟩, (-1)⟩)

def event28285 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨44446⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨42718⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨44445⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨44445⟩⟩) ⟨43862⟩ 28254)

def event28286 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨44446⟩⟩, .relation 28285 0, ⟨[⟨.program ⟨257⟩, ⟨42718⟩⟩], [⟨.program ⟨257⟩, ⟨43862⟩⟩]⟩, (-1)⟩)

def event28287 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨44446⟩⟩, .operator (⟨28280, 0⟩, ⟨28257, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7194⟩⟩, ⟨.program ⟨257⟩, ⟨44445⟩⟩]⟩, (1)⟩)

def exact28288RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7194⟩⟩, ⟨.program ⟨257⟩, ⟨44445⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨42718⟩⟩], [⟨.program ⟨257⟩, ⟨43862⟩⟩]⟩, (-1)⟩]

theorem exact28288RawTermsValid :
    exact28288RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28288 : Event := .resultExact (⟨.program ⟨257⟩, ⟨44446⟩⟩) exact28288RawTerms .large 28283 .exactZero (none)

def event28289 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42888⟩⟩) 0 ⟨42719⟩ 28246

def event28290 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42888⟩⟩) (.authority (.programFamilyFact))

def exact28291RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨42888⟩⟩], []⟩, (1)⟩]

theorem exact28291RawTermsValid :
    exact28291RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28291 : Event := .resultExact (⟨.program ⟨257⟩, ⟨42888⟩⟩) exact28291RawTerms (.finite 52) 28290 .exactZero (none)

def event28292 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42890⟩⟩) 0 ⟨6908⟩ 28268

def event28293 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42890⟩⟩) 1 ⟨42888⟩ 28291

def event28294 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42890⟩⟩) (.product (.predecessor 0 28292 .coefficient) (.predecessor 1 28293 .coefficient) (⟨false, true, none, none, some 1⟩))

def event28295 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨42890⟩⟩, .operator (⟨28268, 0⟩, ⟨28291, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨42888⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact28296RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨42888⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact28296RawTermsValid :
    exact28296RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28296 : Event := .resultExact (⟨.program ⟨257⟩, ⟨42890⟩⟩) exact28296RawTerms .large 28294 .exactZero (none)

def event28297 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7227⟩⟩) 0 ⟨7177⟩ 28250

def event28298 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7227⟩⟩) (.authority (.operator))

def exact28299RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7227⟩⟩]⟩, (1)⟩]

theorem exact28299RawTermsValid :
    exact28299RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28299 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7227⟩⟩) exact28299RawTerms .large 28298 .exactZero (none)

def event28300 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42891⟩⟩) 0 ⟨7227⟩ 28299

def event28301 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42891⟩⟩) 1 ⟨42890⟩ 28296

def event28302 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42891⟩⟩) (.sum [.predecessor 0 28300 .coefficient, .predecessor 1 28301 .coefficient])

def exact28303RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7227⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨42888⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact28303RawTermsValid :
    exact28303RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28303 : Event := .resultExact (⟨.program ⟨257⟩, ⟨42891⟩⟩) exact28303RawTerms .large 28302 .exactZero (none)

def event28304 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44450⟩⟩) 0 ⟨42891⟩ 28303

def event28305 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44450⟩⟩) 1 ⟨44446⟩ 28288

def event28306 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44450⟩⟩) (.sum [.predecessor 0 28304 .coefficient, .predecessor 1 28305 .coefficient])

def exact28307RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7194⟩⟩, ⟨.program ⟨257⟩, ⟨44445⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7227⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨42718⟩⟩], [⟨.program ⟨257⟩, ⟨43862⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨42888⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact28307RawTermsValid :
    exact28307RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28307 : Event := .resultExact (⟨.program ⟨257⟩, ⟨44450⟩⟩) exact28307RawTerms .large 28306 .exactZero (none)

def event28308 : Event := .preFoldPolynomial 28307 [⟨⟨[], [⟨.program ⟨257⟩, ⟨7194⟩⟩, ⟨.program ⟨257⟩, ⟨44445⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7227⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨42718⟩⟩], [⟨.program ⟨257⟩, ⟨43862⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨42888⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩] .exactZero none

def exact28309RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7194⟩⟩, ⟨.program ⟨257⟩, ⟨44445⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7227⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨42718⟩⟩], [⟨.program ⟨257⟩, ⟨43862⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨42888⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

def event28309 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨44450⟩⟩) 28308 exact28309RawTerms .large 28306 .exactZero (none)

def event28310 : Event := .specializationComputed (⟨.program ⟨257⟩, ⟨42719⟩⟩) ⟨⟨106⟩, ⟨89⟩, ⟨135⟩⟩ ⟨28152, 28310⟩

def event28311 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨43361⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43358⟩⟩]⟩) (1) 0 2 (.universal 28310 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43358⟩⟩]⟩) (none) 28309)

def event28312 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨43361⟩⟩, .relation 28311 1, ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7227⟩⟩]⟩, (1)⟩)

def event28313 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨43361⟩⟩, .relation 28311 2, ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨42718⟩⟩], [⟨.program ⟨257⟩, ⟨43862⟩⟩]⟩, (1)⟩)

def event28314 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨43361⟩⟩, .relation 28311 0, ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7194⟩⟩, ⟨.program ⟨257⟩, ⟨44445⟩⟩]⟩, (-1)⟩)

def event28315 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨43361⟩⟩, .relation 28311 3, ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨42888⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact28316RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7194⟩⟩, ⟨.program ⟨257⟩, ⟨44445⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7227⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨42718⟩⟩], [⟨.program ⟨257⟩, ⟨43862⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨42888⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact28316RawTermsValid :
    exact28316RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28316 : Event := .resultExact (⟨.program ⟨257⟩, ⟨43361⟩⟩) exact28316RawTerms .large 28148 (.finite 202072841853861888) (some (28150))

def event28317 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44448⟩⟩) 0 ⟨43361⟩ 28316

def event28318 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44448⟩⟩) 1 ⟨44447⟩ 28138

def event28319 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44448⟩⟩) (.sum [.predecessor 0 28317 .coefficient, .predecessor 1 28318 .coefficient])

def event28320 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨44448⟩⟩, .operator (⟨28316, 2⟩, ⟨28138, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨42718⟩⟩], [⟨.program ⟨257⟩, ⟨43862⟩⟩]⟩, (-1)⟩)

def event28321 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨44448⟩⟩, .operator (⟨28316, 0⟩, ⟨28138, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7194⟩⟩, ⟨.program ⟨257⟩, ⟨44445⟩⟩]⟩, (1)⟩)

def event28322 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44448⟩⟩) (.sum [.result 28316 .summary, .result 28138 .summary])

def exact28323RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7227⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨42888⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact28323RawTermsValid :
    exact28323RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28323 : Event := .resultExact (⟨.program ⟨257⟩, ⟨44448⟩⟩) exact28323RawTerms .large 28319 (.finite 32193718473625891320532869316608) (some (28322))

def event28324 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44449⟩⟩) 0 ⟨44448⟩ 28323

def event28325 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44449⟩⟩) 1 ⟨7154⟩ 15582

def event28326 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44449⟩⟩) (.product (.predecessor 0 28324 .coefficient) (.predecessor 1 28325 .coefficient) (⟨false, false, none, none, none⟩))

def event28327 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44449⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨7153⟩⟩]⟩) [⟨.result 15578 .coefficient, false, none⟩])

def event28328 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44449⟩⟩) (.product (.result 28323 .summary) (.transfer 28327) (⟨false, false, none, none, none⟩))

def event28329 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨44449⟩⟩, .operator (⟨28323, 0⟩, ⟨15582, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7227⟩⟩, ⟨.program ⟨257⟩, ⟨7153⟩⟩]⟩, (1)⟩)

def event28330 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨44449⟩⟩, .operator (⟨28323, 1⟩, ⟨15582, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨42888⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7153⟩⟩]⟩, (-1)⟩)

def event28331 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨44449⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨42888⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7153⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨7153⟩⟩) ⟨7042⟩ 15575)

def event28332 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨44449⟩⟩, .relation 28331 0, ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨42888⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact28333RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7227⟩⟩, ⟨.program ⟨257⟩, ⟨7153⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨42888⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact28333RawTermsValid :
    exact28333RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28333 : Event := .resultExact (⟨.program ⟨257⟩, ⟨44449⟩⟩) exact28333RawTerms .large 28326 (.finite 345677419952135604401347317519683074129920) (some (28328))

def event28334 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41182⟩⟩) 0 ⟨7177⟩ 15500

def event28335 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41182⟩⟩) 1 ⟨41181⟩ 18555

def event28336 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41182⟩⟩) (.authority (.operator))

def exact28337RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨41182⟩⟩]⟩, (1)⟩]

theorem exact28337RawTermsValid :
    exact28337RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28337 : Event := .resultExact (⟨.program ⟨257⟩, ⟨41182⟩⟩) exact28337RawTerms .large 28336 .exactZero (none)

def event28338 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41765⟩⟩) 0 ⟨41182⟩ 28337

def event28339 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41765⟩⟩) (.authority (.operator))

def exact28340RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨41765⟩⟩]⟩, (1)⟩]

theorem exact28340RawTermsValid :
    exact28340RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28340 : Event := .resultExact (⟨.program ⟨257⟩, ⟨41765⟩⟩) exact28340RawTerms (.finite 8192) 28339 .exactZero (none)

def event28341 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41767⟩⟩) 0 ⟨41525⟩ 18858

def event28342 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41767⟩⟩) 1 ⟨41765⟩ 28340

def event28343 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41767⟩⟩) (.product (.predecessor 0 28341 .coefficient) (.predecessor 1 28342 .coefficient) (⟨false, false, none, none, none⟩))

def event28344 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41767⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨41765⟩⟩]⟩) [⟨.result 28340 .coefficient, false, none⟩])

def event28345 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41767⟩⟩) (.product (.result 18858 .summary) (.transfer 28344) (⟨false, false, none, none, none⟩))

def event28346 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨41767⟩⟩, .operator (⟨18858, 1⟩, ⟨28340, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨40038⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨41765⟩⟩]⟩, (-1)⟩)

def event28347 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨41767⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨40038⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨41765⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨41765⟩⟩) ⟨41182⟩ 28337)

def event28348 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨41767⟩⟩, .relation 28347 0, ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨40038⟩⟩], [⟨.program ⟨257⟩, ⟨41182⟩⟩]⟩, (-1)⟩)

def event28349 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨41767⟩⟩, .operator (⟨18858, 0⟩, ⟨28340, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨41765⟩⟩]⟩, (1)⟩)

def exact28350RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨41765⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨40038⟩⟩], [⟨.program ⟨257⟩, ⟨41182⟩⟩]⟩, (-1)⟩]

theorem exact28350RawTermsValid :
    exact28350RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28350 : Event := .resultExact (⟨.program ⟨257⟩, ⟨41767⟩⟩) exact28350RawTerms .large 28343 (.finite 32193129122288627115968346193920) (some (28345))

def event28351 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40678⟩⟩) 0 ⟨40039⟩ 137

def event28352 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨40678⟩⟩) (.authority (.relationPreimageSource ⟨86⟩))

def exact28353RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨40678⟩⟩]⟩, (1)⟩]

theorem exact28353RawTermsValid :
    exact28353RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28353 : Event := .resultExact (⟨.program ⟨257⟩, ⟨40678⟩⟩) exact28353RawTerms (.finite 5647228698) 28352 .exactZero (none)

def event28354 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40680⟩⟩) 0 ⟨40678⟩ 28353

def event28355 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40680⟩⟩) 1 ⟨2370⟩ 4

def event28356 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨40680⟩⟩) (.scale (.predecessor 0 28354 .coefficient) (.value (.predecessor 1 28355 .coefficient)))

def exact28357RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨40678⟩⟩]⟩, (1)⟩]

theorem exact28357RawTermsValid :
    exact28357RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28357 : Event := .resultExact (⟨.program ⟨257⟩, ⟨40680⟩⟩) exact28357RawTerms (.finite 5647228698) 28356 .exactZero (none)

def event28358 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40681⟩⟩) 0 ⟨5443⟩ 17169

def event28359 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40681⟩⟩) 1 ⟨40680⟩ 28357

def event28360 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨40681⟩⟩) (.product (.predecessor 0 28358 .coefficient) (.predecessor 1 28359 .coefficient) (⟨false, false, none, none, none⟩))

def event28361 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨40681⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨40678⟩⟩]⟩) [⟨.result 28353 .coefficient, false, none⟩])

def event28362 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨40681⟩⟩) (.product (.result 17169 .summary) (.transfer 28361) (⟨false, false, none, none, none⟩))

def event28363 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨40681⟩⟩, .operator (⟨17169, 0⟩, ⟨28357, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40678⟩⟩]⟩, (1)⟩)

def event28364 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨40679⟩⟩)

def event28365 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event28366 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event28367 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨140⟩⟩) (.authority (.operator))

def event28368 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨140⟩⟩) (.finite 19)

def event28369 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event28370 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event28371 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event28372 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event28373 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 28372

def event28374 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 28370

def event28375 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 28373 .coefficient) (.value (.predecessor 1 28374 .coefficient)))

def event28376 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event28377 : Event := .predecessor (⟨.program ⟨257⟩, ⟨393⟩⟩) 0 ⟨392⟩ 28376

def event28378 : Event := .predecessor (⟨.program ⟨257⟩, ⟨393⟩⟩) 1 ⟨140⟩ 28368

def event28379 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨393⟩⟩) (.sum [.predecessor 0 28377 .coefficient, .predecessor 1 28378 .coefficient])

def event28380 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨393⟩⟩) (.finite 655359)

def event28381 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5439⟩⟩) 0 ⟨393⟩ 28380

def event28382 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5439⟩⟩) 1 ⟨5426⟩ 28366

def event28383 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5439⟩⟩) (.identity (.predecessor 1 28382 .coefficient))

def event28384 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5439⟩⟩) (.finite 655360)

def event28385 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39586⟩⟩) 0 ⟨5439⟩ 28384

def event28386 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39586⟩⟩) (.authority (.programFamilyFact))

def exact28387RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨39586⟩⟩], []⟩, (1)⟩]

theorem exact28387RawTermsValid :
    exact28387RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28387 : Event := .resultExact (⟨.program ⟨257⟩, ⟨39586⟩⟩) exact28387RawTerms (.finite 46) 28386 .exactZero (none)

def event28388 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14051⟩⟩) 0 ⟨5439⟩ 28384

def event28389 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14051⟩⟩) (.authority (.programFamilyFact))

def exact28390RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14051⟩⟩], []⟩, (1)⟩]

theorem exact28390RawTermsValid :
    exact28390RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28390 : Event := .resultExact (⟨.program ⟨257⟩, ⟨14051⟩⟩) exact28390RawTerms (.finite 46) 28389 .exactZero (none)

def event28391 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39587⟩⟩) 0 ⟨14051⟩ 28390

def event28392 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39587⟩⟩) 1 ⟨39586⟩ 28387

def event28393 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39587⟩⟩) (.product (.predecessor 0 28391 .coefficient) (.predecessor 1 28392 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event28394 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39587⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨14051⟩⟩, ⟨.program ⟨257⟩, ⟨39586⟩⟩], []⟩) [⟨.result 28390 .coefficient, true, some 1⟩, ⟨.result 28387 .coefficient, true, some 1⟩])

def event28395 : Event := .survivorFold (1) 28394

def exact28396RawTerms : List Term := []

theorem exact28396RawTermsValid :
    exact28396RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28396 : Event := .resultExact (⟨.program ⟨257⟩, ⟨39587⟩⟩) exact28396RawTerms (.finite 2116) 28393 (.finite 2116) (some (28394))

def event28397 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39588⟩⟩) 0 ⟨39587⟩ 28396

def event28398 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39588⟩⟩) (.identity (.predecessor 0 28397 .coefficient))

def event28399 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨39588⟩⟩) (.finite 2116)

def event28400 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40038⟩⟩) 0 ⟨39588⟩ 28399

def event28401 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨40038⟩⟩) (.authority (.programFamilyFact))

def exact28402RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨40038⟩⟩], []⟩, (1)⟩]

theorem exact28402RawTermsValid :
    exact28402RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28402 : Event := .resultExact (⟨.program ⟨257⟩, ⟨40038⟩⟩) exact28402RawTerms (.finite 46) 28401 .exactZero (none)

def event28403 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40039⟩⟩) 0 ⟨40038⟩ 28402

def event28404 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨40039⟩⟩) (.identity (.predecessor 0 28403 .coefficient))

def event28405 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨40039⟩⟩) (.finite 46)

def event28406 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40678⟩⟩) 0 ⟨40039⟩ 28405

def event28407 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨40678⟩⟩) (.authority (.relationPreimageSource ⟨86⟩))

def exact28408RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨40678⟩⟩]⟩, (1)⟩]

theorem exact28408RawTermsValid :
    exact28408RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28408 : Event := .resultExact (⟨.program ⟨257⟩, ⟨40678⟩⟩) exact28408RawTerms (.finite 5647228698) 28407 .exactZero (none)

def event28409 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35⟩⟩) (.authority (.operator))

def exact28410RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩]⟩, (1)⟩]

theorem exact28410RawTermsValid :
    exact28410RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28410 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35⟩⟩) exact28410RawTerms .large 28409 .exactZero (none)

def event28411 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40679⟩⟩) 0 ⟨35⟩ 28410

def event28412 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40679⟩⟩) 1 ⟨40678⟩ 28408

def event28413 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨40679⟩⟩) (.product (.predecessor 0 28411 .coefficient) (.predecessor 1 28412 .coefficient) (⟨false, false, none, none, none⟩))

def event28414 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨40679⟩⟩, .operator (⟨28410, 0⟩, ⟨28408, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40678⟩⟩]⟩, (1)⟩)

def exact28415RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40678⟩⟩]⟩, (1)⟩]

theorem exact28415RawTermsValid :
    exact28415RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28415 : Event := .resultExact (⟨.program ⟨257⟩, ⟨40679⟩⟩) exact28415RawTerms .large 28413 .exactZero (none)

def eventLeaf1760 : Array AnnotatedEvent := #[
  { event := event28160
    frameStart := 28152 },
  { event := event28161
    frameStart := 28152 },
  { event := event28162
    frameStart := 28152 },
  { event := event28163
    frameStart := 28152 },
  { event := event28164
    frameStart := 28152 },
  { event := event28165
    frameStart := 28152 },
  { event := event28166
    frameStart := 28152 },
  { event := event28167
    frameStart := 28152 },
  { event := event28168
    frameStart := 28152 },
  { event := event28169
    frameStart := 28152 },
  { event := event28170
    frameStart := 28152 },
  { event := event28171
    frameStart := 28152 },
  { event := event28172
    frameStart := 28152 },
  { event := event28173
    frameStart := 28152 },
  { event := event28174
    frameStart := 28152 },
  { event := event28175
    frameStart := 28152 }
]

def eventLeaf1761 : Array AnnotatedEvent := #[
  { event := event28176
    frameStart := 28152 },
  { event := event28177
    frameStart := 28152 },
  { event := event28178
    frameStart := 28152 },
  { event := event28179
    frameStart := 28152 },
  { event := event28180
    frameStart := 28152 },
  { event := event28181
    frameStart := 28152 },
  { event := event28182
    frameStart := 28152 },
  { event := event28183
    frameStart := 28152 },
  { event := event28184
    frameStart := 28152 },
  { event := event28185
    frameStart := 28152 },
  { event := event28186
    frameStart := 28152 },
  { event := event28187
    frameStart := 28152 },
  { event := event28188
    frameStart := 28152 },
  { event := event28189
    frameStart := 28152 },
  { event := event28190
    frameStart := 28152 },
  { event := event28191
    frameStart := 28152 }
]

def eventLeaf1762 : Array AnnotatedEvent := #[
  { event := event28192
    frameStart := 28152 },
  { event := event28193
    frameStart := 28152 },
  { event := event28194
    frameStart := 28152 },
  { event := event28195
    frameStart := 28152 },
  { event := event28196
    frameStart := 28152 },
  { event := event28197
    frameStart := 28152 },
  { event := event28198
    frameStart := 28152 },
  { event := event28199
    frameStart := 28152 },
  { event := event28200
    frameStart := 28152 },
  { event := event28201
    frameStart := 28152 },
  { event := event28202
    frameStart := 28152 },
  { event := event28203
    frameStart := 28152 },
  { event := event28204
    frameStart := 28152 },
  { event := event28205
    frameStart := 28152 },
  { event := event28206
    frameStart := 28206 },
  { event := event28207
    frameStart := 28206 }
]

def eventLeaf1763 : Array AnnotatedEvent := #[
  { event := event28208
    frameStart := 28206 },
  { event := event28209
    frameStart := 28206 },
  { event := event28210
    frameStart := 28206 },
  { event := event28211
    frameStart := 28206 },
  { event := event28212
    frameStart := 28206 },
  { event := event28213
    frameStart := 28206 },
  { event := event28214
    frameStart := 28206 },
  { event := event28215
    frameStart := 28206 },
  { event := event28216
    frameStart := 28206 },
  { event := event28217
    frameStart := 28206 },
  { event := event28218
    frameStart := 28206 },
  { event := event28219
    frameStart := 28206 },
  { event := event28220
    frameStart := 28206 },
  { event := event28221
    frameStart := 28206 },
  { event := event28222
    frameStart := 28206 },
  { event := event28223
    frameStart := 28206 }
]

def eventLeaf1764 : Array AnnotatedEvent := #[
  { event := event28224
    frameStart := 28206 },
  { event := event28225
    frameStart := 28206 },
  { event := event28226
    frameStart := 28206 },
  { event := event28227
    frameStart := 28206 },
  { event := event28228
    frameStart := 28206 },
  { event := event28229
    frameStart := 28206 },
  { event := event28230
    frameStart := 28206 },
  { event := event28231
    frameStart := 28206 },
  { event := event28232
    frameStart := 28206 },
  { event := event28233
    frameStart := 28206 },
  { event := event28234
    frameStart := 28206 },
  { event := event28235
    frameStart := 28206 },
  { event := event28236
    frameStart := 28206 },
  { event := event28237
    frameStart := 28206 },
  { event := event28238
    frameStart := 28206 },
  { event := event28239
    frameStart := 28206 }
]

def eventLeaf1765 : Array AnnotatedEvent := #[
  { event := event28240
    frameStart := 28206 },
  { event := event28241
    frameStart := 28206 },
  { event := event28242
    frameStart := 28206 },
  { event := event28243
    frameStart := 28206 },
  { event := event28244
    frameStart := 28206 },
  { event := event28245
    frameStart := 28206 },
  { event := event28246
    frameStart := 28206 },
  { event := event28247
    frameStart := 28206 },
  { event := event28248
    frameStart := 28206 },
  { event := event28249
    frameStart := 28206 },
  { event := event28250
    frameStart := 28206 },
  { event := event28251
    frameStart := 28206 },
  { event := event28252
    frameStart := 28206 },
  { event := event28253
    frameStart := 28206 },
  { event := event28254
    frameStart := 28206 },
  { event := event28255
    frameStart := 28206 }
]

def eventLeaf1766 : Array AnnotatedEvent := #[
  { event := event28256
    frameStart := 28206 },
  { event := event28257
    frameStart := 28206 },
  { event := event28258
    frameStart := 28206 },
  { event := event28259
    frameStart := 28206 },
  { event := event28260
    frameStart := 28206 },
  { event := event28261
    frameStart := 28206 },
  { event := event28262
    frameStart := 28206 },
  { event := event28263
    frameStart := 28206 },
  { event := event28264
    frameStart := 28206 },
  { event := event28265
    frameStart := 28206 },
  { event := event28266
    frameStart := 28206 },
  { event := event28267
    frameStart := 28206 },
  { event := event28268
    frameStart := 28206 },
  { event := event28269
    frameStart := 28206 },
  { event := event28270
    frameStart := 28206 },
  { event := event28271
    frameStart := 28206 }
]

def eventLeaf1767 : Array AnnotatedEvent := #[
  { event := event28272
    frameStart := 28206 },
  { event := event28273
    frameStart := 28206 },
  { event := event28274
    frameStart := 28206 },
  { event := event28275
    frameStart := 28206 },
  { event := event28276
    frameStart := 28206 },
  { event := event28277
    frameStart := 28206 },
  { event := event28278
    frameStart := 28206 },
  { event := event28279
    frameStart := 28206 },
  { event := event28280
    frameStart := 28206 },
  { event := event28281
    frameStart := 28206 },
  { event := event28282
    frameStart := 28206 },
  { event := event28283
    frameStart := 28206 },
  { event := event28284
    frameStart := 28206 },
  { event := event28285
    frameStart := 28206 },
  { event := event28286
    frameStart := 28206 },
  { event := event28287
    frameStart := 28206 }
]

def eventLeaf1768 : Array AnnotatedEvent := #[
  { event := event28288
    frameStart := 28206 },
  { event := event28289
    frameStart := 28206 },
  { event := event28290
    frameStart := 28206 },
  { event := event28291
    frameStart := 28206 },
  { event := event28292
    frameStart := 28206 },
  { event := event28293
    frameStart := 28206 },
  { event := event28294
    frameStart := 28206 },
  { event := event28295
    frameStart := 28206 },
  { event := event28296
    frameStart := 28206 },
  { event := event28297
    frameStart := 28206 },
  { event := event28298
    frameStart := 28206 },
  { event := event28299
    frameStart := 28206 },
  { event := event28300
    frameStart := 28206 },
  { event := event28301
    frameStart := 28206 },
  { event := event28302
    frameStart := 28206 },
  { event := event28303
    frameStart := 28206 }
]

def eventLeaf1769 : Array AnnotatedEvent := #[
  { event := event28304
    frameStart := 28206 },
  { event := event28305
    frameStart := 28206 },
  { event := event28306
    frameStart := 28206 },
  { event := event28307
    frameStart := 28206 },
  { event := event28308
    frameStart := 28206 },
  { event := event28309
    frameStart := 28206 },
  { event := event28310
    frameStart := 0 },
  { event := event28311
    frameStart := 0 },
  { event := event28312
    frameStart := 0 },
  { event := event28313
    frameStart := 0 },
  { event := event28314
    frameStart := 0 },
  { event := event28315
    frameStart := 0 },
  { event := event28316
    frameStart := 0 },
  { event := event28317
    frameStart := 0 },
  { event := event28318
    frameStart := 0 },
  { event := event28319
    frameStart := 0 }
]

def eventLeaf1770 : Array AnnotatedEvent := #[
  { event := event28320
    frameStart := 0 },
  { event := event28321
    frameStart := 0 },
  { event := event28322
    frameStart := 0 },
  { event := event28323
    frameStart := 0 },
  { event := event28324
    frameStart := 0 },
  { event := event28325
    frameStart := 0 },
  { event := event28326
    frameStart := 0 },
  { event := event28327
    frameStart := 0 },
  { event := event28328
    frameStart := 0 },
  { event := event28329
    frameStart := 0 },
  { event := event28330
    frameStart := 0 },
  { event := event28331
    frameStart := 0 },
  { event := event28332
    frameStart := 0 },
  { event := event28333
    frameStart := 0 },
  { event := event28334
    frameStart := 0 },
  { event := event28335
    frameStart := 0 }
]

def eventLeaf1771 : Array AnnotatedEvent := #[
  { event := event28336
    frameStart := 0 },
  { event := event28337
    frameStart := 0 },
  { event := event28338
    frameStart := 0 },
  { event := event28339
    frameStart := 0 },
  { event := event28340
    frameStart := 0 },
  { event := event28341
    frameStart := 0 },
  { event := event28342
    frameStart := 0 },
  { event := event28343
    frameStart := 0 },
  { event := event28344
    frameStart := 0 },
  { event := event28345
    frameStart := 0 },
  { event := event28346
    frameStart := 0 },
  { event := event28347
    frameStart := 0 },
  { event := event28348
    frameStart := 0 },
  { event := event28349
    frameStart := 0 },
  { event := event28350
    frameStart := 0 },
  { event := event28351
    frameStart := 0 }
]

def eventLeaf1772 : Array AnnotatedEvent := #[
  { event := event28352
    frameStart := 0 },
  { event := event28353
    frameStart := 0 },
  { event := event28354
    frameStart := 0 },
  { event := event28355
    frameStart := 0 },
  { event := event28356
    frameStart := 0 },
  { event := event28357
    frameStart := 0 },
  { event := event28358
    frameStart := 0 },
  { event := event28359
    frameStart := 0 },
  { event := event28360
    frameStart := 0 },
  { event := event28361
    frameStart := 0 },
  { event := event28362
    frameStart := 0 },
  { event := event28363
    frameStart := 0 },
  { event := event28364
    frameStart := 28364 },
  { event := event28365
    frameStart := 28364 },
  { event := event28366
    frameStart := 28364 },
  { event := event28367
    frameStart := 28364 }
]

def eventLeaf1773 : Array AnnotatedEvent := #[
  { event := event28368
    frameStart := 28364 },
  { event := event28369
    frameStart := 28364 },
  { event := event28370
    frameStart := 28364 },
  { event := event28371
    frameStart := 28364 },
  { event := event28372
    frameStart := 28364 },
  { event := event28373
    frameStart := 28364 },
  { event := event28374
    frameStart := 28364 },
  { event := event28375
    frameStart := 28364 },
  { event := event28376
    frameStart := 28364 },
  { event := event28377
    frameStart := 28364 },
  { event := event28378
    frameStart := 28364 },
  { event := event28379
    frameStart := 28364 },
  { event := event28380
    frameStart := 28364 },
  { event := event28381
    frameStart := 28364 },
  { event := event28382
    frameStart := 28364 },
  { event := event28383
    frameStart := 28364 }
]

def eventLeaf1774 : Array AnnotatedEvent := #[
  { event := event28384
    frameStart := 28364 },
  { event := event28385
    frameStart := 28364 },
  { event := event28386
    frameStart := 28364 },
  { event := event28387
    frameStart := 28364 },
  { event := event28388
    frameStart := 28364 },
  { event := event28389
    frameStart := 28364 },
  { event := event28390
    frameStart := 28364 },
  { event := event28391
    frameStart := 28364 },
  { event := event28392
    frameStart := 28364 },
  { event := event28393
    frameStart := 28364 },
  { event := event28394
    frameStart := 28364 },
  { event := event28395
    frameStart := 28364 },
  { event := event28396
    frameStart := 28364 },
  { event := event28397
    frameStart := 28364 },
  { event := event28398
    frameStart := 28364 },
  { event := event28399
    frameStart := 28364 }
]

def eventLeaf1775 : Array AnnotatedEvent := #[
  { event := event28400
    frameStart := 28364 },
  { event := event28401
    frameStart := 28364 },
  { event := event28402
    frameStart := 28364 },
  { event := event28403
    frameStart := 28364 },
  { event := event28404
    frameStart := 28364 },
  { event := event28405
    frameStart := 28364 },
  { event := event28406
    frameStart := 28364 },
  { event := event28407
    frameStart := 28364 },
  { event := event28408
    frameStart := 28364 },
  { event := event28409
    frameStart := 28364 },
  { event := event28410
    frameStart := 28364 },
  { event := event28411
    frameStart := 28364 },
  { event := event28412
    frameStart := 28364 },
  { event := event28413
    frameStart := 28364 },
  { event := event28414
    frameStart := 28364 },
  { event := event28415
    frameStart := 28364 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events110
