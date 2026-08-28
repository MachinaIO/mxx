import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events567

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event145152 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨43395⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨43392⟩⟩]⟩) [⟨.result 145144 .coefficient, false, none⟩])

def event145153 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨43395⟩⟩) (.product (.result 134495 .summary) (.transfer 145152) (⟨false, false, none, none, none⟩))

def event145154 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨43395⟩⟩, .operator (⟨134495, 0⟩, ⟨145148, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43392⟩⟩]⟩, (1)⟩)

def event145155 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨43393⟩⟩)

def event145156 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event145157 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event145158 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨455⟩⟩) (.authority (.operator))

def event145159 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨455⟩⟩) (.finite 11)

def event145160 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event145161 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event145162 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event145163 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event145164 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 145163

def event145165 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 145161

def event145166 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 145164 .coefficient) (.value (.predecessor 1 145165 .coefficient)))

def event145167 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event145168 : Event := .predecessor (⟨.program ⟨257⟩, ⟨457⟩⟩) 0 ⟨392⟩ 145167

def event145169 : Event := .predecessor (⟨.program ⟨257⟩, ⟨457⟩⟩) 1 ⟨455⟩ 145159

def event145170 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨457⟩⟩) (.sum [.predecessor 0 145168 .coefficient, .predecessor 1 145169 .coefficient])

def event145171 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨457⟩⟩) (.finite 655351)

def event145172 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5469⟩⟩) 0 ⟨457⟩ 145171

def event145173 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5469⟩⟩) 1 ⟨5426⟩ 145157

def event145174 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5469⟩⟩) (.identity (.predecessor 1 145173 .coefficient))

def event145175 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5469⟩⟩) (.finite 655360)

def event145176 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42306⟩⟩) 0 ⟨5469⟩ 145175

def event145177 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42306⟩⟩) (.authority (.programFamilyFact))

def exact145178RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨42306⟩⟩], []⟩, (1)⟩]

theorem exact145178RawTermsValid :
    exact145178RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event145178 : Event := .resultExact (⟨.program ⟨257⟩, ⟨42306⟩⟩) exact145178RawTerms (.finite 52) 145177 .exactZero (none)

def event145179 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14376⟩⟩) 0 ⟨5469⟩ 145175

def event145180 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14376⟩⟩) (.authority (.programFamilyFact))

def exact145181RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14376⟩⟩], []⟩, (1)⟩]

theorem exact145181RawTermsValid :
    exact145181RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event145181 : Event := .resultExact (⟨.program ⟨257⟩, ⟨14376⟩⟩) exact145181RawTerms (.finite 52) 145180 .exactZero (none)

def event145182 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42307⟩⟩) 0 ⟨14376⟩ 145181

def event145183 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42307⟩⟩) 1 ⟨42306⟩ 145178

def event145184 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42307⟩⟩) (.product (.predecessor 0 145182 .coefficient) (.predecessor 1 145183 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event145185 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42307⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨14376⟩⟩, ⟨.program ⟨257⟩, ⟨42306⟩⟩], []⟩) [⟨.result 145181 .coefficient, true, some 1⟩, ⟨.result 145178 .coefficient, true, some 1⟩])

def event145186 : Event := .survivorFold (1) 145185

def exact145187RawTerms : List Term := []

theorem exact145187RawTermsValid :
    exact145187RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event145187 : Event := .resultExact (⟨.program ⟨257⟩, ⟨42307⟩⟩) exact145187RawTerms (.finite 2704) 145184 (.finite 2704) (some (145185))

def event145188 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42308⟩⟩) 0 ⟨42307⟩ 145187

def event145189 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42308⟩⟩) (.identity (.predecessor 0 145188 .coefficient))

def event145190 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨42308⟩⟩) (.finite 2704)

def event145191 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42732⟩⟩) 0 ⟨42308⟩ 145190

def event145192 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42732⟩⟩) (.authority (.programFamilyFact))

def exact145193RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨42732⟩⟩], []⟩, (1)⟩]

theorem exact145193RawTermsValid :
    exact145193RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event145193 : Event := .resultExact (⟨.program ⟨257⟩, ⟨42732⟩⟩) exact145193RawTerms (.finite 52) 145192 .exactZero (none)

def event145194 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42733⟩⟩) 0 ⟨42732⟩ 145193

def event145195 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42733⟩⟩) (.identity (.predecessor 0 145194 .coefficient))

def event145196 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨42733⟩⟩) (.finite 52)

def event145197 : Event := .predecessor (⟨.program ⟨257⟩, ⟨43392⟩⟩) 0 ⟨42733⟩ 145196

def event145198 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨43392⟩⟩) (.authority (.relationPreimageSource ⟨89⟩))

def exact145199RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨43392⟩⟩]⟩, (1)⟩]

theorem exact145199RawTermsValid :
    exact145199RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event145199 : Event := .resultExact (⟨.program ⟨257⟩, ⟨43392⟩⟩) exact145199RawTerms (.finite 5647228698) 145198 .exactZero (none)

def event145200 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35⟩⟩) (.authority (.operator))

def exact145201RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩]⟩, (1)⟩]

theorem exact145201RawTermsValid :
    exact145201RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event145201 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35⟩⟩) exact145201RawTerms .large 145200 .exactZero (none)

def event145202 : Event := .predecessor (⟨.program ⟨257⟩, ⟨43393⟩⟩) 0 ⟨35⟩ 145201

def event145203 : Event := .predecessor (⟨.program ⟨257⟩, ⟨43393⟩⟩) 1 ⟨43392⟩ 145199

def event145204 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨43393⟩⟩) (.product (.predecessor 0 145202 .coefficient) (.predecessor 1 145203 .coefficient) (⟨false, false, none, none, none⟩))

def event145205 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨43393⟩⟩, .operator (⟨145201, 0⟩, ⟨145199, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43392⟩⟩]⟩, (1)⟩)

def exact145206RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43392⟩⟩]⟩, (1)⟩]

theorem exact145206RawTermsValid :
    exact145206RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event145206 : Event := .resultExact (⟨.program ⟨257⟩, ⟨43393⟩⟩) exact145206RawTerms .large 145204 .exactZero (none)

def event145207 : Event := .preFoldPolynomial 145206 [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43392⟩⟩]⟩, (1)⟩] .exactZero none

def exact145208RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43392⟩⟩]⟩, (1)⟩]

def event145208 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨43393⟩⟩) 145207 exact145208RawTerms .large 145204 .exactZero (none)

def event145209 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨44493⟩⟩)

def event145210 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event145211 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event145212 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨455⟩⟩) (.authority (.operator))

def event145213 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨455⟩⟩) (.finite 11)

def event145214 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event145215 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event145216 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event145217 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event145218 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 145217

def event145219 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 145215

def event145220 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 145218 .coefficient) (.value (.predecessor 1 145219 .coefficient)))

def event145221 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event145222 : Event := .predecessor (⟨.program ⟨257⟩, ⟨457⟩⟩) 0 ⟨392⟩ 145221

def event145223 : Event := .predecessor (⟨.program ⟨257⟩, ⟨457⟩⟩) 1 ⟨455⟩ 145213

def event145224 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨457⟩⟩) (.sum [.predecessor 0 145222 .coefficient, .predecessor 1 145223 .coefficient])

def event145225 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨457⟩⟩) (.finite 655351)

def event145226 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5469⟩⟩) 0 ⟨457⟩ 145225

def event145227 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5469⟩⟩) 1 ⟨5426⟩ 145211

def event145228 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5469⟩⟩) (.identity (.predecessor 1 145227 .coefficient))

def event145229 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5469⟩⟩) (.finite 655360)

def event145230 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42306⟩⟩) 0 ⟨5469⟩ 145229

def event145231 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42306⟩⟩) (.authority (.programFamilyFact))

def exact145232RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨42306⟩⟩], []⟩, (1)⟩]

theorem exact145232RawTermsValid :
    exact145232RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event145232 : Event := .resultExact (⟨.program ⟨257⟩, ⟨42306⟩⟩) exact145232RawTerms (.finite 52) 145231 .exactZero (none)

def event145233 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14376⟩⟩) 0 ⟨5469⟩ 145229

def event145234 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14376⟩⟩) (.authority (.programFamilyFact))

def exact145235RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14376⟩⟩], []⟩, (1)⟩]

theorem exact145235RawTermsValid :
    exact145235RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event145235 : Event := .resultExact (⟨.program ⟨257⟩, ⟨14376⟩⟩) exact145235RawTerms (.finite 52) 145234 .exactZero (none)

def event145236 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42307⟩⟩) 0 ⟨14376⟩ 145235

def event145237 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42307⟩⟩) 1 ⟨42306⟩ 145232

def event145238 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42307⟩⟩) (.product (.predecessor 0 145236 .coefficient) (.predecessor 1 145237 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event145239 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨42307⟩⟩, .operator (⟨145235, 0⟩, ⟨145232, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨14376⟩⟩, ⟨.program ⟨257⟩, ⟨42306⟩⟩], []⟩, (1)⟩)

def exact145240RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14376⟩⟩, ⟨.program ⟨257⟩, ⟨42306⟩⟩], []⟩, (1)⟩]

theorem exact145240RawTermsValid :
    exact145240RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event145240 : Event := .resultExact (⟨.program ⟨257⟩, ⟨42307⟩⟩) exact145240RawTerms (.finite 2704) 145238 .exactZero (none)

def event145241 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42308⟩⟩) 0 ⟨42307⟩ 145240

def event145242 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42308⟩⟩) (.identity (.predecessor 0 145241 .coefficient))

def event145243 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨42308⟩⟩) (.finite 2704)

def event145244 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42732⟩⟩) 0 ⟨42308⟩ 145243

def event145245 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42732⟩⟩) (.authority (.programFamilyFact))

def exact145246RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨42732⟩⟩], []⟩, (1)⟩]

theorem exact145246RawTermsValid :
    exact145246RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event145246 : Event := .resultExact (⟨.program ⟨257⟩, ⟨42732⟩⟩) exact145246RawTerms (.finite 52) 145245 .exactZero (none)

def event145247 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42733⟩⟩) 0 ⟨42732⟩ 145246

def event145248 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42733⟩⟩) (.identity (.predecessor 0 145247 .coefficient))

def event145249 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨42733⟩⟩) (.finite 52)

def event145250 : Event := .predecessor (⟨.program ⟨257⟩, ⟨43876⟩⟩) 0 ⟨42733⟩ 145249

def event145251 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨43876⟩⟩) (.authority (.programFamilyFact))

def event145252 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨43876⟩⟩) (.finite 3720)

def event145253 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨7177⟩⟩) .missing

def event145254 : Event := .predecessor (⟨.program ⟨257⟩, ⟨43877⟩⟩) 0 ⟨7177⟩ 145253

def event145255 : Event := .predecessor (⟨.program ⟨257⟩, ⟨43877⟩⟩) 1 ⟨43876⟩ 145252

def event145256 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨43877⟩⟩) (.authority (.operator))

def exact145257RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨43877⟩⟩]⟩, (1)⟩]

theorem exact145257RawTermsValid :
    exact145257RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event145257 : Event := .resultExact (⟨.program ⟨257⟩, ⟨43877⟩⟩) exact145257RawTerms .large 145256 .exactZero (none)

def event145258 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44488⟩⟩) 0 ⟨43877⟩ 145257

def event145259 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44488⟩⟩) (.authority (.operator))

def exact145260RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨44488⟩⟩]⟩, (1)⟩]

theorem exact145260RawTermsValid :
    exact145260RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event145260 : Event := .resultExact (⟨.program ⟨257⟩, ⟨44488⟩⟩) exact145260RawTerms (.finite 8192) 145259 .exactZero (none)

def event145261 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨136⟩⟩) (.authority (.operator))

def event145262 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨136⟩⟩) .exactZero

def event145263 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44118⟩⟩) 0 ⟨42733⟩ 145249

def event145264 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44118⟩⟩) 1 ⟨136⟩ 145262

def event145265 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44118⟩⟩) (.sum [.predecessor 0 145263 .coefficient, .predecessor 1 145264 .coefficient])

def event145266 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨44118⟩⟩) (.finite 52)

def event145267 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44119⟩⟩) 0 ⟨44118⟩ 145266

def event145268 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44119⟩⟩) (.identity (.predecessor 0 145267 .coefficient))

def exact145269RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨42732⟩⟩], []⟩, (1)⟩]

theorem exact145269RawTermsValid :
    exact145269RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event145269 : Event := .resultExact (⟨.program ⟨257⟩, ⟨44119⟩⟩) exact145269RawTerms (.finite 52) 145268 .exactZero (none)

def event145270 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6908⟩⟩) (.authority (.factStore))

def exact145271RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact145271RawTermsValid :
    exact145271RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event145271 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6908⟩⟩) exact145271RawTerms .large 145270 .exactZero (none)

def event145272 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44120⟩⟩) 0 ⟨6908⟩ 145271

def event145273 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44120⟩⟩) 1 ⟨44119⟩ 145269

def event145274 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44120⟩⟩) (.product (.predecessor 0 145272 .coefficient) (.predecessor 1 145273 .coefficient) (⟨false, false, none, none, none⟩))

def event145275 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨44120⟩⟩, .operator (⟨145271, 0⟩, ⟨145269, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨42732⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact145276RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨42732⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact145276RawTermsValid :
    exact145276RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event145276 : Event := .resultExact (⟨.program ⟨257⟩, ⟨44120⟩⟩) exact145276RawTerms .large 145274 .exactZero (none)

def event145277 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7194⟩⟩) 0 ⟨7177⟩ 145253

def event145278 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7194⟩⟩) (.authority (.operator))

def exact145279RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7194⟩⟩]⟩, (1)⟩]

theorem exact145279RawTermsValid :
    exact145279RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event145279 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7194⟩⟩) exact145279RawTerms .large 145278 .exactZero (none)

def event145280 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44121⟩⟩) 0 ⟨7194⟩ 145279

def event145281 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44121⟩⟩) 1 ⟨44120⟩ 145276

def event145282 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44121⟩⟩) (.sum [.predecessor 0 145280 .coefficient, .predecessor 1 145281 .coefficient])

def exact145283RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7194⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨42732⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact145283RawTermsValid :
    exact145283RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event145283 : Event := .resultExact (⟨.program ⟨257⟩, ⟨44121⟩⟩) exact145283RawTerms .large 145282 .exactZero (none)

def event145284 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44489⟩⟩) 0 ⟨44121⟩ 145283

def event145285 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44489⟩⟩) 1 ⟨44488⟩ 145260

def event145286 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44489⟩⟩) (.product (.predecessor 0 145284 .coefficient) (.predecessor 1 145285 .coefficient) (⟨false, false, none, none, none⟩))

def event145287 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨44489⟩⟩, .operator (⟨145283, 0⟩, ⟨145260, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7194⟩⟩, ⟨.program ⟨257⟩, ⟨44488⟩⟩]⟩, (1)⟩)

def event145288 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨44489⟩⟩, .operator (⟨145283, 1⟩, ⟨145260, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨42732⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨44488⟩⟩]⟩, (-1)⟩)

def event145289 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨44489⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨42732⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨44488⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨44488⟩⟩) ⟨43877⟩ 145257)

def event145290 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨44489⟩⟩, .relation 145289 0, ⟨[⟨.program ⟨257⟩, ⟨42732⟩⟩], [⟨.program ⟨257⟩, ⟨43877⟩⟩]⟩, (-1)⟩)

def exact145291RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7194⟩⟩, ⟨.program ⟨257⟩, ⟨44488⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨42732⟩⟩], [⟨.program ⟨257⟩, ⟨43877⟩⟩]⟩, (-1)⟩]

theorem exact145291RawTermsValid :
    exact145291RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event145291 : Event := .resultExact (⟨.program ⟨257⟩, ⟨44489⟩⟩) exact145291RawTerms .large 145286 .exactZero (none)

def event145292 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42911⟩⟩) 0 ⟨42733⟩ 145249

def event145293 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42911⟩⟩) (.authority (.programFamilyFact))

def exact145294RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨42911⟩⟩], []⟩, (1)⟩]

theorem exact145294RawTermsValid :
    exact145294RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event145294 : Event := .resultExact (⟨.program ⟨257⟩, ⟨42911⟩⟩) exact145294RawTerms (.finite 52) 145293 .exactZero (none)

def event145295 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42913⟩⟩) 0 ⟨6908⟩ 145271

def event145296 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42913⟩⟩) 1 ⟨42911⟩ 145294

def event145297 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42913⟩⟩) (.product (.predecessor 0 145295 .coefficient) (.predecessor 1 145296 .coefficient) (⟨false, true, none, none, some 1⟩))

def event145298 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨42913⟩⟩, .operator (⟨145271, 0⟩, ⟨145294, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨42911⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact145299RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨42911⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact145299RawTermsValid :
    exact145299RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event145299 : Event := .resultExact (⟨.program ⟨257⟩, ⟨42913⟩⟩) exact145299RawTerms .large 145297 .exactZero (none)

def event145300 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7227⟩⟩) 0 ⟨7177⟩ 145253

def event145301 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7227⟩⟩) (.authority (.operator))

def exact145302RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7227⟩⟩]⟩, (1)⟩]

theorem exact145302RawTermsValid :
    exact145302RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event145302 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7227⟩⟩) exact145302RawTerms .large 145301 .exactZero (none)

def event145303 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42914⟩⟩) 0 ⟨7227⟩ 145302

def event145304 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42914⟩⟩) 1 ⟨42913⟩ 145299

def event145305 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42914⟩⟩) (.sum [.predecessor 0 145303 .coefficient, .predecessor 1 145304 .coefficient])

def exact145306RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7227⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨42911⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact145306RawTermsValid :
    exact145306RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event145306 : Event := .resultExact (⟨.program ⟨257⟩, ⟨42914⟩⟩) exact145306RawTerms .large 145305 .exactZero (none)

def event145307 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44493⟩⟩) 0 ⟨42914⟩ 145306

def event145308 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44493⟩⟩) 1 ⟨44489⟩ 145291

def event145309 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44493⟩⟩) (.sum [.predecessor 0 145307 .coefficient, .predecessor 1 145308 .coefficient])

def exact145310RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7194⟩⟩, ⟨.program ⟨257⟩, ⟨44488⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7227⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨42732⟩⟩], [⟨.program ⟨257⟩, ⟨43877⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨42911⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact145310RawTermsValid :
    exact145310RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event145310 : Event := .resultExact (⟨.program ⟨257⟩, ⟨44493⟩⟩) exact145310RawTerms .large 145309 .exactZero (none)

def event145311 : Event := .preFoldPolynomial 145310 [⟨⟨[], [⟨.program ⟨257⟩, ⟨7194⟩⟩, ⟨.program ⟨257⟩, ⟨44488⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7227⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨42732⟩⟩], [⟨.program ⟨257⟩, ⟨43877⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨42911⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩] .exactZero none

def exact145312RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7194⟩⟩, ⟨.program ⟨257⟩, ⟨44488⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7227⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨42732⟩⟩], [⟨.program ⟨257⟩, ⟨43877⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨42911⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

def event145312 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨44493⟩⟩) 145311 exact145312RawTerms .large 145309 .exactZero (none)

def event145313 : Event := .specializationComputed (⟨.program ⟨257⟩, ⟨42733⟩⟩) ⟨⟨106⟩, ⟨89⟩, ⟨135⟩⟩ ⟨145155, 145313⟩

def event145314 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨43395⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43392⟩⟩]⟩) (1) 0 2 (.universal 145313 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43392⟩⟩]⟩) (none) 145312)

def event145315 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨43395⟩⟩, .relation 145314 1, ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7227⟩⟩]⟩, (1)⟩)

def event145316 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨43395⟩⟩, .relation 145314 0, ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7194⟩⟩, ⟨.program ⟨257⟩, ⟨44488⟩⟩]⟩, (-1)⟩)

def event145317 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨43395⟩⟩, .relation 145314 2, ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨42732⟩⟩], [⟨.program ⟨257⟩, ⟨43877⟩⟩]⟩, (1)⟩)

def event145318 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨43395⟩⟩, .relation 145314 3, ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨42911⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact145319RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7194⟩⟩, ⟨.program ⟨257⟩, ⟨44488⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7227⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨42732⟩⟩], [⟨.program ⟨257⟩, ⟨43877⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨42911⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact145319RawTermsValid :
    exact145319RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event145319 : Event := .resultExact (⟨.program ⟨257⟩, ⟨43395⟩⟩) exact145319RawTerms .large 145151 (.finite 202072841853861888) (some (145153))

def event145320 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44491⟩⟩) 0 ⟨43395⟩ 145319

def event145321 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44491⟩⟩) 1 ⟨44490⟩ 145141

def event145322 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44491⟩⟩) (.sum [.predecessor 0 145320 .coefficient, .predecessor 1 145321 .coefficient])

def event145323 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨44491⟩⟩, .operator (⟨145319, 0⟩, ⟨145141, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7194⟩⟩, ⟨.program ⟨257⟩, ⟨44488⟩⟩]⟩, (1)⟩)

def event145324 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨44491⟩⟩, .operator (⟨145319, 2⟩, ⟨145141, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨42732⟩⟩], [⟨.program ⟨257⟩, ⟨43877⟩⟩]⟩, (-1)⟩)

def event145325 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44491⟩⟩) (.sum [.result 145319 .summary, .result 145141 .summary])

def exact145326RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7227⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨42911⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact145326RawTermsValid :
    exact145326RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event145326 : Event := .resultExact (⟨.program ⟨257⟩, ⟨44491⟩⟩) exact145326RawTerms .large 145322 (.finite 32193718473625891320532869316608) (some (145325))

def event145327 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44492⟩⟩) 0 ⟨44491⟩ 145326

def event145328 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44492⟩⟩) 1 ⟨7154⟩ 15582

def event145329 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44492⟩⟩) (.product (.predecessor 0 145327 .coefficient) (.predecessor 1 145328 .coefficient) (⟨false, false, none, none, none⟩))

def event145330 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44492⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨7153⟩⟩]⟩) [⟨.result 15578 .coefficient, false, none⟩])

def event145331 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44492⟩⟩) (.product (.result 145326 .summary) (.transfer 145330) (⟨false, false, none, none, none⟩))

def event145332 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨44492⟩⟩, .operator (⟨145326, 0⟩, ⟨15582, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7227⟩⟩, ⟨.program ⟨257⟩, ⟨7153⟩⟩]⟩, (1)⟩)

def event145333 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨44492⟩⟩, .operator (⟨145326, 1⟩, ⟨15582, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨42911⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7153⟩⟩]⟩, (-1)⟩)

def event145334 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨44492⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨42911⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7153⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨7153⟩⟩) ⟨7042⟩ 15575)

def event145335 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨44492⟩⟩, .relation 145334 0, ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨42911⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact145336RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7227⟩⟩, ⟨.program ⟨257⟩, ⟨7153⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨42911⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact145336RawTermsValid :
    exact145336RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event145336 : Event := .resultExact (⟨.program ⟨257⟩, ⟨44492⟩⟩) exact145336RawTerms .large 145329 (.finite 345677419952135604401347317519683074129920) (some (145331))

def event145337 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41197⟩⟩) 0 ⟨7177⟩ 15500

def event145338 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41197⟩⟩) 1 ⟨41196⟩ 135843

def event145339 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41197⟩⟩) (.authority (.operator))

def exact145340RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨41197⟩⟩]⟩, (1)⟩]

theorem exact145340RawTermsValid :
    exact145340RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event145340 : Event := .resultExact (⟨.program ⟨257⟩, ⟨41197⟩⟩) exact145340RawTerms .large 145339 .exactZero (none)

def event145341 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41808⟩⟩) 0 ⟨41197⟩ 145340

def event145342 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41808⟩⟩) (.authority (.operator))

def exact145343RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨41808⟩⟩]⟩, (1)⟩]

theorem exact145343RawTermsValid :
    exact145343RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event145343 : Event := .resultExact (⟨.program ⟨257⟩, ⟨41808⟩⟩) exact145343RawTerms (.finite 8192) 145342 .exactZero (none)

def event145344 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41810⟩⟩) 0 ⟨41544⟩ 136127

def event145345 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41810⟩⟩) 1 ⟨41808⟩ 145343

def event145346 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41810⟩⟩) (.product (.predecessor 0 145344 .coefficient) (.predecessor 1 145345 .coefficient) (⟨false, false, none, none, none⟩))

def event145347 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41810⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨41808⟩⟩]⟩) [⟨.result 145343 .coefficient, false, none⟩])

def event145348 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41810⟩⟩) (.product (.result 136127 .summary) (.transfer 145347) (⟨false, false, none, none, none⟩))

def event145349 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨41810⟩⟩, .operator (⟨136127, 0⟩, ⟨145343, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨41808⟩⟩]⟩, (1)⟩)

def event145350 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨41810⟩⟩, .operator (⟨136127, 1⟩, ⟨145343, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨40052⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨41808⟩⟩]⟩, (-1)⟩)

def event145351 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨41810⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨40052⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨41808⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨41808⟩⟩) ⟨41197⟩ 145340)

def event145352 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨41810⟩⟩, .relation 145351 0, ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨40052⟩⟩], [⟨.program ⟨257⟩, ⟨41197⟩⟩]⟩, (-1)⟩)

def exact145353RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨41808⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨40052⟩⟩], [⟨.program ⟨257⟩, ⟨41197⟩⟩]⟩, (-1)⟩]

theorem exact145353RawTermsValid :
    exact145353RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event145353 : Event := .resultExact (⟨.program ⟨257⟩, ⟨41810⟩⟩) exact145353RawTerms .large 145346 (.finite 32193129122288627115968346193920) (some (145348))

def event145354 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40712⟩⟩) 0 ⟨40053⟩ 6164

def event145355 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨40712⟩⟩) (.authority (.relationPreimageSource ⟨86⟩))

def exact145356RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨40712⟩⟩]⟩, (1)⟩]

theorem exact145356RawTermsValid :
    exact145356RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event145356 : Event := .resultExact (⟨.program ⟨257⟩, ⟨40712⟩⟩) exact145356RawTerms (.finite 5647228698) 145355 .exactZero (none)

def event145357 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40714⟩⟩) 0 ⟨40712⟩ 145356

def event145358 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40714⟩⟩) 1 ⟨2370⟩ 4

def event145359 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨40714⟩⟩) (.scale (.predecessor 0 145357 .coefficient) (.value (.predecessor 1 145358 .coefficient)))

def exact145360RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨40712⟩⟩]⟩, (1)⟩]

theorem exact145360RawTermsValid :
    exact145360RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event145360 : Event := .resultExact (⟨.program ⟨257⟩, ⟨40714⟩⟩) exact145360RawTerms (.finite 5647228698) 145359 .exactZero (none)

def event145361 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40715⟩⟩) 0 ⟨5473⟩ 134495

def event145362 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40715⟩⟩) 1 ⟨40714⟩ 145360

def event145363 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨40715⟩⟩) (.product (.predecessor 0 145361 .coefficient) (.predecessor 1 145362 .coefficient) (⟨false, false, none, none, none⟩))

def event145364 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨40715⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨40712⟩⟩]⟩) [⟨.result 145356 .coefficient, false, none⟩])

def event145365 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨40715⟩⟩) (.product (.result 134495 .summary) (.transfer 145364) (⟨false, false, none, none, none⟩))

def event145366 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨40715⟩⟩, .operator (⟨134495, 0⟩, ⟨145360, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40712⟩⟩]⟩, (1)⟩)

def event145367 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨40713⟩⟩)

def event145368 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event145369 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event145370 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨455⟩⟩) (.authority (.operator))

def event145371 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨455⟩⟩) (.finite 11)

def event145372 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event145373 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event145374 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event145375 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event145376 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 145375

def event145377 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 145373

def event145378 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 145376 .coefficient) (.value (.predecessor 1 145377 .coefficient)))

def event145379 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event145380 : Event := .predecessor (⟨.program ⟨257⟩, ⟨457⟩⟩) 0 ⟨392⟩ 145379

def event145381 : Event := .predecessor (⟨.program ⟨257⟩, ⟨457⟩⟩) 1 ⟨455⟩ 145371

def event145382 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨457⟩⟩) (.sum [.predecessor 0 145380 .coefficient, .predecessor 1 145381 .coefficient])

def event145383 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨457⟩⟩) (.finite 655351)

def event145384 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5469⟩⟩) 0 ⟨457⟩ 145383

def event145385 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5469⟩⟩) 1 ⟨5426⟩ 145369

def event145386 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5469⟩⟩) (.identity (.predecessor 1 145385 .coefficient))

def event145387 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5469⟩⟩) (.finite 655360)

def event145388 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39626⟩⟩) 0 ⟨5469⟩ 145387

def event145389 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39626⟩⟩) (.authority (.programFamilyFact))

def exact145390RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨39626⟩⟩], []⟩, (1)⟩]

theorem exact145390RawTermsValid :
    exact145390RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event145390 : Event := .resultExact (⟨.program ⟨257⟩, ⟨39626⟩⟩) exact145390RawTerms (.finite 46) 145389 .exactZero (none)

def event145391 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14076⟩⟩) 0 ⟨5469⟩ 145387

def event145392 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14076⟩⟩) (.authority (.programFamilyFact))

def exact145393RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14076⟩⟩], []⟩, (1)⟩]

theorem exact145393RawTermsValid :
    exact145393RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event145393 : Event := .resultExact (⟨.program ⟨257⟩, ⟨14076⟩⟩) exact145393RawTerms (.finite 46) 145392 .exactZero (none)

def event145394 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39627⟩⟩) 0 ⟨14076⟩ 145393

def event145395 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39627⟩⟩) 1 ⟨39626⟩ 145390

def event145396 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39627⟩⟩) (.product (.predecessor 0 145394 .coefficient) (.predecessor 1 145395 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event145397 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39627⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨14076⟩⟩, ⟨.program ⟨257⟩, ⟨39626⟩⟩], []⟩) [⟨.result 145393 .coefficient, true, some 1⟩, ⟨.result 145390 .coefficient, true, some 1⟩])

def event145398 : Event := .survivorFold (1) 145397

def exact145399RawTerms : List Term := []

theorem exact145399RawTermsValid :
    exact145399RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event145399 : Event := .resultExact (⟨.program ⟨257⟩, ⟨39627⟩⟩) exact145399RawTerms (.finite 2116) 145396 (.finite 2116) (some (145397))

def event145400 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39628⟩⟩) 0 ⟨39627⟩ 145399

def event145401 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39628⟩⟩) (.identity (.predecessor 0 145400 .coefficient))

def event145402 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨39628⟩⟩) (.finite 2116)

def event145403 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40052⟩⟩) 0 ⟨39628⟩ 145402

def event145404 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨40052⟩⟩) (.authority (.programFamilyFact))

def exact145405RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨40052⟩⟩], []⟩, (1)⟩]

theorem exact145405RawTermsValid :
    exact145405RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event145405 : Event := .resultExact (⟨.program ⟨257⟩, ⟨40052⟩⟩) exact145405RawTerms (.finite 46) 145404 .exactZero (none)

def event145406 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40053⟩⟩) 0 ⟨40052⟩ 145405

def event145407 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨40053⟩⟩) (.identity (.predecessor 0 145406 .coefficient))

def eventLeaf9072 : Array AnnotatedEvent := #[
  { event := event145152
    frameStart := 0 },
  { event := event145153
    frameStart := 0 },
  { event := event145154
    frameStart := 0 },
  { event := event145155
    frameStart := 145155 },
  { event := event145156
    frameStart := 145155 },
  { event := event145157
    frameStart := 145155 },
  { event := event145158
    frameStart := 145155 },
  { event := event145159
    frameStart := 145155 },
  { event := event145160
    frameStart := 145155 },
  { event := event145161
    frameStart := 145155 },
  { event := event145162
    frameStart := 145155 },
  { event := event145163
    frameStart := 145155 },
  { event := event145164
    frameStart := 145155 },
  { event := event145165
    frameStart := 145155 },
  { event := event145166
    frameStart := 145155 },
  { event := event145167
    frameStart := 145155 }
]

def eventLeaf9073 : Array AnnotatedEvent := #[
  { event := event145168
    frameStart := 145155 },
  { event := event145169
    frameStart := 145155 },
  { event := event145170
    frameStart := 145155 },
  { event := event145171
    frameStart := 145155 },
  { event := event145172
    frameStart := 145155 },
  { event := event145173
    frameStart := 145155 },
  { event := event145174
    frameStart := 145155 },
  { event := event145175
    frameStart := 145155 },
  { event := event145176
    frameStart := 145155 },
  { event := event145177
    frameStart := 145155 },
  { event := event145178
    frameStart := 145155 },
  { event := event145179
    frameStart := 145155 },
  { event := event145180
    frameStart := 145155 },
  { event := event145181
    frameStart := 145155 },
  { event := event145182
    frameStart := 145155 },
  { event := event145183
    frameStart := 145155 }
]

def eventLeaf9074 : Array AnnotatedEvent := #[
  { event := event145184
    frameStart := 145155 },
  { event := event145185
    frameStart := 145155 },
  { event := event145186
    frameStart := 145155 },
  { event := event145187
    frameStart := 145155 },
  { event := event145188
    frameStart := 145155 },
  { event := event145189
    frameStart := 145155 },
  { event := event145190
    frameStart := 145155 },
  { event := event145191
    frameStart := 145155 },
  { event := event145192
    frameStart := 145155 },
  { event := event145193
    frameStart := 145155 },
  { event := event145194
    frameStart := 145155 },
  { event := event145195
    frameStart := 145155 },
  { event := event145196
    frameStart := 145155 },
  { event := event145197
    frameStart := 145155 },
  { event := event145198
    frameStart := 145155 },
  { event := event145199
    frameStart := 145155 }
]

def eventLeaf9075 : Array AnnotatedEvent := #[
  { event := event145200
    frameStart := 145155 },
  { event := event145201
    frameStart := 145155 },
  { event := event145202
    frameStart := 145155 },
  { event := event145203
    frameStart := 145155 },
  { event := event145204
    frameStart := 145155 },
  { event := event145205
    frameStart := 145155 },
  { event := event145206
    frameStart := 145155 },
  { event := event145207
    frameStart := 145155 },
  { event := event145208
    frameStart := 145155 },
  { event := event145209
    frameStart := 145209 },
  { event := event145210
    frameStart := 145209 },
  { event := event145211
    frameStart := 145209 },
  { event := event145212
    frameStart := 145209 },
  { event := event145213
    frameStart := 145209 },
  { event := event145214
    frameStart := 145209 },
  { event := event145215
    frameStart := 145209 }
]

def eventLeaf9076 : Array AnnotatedEvent := #[
  { event := event145216
    frameStart := 145209 },
  { event := event145217
    frameStart := 145209 },
  { event := event145218
    frameStart := 145209 },
  { event := event145219
    frameStart := 145209 },
  { event := event145220
    frameStart := 145209 },
  { event := event145221
    frameStart := 145209 },
  { event := event145222
    frameStart := 145209 },
  { event := event145223
    frameStart := 145209 },
  { event := event145224
    frameStart := 145209 },
  { event := event145225
    frameStart := 145209 },
  { event := event145226
    frameStart := 145209 },
  { event := event145227
    frameStart := 145209 },
  { event := event145228
    frameStart := 145209 },
  { event := event145229
    frameStart := 145209 },
  { event := event145230
    frameStart := 145209 },
  { event := event145231
    frameStart := 145209 }
]

def eventLeaf9077 : Array AnnotatedEvent := #[
  { event := event145232
    frameStart := 145209 },
  { event := event145233
    frameStart := 145209 },
  { event := event145234
    frameStart := 145209 },
  { event := event145235
    frameStart := 145209 },
  { event := event145236
    frameStart := 145209 },
  { event := event145237
    frameStart := 145209 },
  { event := event145238
    frameStart := 145209 },
  { event := event145239
    frameStart := 145209 },
  { event := event145240
    frameStart := 145209 },
  { event := event145241
    frameStart := 145209 },
  { event := event145242
    frameStart := 145209 },
  { event := event145243
    frameStart := 145209 },
  { event := event145244
    frameStart := 145209 },
  { event := event145245
    frameStart := 145209 },
  { event := event145246
    frameStart := 145209 },
  { event := event145247
    frameStart := 145209 }
]

def eventLeaf9078 : Array AnnotatedEvent := #[
  { event := event145248
    frameStart := 145209 },
  { event := event145249
    frameStart := 145209 },
  { event := event145250
    frameStart := 145209 },
  { event := event145251
    frameStart := 145209 },
  { event := event145252
    frameStart := 145209 },
  { event := event145253
    frameStart := 145209 },
  { event := event145254
    frameStart := 145209 },
  { event := event145255
    frameStart := 145209 },
  { event := event145256
    frameStart := 145209 },
  { event := event145257
    frameStart := 145209 },
  { event := event145258
    frameStart := 145209 },
  { event := event145259
    frameStart := 145209 },
  { event := event145260
    frameStart := 145209 },
  { event := event145261
    frameStart := 145209 },
  { event := event145262
    frameStart := 145209 },
  { event := event145263
    frameStart := 145209 }
]

def eventLeaf9079 : Array AnnotatedEvent := #[
  { event := event145264
    frameStart := 145209 },
  { event := event145265
    frameStart := 145209 },
  { event := event145266
    frameStart := 145209 },
  { event := event145267
    frameStart := 145209 },
  { event := event145268
    frameStart := 145209 },
  { event := event145269
    frameStart := 145209 },
  { event := event145270
    frameStart := 145209 },
  { event := event145271
    frameStart := 145209 },
  { event := event145272
    frameStart := 145209 },
  { event := event145273
    frameStart := 145209 },
  { event := event145274
    frameStart := 145209 },
  { event := event145275
    frameStart := 145209 },
  { event := event145276
    frameStart := 145209 },
  { event := event145277
    frameStart := 145209 },
  { event := event145278
    frameStart := 145209 },
  { event := event145279
    frameStart := 145209 }
]

def eventLeaf9080 : Array AnnotatedEvent := #[
  { event := event145280
    frameStart := 145209 },
  { event := event145281
    frameStart := 145209 },
  { event := event145282
    frameStart := 145209 },
  { event := event145283
    frameStart := 145209 },
  { event := event145284
    frameStart := 145209 },
  { event := event145285
    frameStart := 145209 },
  { event := event145286
    frameStart := 145209 },
  { event := event145287
    frameStart := 145209 },
  { event := event145288
    frameStart := 145209 },
  { event := event145289
    frameStart := 145209 },
  { event := event145290
    frameStart := 145209 },
  { event := event145291
    frameStart := 145209 },
  { event := event145292
    frameStart := 145209 },
  { event := event145293
    frameStart := 145209 },
  { event := event145294
    frameStart := 145209 },
  { event := event145295
    frameStart := 145209 }
]

def eventLeaf9081 : Array AnnotatedEvent := #[
  { event := event145296
    frameStart := 145209 },
  { event := event145297
    frameStart := 145209 },
  { event := event145298
    frameStart := 145209 },
  { event := event145299
    frameStart := 145209 },
  { event := event145300
    frameStart := 145209 },
  { event := event145301
    frameStart := 145209 },
  { event := event145302
    frameStart := 145209 },
  { event := event145303
    frameStart := 145209 },
  { event := event145304
    frameStart := 145209 },
  { event := event145305
    frameStart := 145209 },
  { event := event145306
    frameStart := 145209 },
  { event := event145307
    frameStart := 145209 },
  { event := event145308
    frameStart := 145209 },
  { event := event145309
    frameStart := 145209 },
  { event := event145310
    frameStart := 145209 },
  { event := event145311
    frameStart := 145209 }
]

def eventLeaf9082 : Array AnnotatedEvent := #[
  { event := event145312
    frameStart := 145209 },
  { event := event145313
    frameStart := 0 },
  { event := event145314
    frameStart := 0 },
  { event := event145315
    frameStart := 0 },
  { event := event145316
    frameStart := 0 },
  { event := event145317
    frameStart := 0 },
  { event := event145318
    frameStart := 0 },
  { event := event145319
    frameStart := 0 },
  { event := event145320
    frameStart := 0 },
  { event := event145321
    frameStart := 0 },
  { event := event145322
    frameStart := 0 },
  { event := event145323
    frameStart := 0 },
  { event := event145324
    frameStart := 0 },
  { event := event145325
    frameStart := 0 },
  { event := event145326
    frameStart := 0 },
  { event := event145327
    frameStart := 0 }
]

def eventLeaf9083 : Array AnnotatedEvent := #[
  { event := event145328
    frameStart := 0 },
  { event := event145329
    frameStart := 0 },
  { event := event145330
    frameStart := 0 },
  { event := event145331
    frameStart := 0 },
  { event := event145332
    frameStart := 0 },
  { event := event145333
    frameStart := 0 },
  { event := event145334
    frameStart := 0 },
  { event := event145335
    frameStart := 0 },
  { event := event145336
    frameStart := 0 },
  { event := event145337
    frameStart := 0 },
  { event := event145338
    frameStart := 0 },
  { event := event145339
    frameStart := 0 },
  { event := event145340
    frameStart := 0 },
  { event := event145341
    frameStart := 0 },
  { event := event145342
    frameStart := 0 },
  { event := event145343
    frameStart := 0 }
]

def eventLeaf9084 : Array AnnotatedEvent := #[
  { event := event145344
    frameStart := 0 },
  { event := event145345
    frameStart := 0 },
  { event := event145346
    frameStart := 0 },
  { event := event145347
    frameStart := 0 },
  { event := event145348
    frameStart := 0 },
  { event := event145349
    frameStart := 0 },
  { event := event145350
    frameStart := 0 },
  { event := event145351
    frameStart := 0 },
  { event := event145352
    frameStart := 0 },
  { event := event145353
    frameStart := 0 },
  { event := event145354
    frameStart := 0 },
  { event := event145355
    frameStart := 0 },
  { event := event145356
    frameStart := 0 },
  { event := event145357
    frameStart := 0 },
  { event := event145358
    frameStart := 0 },
  { event := event145359
    frameStart := 0 }
]

def eventLeaf9085 : Array AnnotatedEvent := #[
  { event := event145360
    frameStart := 0 },
  { event := event145361
    frameStart := 0 },
  { event := event145362
    frameStart := 0 },
  { event := event145363
    frameStart := 0 },
  { event := event145364
    frameStart := 0 },
  { event := event145365
    frameStart := 0 },
  { event := event145366
    frameStart := 0 },
  { event := event145367
    frameStart := 145367 },
  { event := event145368
    frameStart := 145367 },
  { event := event145369
    frameStart := 145367 },
  { event := event145370
    frameStart := 145367 },
  { event := event145371
    frameStart := 145367 },
  { event := event145372
    frameStart := 145367 },
  { event := event145373
    frameStart := 145367 },
  { event := event145374
    frameStart := 145367 },
  { event := event145375
    frameStart := 145367 }
]

def eventLeaf9086 : Array AnnotatedEvent := #[
  { event := event145376
    frameStart := 145367 },
  { event := event145377
    frameStart := 145367 },
  { event := event145378
    frameStart := 145367 },
  { event := event145379
    frameStart := 145367 },
  { event := event145380
    frameStart := 145367 },
  { event := event145381
    frameStart := 145367 },
  { event := event145382
    frameStart := 145367 },
  { event := event145383
    frameStart := 145367 },
  { event := event145384
    frameStart := 145367 },
  { event := event145385
    frameStart := 145367 },
  { event := event145386
    frameStart := 145367 },
  { event := event145387
    frameStart := 145367 },
  { event := event145388
    frameStart := 145367 },
  { event := event145389
    frameStart := 145367 },
  { event := event145390
    frameStart := 145367 },
  { event := event145391
    frameStart := 145367 }
]

def eventLeaf9087 : Array AnnotatedEvent := #[
  { event := event145392
    frameStart := 145367 },
  { event := event145393
    frameStart := 145367 },
  { event := event145394
    frameStart := 145367 },
  { event := event145395
    frameStart := 145367 },
  { event := event145396
    frameStart := 145367 },
  { event := event145397
    frameStart := 145367 },
  { event := event145398
    frameStart := 145367 },
  { event := event145399
    frameStart := 145367 },
  { event := event145400
    frameStart := 145367 },
  { event := event145401
    frameStart := 145367 },
  { event := event145402
    frameStart := 145367 },
  { event := event145403
    frameStart := 145367 },
  { event := event145404
    frameStart := 145367 },
  { event := event145405
    frameStart := 145367 },
  { event := event145406
    frameStart := 145367 },
  { event := event145407
    frameStart := 145367 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events567
