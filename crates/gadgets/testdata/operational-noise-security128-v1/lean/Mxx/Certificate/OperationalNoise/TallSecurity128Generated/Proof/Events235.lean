import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events235

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event60160 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨22835⟩⟩, .operator (⟨46745, 0⟩, ⟨60154, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22832⟩⟩]⟩, (1)⟩)

def event60161 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨22833⟩⟩)

def event60162 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event60163 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event60164 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨11115⟩⟩) (.authority (.operator))

def event60165 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨11115⟩⟩) (.finite 17)

def event60166 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event60167 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event60168 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event60169 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event60170 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 60169

def event60171 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 60167

def event60172 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 60170 .coefficient) (.value (.predecessor 1 60171 .coefficient)))

def event60173 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event60174 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11117⟩⟩) 0 ⟨392⟩ 60173

def event60175 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11117⟩⟩) 1 ⟨11115⟩ 60165

def event60176 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨11117⟩⟩) (.sum [.predecessor 0 60174 .coefficient, .predecessor 1 60175 .coefficient])

def event60177 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨11117⟩⟩) (.finite 655357)

def event60178 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11173⟩⟩) 0 ⟨11117⟩ 60177

def event60179 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11173⟩⟩) 1 ⟨5426⟩ 60163

def event60180 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨11173⟩⟩) (.identity (.predecessor 1 60179 .coefficient))

def event60181 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨11173⟩⟩) (.finite 655360)

def event60182 : Event := .predecessor (⟨.program ⟨257⟩, ⟨21686⟩⟩) 0 ⟨11173⟩ 60181

def event60183 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨21686⟩⟩) (.authority (.programFamilyFact))

def exact60184RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨21686⟩⟩], []⟩, (1)⟩]

theorem exact60184RawTermsValid :
    exact60184RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event60184 : Event := .resultExact (⟨.program ⟨257⟩, ⟨21686⟩⟩) exact60184RawTerms (.finite 4) 60183 .exactZero (none)

def event60185 : Event := .predecessor (⟨.program ⟨257⟩, ⟨21221⟩⟩) 0 ⟨11173⟩ 60181

def event60186 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨21221⟩⟩) (.authority (.programFamilyFact))

def exact60187RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨21221⟩⟩], []⟩, (1)⟩]

theorem exact60187RawTermsValid :
    exact60187RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event60187 : Event := .resultExact (⟨.program ⟨257⟩, ⟨21221⟩⟩) exact60187RawTerms (.finite 4) 60186 .exactZero (none)

def event60188 : Event := .predecessor (⟨.program ⟨257⟩, ⟨21687⟩⟩) 0 ⟨21221⟩ 60187

def event60189 : Event := .predecessor (⟨.program ⟨257⟩, ⟨21687⟩⟩) 1 ⟨21686⟩ 60184

def event60190 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨21687⟩⟩) (.product (.predecessor 0 60188 .coefficient) (.predecessor 1 60189 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event60191 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨21687⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨21221⟩⟩, ⟨.program ⟨257⟩, ⟨21686⟩⟩], []⟩) [⟨.result 60187 .coefficient, true, some 1⟩, ⟨.result 60184 .coefficient, true, some 1⟩])

def event60192 : Event := .survivorFold (1) 60191

def exact60193RawTerms : List Term := []

theorem exact60193RawTermsValid :
    exact60193RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event60193 : Event := .resultExact (⟨.program ⟨257⟩, ⟨21687⟩⟩) exact60193RawTerms (.finite 16) 60190 (.finite 16) (some (60191))

def event60194 : Event := .predecessor (⟨.program ⟨257⟩, ⟨21688⟩⟩) 0 ⟨21687⟩ 60193

def event60195 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨21688⟩⟩) (.identity (.predecessor 0 60194 .coefficient))

def event60196 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨21688⟩⟩) (.finite 16)

def event60197 : Event := .predecessor (⟨.program ⟨257⟩, ⟨21872⟩⟩) 0 ⟨21688⟩ 60196

def event60198 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨21872⟩⟩) (.authority (.programFamilyFact))

def exact60199RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨21872⟩⟩], []⟩, (1)⟩]

theorem exact60199RawTermsValid :
    exact60199RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event60199 : Event := .resultExact (⟨.program ⟨257⟩, ⟨21872⟩⟩) exact60199RawTerms (.finite 4) 60198 .exactZero (none)

def event60200 : Event := .predecessor (⟨.program ⟨257⟩, ⟨21873⟩⟩) 0 ⟨21872⟩ 60199

def event60201 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨21873⟩⟩) (.identity (.predecessor 0 60200 .coefficient))

def event60202 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨21873⟩⟩) (.finite 4)

def event60203 : Event := .predecessor (⟨.program ⟨257⟩, ⟨22832⟩⟩) 0 ⟨21873⟩ 60202

def event60204 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨22832⟩⟩) (.authority (.relationPreimageSource ⟨60⟩))

def exact60205RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨22832⟩⟩]⟩, (1)⟩]

theorem exact60205RawTermsValid :
    exact60205RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event60205 : Event := .resultExact (⟨.program ⟨257⟩, ⟨22832⟩⟩) exact60205RawTerms (.finite 5647228698) 60204 .exactZero (none)

def event60206 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35⟩⟩) (.authority (.operator))

def exact60207RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩]⟩, (1)⟩]

theorem exact60207RawTermsValid :
    exact60207RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event60207 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35⟩⟩) exact60207RawTerms .large 60206 .exactZero (none)

def event60208 : Event := .predecessor (⟨.program ⟨257⟩, ⟨22833⟩⟩) 0 ⟨35⟩ 60207

def event60209 : Event := .predecessor (⟨.program ⟨257⟩, ⟨22833⟩⟩) 1 ⟨22832⟩ 60205

def event60210 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨22833⟩⟩) (.product (.predecessor 0 60208 .coefficient) (.predecessor 1 60209 .coefficient) (⟨false, false, none, none, none⟩))

def event60211 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨22833⟩⟩, .operator (⟨60207, 0⟩, ⟨60205, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22832⟩⟩]⟩, (1)⟩)

def exact60212RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22832⟩⟩]⟩, (1)⟩]

theorem exact60212RawTermsValid :
    exact60212RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event60212 : Event := .resultExact (⟨.program ⟨257⟩, ⟨22833⟩⟩) exact60212RawTerms .large 60210 .exactZero (none)

def event60213 : Event := .preFoldPolynomial 60212 [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22832⟩⟩]⟩, (1)⟩] .exactZero none

def exact60214RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22832⟩⟩]⟩, (1)⟩]

def event60214 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨22833⟩⟩) 60213 exact60214RawTerms .large 60210 .exactZero (none)

def event60215 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨24119⟩⟩)

def event60216 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event60217 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event60218 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨11115⟩⟩) (.authority (.operator))

def event60219 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨11115⟩⟩) (.finite 17)

def event60220 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event60221 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event60222 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event60223 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event60224 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 60223

def event60225 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 60221

def event60226 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 60224 .coefficient) (.value (.predecessor 1 60225 .coefficient)))

def event60227 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event60228 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11117⟩⟩) 0 ⟨392⟩ 60227

def event60229 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11117⟩⟩) 1 ⟨11115⟩ 60219

def event60230 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨11117⟩⟩) (.sum [.predecessor 0 60228 .coefficient, .predecessor 1 60229 .coefficient])

def event60231 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨11117⟩⟩) (.finite 655357)

def event60232 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11173⟩⟩) 0 ⟨11117⟩ 60231

def event60233 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11173⟩⟩) 1 ⟨5426⟩ 60217

def event60234 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨11173⟩⟩) (.identity (.predecessor 1 60233 .coefficient))

def event60235 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨11173⟩⟩) (.finite 655360)

def event60236 : Event := .predecessor (⟨.program ⟨257⟩, ⟨21686⟩⟩) 0 ⟨11173⟩ 60235

def event60237 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨21686⟩⟩) (.authority (.programFamilyFact))

def exact60238RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨21686⟩⟩], []⟩, (1)⟩]

theorem exact60238RawTermsValid :
    exact60238RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event60238 : Event := .resultExact (⟨.program ⟨257⟩, ⟨21686⟩⟩) exact60238RawTerms (.finite 4) 60237 .exactZero (none)

def event60239 : Event := .predecessor (⟨.program ⟨257⟩, ⟨21221⟩⟩) 0 ⟨11173⟩ 60235

def event60240 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨21221⟩⟩) (.authority (.programFamilyFact))

def exact60241RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨21221⟩⟩], []⟩, (1)⟩]

theorem exact60241RawTermsValid :
    exact60241RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event60241 : Event := .resultExact (⟨.program ⟨257⟩, ⟨21221⟩⟩) exact60241RawTerms (.finite 4) 60240 .exactZero (none)

def event60242 : Event := .predecessor (⟨.program ⟨257⟩, ⟨21687⟩⟩) 0 ⟨21221⟩ 60241

def event60243 : Event := .predecessor (⟨.program ⟨257⟩, ⟨21687⟩⟩) 1 ⟨21686⟩ 60238

def event60244 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨21687⟩⟩) (.product (.predecessor 0 60242 .coefficient) (.predecessor 1 60243 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event60245 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨21687⟩⟩, .operator (⟨60241, 0⟩, ⟨60238, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨21221⟩⟩, ⟨.program ⟨257⟩, ⟨21686⟩⟩], []⟩, (1)⟩)

def exact60246RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨21221⟩⟩, ⟨.program ⟨257⟩, ⟨21686⟩⟩], []⟩, (1)⟩]

theorem exact60246RawTermsValid :
    exact60246RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event60246 : Event := .resultExact (⟨.program ⟨257⟩, ⟨21687⟩⟩) exact60246RawTerms (.finite 16) 60244 .exactZero (none)

def event60247 : Event := .predecessor (⟨.program ⟨257⟩, ⟨21688⟩⟩) 0 ⟨21687⟩ 60246

def event60248 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨21688⟩⟩) (.identity (.predecessor 0 60247 .coefficient))

def event60249 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨21688⟩⟩) (.finite 16)

def event60250 : Event := .predecessor (⟨.program ⟨257⟩, ⟨21872⟩⟩) 0 ⟨21688⟩ 60249

def event60251 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨21872⟩⟩) (.authority (.programFamilyFact))

def exact60252RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨21872⟩⟩], []⟩, (1)⟩]

theorem exact60252RawTermsValid :
    exact60252RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event60252 : Event := .resultExact (⟨.program ⟨257⟩, ⟨21872⟩⟩) exact60252RawTerms (.finite 4) 60251 .exactZero (none)

def event60253 : Event := .predecessor (⟨.program ⟨257⟩, ⟨21873⟩⟩) 0 ⟨21872⟩ 60252

def event60254 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨21873⟩⟩) (.identity (.predecessor 0 60253 .coefficient))

def event60255 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨21873⟩⟩) (.finite 4)

def event60256 : Event := .predecessor (⟨.program ⟨257⟩, ⟨23151⟩⟩) 0 ⟨21873⟩ 60255

def event60257 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨23151⟩⟩) (.authority (.programFamilyFact))

def event60258 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨23151⟩⟩) (.finite 3720)

def event60259 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨7177⟩⟩) .missing

def event60260 : Event := .predecessor (⟨.program ⟨257⟩, ⟨23152⟩⟩) 0 ⟨7177⟩ 60259

def event60261 : Event := .predecessor (⟨.program ⟨257⟩, ⟨23152⟩⟩) 1 ⟨23151⟩ 60258

def event60262 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨23152⟩⟩) (.authority (.operator))

def exact60263RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨23152⟩⟩]⟩, (1)⟩]

theorem exact60263RawTermsValid :
    exact60263RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event60263 : Event := .resultExact (⟨.program ⟨257⟩, ⟨23152⟩⟩) exact60263RawTerms .large 60262 .exactZero (none)

def event60264 : Event := .predecessor (⟨.program ⟨257⟩, ⟨24113⟩⟩) 0 ⟨23152⟩ 60263

def event60265 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨24113⟩⟩) (.authority (.operator))

def exact60266RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨24113⟩⟩]⟩, (1)⟩]

theorem exact60266RawTermsValid :
    exact60266RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event60266 : Event := .resultExact (⟨.program ⟨257⟩, ⟨24113⟩⟩) exact60266RawTerms (.finite 8192) 60265 .exactZero (none)

def event60267 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨136⟩⟩) (.authority (.operator))

def event60268 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨136⟩⟩) .exactZero

def event60269 : Event := .predecessor (⟨.program ⟨257⟩, ⟨23318⟩⟩) 0 ⟨21873⟩ 60255

def event60270 : Event := .predecessor (⟨.program ⟨257⟩, ⟨23318⟩⟩) 1 ⟨136⟩ 60268

def event60271 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨23318⟩⟩) (.sum [.predecessor 0 60269 .coefficient, .predecessor 1 60270 .coefficient])

def event60272 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨23318⟩⟩) (.finite 4)

def event60273 : Event := .predecessor (⟨.program ⟨257⟩, ⟨23319⟩⟩) 0 ⟨23318⟩ 60272

def event60274 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨23319⟩⟩) (.identity (.predecessor 0 60273 .coefficient))

def exact60275RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨21872⟩⟩], []⟩, (1)⟩]

theorem exact60275RawTermsValid :
    exact60275RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event60275 : Event := .resultExact (⟨.program ⟨257⟩, ⟨23319⟩⟩) exact60275RawTerms (.finite 4) 60274 .exactZero (none)

def event60276 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6908⟩⟩) (.authority (.factStore))

def exact60277RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact60277RawTermsValid :
    exact60277RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event60277 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6908⟩⟩) exact60277RawTerms .large 60276 .exactZero (none)

def event60278 : Event := .predecessor (⟨.program ⟨257⟩, ⟨23320⟩⟩) 0 ⟨6908⟩ 60277

def event60279 : Event := .predecessor (⟨.program ⟨257⟩, ⟨23320⟩⟩) 1 ⟨23319⟩ 60275

def event60280 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨23320⟩⟩) (.product (.predecessor 0 60278 .coefficient) (.predecessor 1 60279 .coefficient) (⟨false, false, none, none, none⟩))

def event60281 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨23320⟩⟩, .operator (⟨60277, 0⟩, ⟨60275, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨21872⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact60282RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨21872⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact60282RawTermsValid :
    exact60282RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event60282 : Event := .resultExact (⟨.program ⟨257⟩, ⟨23320⟩⟩) exact60282RawTerms .large 60280 .exactZero (none)

def event60283 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7181⟩⟩) 0 ⟨7177⟩ 60259

def event60284 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7181⟩⟩) (.authority (.operator))

def exact60285RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7181⟩⟩]⟩, (1)⟩]

theorem exact60285RawTermsValid :
    exact60285RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event60285 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7181⟩⟩) exact60285RawTerms .large 60284 .exactZero (none)

def event60286 : Event := .predecessor (⟨.program ⟨257⟩, ⟨23321⟩⟩) 0 ⟨7181⟩ 60285

def event60287 : Event := .predecessor (⟨.program ⟨257⟩, ⟨23321⟩⟩) 1 ⟨23320⟩ 60282

def event60288 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨23321⟩⟩) (.sum [.predecessor 0 60286 .coefficient, .predecessor 1 60287 .coefficient])

def exact60289RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7181⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨21872⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact60289RawTermsValid :
    exact60289RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event60289 : Event := .resultExact (⟨.program ⟨257⟩, ⟨23321⟩⟩) exact60289RawTerms .large 60288 .exactZero (none)

def event60290 : Event := .predecessor (⟨.program ⟨257⟩, ⟨24114⟩⟩) 0 ⟨23321⟩ 60289

def event60291 : Event := .predecessor (⟨.program ⟨257⟩, ⟨24114⟩⟩) 1 ⟨24113⟩ 60266

def event60292 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨24114⟩⟩) (.product (.predecessor 0 60290 .coefficient) (.predecessor 1 60291 .coefficient) (⟨false, false, none, none, none⟩))

def event60293 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨24114⟩⟩, .operator (⟨60289, 0⟩, ⟨60266, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7181⟩⟩, ⟨.program ⟨257⟩, ⟨24113⟩⟩]⟩, (1)⟩)

def event60294 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨24114⟩⟩, .operator (⟨60289, 1⟩, ⟨60266, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨21872⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨24113⟩⟩]⟩, (-1)⟩)

def event60295 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨24114⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨21872⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨24113⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨24113⟩⟩) ⟨23152⟩ 60263)

def event60296 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨24114⟩⟩, .relation 60295 0, ⟨[⟨.program ⟨257⟩, ⟨21872⟩⟩], [⟨.program ⟨257⟩, ⟨23152⟩⟩]⟩, (-1)⟩)

def exact60297RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7181⟩⟩, ⟨.program ⟨257⟩, ⟨24113⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨21872⟩⟩], [⟨.program ⟨257⟩, ⟨23152⟩⟩]⟩, (-1)⟩]

theorem exact60297RawTermsValid :
    exact60297RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event60297 : Event := .resultExact (⟨.program ⟨257⟩, ⟨24114⟩⟩) exact60297RawTerms .large 60292 .exactZero (none)

def event60298 : Event := .predecessor (⟨.program ⟨257⟩, ⟨22233⟩⟩) 0 ⟨21873⟩ 60255

def event60299 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨22233⟩⟩) (.authority (.programFamilyFact))

def exact60300RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨22233⟩⟩], []⟩, (1)⟩]

theorem exact60300RawTermsValid :
    exact60300RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event60300 : Event := .resultExact (⟨.program ⟨257⟩, ⟨22233⟩⟩) exact60300RawTerms (.finite 4) 60299 .exactZero (none)

def event60301 : Event := .predecessor (⟨.program ⟨257⟩, ⟨22236⟩⟩) 0 ⟨6908⟩ 60277

def event60302 : Event := .predecessor (⟨.program ⟨257⟩, ⟨22236⟩⟩) 1 ⟨22233⟩ 60300

def event60303 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨22236⟩⟩) (.product (.predecessor 0 60301 .coefficient) (.predecessor 1 60302 .coefficient) (⟨false, true, none, none, some 1⟩))

def event60304 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨22236⟩⟩, .operator (⟨60277, 0⟩, ⟨60300, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨22233⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact60305RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨22233⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact60305RawTermsValid :
    exact60305RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event60305 : Event := .resultExact (⟨.program ⟨257⟩, ⟨22236⟩⟩) exact60305RawTerms .large 60303 .exactZero (none)

def event60306 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7201⟩⟩) 0 ⟨7177⟩ 60259

def event60307 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7201⟩⟩) (.authority (.operator))

def exact60308RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7201⟩⟩]⟩, (1)⟩]

theorem exact60308RawTermsValid :
    exact60308RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event60308 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7201⟩⟩) exact60308RawTerms .large 60307 .exactZero (none)

def event60309 : Event := .predecessor (⟨.program ⟨257⟩, ⟨22237⟩⟩) 0 ⟨7201⟩ 60308

def event60310 : Event := .predecessor (⟨.program ⟨257⟩, ⟨22237⟩⟩) 1 ⟨22236⟩ 60305

def event60311 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨22237⟩⟩) (.sum [.predecessor 0 60309 .coefficient, .predecessor 1 60310 .coefficient])

def exact60312RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7201⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨22233⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact60312RawTermsValid :
    exact60312RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event60312 : Event := .resultExact (⟨.program ⟨257⟩, ⟨22237⟩⟩) exact60312RawTerms .large 60311 .exactZero (none)

def event60313 : Event := .predecessor (⟨.program ⟨257⟩, ⟨24119⟩⟩) 0 ⟨22237⟩ 60312

def event60314 : Event := .predecessor (⟨.program ⟨257⟩, ⟨24119⟩⟩) 1 ⟨24114⟩ 60297

def event60315 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨24119⟩⟩) (.sum [.predecessor 0 60313 .coefficient, .predecessor 1 60314 .coefficient])

def exact60316RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7181⟩⟩, ⟨.program ⟨257⟩, ⟨24113⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7201⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨21872⟩⟩], [⟨.program ⟨257⟩, ⟨23152⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨22233⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact60316RawTermsValid :
    exact60316RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event60316 : Event := .resultExact (⟨.program ⟨257⟩, ⟨24119⟩⟩) exact60316RawTerms .large 60315 .exactZero (none)

def event60317 : Event := .preFoldPolynomial 60316 [⟨⟨[], [⟨.program ⟨257⟩, ⟨7181⟩⟩, ⟨.program ⟨257⟩, ⟨24113⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7201⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨21872⟩⟩], [⟨.program ⟨257⟩, ⟨23152⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨22233⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩] .exactZero none

def exact60318RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7181⟩⟩, ⟨.program ⟨257⟩, ⟨24113⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7201⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨21872⟩⟩], [⟨.program ⟨257⟩, ⟨23152⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨22233⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

def event60318 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨24119⟩⟩) 60317 exact60318RawTerms .large 60315 .exactZero (none)

def event60319 : Event := .specializationComputed (⟨.program ⟨257⟩, ⟨21873⟩⟩) ⟨⟨80⟩, ⟨60⟩, ⟨135⟩⟩ ⟨60161, 60319⟩

def event60320 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨22835⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22832⟩⟩]⟩) (1) 0 2 (.universal 60319 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22832⟩⟩]⟩) (none) 60318)

def event60321 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨22835⟩⟩, .relation 60320 1, ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩]⟩, (1)⟩)

def event60322 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨22835⟩⟩, .relation 60320 0, ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7181⟩⟩, ⟨.program ⟨257⟩, ⟨24113⟩⟩]⟩, (-1)⟩)

def event60323 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨22835⟩⟩, .relation 60320 2, ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨21872⟩⟩], [⟨.program ⟨257⟩, ⟨23152⟩⟩]⟩, (1)⟩)

def event60324 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨22835⟩⟩, .relation 60320 3, ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨22233⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact60325RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7181⟩⟩, ⟨.program ⟨257⟩, ⟨24113⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨21872⟩⟩], [⟨.program ⟨257⟩, ⟨23152⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨22233⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact60325RawTermsValid :
    exact60325RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event60325 : Event := .resultExact (⟨.program ⟨257⟩, ⟨22835⟩⟩) exact60325RawTerms .large 60157 (.finite 202072841853861888) (some (60159))

def event60326 : Event := .predecessor (⟨.program ⟨257⟩, ⟨24116⟩⟩) 0 ⟨22835⟩ 60325

def event60327 : Event := .predecessor (⟨.program ⟨257⟩, ⟨24116⟩⟩) 1 ⟨24115⟩ 60147

def event60328 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨24116⟩⟩) (.sum [.predecessor 0 60326 .coefficient, .predecessor 1 60327 .coefficient])

def event60329 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨24116⟩⟩, .operator (⟨60325, 0⟩, ⟨60147, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7181⟩⟩, ⟨.program ⟨257⟩, ⟨24113⟩⟩]⟩, (1)⟩)

def event60330 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨24116⟩⟩, .operator (⟨60325, 2⟩, ⟨60147, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨21872⟩⟩], [⟨.program ⟨257⟩, ⟨23152⟩⟩]⟩, (-1)⟩)

def event60331 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨24116⟩⟩) (.sum [.result 60325 .summary, .result 60147 .summary])

def exact60332RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨22233⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact60332RawTermsValid :
    exact60332RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event60332 : Event := .resultExact (⟨.program ⟨257⟩, ⟨24116⟩⟩) exact60332RawTerms .large 60328 (.finite 32189003662929394266751515230208) (some (60331))

def event60333 : Event := .predecessor (⟨.program ⟨257⟩, ⟨24117⟩⟩) 0 ⟨24116⟩ 60332

def event60334 : Event := .predecessor (⟨.program ⟨257⟩, ⟨24117⟩⟩) 1 ⟨7156⟩ 15842

def event60335 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨24117⟩⟩) (.product (.predecessor 0 60333 .coefficient) (.predecessor 1 60334 .coefficient) (⟨false, false, none, none, none⟩))

def event60336 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨24117⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩) [⟨.result 15838 .coefficient, false, none⟩])

def event60337 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨24117⟩⟩) (.product (.result 60332 .summary) (.transfer 60336) (⟨false, false, none, none, none⟩))

def event60338 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨24117⟩⟩, .operator (⟨60332, 0⟩, ⟨15842, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩, (1)⟩)

def event60339 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨24117⟩⟩, .operator (⟨60332, 1⟩, ⟨15842, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨22233⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩, (-1)⟩)

def event60340 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨24117⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨22233⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨7155⟩⟩) ⟨7043⟩ 15835)

def event60341 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨24117⟩⟩, .relation 60340 0, ⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨22233⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact60342RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨22233⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩, (1)⟩]

theorem exact60342RawTermsValid :
    exact60342RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event60342 : Event := .resultExact (⟨.program ⟨257⟩, ⟨24117⟩⟩) exact60342RawTerms .large 60335 (.finite 345626795057764889831969145180473178193920) (some (60337))

def event60343 : Event := .predecessor (⟨.program ⟨257⟩, ⟨19932⟩⟩) 0 ⟨7177⟩ 15500

def event60344 : Event := .predecessor (⟨.program ⟨257⟩, ⟨19932⟩⟩) 1 ⟨19931⟩ 54359

def event60345 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨19932⟩⟩) (.authority (.operator))

def exact60346RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨19932⟩⟩]⟩, (1)⟩]

theorem exact60346RawTermsValid :
    exact60346RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event60346 : Event := .resultExact (⟨.program ⟨257⟩, ⟨19932⟩⟩) exact60346RawTerms .large 60345 .exactZero (none)

def event60347 : Event := .predecessor (⟨.program ⟨257⟩, ⟨20893⟩⟩) 0 ⟨19932⟩ 60346

def event60348 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨20893⟩⟩) (.authority (.operator))

def exact60349RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨20893⟩⟩]⟩, (1)⟩]

theorem exact60349RawTermsValid :
    exact60349RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event60349 : Event := .resultExact (⟨.program ⟨257⟩, ⟨20893⟩⟩) exact60349RawTerms (.finite 8192) 60348 .exactZero (none)

def event60350 : Event := .predecessor (⟨.program ⟨257⟩, ⟨20895⟩⟩) 0 ⟨20309⟩ 54643

def event60351 : Event := .predecessor (⟨.program ⟨257⟩, ⟨20895⟩⟩) 1 ⟨20893⟩ 60349

def event60352 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨20895⟩⟩) (.product (.predecessor 0 60350 .coefficient) (.predecessor 1 60351 .coefficient) (⟨false, false, none, none, none⟩))

def event60353 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨20895⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨20893⟩⟩]⟩) [⟨.result 60349 .coefficient, false, none⟩])

def event60354 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨20895⟩⟩) (.product (.result 54643 .summary) (.transfer 60353) (⟨false, false, none, none, none⟩))

def event60355 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨20895⟩⟩, .operator (⟨54643, 0⟩, ⟨60349, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7180⟩⟩, ⟨.program ⟨257⟩, ⟨20893⟩⟩]⟩, (1)⟩)

def event60356 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨20895⟩⟩, .operator (⟨54643, 1⟩, ⟨60349, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨18652⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨20893⟩⟩]⟩, (-1)⟩)

def event60357 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨20895⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨18652⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨20893⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨20893⟩⟩) ⟨19932⟩ 60346)

def event60358 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨20895⟩⟩, .relation 60357 0, ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨18652⟩⟩], [⟨.program ⟨257⟩, ⟨19932⟩⟩]⟩, (-1)⟩)

def exact60359RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7180⟩⟩, ⟨.program ⟨257⟩, ⟨20893⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨18652⟩⟩], [⟨.program ⟨257⟩, ⟨19932⟩⟩]⟩, (-1)⟩]

theorem exact60359RawTermsValid :
    exact60359RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event60359 : Event := .resultExact (⟨.program ⟨257⟩, ⟨20895⟩⟩) exact60359RawTerms .large 60352 (.finite 32188905437706348505289216491520) (some (60354))

def event60360 : Event := .predecessor (⟨.program ⟨257⟩, ⟨19612⟩⟩) 0 ⟨18653⟩ 1975

def event60361 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨19612⟩⟩) (.authority (.relationPreimageSource ⟨58⟩))

def exact60362RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨19612⟩⟩]⟩, (1)⟩]

theorem exact60362RawTermsValid :
    exact60362RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event60362 : Event := .resultExact (⟨.program ⟨257⟩, ⟨19612⟩⟩) exact60362RawTerms (.finite 5647228698) 60361 .exactZero (none)

def event60363 : Event := .predecessor (⟨.program ⟨257⟩, ⟨19614⟩⟩) 0 ⟨19612⟩ 60362

def event60364 : Event := .predecessor (⟨.program ⟨257⟩, ⟨19614⟩⟩) 1 ⟨2370⟩ 4

def event60365 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨19614⟩⟩) (.scale (.predecessor 0 60363 .coefficient) (.value (.predecessor 1 60364 .coefficient)))

def exact60366RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨19612⟩⟩]⟩, (1)⟩]

theorem exact60366RawTermsValid :
    exact60366RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event60366 : Event := .resultExact (⟨.program ⟨257⟩, ⟨19614⟩⟩) exact60366RawTerms (.finite 5647228698) 60365 .exactZero (none)

def event60367 : Event := .predecessor (⟨.program ⟨257⟩, ⟨19615⟩⟩) 0 ⟨11216⟩ 46745

def event60368 : Event := .predecessor (⟨.program ⟨257⟩, ⟨19615⟩⟩) 1 ⟨19614⟩ 60366

def event60369 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨19615⟩⟩) (.product (.predecessor 0 60367 .coefficient) (.predecessor 1 60368 .coefficient) (⟨false, false, none, none, none⟩))

def event60370 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨19615⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨19612⟩⟩]⟩) [⟨.result 60362 .coefficient, false, none⟩])

def event60371 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨19615⟩⟩) (.product (.result 46745 .summary) (.transfer 60370) (⟨false, false, none, none, none⟩))

def event60372 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨19615⟩⟩, .operator (⟨46745, 0⟩, ⟨60366, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19612⟩⟩]⟩, (1)⟩)

def event60373 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨19613⟩⟩)

def event60374 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event60375 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event60376 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨11115⟩⟩) (.authority (.operator))

def event60377 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨11115⟩⟩) (.finite 17)

def event60378 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event60379 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event60380 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event60381 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event60382 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 60381

def event60383 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 60379

def event60384 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 60382 .coefficient) (.value (.predecessor 1 60383 .coefficient)))

def event60385 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event60386 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11117⟩⟩) 0 ⟨392⟩ 60385

def event60387 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11117⟩⟩) 1 ⟨11115⟩ 60377

def event60388 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨11117⟩⟩) (.sum [.predecessor 0 60386 .coefficient, .predecessor 1 60387 .coefficient])

def event60389 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨11117⟩⟩) (.finite 655357)

def event60390 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11173⟩⟩) 0 ⟨11117⟩ 60389

def event60391 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11173⟩⟩) 1 ⟨5426⟩ 60375

def event60392 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨11173⟩⟩) (.identity (.predecessor 1 60391 .coefficient))

def event60393 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨11173⟩⟩) (.finite 655360)

def event60394 : Event := .predecessor (⟨.program ⟨257⟩, ⟨18466⟩⟩) 0 ⟨11173⟩ 60393

def event60395 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨18466⟩⟩) (.authority (.programFamilyFact))

def exact60396RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨18466⟩⟩], []⟩, (1)⟩]

theorem exact60396RawTermsValid :
    exact60396RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event60396 : Event := .resultExact (⟨.program ⟨257⟩, ⟨18466⟩⟩) exact60396RawTerms (.finite 3) 60395 .exactZero (none)

def event60397 : Event := .predecessor (⟨.program ⟨257⟩, ⟨12801⟩⟩) 0 ⟨11173⟩ 60393

def event60398 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨12801⟩⟩) (.authority (.programFamilyFact))

def exact60399RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨12801⟩⟩], []⟩, (1)⟩]

theorem exact60399RawTermsValid :
    exact60399RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event60399 : Event := .resultExact (⟨.program ⟨257⟩, ⟨12801⟩⟩) exact60399RawTerms (.finite 3) 60398 .exactZero (none)

def event60400 : Event := .predecessor (⟨.program ⟨257⟩, ⟨18467⟩⟩) 0 ⟨12801⟩ 60399

def event60401 : Event := .predecessor (⟨.program ⟨257⟩, ⟨18467⟩⟩) 1 ⟨18466⟩ 60396

def event60402 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨18467⟩⟩) (.product (.predecessor 0 60400 .coefficient) (.predecessor 1 60401 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event60403 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨18467⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨12801⟩⟩, ⟨.program ⟨257⟩, ⟨18466⟩⟩], []⟩) [⟨.result 60399 .coefficient, true, some 1⟩, ⟨.result 60396 .coefficient, true, some 1⟩])

def event60404 : Event := .survivorFold (1) 60403

def exact60405RawTerms : List Term := []

theorem exact60405RawTermsValid :
    exact60405RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event60405 : Event := .resultExact (⟨.program ⟨257⟩, ⟨18467⟩⟩) exact60405RawTerms (.finite 9) 60402 (.finite 9) (some (60403))

def event60406 : Event := .predecessor (⟨.program ⟨257⟩, ⟨18468⟩⟩) 0 ⟨18467⟩ 60405

def event60407 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨18468⟩⟩) (.identity (.predecessor 0 60406 .coefficient))

def event60408 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨18468⟩⟩) (.finite 9)

def event60409 : Event := .predecessor (⟨.program ⟨257⟩, ⟨18652⟩⟩) 0 ⟨18468⟩ 60408

def event60410 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨18652⟩⟩) (.authority (.programFamilyFact))

def exact60411RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨18652⟩⟩], []⟩, (1)⟩]

theorem exact60411RawTermsValid :
    exact60411RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event60411 : Event := .resultExact (⟨.program ⟨257⟩, ⟨18652⟩⟩) exact60411RawTerms (.finite 3) 60410 .exactZero (none)

def event60412 : Event := .predecessor (⟨.program ⟨257⟩, ⟨18653⟩⟩) 0 ⟨18652⟩ 60411

def event60413 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨18653⟩⟩) (.identity (.predecessor 0 60412 .coefficient))

def event60414 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨18653⟩⟩) (.finite 3)

def event60415 : Event := .predecessor (⟨.program ⟨257⟩, ⟨19612⟩⟩) 0 ⟨18653⟩ 60414

def eventLeaf3760 : Array AnnotatedEvent := #[
  { event := event60160
    frameStart := 0 },
  { event := event60161
    frameStart := 60161 },
  { event := event60162
    frameStart := 60161 },
  { event := event60163
    frameStart := 60161 },
  { event := event60164
    frameStart := 60161 },
  { event := event60165
    frameStart := 60161 },
  { event := event60166
    frameStart := 60161 },
  { event := event60167
    frameStart := 60161 },
  { event := event60168
    frameStart := 60161 },
  { event := event60169
    frameStart := 60161 },
  { event := event60170
    frameStart := 60161 },
  { event := event60171
    frameStart := 60161 },
  { event := event60172
    frameStart := 60161 },
  { event := event60173
    frameStart := 60161 },
  { event := event60174
    frameStart := 60161 },
  { event := event60175
    frameStart := 60161 }
]

def eventLeaf3761 : Array AnnotatedEvent := #[
  { event := event60176
    frameStart := 60161 },
  { event := event60177
    frameStart := 60161 },
  { event := event60178
    frameStart := 60161 },
  { event := event60179
    frameStart := 60161 },
  { event := event60180
    frameStart := 60161 },
  { event := event60181
    frameStart := 60161 },
  { event := event60182
    frameStart := 60161 },
  { event := event60183
    frameStart := 60161 },
  { event := event60184
    frameStart := 60161 },
  { event := event60185
    frameStart := 60161 },
  { event := event60186
    frameStart := 60161 },
  { event := event60187
    frameStart := 60161 },
  { event := event60188
    frameStart := 60161 },
  { event := event60189
    frameStart := 60161 },
  { event := event60190
    frameStart := 60161 },
  { event := event60191
    frameStart := 60161 }
]

def eventLeaf3762 : Array AnnotatedEvent := #[
  { event := event60192
    frameStart := 60161 },
  { event := event60193
    frameStart := 60161 },
  { event := event60194
    frameStart := 60161 },
  { event := event60195
    frameStart := 60161 },
  { event := event60196
    frameStart := 60161 },
  { event := event60197
    frameStart := 60161 },
  { event := event60198
    frameStart := 60161 },
  { event := event60199
    frameStart := 60161 },
  { event := event60200
    frameStart := 60161 },
  { event := event60201
    frameStart := 60161 },
  { event := event60202
    frameStart := 60161 },
  { event := event60203
    frameStart := 60161 },
  { event := event60204
    frameStart := 60161 },
  { event := event60205
    frameStart := 60161 },
  { event := event60206
    frameStart := 60161 },
  { event := event60207
    frameStart := 60161 }
]

def eventLeaf3763 : Array AnnotatedEvent := #[
  { event := event60208
    frameStart := 60161 },
  { event := event60209
    frameStart := 60161 },
  { event := event60210
    frameStart := 60161 },
  { event := event60211
    frameStart := 60161 },
  { event := event60212
    frameStart := 60161 },
  { event := event60213
    frameStart := 60161 },
  { event := event60214
    frameStart := 60161 },
  { event := event60215
    frameStart := 60215 },
  { event := event60216
    frameStart := 60215 },
  { event := event60217
    frameStart := 60215 },
  { event := event60218
    frameStart := 60215 },
  { event := event60219
    frameStart := 60215 },
  { event := event60220
    frameStart := 60215 },
  { event := event60221
    frameStart := 60215 },
  { event := event60222
    frameStart := 60215 },
  { event := event60223
    frameStart := 60215 }
]

def eventLeaf3764 : Array AnnotatedEvent := #[
  { event := event60224
    frameStart := 60215 },
  { event := event60225
    frameStart := 60215 },
  { event := event60226
    frameStart := 60215 },
  { event := event60227
    frameStart := 60215 },
  { event := event60228
    frameStart := 60215 },
  { event := event60229
    frameStart := 60215 },
  { event := event60230
    frameStart := 60215 },
  { event := event60231
    frameStart := 60215 },
  { event := event60232
    frameStart := 60215 },
  { event := event60233
    frameStart := 60215 },
  { event := event60234
    frameStart := 60215 },
  { event := event60235
    frameStart := 60215 },
  { event := event60236
    frameStart := 60215 },
  { event := event60237
    frameStart := 60215 },
  { event := event60238
    frameStart := 60215 },
  { event := event60239
    frameStart := 60215 }
]

def eventLeaf3765 : Array AnnotatedEvent := #[
  { event := event60240
    frameStart := 60215 },
  { event := event60241
    frameStart := 60215 },
  { event := event60242
    frameStart := 60215 },
  { event := event60243
    frameStart := 60215 },
  { event := event60244
    frameStart := 60215 },
  { event := event60245
    frameStart := 60215 },
  { event := event60246
    frameStart := 60215 },
  { event := event60247
    frameStart := 60215 },
  { event := event60248
    frameStart := 60215 },
  { event := event60249
    frameStart := 60215 },
  { event := event60250
    frameStart := 60215 },
  { event := event60251
    frameStart := 60215 },
  { event := event60252
    frameStart := 60215 },
  { event := event60253
    frameStart := 60215 },
  { event := event60254
    frameStart := 60215 },
  { event := event60255
    frameStart := 60215 }
]

def eventLeaf3766 : Array AnnotatedEvent := #[
  { event := event60256
    frameStart := 60215 },
  { event := event60257
    frameStart := 60215 },
  { event := event60258
    frameStart := 60215 },
  { event := event60259
    frameStart := 60215 },
  { event := event60260
    frameStart := 60215 },
  { event := event60261
    frameStart := 60215 },
  { event := event60262
    frameStart := 60215 },
  { event := event60263
    frameStart := 60215 },
  { event := event60264
    frameStart := 60215 },
  { event := event60265
    frameStart := 60215 },
  { event := event60266
    frameStart := 60215 },
  { event := event60267
    frameStart := 60215 },
  { event := event60268
    frameStart := 60215 },
  { event := event60269
    frameStart := 60215 },
  { event := event60270
    frameStart := 60215 },
  { event := event60271
    frameStart := 60215 }
]

def eventLeaf3767 : Array AnnotatedEvent := #[
  { event := event60272
    frameStart := 60215 },
  { event := event60273
    frameStart := 60215 },
  { event := event60274
    frameStart := 60215 },
  { event := event60275
    frameStart := 60215 },
  { event := event60276
    frameStart := 60215 },
  { event := event60277
    frameStart := 60215 },
  { event := event60278
    frameStart := 60215 },
  { event := event60279
    frameStart := 60215 },
  { event := event60280
    frameStart := 60215 },
  { event := event60281
    frameStart := 60215 },
  { event := event60282
    frameStart := 60215 },
  { event := event60283
    frameStart := 60215 },
  { event := event60284
    frameStart := 60215 },
  { event := event60285
    frameStart := 60215 },
  { event := event60286
    frameStart := 60215 },
  { event := event60287
    frameStart := 60215 }
]

def eventLeaf3768 : Array AnnotatedEvent := #[
  { event := event60288
    frameStart := 60215 },
  { event := event60289
    frameStart := 60215 },
  { event := event60290
    frameStart := 60215 },
  { event := event60291
    frameStart := 60215 },
  { event := event60292
    frameStart := 60215 },
  { event := event60293
    frameStart := 60215 },
  { event := event60294
    frameStart := 60215 },
  { event := event60295
    frameStart := 60215 },
  { event := event60296
    frameStart := 60215 },
  { event := event60297
    frameStart := 60215 },
  { event := event60298
    frameStart := 60215 },
  { event := event60299
    frameStart := 60215 },
  { event := event60300
    frameStart := 60215 },
  { event := event60301
    frameStart := 60215 },
  { event := event60302
    frameStart := 60215 },
  { event := event60303
    frameStart := 60215 }
]

def eventLeaf3769 : Array AnnotatedEvent := #[
  { event := event60304
    frameStart := 60215 },
  { event := event60305
    frameStart := 60215 },
  { event := event60306
    frameStart := 60215 },
  { event := event60307
    frameStart := 60215 },
  { event := event60308
    frameStart := 60215 },
  { event := event60309
    frameStart := 60215 },
  { event := event60310
    frameStart := 60215 },
  { event := event60311
    frameStart := 60215 },
  { event := event60312
    frameStart := 60215 },
  { event := event60313
    frameStart := 60215 },
  { event := event60314
    frameStart := 60215 },
  { event := event60315
    frameStart := 60215 },
  { event := event60316
    frameStart := 60215 },
  { event := event60317
    frameStart := 60215 },
  { event := event60318
    frameStart := 60215 },
  { event := event60319
    frameStart := 0 }
]

def eventLeaf3770 : Array AnnotatedEvent := #[
  { event := event60320
    frameStart := 0 },
  { event := event60321
    frameStart := 0 },
  { event := event60322
    frameStart := 0 },
  { event := event60323
    frameStart := 0 },
  { event := event60324
    frameStart := 0 },
  { event := event60325
    frameStart := 0 },
  { event := event60326
    frameStart := 0 },
  { event := event60327
    frameStart := 0 },
  { event := event60328
    frameStart := 0 },
  { event := event60329
    frameStart := 0 },
  { event := event60330
    frameStart := 0 },
  { event := event60331
    frameStart := 0 },
  { event := event60332
    frameStart := 0 },
  { event := event60333
    frameStart := 0 },
  { event := event60334
    frameStart := 0 },
  { event := event60335
    frameStart := 0 }
]

def eventLeaf3771 : Array AnnotatedEvent := #[
  { event := event60336
    frameStart := 0 },
  { event := event60337
    frameStart := 0 },
  { event := event60338
    frameStart := 0 },
  { event := event60339
    frameStart := 0 },
  { event := event60340
    frameStart := 0 },
  { event := event60341
    frameStart := 0 },
  { event := event60342
    frameStart := 0 },
  { event := event60343
    frameStart := 0 },
  { event := event60344
    frameStart := 0 },
  { event := event60345
    frameStart := 0 },
  { event := event60346
    frameStart := 0 },
  { event := event60347
    frameStart := 0 },
  { event := event60348
    frameStart := 0 },
  { event := event60349
    frameStart := 0 },
  { event := event60350
    frameStart := 0 },
  { event := event60351
    frameStart := 0 }
]

def eventLeaf3772 : Array AnnotatedEvent := #[
  { event := event60352
    frameStart := 0 },
  { event := event60353
    frameStart := 0 },
  { event := event60354
    frameStart := 0 },
  { event := event60355
    frameStart := 0 },
  { event := event60356
    frameStart := 0 },
  { event := event60357
    frameStart := 0 },
  { event := event60358
    frameStart := 0 },
  { event := event60359
    frameStart := 0 },
  { event := event60360
    frameStart := 0 },
  { event := event60361
    frameStart := 0 },
  { event := event60362
    frameStart := 0 },
  { event := event60363
    frameStart := 0 },
  { event := event60364
    frameStart := 0 },
  { event := event60365
    frameStart := 0 },
  { event := event60366
    frameStart := 0 },
  { event := event60367
    frameStart := 0 }
]

def eventLeaf3773 : Array AnnotatedEvent := #[
  { event := event60368
    frameStart := 0 },
  { event := event60369
    frameStart := 0 },
  { event := event60370
    frameStart := 0 },
  { event := event60371
    frameStart := 0 },
  { event := event60372
    frameStart := 0 },
  { event := event60373
    frameStart := 60373 },
  { event := event60374
    frameStart := 60373 },
  { event := event60375
    frameStart := 60373 },
  { event := event60376
    frameStart := 60373 },
  { event := event60377
    frameStart := 60373 },
  { event := event60378
    frameStart := 60373 },
  { event := event60379
    frameStart := 60373 },
  { event := event60380
    frameStart := 60373 },
  { event := event60381
    frameStart := 60373 },
  { event := event60382
    frameStart := 60373 },
  { event := event60383
    frameStart := 60373 }
]

def eventLeaf3774 : Array AnnotatedEvent := #[
  { event := event60384
    frameStart := 60373 },
  { event := event60385
    frameStart := 60373 },
  { event := event60386
    frameStart := 60373 },
  { event := event60387
    frameStart := 60373 },
  { event := event60388
    frameStart := 60373 },
  { event := event60389
    frameStart := 60373 },
  { event := event60390
    frameStart := 60373 },
  { event := event60391
    frameStart := 60373 },
  { event := event60392
    frameStart := 60373 },
  { event := event60393
    frameStart := 60373 },
  { event := event60394
    frameStart := 60373 },
  { event := event60395
    frameStart := 60373 },
  { event := event60396
    frameStart := 60373 },
  { event := event60397
    frameStart := 60373 },
  { event := event60398
    frameStart := 60373 },
  { event := event60399
    frameStart := 60373 }
]

def eventLeaf3775 : Array AnnotatedEvent := #[
  { event := event60400
    frameStart := 60373 },
  { event := event60401
    frameStart := 60373 },
  { event := event60402
    frameStart := 60373 },
  { event := event60403
    frameStart := 60373 },
  { event := event60404
    frameStart := 60373 },
  { event := event60405
    frameStart := 60373 },
  { event := event60406
    frameStart := 60373 },
  { event := event60407
    frameStart := 60373 },
  { event := event60408
    frameStart := 60373 },
  { event := event60409
    frameStart := 60373 },
  { event := event60410
    frameStart := 60373 },
  { event := event60411
    frameStart := 60373 },
  { event := event60412
    frameStart := 60373 },
  { event := event60413
    frameStart := 60373 },
  { event := event60414
    frameStart := 60373 },
  { event := event60415
    frameStart := 60373 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events235
