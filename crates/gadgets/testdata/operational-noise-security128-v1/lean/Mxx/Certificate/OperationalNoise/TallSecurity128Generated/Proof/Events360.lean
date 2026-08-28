import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events360

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event92160 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41139⟩⟩) 0 ⟨7177⟩ 92159

def event92161 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41139⟩⟩) 1 ⟨41138⟩ 92158

def event92162 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41139⟩⟩) (.authority (.operator))

def exact92163RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨41139⟩⟩]⟩, (1)⟩]

theorem exact92163RawTermsValid :
    exact92163RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event92163 : Event := .resultExact (⟨.program ⟨257⟩, ⟨41139⟩⟩) exact92163RawTerms .large 92162 .exactZero (none)

def event92164 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41674⟩⟩) 0 ⟨41139⟩ 92163

def event92165 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41674⟩⟩) (.authority (.operator))

def exact92166RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨41674⟩⟩]⟩, (1)⟩]

theorem exact92166RawTermsValid :
    exact92166RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event92166 : Event := .resultExact (⟨.program ⟨257⟩, ⟨41674⟩⟩) exact92166RawTerms (.finite 8192) 92165 .exactZero (none)

def event92167 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨136⟩⟩) (.authority (.operator))

def event92168 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨136⟩⟩) .exactZero

def event92169 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41406⟩⟩) 0 ⟨39916⟩ 92155

def event92170 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41406⟩⟩) 1 ⟨136⟩ 92168

def event92171 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41406⟩⟩) (.sum [.predecessor 0 92169 .coefficient, .predecessor 1 92170 .coefficient])

def event92172 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨41406⟩⟩) (.finite 2116)

def event92173 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41407⟩⟩) 0 ⟨41406⟩ 92172

def event92174 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41407⟩⟩) (.identity (.predecessor 0 92173 .coefficient))

def exact92175RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14256⟩⟩, ⟨.program ⟨257⟩, ⟨39914⟩⟩], []⟩, (1)⟩]

theorem exact92175RawTermsValid :
    exact92175RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event92175 : Event := .resultExact (⟨.program ⟨257⟩, ⟨41407⟩⟩) exact92175RawTerms (.finite 2116) 92174 .exactZero (none)

def event92176 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6908⟩⟩) (.authority (.factStore))

def exact92177RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact92177RawTermsValid :
    exact92177RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event92177 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6908⟩⟩) exact92177RawTerms .large 92176 .exactZero (none)

def event92178 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41408⟩⟩) 0 ⟨6908⟩ 92177

def event92179 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41408⟩⟩) 1 ⟨41407⟩ 92175

def event92180 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41408⟩⟩) (.product (.predecessor 0 92178 .coefficient) (.predecessor 1 92179 .coefficient) (⟨false, false, none, none, none⟩))

def event92181 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨41408⟩⟩, .operator (⟨92177, 0⟩, ⟨92175, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨14256⟩⟩, ⟨.program ⟨257⟩, ⟨39914⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact92182RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14256⟩⟩, ⟨.program ⟨257⟩, ⟨39914⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact92182RawTermsValid :
    exact92182RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event92182 : Event := .resultExact (⟨.program ⟨257⟩, ⟨41408⟩⟩) exact92182RawTerms .large 92180 .exactZero (none)

def event92183 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨2370⟩⟩) (.authority (.operator))

def event92184 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨2370⟩⟩) (.finite 1)

def event92185 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7178⟩⟩) 0 ⟨7177⟩ 92159

def event92186 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7178⟩⟩) (.authority (.operator))

def exact92187RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7178⟩⟩]⟩, (1)⟩]

theorem exact92187RawTermsValid :
    exact92187RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event92187 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7178⟩⟩) exact92187RawTerms .large 92186 .exactZero (none)

def event92188 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7282⟩⟩) 0 ⟨7178⟩ 92187

def event92189 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7282⟩⟩) (.identity (.predecessor 0 92188 .coefficient))

def exact92190RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7282⟩⟩]⟩, (1)⟩]

theorem exact92190RawTermsValid :
    exact92190RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event92190 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7282⟩⟩) exact92190RawTerms .large 92189 .exactZero (none)

def event92191 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9556⟩⟩) 0 ⟨7282⟩ 92190

def event92192 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9556⟩⟩) (.authority (.operator))

def exact92193RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨9556⟩⟩]⟩, (1)⟩]

theorem exact92193RawTermsValid :
    exact92193RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event92193 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9556⟩⟩) exact92193RawTerms (.finite 8192) 92192 .exactZero (none)

def event92194 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9557⟩⟩) 0 ⟨9556⟩ 92193

def event92195 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9557⟩⟩) 1 ⟨2370⟩ 92184

def event92196 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9557⟩⟩) (.scale (.predecessor 0 92194 .coefficient) (.value (.predecessor 1 92195 .coefficient)))

def exact92197RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨9556⟩⟩]⟩, (1)⟩]

theorem exact92197RawTermsValid :
    exact92197RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event92197 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9557⟩⟩) exact92197RawTerms (.finite 8192) 92196 .exactZero (none)

def event92198 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7299⟩⟩) 0 ⟨7178⟩ 92187

def event92199 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7299⟩⟩) (.identity (.predecessor 0 92198 .coefficient))

def exact92200RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7299⟩⟩]⟩, (1)⟩]

theorem exact92200RawTermsValid :
    exact92200RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event92200 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7299⟩⟩) exact92200RawTerms .large 92199 .exactZero (none)

def event92201 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9558⟩⟩) 0 ⟨7299⟩ 92200

def event92202 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9558⟩⟩) 1 ⟨9557⟩ 92197

def event92203 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9558⟩⟩) (.product (.predecessor 0 92201 .coefficient) (.predecessor 1 92202 .coefficient) (⟨false, false, none, none, none⟩))

def event92204 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨9558⟩⟩, .operator (⟨92200, 0⟩, ⟨92197, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩]⟩, (1)⟩)

def exact92205RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩]⟩, (1)⟩]

theorem exact92205RawTermsValid :
    exact92205RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event92205 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9558⟩⟩) exact92205RawTerms .large 92203 .exactZero (none)

def event92206 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41409⟩⟩) 0 ⟨9558⟩ 92205

def event92207 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41409⟩⟩) 1 ⟨41408⟩ 92182

def event92208 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41409⟩⟩) (.sum [.predecessor 0 92206 .coefficient, .predecessor 1 92207 .coefficient])

def exact92209RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨14256⟩⟩, ⟨.program ⟨257⟩, ⟨39914⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact92209RawTermsValid :
    exact92209RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event92209 : Event := .resultExact (⟨.program ⟨257⟩, ⟨41409⟩⟩) exact92209RawTerms .large 92208 .exactZero (none)

def event92210 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41677⟩⟩) 0 ⟨41409⟩ 92209

def event92211 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41677⟩⟩) 1 ⟨41674⟩ 92166

def event92212 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41677⟩⟩) (.product (.predecessor 0 92210 .coefficient) (.predecessor 1 92211 .coefficient) (⟨false, false, none, none, none⟩))

def event92213 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨41677⟩⟩, .operator (⟨92209, 0⟩, ⟨92166, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41674⟩⟩]⟩, (1)⟩)

def event92214 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨41677⟩⟩, .operator (⟨92209, 1⟩, ⟨92166, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨14256⟩⟩, ⟨.program ⟨257⟩, ⟨39914⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨41674⟩⟩]⟩, (-1)⟩)

def event92215 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨41677⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨14256⟩⟩, ⟨.program ⟨257⟩, ⟨39914⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨41674⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨41674⟩⟩) ⟨41139⟩ 92163)

def event92216 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨41677⟩⟩, .relation 92215 0, ⟨[⟨.program ⟨257⟩, ⟨14256⟩⟩, ⟨.program ⟨257⟩, ⟨39914⟩⟩], [⟨.program ⟨257⟩, ⟨41139⟩⟩]⟩, (-1)⟩)

def exact92217RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41674⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨14256⟩⟩, ⟨.program ⟨257⟩, ⟨39914⟩⟩], [⟨.program ⟨257⟩, ⟨41139⟩⟩]⟩, (-1)⟩]

theorem exact92217RawTermsValid :
    exact92217RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event92217 : Event := .resultExact (⟨.program ⟨257⟩, ⟨41677⟩⟩) exact92217RawTerms .large 92212 .exactZero (none)

def event92218 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40148⟩⟩) 0 ⟨39916⟩ 92155

def event92219 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨40148⟩⟩) (.authority (.programFamilyFact))

def exact92220RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨40148⟩⟩], []⟩, (1)⟩]

theorem exact92220RawTermsValid :
    exact92220RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event92220 : Event := .resultExact (⟨.program ⟨257⟩, ⟨40148⟩⟩) exact92220RawTerms (.finite 46) 92219 .exactZero (none)

def event92221 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40150⟩⟩) 0 ⟨6908⟩ 92177

def event92222 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40150⟩⟩) 1 ⟨40148⟩ 92220

def event92223 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨40150⟩⟩) (.product (.predecessor 0 92221 .coefficient) (.predecessor 1 92222 .coefficient) (⟨false, true, none, none, some 1⟩))

def event92224 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨40150⟩⟩, .operator (⟨92177, 0⟩, ⟨92220, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨40148⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact92225RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨40148⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact92225RawTermsValid :
    exact92225RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event92225 : Event := .resultExact (⟨.program ⟨257⟩, ⟨40150⟩⟩) exact92225RawTerms .large 92223 .exactZero (none)

def event92226 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7193⟩⟩) 0 ⟨7177⟩ 92159

def event92227 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7193⟩⟩) (.authority (.operator))

def exact92228RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7193⟩⟩]⟩, (1)⟩]

theorem exact92228RawTermsValid :
    exact92228RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event92228 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7193⟩⟩) exact92228RawTerms .large 92227 .exactZero (none)

def event92229 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40151⟩⟩) 0 ⟨7193⟩ 92228

def event92230 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40151⟩⟩) 1 ⟨40150⟩ 92225

def event92231 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨40151⟩⟩) (.sum [.predecessor 0 92229 .coefficient, .predecessor 1 92230 .coefficient])

def exact92232RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7193⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨40148⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact92232RawTermsValid :
    exact92232RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event92232 : Event := .resultExact (⟨.program ⟨257⟩, ⟨40151⟩⟩) exact92232RawTerms .large 92231 .exactZero (none)

def event92233 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41678⟩⟩) 0 ⟨40151⟩ 92232

def event92234 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41678⟩⟩) 1 ⟨41677⟩ 92217

def event92235 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41678⟩⟩) (.sum [.predecessor 0 92233 .coefficient, .predecessor 1 92234 .coefficient])

def exact92236RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7193⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41674⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨14256⟩⟩, ⟨.program ⟨257⟩, ⟨39914⟩⟩], [⟨.program ⟨257⟩, ⟨41139⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨40148⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact92236RawTermsValid :
    exact92236RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event92236 : Event := .resultExact (⟨.program ⟨257⟩, ⟨41678⟩⟩) exact92236RawTerms .large 92235 .exactZero (none)

def event92237 : Event := .preFoldPolynomial 92236 [⟨⟨[], [⟨.program ⟨257⟩, ⟨7193⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41674⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨14256⟩⟩, ⟨.program ⟨257⟩, ⟨39914⟩⟩], [⟨.program ⟨257⟩, ⟨41139⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨40148⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩] .exactZero none

def exact92238RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7193⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41674⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨14256⟩⟩, ⟨.program ⟨257⟩, ⟨39914⟩⟩], [⟨.program ⟨257⟩, ⟨41139⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨40148⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

def event92238 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨41678⟩⟩) 92237 exact92238RawTerms .large 92235 .exactZero (none)

def event92239 : Event := .specializationComputed (⟨.program ⟨257⟩, ⟨39916⟩⟩) ⟨⟨72⟩, ⟨51⟩, ⟨135⟩⟩ ⟨92073, 92239⟩

def event92240 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨40602⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40599⟩⟩]⟩) (1) 0 2 (.universal 92239 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40599⟩⟩]⟩) (none) 92238)

def event92241 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨40602⟩⟩, .relation 92240 0, ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7193⟩⟩]⟩, (1)⟩)

def event92242 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨40602⟩⟩, .relation 92240 1, ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41674⟩⟩]⟩, (-1)⟩)

def event92243 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨40602⟩⟩, .relation 92240 2, ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨14256⟩⟩, ⟨.program ⟨257⟩, ⟨39914⟩⟩], [⟨.program ⟨257⟩, ⟨41139⟩⟩]⟩, (1)⟩)

def event92244 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨40602⟩⟩, .relation 92240 3, ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨40148⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact92245RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7193⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41674⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨14256⟩⟩, ⟨.program ⟨257⟩, ⟨39914⟩⟩], [⟨.program ⟨257⟩, ⟨41139⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨40148⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact92245RawTermsValid :
    exact92245RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event92245 : Event := .resultExact (⟨.program ⟨257⟩, ⟨40602⟩⟩) exact92245RawTerms .large 92069 (.finite 202072841853861888) (some (92071))

def event92246 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41676⟩⟩) 0 ⟨40602⟩ 92245

def event92247 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41676⟩⟩) 1 ⟨41675⟩ 92059

def event92248 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41676⟩⟩) (.sum [.predecessor 0 92246 .coefficient, .predecessor 1 92247 .coefficient])

def event92249 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨41676⟩⟩, .operator (⟨92245, 2⟩, ⟨92059, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨14256⟩⟩, ⟨.program ⟨257⟩, ⟨39914⟩⟩], [⟨.program ⟨257⟩, ⟨41139⟩⟩]⟩, (-1)⟩)

def event92250 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨41676⟩⟩, .operator (⟨92245, 1⟩, ⟨92059, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41674⟩⟩]⟩, (1)⟩)

def event92251 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41676⟩⟩) (.sum [.result 92245 .summary, .result 92059 .summary])

def exact92252RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7193⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨40148⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact92252RawTermsValid :
    exact92252RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event92252 : Event := .resultExact (⟨.program ⟨257⟩, ⟨41676⟩⟩) exact92252RawTerms .large 92248 (.finite 2998218789909838430208) (some (92251))

def event92253 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42116⟩⟩) 0 ⟨41676⟩ 92252

def event92254 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42116⟩⟩) 1 ⟨42114⟩ 91975

def event92255 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42116⟩⟩) (.product (.predecessor 0 92253 .coefficient) (.predecessor 1 92254 .coefficient) (⟨false, false, none, none, none⟩))

def event92256 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42116⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨42114⟩⟩]⟩) [⟨.result 91975 .coefficient, false, none⟩])

def event92257 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42116⟩⟩) (.product (.result 92252 .summary) (.transfer 92256) (⟨false, false, none, none, none⟩))

def event92258 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨42116⟩⟩, .operator (⟨92252, 0⟩, ⟨91975, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨42114⟩⟩]⟩, (1)⟩)

def event92259 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨42116⟩⟩, .operator (⟨92252, 1⟩, ⟨91975, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨40148⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨42114⟩⟩]⟩, (-1)⟩)

def event92260 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨42116⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨40148⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨42114⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨42114⟩⟩) ⟨41306⟩ 91972)

def event92261 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨42116⟩⟩, .relation 92260 0, ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨40148⟩⟩], [⟨.program ⟨257⟩, ⟨41306⟩⟩]⟩, (-1)⟩)

def exact92262RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨42114⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨40148⟩⟩], [⟨.program ⟨257⟩, ⟨41306⟩⟩]⟩, (-1)⟩]

theorem exact92262RawTermsValid :
    exact92262RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event92262 : Event := .resultExact (⟨.program ⟨257⟩, ⟨42116⟩⟩) exact92262RawTerms .large 92255 (.finite 32193129122288627115968346193920) (some (92257))

def event92263 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40956⟩⟩) 0 ⟨40149⟩ 3920

def event92264 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨40956⟩⟩) (.authority (.relationPreimageSource ⟨87⟩))

def exact92265RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨40956⟩⟩]⟩, (1)⟩]

theorem exact92265RawTermsValid :
    exact92265RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event92265 : Event := .resultExact (⟨.program ⟨257⟩, ⟨40956⟩⟩) exact92265RawTerms (.finite 5647228698) 92264 .exactZero (none)

def event92266 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40958⟩⟩) 0 ⟨40956⟩ 92265

def event92267 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40958⟩⟩) 1 ⟨2370⟩ 4

def event92268 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨40958⟩⟩) (.scale (.predecessor 0 92266 .coefficient) (.value (.predecessor 1 92267 .coefficient)))

def exact92269RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨40956⟩⟩]⟩, (1)⟩]

theorem exact92269RawTermsValid :
    exact92269RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event92269 : Event := .resultExact (⟨.program ⟨257⟩, ⟨40958⟩⟩) exact92269RawTerms (.finite 5647228698) 92268 .exactZero (none)

def event92270 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40959⟩⟩) 0 ⟨9944⟩ 90620

def event92271 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40959⟩⟩) 1 ⟨40958⟩ 92269

def event92272 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨40959⟩⟩) (.product (.predecessor 0 92270 .coefficient) (.predecessor 1 92271 .coefficient) (⟨false, false, none, none, none⟩))

def event92273 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨40959⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨40956⟩⟩]⟩) [⟨.result 92265 .coefficient, false, none⟩])

def event92274 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨40959⟩⟩) (.product (.result 90620 .summary) (.transfer 92273) (⟨false, false, none, none, none⟩))

def event92275 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨40959⟩⟩, .operator (⟨90620, 0⟩, ⟨92269, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40956⟩⟩]⟩, (1)⟩)

def event92276 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨40957⟩⟩)

def event92277 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event92278 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event92279 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9843⟩⟩) (.authority (.operator))

def event92280 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨9843⟩⟩) (.finite 14)

def event92281 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event92282 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event92283 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event92284 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event92285 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 92284

def event92286 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 92282

def event92287 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 92285 .coefficient) (.value (.predecessor 1 92286 .coefficient)))

def event92288 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event92289 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9845⟩⟩) 0 ⟨392⟩ 92288

def event92290 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9845⟩⟩) 1 ⟨9843⟩ 92280

def event92291 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9845⟩⟩) (.sum [.predecessor 0 92289 .coefficient, .predecessor 1 92290 .coefficient])

def event92292 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨9845⟩⟩) (.finite 655354)

def event92293 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9901⟩⟩) 0 ⟨9845⟩ 92292

def event92294 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9901⟩⟩) 1 ⟨5426⟩ 92278

def event92295 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9901⟩⟩) (.identity (.predecessor 1 92294 .coefficient))

def event92296 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨9901⟩⟩) (.finite 655360)

def event92297 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39914⟩⟩) 0 ⟨9901⟩ 92296

def event92298 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39914⟩⟩) (.authority (.programFamilyFact))

def exact92299RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨39914⟩⟩], []⟩, (1)⟩]

theorem exact92299RawTermsValid :
    exact92299RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event92299 : Event := .resultExact (⟨.program ⟨257⟩, ⟨39914⟩⟩) exact92299RawTerms (.finite 46) 92298 .exactZero (none)

def event92300 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14256⟩⟩) 0 ⟨9901⟩ 92296

def event92301 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14256⟩⟩) (.authority (.programFamilyFact))

def exact92302RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14256⟩⟩], []⟩, (1)⟩]

theorem exact92302RawTermsValid :
    exact92302RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event92302 : Event := .resultExact (⟨.program ⟨257⟩, ⟨14256⟩⟩) exact92302RawTerms (.finite 46) 92301 .exactZero (none)

def event92303 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39915⟩⟩) 0 ⟨14256⟩ 92302

def event92304 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39915⟩⟩) 1 ⟨39914⟩ 92299

def event92305 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39915⟩⟩) (.product (.predecessor 0 92303 .coefficient) (.predecessor 1 92304 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event92306 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39915⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨14256⟩⟩, ⟨.program ⟨257⟩, ⟨39914⟩⟩], []⟩) [⟨.result 92302 .coefficient, true, some 1⟩, ⟨.result 92299 .coefficient, true, some 1⟩])

def event92307 : Event := .survivorFold (1) 92306

def exact92308RawTerms : List Term := []

theorem exact92308RawTermsValid :
    exact92308RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event92308 : Event := .resultExact (⟨.program ⟨257⟩, ⟨39915⟩⟩) exact92308RawTerms (.finite 2116) 92305 (.finite 2116) (some (92306))

def event92309 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39916⟩⟩) 0 ⟨39915⟩ 92308

def event92310 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39916⟩⟩) (.identity (.predecessor 0 92309 .coefficient))

def event92311 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨39916⟩⟩) (.finite 2116)

def event92312 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40148⟩⟩) 0 ⟨39916⟩ 92311

def event92313 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨40148⟩⟩) (.authority (.programFamilyFact))

def exact92314RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨40148⟩⟩], []⟩, (1)⟩]

theorem exact92314RawTermsValid :
    exact92314RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event92314 : Event := .resultExact (⟨.program ⟨257⟩, ⟨40148⟩⟩) exact92314RawTerms (.finite 46) 92313 .exactZero (none)

def event92315 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40149⟩⟩) 0 ⟨40148⟩ 92314

def event92316 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨40149⟩⟩) (.identity (.predecessor 0 92315 .coefficient))

def event92317 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨40149⟩⟩) (.finite 46)

def event92318 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40956⟩⟩) 0 ⟨40149⟩ 92317

def event92319 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨40956⟩⟩) (.authority (.relationPreimageSource ⟨87⟩))

def exact92320RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨40956⟩⟩]⟩, (1)⟩]

theorem exact92320RawTermsValid :
    exact92320RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event92320 : Event := .resultExact (⟨.program ⟨257⟩, ⟨40956⟩⟩) exact92320RawTerms (.finite 5647228698) 92319 .exactZero (none)

def event92321 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35⟩⟩) (.authority (.operator))

def exact92322RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩]⟩, (1)⟩]

theorem exact92322RawTermsValid :
    exact92322RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event92322 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35⟩⟩) exact92322RawTerms .large 92321 .exactZero (none)

def event92323 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40957⟩⟩) 0 ⟨35⟩ 92322

def event92324 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40957⟩⟩) 1 ⟨40956⟩ 92320

def event92325 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨40957⟩⟩) (.product (.predecessor 0 92323 .coefficient) (.predecessor 1 92324 .coefficient) (⟨false, false, none, none, none⟩))

def event92326 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨40957⟩⟩, .operator (⟨92322, 0⟩, ⟨92320, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40956⟩⟩]⟩, (1)⟩)

def exact92327RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40956⟩⟩]⟩, (1)⟩]

theorem exact92327RawTermsValid :
    exact92327RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event92327 : Event := .resultExact (⟨.program ⟨257⟩, ⟨40957⟩⟩) exact92327RawTerms .large 92325 .exactZero (none)

def event92328 : Event := .preFoldPolynomial 92327 [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40956⟩⟩]⟩, (1)⟩] .exactZero none

def exact92329RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40956⟩⟩]⟩, (1)⟩]

def event92329 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨40957⟩⟩) 92328 exact92329RawTerms .large 92325 .exactZero (none)

def event92330 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨42118⟩⟩)

def event92331 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event92332 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event92333 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9843⟩⟩) (.authority (.operator))

def event92334 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨9843⟩⟩) (.finite 14)

def event92335 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event92336 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event92337 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event92338 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event92339 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 92338

def event92340 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 92336

def event92341 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 92339 .coefficient) (.value (.predecessor 1 92340 .coefficient)))

def event92342 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event92343 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9845⟩⟩) 0 ⟨392⟩ 92342

def event92344 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9845⟩⟩) 1 ⟨9843⟩ 92334

def event92345 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9845⟩⟩) (.sum [.predecessor 0 92343 .coefficient, .predecessor 1 92344 .coefficient])

def event92346 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨9845⟩⟩) (.finite 655354)

def event92347 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9901⟩⟩) 0 ⟨9845⟩ 92346

def event92348 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9901⟩⟩) 1 ⟨5426⟩ 92332

def event92349 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9901⟩⟩) (.identity (.predecessor 1 92348 .coefficient))

def event92350 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨9901⟩⟩) (.finite 655360)

def event92351 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39914⟩⟩) 0 ⟨9901⟩ 92350

def event92352 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39914⟩⟩) (.authority (.programFamilyFact))

def exact92353RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨39914⟩⟩], []⟩, (1)⟩]

theorem exact92353RawTermsValid :
    exact92353RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event92353 : Event := .resultExact (⟨.program ⟨257⟩, ⟨39914⟩⟩) exact92353RawTerms (.finite 46) 92352 .exactZero (none)

def event92354 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14256⟩⟩) 0 ⟨9901⟩ 92350

def event92355 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14256⟩⟩) (.authority (.programFamilyFact))

def exact92356RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14256⟩⟩], []⟩, (1)⟩]

theorem exact92356RawTermsValid :
    exact92356RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event92356 : Event := .resultExact (⟨.program ⟨257⟩, ⟨14256⟩⟩) exact92356RawTerms (.finite 46) 92355 .exactZero (none)

def event92357 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39915⟩⟩) 0 ⟨14256⟩ 92356

def event92358 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39915⟩⟩) 1 ⟨39914⟩ 92353

def event92359 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39915⟩⟩) (.product (.predecessor 0 92357 .coefficient) (.predecessor 1 92358 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event92360 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨39915⟩⟩, .operator (⟨92356, 0⟩, ⟨92353, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨14256⟩⟩, ⟨.program ⟨257⟩, ⟨39914⟩⟩], []⟩, (1)⟩)

def exact92361RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14256⟩⟩, ⟨.program ⟨257⟩, ⟨39914⟩⟩], []⟩, (1)⟩]

theorem exact92361RawTermsValid :
    exact92361RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event92361 : Event := .resultExact (⟨.program ⟨257⟩, ⟨39915⟩⟩) exact92361RawTerms (.finite 2116) 92359 .exactZero (none)

def event92362 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39916⟩⟩) 0 ⟨39915⟩ 92361

def event92363 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39916⟩⟩) (.identity (.predecessor 0 92362 .coefficient))

def event92364 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨39916⟩⟩) (.finite 2116)

def event92365 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40148⟩⟩) 0 ⟨39916⟩ 92364

def event92366 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨40148⟩⟩) (.authority (.programFamilyFact))

def exact92367RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨40148⟩⟩], []⟩, (1)⟩]

theorem exact92367RawTermsValid :
    exact92367RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event92367 : Event := .resultExact (⟨.program ⟨257⟩, ⟨40148⟩⟩) exact92367RawTerms (.finite 46) 92366 .exactZero (none)

def event92368 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40149⟩⟩) 0 ⟨40148⟩ 92367

def event92369 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨40149⟩⟩) (.identity (.predecessor 0 92368 .coefficient))

def event92370 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨40149⟩⟩) (.finite 46)

def event92371 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41304⟩⟩) 0 ⟨40149⟩ 92370

def event92372 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41304⟩⟩) (.authority (.programFamilyFact))

def event92373 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨41304⟩⟩) (.finite 3720)

def event92374 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨7177⟩⟩) .missing

def event92375 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41306⟩⟩) 0 ⟨7177⟩ 92374

def event92376 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41306⟩⟩) 1 ⟨41304⟩ 92373

def event92377 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41306⟩⟩) (.authority (.operator))

def exact92378RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨41306⟩⟩]⟩, (1)⟩]

theorem exact92378RawTermsValid :
    exact92378RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event92378 : Event := .resultExact (⟨.program ⟨257⟩, ⟨41306⟩⟩) exact92378RawTerms .large 92377 .exactZero (none)

def event92379 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42114⟩⟩) 0 ⟨41306⟩ 92378

def event92380 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42114⟩⟩) (.authority (.operator))

def exact92381RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨42114⟩⟩]⟩, (1)⟩]

theorem exact92381RawTermsValid :
    exact92381RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event92381 : Event := .resultExact (⟨.program ⟨257⟩, ⟨42114⟩⟩) exact92381RawTerms (.finite 8192) 92380 .exactZero (none)

def event92382 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨136⟩⟩) (.authority (.operator))

def event92383 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨136⟩⟩) .exactZero

def event92384 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41486⟩⟩) 0 ⟨40149⟩ 92370

def event92385 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41486⟩⟩) 1 ⟨136⟩ 92383

def event92386 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41486⟩⟩) (.sum [.predecessor 0 92384 .coefficient, .predecessor 1 92385 .coefficient])

def event92387 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨41486⟩⟩) (.finite 46)

def event92388 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41487⟩⟩) 0 ⟨41486⟩ 92387

def event92389 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41487⟩⟩) (.identity (.predecessor 0 92388 .coefficient))

def exact92390RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨40148⟩⟩], []⟩, (1)⟩]

theorem exact92390RawTermsValid :
    exact92390RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event92390 : Event := .resultExact (⟨.program ⟨257⟩, ⟨41487⟩⟩) exact92390RawTerms (.finite 46) 92389 .exactZero (none)

def event92391 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6908⟩⟩) (.authority (.factStore))

def exact92392RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact92392RawTermsValid :
    exact92392RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event92392 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6908⟩⟩) exact92392RawTerms .large 92391 .exactZero (none)

def event92393 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41488⟩⟩) 0 ⟨6908⟩ 92392

def event92394 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41488⟩⟩) 1 ⟨41487⟩ 92390

def event92395 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41488⟩⟩) (.product (.predecessor 0 92393 .coefficient) (.predecessor 1 92394 .coefficient) (⟨false, false, none, none, none⟩))

def event92396 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨41488⟩⟩, .operator (⟨92392, 0⟩, ⟨92390, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨40148⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact92397RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨40148⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact92397RawTermsValid :
    exact92397RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event92397 : Event := .resultExact (⟨.program ⟨257⟩, ⟨41488⟩⟩) exact92397RawTerms .large 92395 .exactZero (none)

def event92398 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7193⟩⟩) 0 ⟨7177⟩ 92374

def event92399 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7193⟩⟩) (.authority (.operator))

def exact92400RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7193⟩⟩]⟩, (1)⟩]

theorem exact92400RawTermsValid :
    exact92400RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event92400 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7193⟩⟩) exact92400RawTerms .large 92399 .exactZero (none)

def event92401 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41489⟩⟩) 0 ⟨7193⟩ 92400

def event92402 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41489⟩⟩) 1 ⟨41488⟩ 92397

def event92403 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41489⟩⟩) (.sum [.predecessor 0 92401 .coefficient, .predecessor 1 92402 .coefficient])

def exact92404RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7193⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨40148⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact92404RawTermsValid :
    exact92404RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event92404 : Event := .resultExact (⟨.program ⟨257⟩, ⟨41489⟩⟩) exact92404RawTerms .large 92403 .exactZero (none)

def event92405 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42115⟩⟩) 0 ⟨41489⟩ 92404

def event92406 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42115⟩⟩) 1 ⟨42114⟩ 92381

def event92407 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42115⟩⟩) (.product (.predecessor 0 92405 .coefficient) (.predecessor 1 92406 .coefficient) (⟨false, false, none, none, none⟩))

def event92408 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨42115⟩⟩, .operator (⟨92404, 0⟩, ⟨92381, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨42114⟩⟩]⟩, (1)⟩)

def event92409 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨42115⟩⟩, .operator (⟨92404, 1⟩, ⟨92381, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨40148⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨42114⟩⟩]⟩, (-1)⟩)

def event92410 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨42115⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨40148⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨42114⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨42114⟩⟩) ⟨41306⟩ 92378)

def event92411 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨42115⟩⟩, .relation 92410 0, ⟨[⟨.program ⟨257⟩, ⟨40148⟩⟩], [⟨.program ⟨257⟩, ⟨41306⟩⟩]⟩, (-1)⟩)

def exact92412RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨42114⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨40148⟩⟩], [⟨.program ⟨257⟩, ⟨41306⟩⟩]⟩, (-1)⟩]

theorem exact92412RawTermsValid :
    exact92412RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event92412 : Event := .resultExact (⟨.program ⟨257⟩, ⟨42115⟩⟩) exact92412RawTerms .large 92407 .exactZero (none)

def event92413 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40384⟩⟩) 0 ⟨40149⟩ 92370

def event92414 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨40384⟩⟩) (.authority (.programFamilyFact))

def exact92415RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨40384⟩⟩], []⟩, (1)⟩]

theorem exact92415RawTermsValid :
    exact92415RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event92415 : Event := .resultExact (⟨.program ⟨257⟩, ⟨40384⟩⟩) exact92415RawTerms (.finite 63) 92414 .exactZero (none)

def eventLeaf5760 : Array AnnotatedEvent := #[
  { event := event92160
    frameStart := 92121 },
  { event := event92161
    frameStart := 92121 },
  { event := event92162
    frameStart := 92121 },
  { event := event92163
    frameStart := 92121 },
  { event := event92164
    frameStart := 92121 },
  { event := event92165
    frameStart := 92121 },
  { event := event92166
    frameStart := 92121 },
  { event := event92167
    frameStart := 92121 },
  { event := event92168
    frameStart := 92121 },
  { event := event92169
    frameStart := 92121 },
  { event := event92170
    frameStart := 92121 },
  { event := event92171
    frameStart := 92121 },
  { event := event92172
    frameStart := 92121 },
  { event := event92173
    frameStart := 92121 },
  { event := event92174
    frameStart := 92121 },
  { event := event92175
    frameStart := 92121 }
]

def eventLeaf5761 : Array AnnotatedEvent := #[
  { event := event92176
    frameStart := 92121 },
  { event := event92177
    frameStart := 92121 },
  { event := event92178
    frameStart := 92121 },
  { event := event92179
    frameStart := 92121 },
  { event := event92180
    frameStart := 92121 },
  { event := event92181
    frameStart := 92121 },
  { event := event92182
    frameStart := 92121 },
  { event := event92183
    frameStart := 92121 },
  { event := event92184
    frameStart := 92121 },
  { event := event92185
    frameStart := 92121 },
  { event := event92186
    frameStart := 92121 },
  { event := event92187
    frameStart := 92121 },
  { event := event92188
    frameStart := 92121 },
  { event := event92189
    frameStart := 92121 },
  { event := event92190
    frameStart := 92121 },
  { event := event92191
    frameStart := 92121 }
]

def eventLeaf5762 : Array AnnotatedEvent := #[
  { event := event92192
    frameStart := 92121 },
  { event := event92193
    frameStart := 92121 },
  { event := event92194
    frameStart := 92121 },
  { event := event92195
    frameStart := 92121 },
  { event := event92196
    frameStart := 92121 },
  { event := event92197
    frameStart := 92121 },
  { event := event92198
    frameStart := 92121 },
  { event := event92199
    frameStart := 92121 },
  { event := event92200
    frameStart := 92121 },
  { event := event92201
    frameStart := 92121 },
  { event := event92202
    frameStart := 92121 },
  { event := event92203
    frameStart := 92121 },
  { event := event92204
    frameStart := 92121 },
  { event := event92205
    frameStart := 92121 },
  { event := event92206
    frameStart := 92121 },
  { event := event92207
    frameStart := 92121 }
]

def eventLeaf5763 : Array AnnotatedEvent := #[
  { event := event92208
    frameStart := 92121 },
  { event := event92209
    frameStart := 92121 },
  { event := event92210
    frameStart := 92121 },
  { event := event92211
    frameStart := 92121 },
  { event := event92212
    frameStart := 92121 },
  { event := event92213
    frameStart := 92121 },
  { event := event92214
    frameStart := 92121 },
  { event := event92215
    frameStart := 92121 },
  { event := event92216
    frameStart := 92121 },
  { event := event92217
    frameStart := 92121 },
  { event := event92218
    frameStart := 92121 },
  { event := event92219
    frameStart := 92121 },
  { event := event92220
    frameStart := 92121 },
  { event := event92221
    frameStart := 92121 },
  { event := event92222
    frameStart := 92121 },
  { event := event92223
    frameStart := 92121 }
]

def eventLeaf5764 : Array AnnotatedEvent := #[
  { event := event92224
    frameStart := 92121 },
  { event := event92225
    frameStart := 92121 },
  { event := event92226
    frameStart := 92121 },
  { event := event92227
    frameStart := 92121 },
  { event := event92228
    frameStart := 92121 },
  { event := event92229
    frameStart := 92121 },
  { event := event92230
    frameStart := 92121 },
  { event := event92231
    frameStart := 92121 },
  { event := event92232
    frameStart := 92121 },
  { event := event92233
    frameStart := 92121 },
  { event := event92234
    frameStart := 92121 },
  { event := event92235
    frameStart := 92121 },
  { event := event92236
    frameStart := 92121 },
  { event := event92237
    frameStart := 92121 },
  { event := event92238
    frameStart := 92121 },
  { event := event92239
    frameStart := 0 }
]

def eventLeaf5765 : Array AnnotatedEvent := #[
  { event := event92240
    frameStart := 0 },
  { event := event92241
    frameStart := 0 },
  { event := event92242
    frameStart := 0 },
  { event := event92243
    frameStart := 0 },
  { event := event92244
    frameStart := 0 },
  { event := event92245
    frameStart := 0 },
  { event := event92246
    frameStart := 0 },
  { event := event92247
    frameStart := 0 },
  { event := event92248
    frameStart := 0 },
  { event := event92249
    frameStart := 0 },
  { event := event92250
    frameStart := 0 },
  { event := event92251
    frameStart := 0 },
  { event := event92252
    frameStart := 0 },
  { event := event92253
    frameStart := 0 },
  { event := event92254
    frameStart := 0 },
  { event := event92255
    frameStart := 0 }
]

def eventLeaf5766 : Array AnnotatedEvent := #[
  { event := event92256
    frameStart := 0 },
  { event := event92257
    frameStart := 0 },
  { event := event92258
    frameStart := 0 },
  { event := event92259
    frameStart := 0 },
  { event := event92260
    frameStart := 0 },
  { event := event92261
    frameStart := 0 },
  { event := event92262
    frameStart := 0 },
  { event := event92263
    frameStart := 0 },
  { event := event92264
    frameStart := 0 },
  { event := event92265
    frameStart := 0 },
  { event := event92266
    frameStart := 0 },
  { event := event92267
    frameStart := 0 },
  { event := event92268
    frameStart := 0 },
  { event := event92269
    frameStart := 0 },
  { event := event92270
    frameStart := 0 },
  { event := event92271
    frameStart := 0 }
]

def eventLeaf5767 : Array AnnotatedEvent := #[
  { event := event92272
    frameStart := 0 },
  { event := event92273
    frameStart := 0 },
  { event := event92274
    frameStart := 0 },
  { event := event92275
    frameStart := 0 },
  { event := event92276
    frameStart := 92276 },
  { event := event92277
    frameStart := 92276 },
  { event := event92278
    frameStart := 92276 },
  { event := event92279
    frameStart := 92276 },
  { event := event92280
    frameStart := 92276 },
  { event := event92281
    frameStart := 92276 },
  { event := event92282
    frameStart := 92276 },
  { event := event92283
    frameStart := 92276 },
  { event := event92284
    frameStart := 92276 },
  { event := event92285
    frameStart := 92276 },
  { event := event92286
    frameStart := 92276 },
  { event := event92287
    frameStart := 92276 }
]

def eventLeaf5768 : Array AnnotatedEvent := #[
  { event := event92288
    frameStart := 92276 },
  { event := event92289
    frameStart := 92276 },
  { event := event92290
    frameStart := 92276 },
  { event := event92291
    frameStart := 92276 },
  { event := event92292
    frameStart := 92276 },
  { event := event92293
    frameStart := 92276 },
  { event := event92294
    frameStart := 92276 },
  { event := event92295
    frameStart := 92276 },
  { event := event92296
    frameStart := 92276 },
  { event := event92297
    frameStart := 92276 },
  { event := event92298
    frameStart := 92276 },
  { event := event92299
    frameStart := 92276 },
  { event := event92300
    frameStart := 92276 },
  { event := event92301
    frameStart := 92276 },
  { event := event92302
    frameStart := 92276 },
  { event := event92303
    frameStart := 92276 }
]

def eventLeaf5769 : Array AnnotatedEvent := #[
  { event := event92304
    frameStart := 92276 },
  { event := event92305
    frameStart := 92276 },
  { event := event92306
    frameStart := 92276 },
  { event := event92307
    frameStart := 92276 },
  { event := event92308
    frameStart := 92276 },
  { event := event92309
    frameStart := 92276 },
  { event := event92310
    frameStart := 92276 },
  { event := event92311
    frameStart := 92276 },
  { event := event92312
    frameStart := 92276 },
  { event := event92313
    frameStart := 92276 },
  { event := event92314
    frameStart := 92276 },
  { event := event92315
    frameStart := 92276 },
  { event := event92316
    frameStart := 92276 },
  { event := event92317
    frameStart := 92276 },
  { event := event92318
    frameStart := 92276 },
  { event := event92319
    frameStart := 92276 }
]

def eventLeaf5770 : Array AnnotatedEvent := #[
  { event := event92320
    frameStart := 92276 },
  { event := event92321
    frameStart := 92276 },
  { event := event92322
    frameStart := 92276 },
  { event := event92323
    frameStart := 92276 },
  { event := event92324
    frameStart := 92276 },
  { event := event92325
    frameStart := 92276 },
  { event := event92326
    frameStart := 92276 },
  { event := event92327
    frameStart := 92276 },
  { event := event92328
    frameStart := 92276 },
  { event := event92329
    frameStart := 92276 },
  { event := event92330
    frameStart := 92330 },
  { event := event92331
    frameStart := 92330 },
  { event := event92332
    frameStart := 92330 },
  { event := event92333
    frameStart := 92330 },
  { event := event92334
    frameStart := 92330 },
  { event := event92335
    frameStart := 92330 }
]

def eventLeaf5771 : Array AnnotatedEvent := #[
  { event := event92336
    frameStart := 92330 },
  { event := event92337
    frameStart := 92330 },
  { event := event92338
    frameStart := 92330 },
  { event := event92339
    frameStart := 92330 },
  { event := event92340
    frameStart := 92330 },
  { event := event92341
    frameStart := 92330 },
  { event := event92342
    frameStart := 92330 },
  { event := event92343
    frameStart := 92330 },
  { event := event92344
    frameStart := 92330 },
  { event := event92345
    frameStart := 92330 },
  { event := event92346
    frameStart := 92330 },
  { event := event92347
    frameStart := 92330 },
  { event := event92348
    frameStart := 92330 },
  { event := event92349
    frameStart := 92330 },
  { event := event92350
    frameStart := 92330 },
  { event := event92351
    frameStart := 92330 }
]

def eventLeaf5772 : Array AnnotatedEvent := #[
  { event := event92352
    frameStart := 92330 },
  { event := event92353
    frameStart := 92330 },
  { event := event92354
    frameStart := 92330 },
  { event := event92355
    frameStart := 92330 },
  { event := event92356
    frameStart := 92330 },
  { event := event92357
    frameStart := 92330 },
  { event := event92358
    frameStart := 92330 },
  { event := event92359
    frameStart := 92330 },
  { event := event92360
    frameStart := 92330 },
  { event := event92361
    frameStart := 92330 },
  { event := event92362
    frameStart := 92330 },
  { event := event92363
    frameStart := 92330 },
  { event := event92364
    frameStart := 92330 },
  { event := event92365
    frameStart := 92330 },
  { event := event92366
    frameStart := 92330 },
  { event := event92367
    frameStart := 92330 }
]

def eventLeaf5773 : Array AnnotatedEvent := #[
  { event := event92368
    frameStart := 92330 },
  { event := event92369
    frameStart := 92330 },
  { event := event92370
    frameStart := 92330 },
  { event := event92371
    frameStart := 92330 },
  { event := event92372
    frameStart := 92330 },
  { event := event92373
    frameStart := 92330 },
  { event := event92374
    frameStart := 92330 },
  { event := event92375
    frameStart := 92330 },
  { event := event92376
    frameStart := 92330 },
  { event := event92377
    frameStart := 92330 },
  { event := event92378
    frameStart := 92330 },
  { event := event92379
    frameStart := 92330 },
  { event := event92380
    frameStart := 92330 },
  { event := event92381
    frameStart := 92330 },
  { event := event92382
    frameStart := 92330 },
  { event := event92383
    frameStart := 92330 }
]

def eventLeaf5774 : Array AnnotatedEvent := #[
  { event := event92384
    frameStart := 92330 },
  { event := event92385
    frameStart := 92330 },
  { event := event92386
    frameStart := 92330 },
  { event := event92387
    frameStart := 92330 },
  { event := event92388
    frameStart := 92330 },
  { event := event92389
    frameStart := 92330 },
  { event := event92390
    frameStart := 92330 },
  { event := event92391
    frameStart := 92330 },
  { event := event92392
    frameStart := 92330 },
  { event := event92393
    frameStart := 92330 },
  { event := event92394
    frameStart := 92330 },
  { event := event92395
    frameStart := 92330 },
  { event := event92396
    frameStart := 92330 },
  { event := event92397
    frameStart := 92330 },
  { event := event92398
    frameStart := 92330 },
  { event := event92399
    frameStart := 92330 }
]

def eventLeaf5775 : Array AnnotatedEvent := #[
  { event := event92400
    frameStart := 92330 },
  { event := event92401
    frameStart := 92330 },
  { event := event92402
    frameStart := 92330 },
  { event := event92403
    frameStart := 92330 },
  { event := event92404
    frameStart := 92330 },
  { event := event92405
    frameStart := 92330 },
  { event := event92406
    frameStart := 92330 },
  { event := event92407
    frameStart := 92330 },
  { event := event92408
    frameStart := 92330 },
  { event := event92409
    frameStart := 92330 },
  { event := event92410
    frameStart := 92330 },
  { event := event92411
    frameStart := 92330 },
  { event := event92412
    frameStart := 92330 },
  { event := event92413
    frameStart := 92330 },
  { event := event92414
    frameStart := 92330 },
  { event := event92415
    frameStart := 92330 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events360
