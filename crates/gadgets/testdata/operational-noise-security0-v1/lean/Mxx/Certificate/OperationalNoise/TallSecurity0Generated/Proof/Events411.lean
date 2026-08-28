import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events411

open Mxx.Certificate.OperationalNoise
open CertificateABI

def exact105216RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16252⟩⟩], []⟩, (1)⟩]

theorem exact105216RawTermsValid :
    exact105216RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event105216 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16330⟩⟩) exact105216RawTerms (.finite 30) 105215 .exactZero (none)

def event105217 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6544⟩⟩) (.authority (.factStore))

def exact105218RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact105218RawTermsValid :
    exact105218RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event105218 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6544⟩⟩) exact105218RawTerms .large 105217 .exactZero (none)

def event105219 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16331⟩⟩) 0 ⟨6544⟩ 105218

def event105220 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16331⟩⟩) 1 ⟨16330⟩ 105216

def event105221 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16331⟩⟩) (.product (.predecessor 0 105219 .coefficient) (.predecessor 1 105220 .coefficient) (⟨false, false, none, none, none⟩))

def event105222 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨16331⟩⟩, .operator (⟨105218, 0⟩, ⟨105216, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨16252⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact105223RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16252⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact105223RawTermsValid :
    exact105223RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event105223 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16331⟩⟩) exact105223RawTerms .large 105221 .exactZero (none)

def event105224 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6700⟩⟩) 0 ⟨6689⟩ 105200

def event105225 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6700⟩⟩) (.authority (.operator))

def exact105226RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6700⟩⟩]⟩, (1)⟩]

theorem exact105226RawTermsValid :
    exact105226RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event105226 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6700⟩⟩) exact105226RawTerms .large 105225 .exactZero (none)

def event105227 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16332⟩⟩) 0 ⟨6700⟩ 105226

def event105228 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16332⟩⟩) 1 ⟨16331⟩ 105223

def event105229 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16332⟩⟩) (.sum [.predecessor 0 105227 .coefficient, .predecessor 1 105228 .coefficient])

def exact105230RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6700⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16252⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact105230RawTermsValid :
    exact105230RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event105230 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16332⟩⟩) exact105230RawTerms .large 105229 .exactZero (none)

def event105231 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28476⟩⟩) 0 ⟨16332⟩ 105230

def event105232 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28476⟩⟩) 1 ⟨28475⟩ 105207

def event105233 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28476⟩⟩) (.product (.predecessor 0 105231 .coefficient) (.predecessor 1 105232 .coefficient) (⟨false, false, none, none, none⟩))

def event105234 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28476⟩⟩, .operator (⟨105230, 0⟩, ⟨105207, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6700⟩⟩, ⟨.program ⟨214⟩, ⟨28475⟩⟩]⟩, (1)⟩)

def event105235 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28476⟩⟩, .operator (⟨105230, 1⟩, ⟨105207, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨16252⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28475⟩⟩]⟩, (-1)⟩)

def event105236 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨28476⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨16252⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28475⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨28475⟩⟩) ⟨24341⟩ 105204)

def event105237 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28476⟩⟩, .relation 105236 0, ⟨[⟨.program ⟨214⟩, ⟨16252⟩⟩], [⟨.program ⟨214⟩, ⟨24341⟩⟩]⟩, (-1)⟩)

def exact105238RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6700⟩⟩, ⟨.program ⟨214⟩, ⟨28475⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16252⟩⟩], [⟨.program ⟨214⟩, ⟨24341⟩⟩]⟩, (-1)⟩]

theorem exact105238RawTermsValid :
    exact105238RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event105238 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28476⟩⟩) exact105238RawTerms .large 105233 .exactZero (none)

def event105239 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17596⟩⟩) 0 ⟨16253⟩ 105196

def event105240 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨17596⟩⟩) (.authority (.programFamilyFact))

def exact105241RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨17596⟩⟩], []⟩, (1)⟩]

theorem exact105241RawTermsValid :
    exact105241RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event105241 : Event := .resultExact (⟨.program ⟨214⟩, ⟨17596⟩⟩) exact105241RawTerms (.finite 30) 105240 .exactZero (none)

def event105242 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17598⟩⟩) 0 ⟨6544⟩ 105218

def event105243 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17598⟩⟩) 1 ⟨17596⟩ 105241

def event105244 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨17598⟩⟩) (.product (.predecessor 0 105242 .coefficient) (.predecessor 1 105243 .coefficient) (⟨false, true, none, none, some 1⟩))

def event105245 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨17598⟩⟩, .operator (⟨105218, 0⟩, ⟨105241, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨17596⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact105246RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨17596⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact105246RawTermsValid :
    exact105246RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event105246 : Event := .resultExact (⟨.program ⟨214⟩, ⟨17598⟩⟩) exact105246RawTerms .large 105244 .exactZero (none)

def event105247 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6728⟩⟩) 0 ⟨6689⟩ 105200

def event105248 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6728⟩⟩) (.authority (.operator))

def exact105249RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6728⟩⟩]⟩, (1)⟩]

theorem exact105249RawTermsValid :
    exact105249RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event105249 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6728⟩⟩) exact105249RawTerms .large 105248 .exactZero (none)

def event105250 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17599⟩⟩) 0 ⟨6728⟩ 105249

def event105251 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17599⟩⟩) 1 ⟨17598⟩ 105246

def event105252 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨17599⟩⟩) (.sum [.predecessor 0 105250 .coefficient, .predecessor 1 105251 .coefficient])

def exact105253RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6728⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17596⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact105253RawTermsValid :
    exact105253RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event105253 : Event := .resultExact (⟨.program ⟨214⟩, ⟨17599⟩⟩) exact105253RawTerms .large 105252 .exactZero (none)

def event105254 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28481⟩⟩) 0 ⟨17599⟩ 105253

def event105255 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28481⟩⟩) 1 ⟨28476⟩ 105238

def event105256 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28481⟩⟩) (.sum [.predecessor 0 105254 .coefficient, .predecessor 1 105255 .coefficient])

def exact105257RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6700⟩⟩, ⟨.program ⟨214⟩, ⟨28475⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6728⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16252⟩⟩], [⟨.program ⟨214⟩, ⟨24341⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17596⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact105257RawTermsValid :
    exact105257RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event105257 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28481⟩⟩) exact105257RawTerms .large 105256 .exactZero (none)

def event105258 : Event := .preFoldPolynomial 105257 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6700⟩⟩, ⟨.program ⟨214⟩, ⟨28475⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6728⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16252⟩⟩], [⟨.program ⟨214⟩, ⟨24341⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17596⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩] .exactZero none

def exact105259RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6700⟩⟩, ⟨.program ⟨214⟩, ⟨28475⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6728⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16252⟩⟩], [⟨.program ⟨214⟩, ⟨24341⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17596⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

def event105259 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨28481⟩⟩) 105258 exact105259RawTerms .large 105256 .exactZero (none)

def event105260 : Event := .specializationComputed (⟨.program ⟨214⟩, ⟨16253⟩⟩) ⟨⟨141⟩, ⟨49⟩, ⟨109⟩⟩ ⟨105126, 105260⟩

def event105261 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨21752⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21749⟩⟩]⟩) (1) 0 2 (.universal 105260 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21749⟩⟩]⟩) (none) 105259)

def event105262 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21752⟩⟩, .relation 105261 1, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6728⟩⟩]⟩, (1)⟩)

def event105263 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21752⟩⟩, .relation 105261 0, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6700⟩⟩, ⟨.program ⟨214⟩, ⟨28475⟩⟩]⟩, (-1)⟩)

def event105264 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21752⟩⟩, .relation 105261 2, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16252⟩⟩], [⟨.program ⟨214⟩, ⟨24341⟩⟩]⟩, (1)⟩)

def event105265 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21752⟩⟩, .relation 105261 3, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨17596⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def exact105266RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6700⟩⟩, ⟨.program ⟨214⟩, ⟨28475⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6728⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16252⟩⟩], [⟨.program ⟨214⟩, ⟨24341⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨17596⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact105266RawTermsValid :
    exact105266RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event105266 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21752⟩⟩) exact105266RawTerms .large 105122 (.finite 1811303510016) (some (105124))

def event105267 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28478⟩⟩) 0 ⟨21752⟩ 105266

def event105268 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28478⟩⟩) 1 ⟨28477⟩ 105112

def event105269 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28478⟩⟩) (.sum [.predecessor 0 105267 .coefficient, .predecessor 1 105268 .coefficient])

def event105270 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28478⟩⟩, .operator (⟨105266, 0⟩, ⟨105112, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6700⟩⟩, ⟨.program ⟨214⟩, ⟨28475⟩⟩]⟩, (1)⟩)

def event105271 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28478⟩⟩, .operator (⟨105266, 2⟩, ⟨105112, 1⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16252⟩⟩], [⟨.program ⟨214⟩, ⟨24341⟩⟩]⟩, (-1)⟩)

def event105272 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28478⟩⟩) (.sum [.result 105266 .summary, .result 105112 .summary])

def exact105273RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6728⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨17596⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact105273RawTermsValid :
    exact105273RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event105273 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28478⟩⟩) exact105273RawTerms .large 105269 (.finite 1292202948609709846528) (some (105272))

def event105274 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28479⟩⟩) 0 ⟨28478⟩ 105273

def event105275 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28479⟩⟩) 1 ⟨6678⟩ 5659

def event105276 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28479⟩⟩) (.product (.predecessor 0 105274 .coefficient) (.predecessor 1 105275 .coefficient) (⟨false, false, none, none, none⟩))

def event105277 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28479⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨6677⟩⟩]⟩) [⟨.result 5655 .coefficient, false, none⟩])

def event105278 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28479⟩⟩) (.product (.result 105273 .summary) (.transfer 105277) (⟨false, false, none, none, none⟩))

def event105279 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28479⟩⟩, .operator (⟨105273, 0⟩, ⟨5659, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6728⟩⟩, ⟨.program ⟨214⟩, ⟨6677⟩⟩]⟩, (1)⟩)

def event105280 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28479⟩⟩, .operator (⟨105273, 1⟩, ⟨5659, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨17596⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨6677⟩⟩]⟩, (-1)⟩)

def event105281 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨28479⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨17596⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨6677⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨6677⟩⟩) ⟨6610⟩ 5652)

def event105282 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28479⟩⟩, .relation 105281 0, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6494⟩⟩, ⟨.program ⟨214⟩, ⟨17596⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def exact105283RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6728⟩⟩, ⟨.program ⟨214⟩, ⟨6677⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6494⟩⟩, ⟨.program ⟨214⟩, ⟨17596⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact105283RawTermsValid :
    exact105283RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event105283 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28479⟩⟩) exact105283RawTerms .large 105276 (.finite 4742405496644812892115304448) (some (105278))

def event105284 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24278⟩⟩) 0 ⟨6689⟩ 5477

def event105285 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24278⟩⟩) 1 ⟨24277⟩ 97836

def event105286 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24278⟩⟩) (.authority (.operator))

def exact105287RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨24278⟩⟩]⟩, (1)⟩]

theorem exact105287RawTermsValid :
    exact105287RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event105287 : Event := .resultExact (⟨.program ⟨214⟩, ⟨24278⟩⟩) exact105287RawTerms .large 105286 .exactZero (none)

def event105288 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28258⟩⟩) 0 ⟨24278⟩ 105287

def event105289 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28258⟩⟩) (.authority (.operator))

def exact105290RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨28258⟩⟩]⟩, (1)⟩]

theorem exact105290RawTermsValid :
    exact105290RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event105290 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28258⟩⟩) exact105290RawTerms (.finite 8192) 105289 .exactZero (none)

def event105291 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28260⟩⟩) 0 ⟨26209⟩ 98096

def event105292 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28260⟩⟩) 1 ⟨28258⟩ 105290

def event105293 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28260⟩⟩) (.product (.predecessor 0 105291 .coefficient) (.predecessor 1 105292 .coefficient) (⟨false, false, none, none, none⟩))

def event105294 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28260⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨28258⟩⟩]⟩) [⟨.result 105290 .coefficient, false, none⟩])

def event105295 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28260⟩⟩) (.product (.result 98096 .summary) (.transfer 105294) (⟨false, false, none, none, none⟩))

def event105296 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28260⟩⟩, .operator (⟨98096, 0⟩, ⟨105290, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6699⟩⟩, ⟨.program ⟨214⟩, ⟨28258⟩⟩]⟩, (1)⟩)

def event105297 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28260⟩⟩, .operator (⟨98096, 1⟩, ⟨105290, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16168⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28258⟩⟩]⟩, (-1)⟩)

def event105298 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨28260⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16168⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28258⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨28258⟩⟩) ⟨24278⟩ 105287)

def event105299 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28260⟩⟩, .relation 105298 0, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16168⟩⟩], [⟨.program ⟨214⟩, ⟨24278⟩⟩]⟩, (-1)⟩)

def exact105300RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6699⟩⟩, ⟨.program ⟨214⟩, ⟨28258⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16168⟩⟩], [⟨.program ⟨214⟩, ⟨24278⟩⟩]⟩, (-1)⟩]

theorem exact105300RawTermsValid :
    exact105300RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event105300 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28260⟩⟩) exact105300RawTerms .large 105293 (.finite 1292180534353385750528) (some (105295))

def event105301 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21605⟩⟩) 0 ⟨16169⟩ 4767

def event105302 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21605⟩⟩) (.authority (.relationPreimageSource ⟨47⟩))

def exact105303RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨21605⟩⟩]⟩, (1)⟩]

theorem exact105303RawTermsValid :
    exact105303RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event105303 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21605⟩⟩) exact105303RawTerms (.finite 136065468) 105302 .exactZero (none)

def event105304 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21607⟩⟩) 0 ⟨21605⟩ 105303

def event105305 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21607⟩⟩) 1 ⟨2348⟩ 4

def event105306 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21607⟩⟩) (.scale (.predecessor 0 105304 .coefficient) (.value (.predecessor 1 105305 .coefficient)))

def exact105307RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨21605⟩⟩]⟩, (1)⟩]

theorem exact105307RawTermsValid :
    exact105307RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event105307 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21607⟩⟩) exact105307RawTerms (.finite 136065468) 105306 .exactZero (none)

def event105308 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21608⟩⟩) 0 ⟨5509⟩ 94462

def event105309 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21608⟩⟩) 1 ⟨21607⟩ 105307

def event105310 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21608⟩⟩) (.product (.predecessor 0 105308 .coefficient) (.predecessor 1 105309 .coefficient) (⟨false, false, none, none, none⟩))

def event105311 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21608⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨21605⟩⟩]⟩) [⟨.result 105303 .coefficient, false, none⟩])

def event105312 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21608⟩⟩) (.product (.result 94462 .summary) (.transfer 105311) (⟨false, false, none, none, none⟩))

def event105313 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21608⟩⟩, .operator (⟨94462, 0⟩, ⟨105307, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21605⟩⟩]⟩, (1)⟩)

def event105314 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨21606⟩⟩)

def event105315 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event105316 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event105317 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event105318 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event105319 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 105318

def event105320 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 105316

def event105321 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 105319 .coefficient) (.value (.predecessor 1 105320 .coefficient)))

def event105322 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event105323 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11625⟩⟩) 0 ⟨5503⟩ 105322

def event105324 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11625⟩⟩) (.authority (.programFamilyFact))

def exact105325RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11625⟩⟩], []⟩, (1)⟩]

theorem exact105325RawTermsValid :
    exact105325RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event105325 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11625⟩⟩) exact105325RawTerms (.finite 28) 105324 .exactZero (none)

def event105326 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14614⟩⟩) 0 ⟨5503⟩ 105322

def event105327 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14614⟩⟩) (.authority (.programFamilyFact))

def exact105328RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨14614⟩⟩], []⟩, (1)⟩]

theorem exact105328RawTermsValid :
    exact105328RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event105328 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14614⟩⟩) exact105328RawTerms (.finite 28) 105327 .exactZero (none)

def event105329 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14615⟩⟩) 0 ⟨14614⟩ 105328

def event105330 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14615⟩⟩) 1 ⟨11625⟩ 105325

def event105331 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14615⟩⟩) (.product (.predecessor 0 105329 .coefficient) (.predecessor 1 105330 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event105332 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14615⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨11625⟩⟩, ⟨.program ⟨214⟩, ⟨14614⟩⟩], []⟩) [⟨.result 105328 .coefficient, true, some 1⟩, ⟨.result 105325 .coefficient, true, some 1⟩])

def event105333 : Event := .survivorFold (1) 105332

def exact105334RawTerms : List Term := []

theorem exact105334RawTermsValid :
    exact105334RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event105334 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14615⟩⟩) exact105334RawTerms (.finite 784) 105331 (.finite 784) (some (105332))

def event105335 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14616⟩⟩) 0 ⟨14615⟩ 105334

def event105336 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14616⟩⟩) (.identity (.predecessor 0 105335 .coefficient))

def event105337 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨14616⟩⟩) (.finite 784)

def event105338 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16168⟩⟩) 0 ⟨14616⟩ 105337

def event105339 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16168⟩⟩) (.authority (.programFamilyFact))

def exact105340RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16168⟩⟩], []⟩, (1)⟩]

theorem exact105340RawTermsValid :
    exact105340RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event105340 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16168⟩⟩) exact105340RawTerms (.finite 28) 105339 .exactZero (none)

def event105341 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16169⟩⟩) 0 ⟨16168⟩ 105340

def event105342 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16169⟩⟩) (.identity (.predecessor 0 105341 .coefficient))

def event105343 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨16169⟩⟩) (.finite 28)

def event105344 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21605⟩⟩) 0 ⟨16169⟩ 105343

def event105345 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21605⟩⟩) (.authority (.relationPreimageSource ⟨47⟩))

def exact105346RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨21605⟩⟩]⟩, (1)⟩]

theorem exact105346RawTermsValid :
    exact105346RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event105346 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21605⟩⟩) exact105346RawTerms (.finite 136065468) 105345 .exactZero (none)

def event105347 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6⟩⟩) (.authority (.operator))

def exact105348RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩]⟩, (1)⟩]

theorem exact105348RawTermsValid :
    exact105348RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event105348 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6⟩⟩) exact105348RawTerms .large 105347 .exactZero (none)

def event105349 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21606⟩⟩) 0 ⟨6⟩ 105348

def event105350 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21606⟩⟩) 1 ⟨21605⟩ 105346

def event105351 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21606⟩⟩) (.product (.predecessor 0 105349 .coefficient) (.predecessor 1 105350 .coefficient) (⟨false, false, none, none, none⟩))

def event105352 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21606⟩⟩, .operator (⟨105348, 0⟩, ⟨105346, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21605⟩⟩]⟩, (1)⟩)

def exact105353RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21605⟩⟩]⟩, (1)⟩]

theorem exact105353RawTermsValid :
    exact105353RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event105353 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21606⟩⟩) exact105353RawTerms .large 105351 .exactZero (none)

def event105354 : Event := .preFoldPolynomial 105353 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21605⟩⟩]⟩, (1)⟩] .exactZero none

def exact105355RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21605⟩⟩]⟩, (1)⟩]

def event105355 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨21606⟩⟩) 105354 exact105355RawTerms .large 105351 .exactZero (none)

def event105356 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨28264⟩⟩)

def event105357 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event105358 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event105359 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event105360 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event105361 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 105360

def event105362 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 105358

def event105363 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 105361 .coefficient) (.value (.predecessor 1 105362 .coefficient)))

def event105364 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event105365 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11625⟩⟩) 0 ⟨5503⟩ 105364

def event105366 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11625⟩⟩) (.authority (.programFamilyFact))

def exact105367RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11625⟩⟩], []⟩, (1)⟩]

theorem exact105367RawTermsValid :
    exact105367RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event105367 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11625⟩⟩) exact105367RawTerms (.finite 28) 105366 .exactZero (none)

def event105368 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14614⟩⟩) 0 ⟨5503⟩ 105364

def event105369 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14614⟩⟩) (.authority (.programFamilyFact))

def exact105370RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨14614⟩⟩], []⟩, (1)⟩]

theorem exact105370RawTermsValid :
    exact105370RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event105370 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14614⟩⟩) exact105370RawTerms (.finite 28) 105369 .exactZero (none)

def event105371 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14615⟩⟩) 0 ⟨14614⟩ 105370

def event105372 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14615⟩⟩) 1 ⟨11625⟩ 105367

def event105373 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14615⟩⟩) (.product (.predecessor 0 105371 .coefficient) (.predecessor 1 105372 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event105374 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨14615⟩⟩, .operator (⟨105370, 0⟩, ⟨105367, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨11625⟩⟩, ⟨.program ⟨214⟩, ⟨14614⟩⟩], []⟩, (1)⟩)

def exact105375RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11625⟩⟩, ⟨.program ⟨214⟩, ⟨14614⟩⟩], []⟩, (1)⟩]

theorem exact105375RawTermsValid :
    exact105375RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event105375 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14615⟩⟩) exact105375RawTerms (.finite 784) 105373 .exactZero (none)

def event105376 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14616⟩⟩) 0 ⟨14615⟩ 105375

def event105377 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14616⟩⟩) (.identity (.predecessor 0 105376 .coefficient))

def event105378 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨14616⟩⟩) (.finite 784)

def event105379 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16168⟩⟩) 0 ⟨14616⟩ 105378

def event105380 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16168⟩⟩) (.authority (.programFamilyFact))

def exact105381RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16168⟩⟩], []⟩, (1)⟩]

theorem exact105381RawTermsValid :
    exact105381RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event105381 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16168⟩⟩) exact105381RawTerms (.finite 28) 105380 .exactZero (none)

def event105382 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16169⟩⟩) 0 ⟨16168⟩ 105381

def event105383 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16169⟩⟩) (.identity (.predecessor 0 105382 .coefficient))

def event105384 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨16169⟩⟩) (.finite 28)

def event105385 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24277⟩⟩) 0 ⟨16169⟩ 105384

def event105386 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24277⟩⟩) (.authority (.programFamilyFact))

def event105387 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨24277⟩⟩) (.finite 3720)

def event105388 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨6689⟩⟩) .missing

def event105389 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24278⟩⟩) 0 ⟨6689⟩ 105388

def event105390 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24278⟩⟩) 1 ⟨24277⟩ 105387

def event105391 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24278⟩⟩) (.authority (.operator))

def exact105392RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨24278⟩⟩]⟩, (1)⟩]

theorem exact105392RawTermsValid :
    exact105392RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event105392 : Event := .resultExact (⟨.program ⟨214⟩, ⟨24278⟩⟩) exact105392RawTerms .large 105391 .exactZero (none)

def event105393 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28258⟩⟩) 0 ⟨24278⟩ 105392

def event105394 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28258⟩⟩) (.authority (.operator))

def exact105395RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨28258⟩⟩]⟩, (1)⟩]

theorem exact105395RawTermsValid :
    exact105395RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event105395 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28258⟩⟩) exact105395RawTerms (.finite 8192) 105394 .exactZero (none)

def event105396 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨110⟩⟩) (.authority (.operator))

def event105397 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨110⟩⟩) .exactZero

def event105398 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16210⟩⟩) 0 ⟨16169⟩ 105384

def event105399 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16210⟩⟩) 1 ⟨110⟩ 105397

def event105400 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16210⟩⟩) (.sum [.predecessor 0 105398 .coefficient, .predecessor 1 105399 .coefficient])

def event105401 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨16210⟩⟩) (.finite 28)

def event105402 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16211⟩⟩) 0 ⟨16210⟩ 105401

def event105403 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16211⟩⟩) (.identity (.predecessor 0 105402 .coefficient))

def exact105404RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16168⟩⟩], []⟩, (1)⟩]

theorem exact105404RawTermsValid :
    exact105404RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event105404 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16211⟩⟩) exact105404RawTerms (.finite 28) 105403 .exactZero (none)

def event105405 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6544⟩⟩) (.authority (.factStore))

def exact105406RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact105406RawTermsValid :
    exact105406RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event105406 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6544⟩⟩) exact105406RawTerms .large 105405 .exactZero (none)

def event105407 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16212⟩⟩) 0 ⟨6544⟩ 105406

def event105408 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16212⟩⟩) 1 ⟨16211⟩ 105404

def event105409 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16212⟩⟩) (.product (.predecessor 0 105407 .coefficient) (.predecessor 1 105408 .coefficient) (⟨false, false, none, none, none⟩))

def event105410 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨16212⟩⟩, .operator (⟨105406, 0⟩, ⟨105404, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨16168⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact105411RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16168⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact105411RawTermsValid :
    exact105411RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event105411 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16212⟩⟩) exact105411RawTerms .large 105409 .exactZero (none)

def event105412 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6699⟩⟩) 0 ⟨6689⟩ 105388

def event105413 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6699⟩⟩) (.authority (.operator))

def exact105414RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6699⟩⟩]⟩, (1)⟩]

theorem exact105414RawTermsValid :
    exact105414RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event105414 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6699⟩⟩) exact105414RawTerms .large 105413 .exactZero (none)

def event105415 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16213⟩⟩) 0 ⟨6699⟩ 105414

def event105416 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16213⟩⟩) 1 ⟨16212⟩ 105411

def event105417 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16213⟩⟩) (.sum [.predecessor 0 105415 .coefficient, .predecessor 1 105416 .coefficient])

def exact105418RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6699⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16168⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact105418RawTermsValid :
    exact105418RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event105418 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16213⟩⟩) exact105418RawTerms .large 105417 .exactZero (none)

def event105419 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28259⟩⟩) 0 ⟨16213⟩ 105418

def event105420 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28259⟩⟩) 1 ⟨28258⟩ 105395

def event105421 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28259⟩⟩) (.product (.predecessor 0 105419 .coefficient) (.predecessor 1 105420 .coefficient) (⟨false, false, none, none, none⟩))

def event105422 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28259⟩⟩, .operator (⟨105418, 0⟩, ⟨105395, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6699⟩⟩, ⟨.program ⟨214⟩, ⟨28258⟩⟩]⟩, (1)⟩)

def event105423 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28259⟩⟩, .operator (⟨105418, 1⟩, ⟨105395, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨16168⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28258⟩⟩]⟩, (-1)⟩)

def event105424 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨28259⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨16168⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28258⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨28258⟩⟩) ⟨24278⟩ 105392)

def event105425 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28259⟩⟩, .relation 105424 0, ⟨[⟨.program ⟨214⟩, ⟨16168⟩⟩], [⟨.program ⟨214⟩, ⟨24278⟩⟩]⟩, (-1)⟩)

def exact105426RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6699⟩⟩, ⟨.program ⟨214⟩, ⟨28258⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16168⟩⟩], [⟨.program ⟨214⟩, ⟨24278⟩⟩]⟩, (-1)⟩]

theorem exact105426RawTermsValid :
    exact105426RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event105426 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28259⟩⟩) exact105426RawTerms .large 105421 .exactZero (none)

def event105427 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17652⟩⟩) 0 ⟨16169⟩ 105384

def event105428 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨17652⟩⟩) (.authority (.programFamilyFact))

def exact105429RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨17652⟩⟩], []⟩, (1)⟩]

theorem exact105429RawTermsValid :
    exact105429RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event105429 : Event := .resultExact (⟨.program ⟨214⟩, ⟨17652⟩⟩) exact105429RawTerms (.finite 28) 105428 .exactZero (none)

def event105430 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17654⟩⟩) 0 ⟨6544⟩ 105406

def event105431 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17654⟩⟩) 1 ⟨17652⟩ 105429

def event105432 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨17654⟩⟩) (.product (.predecessor 0 105430 .coefficient) (.predecessor 1 105431 .coefficient) (⟨false, true, none, none, some 1⟩))

def event105433 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨17654⟩⟩, .operator (⟨105406, 0⟩, ⟨105429, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨17652⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact105434RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨17652⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact105434RawTermsValid :
    exact105434RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event105434 : Event := .resultExact (⟨.program ⟨214⟩, ⟨17654⟩⟩) exact105434RawTerms .large 105432 .exactZero (none)

def event105435 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6726⟩⟩) 0 ⟨6689⟩ 105388

def event105436 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6726⟩⟩) (.authority (.operator))

def exact105437RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6726⟩⟩]⟩, (1)⟩]

theorem exact105437RawTermsValid :
    exact105437RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event105437 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6726⟩⟩) exact105437RawTerms .large 105436 .exactZero (none)

def event105438 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17655⟩⟩) 0 ⟨6726⟩ 105437

def event105439 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17655⟩⟩) 1 ⟨17654⟩ 105434

def event105440 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨17655⟩⟩) (.sum [.predecessor 0 105438 .coefficient, .predecessor 1 105439 .coefficient])

def exact105441RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6726⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17652⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact105441RawTermsValid :
    exact105441RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event105441 : Event := .resultExact (⟨.program ⟨214⟩, ⟨17655⟩⟩) exact105441RawTerms .large 105440 .exactZero (none)

def event105442 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28264⟩⟩) 0 ⟨17655⟩ 105441

def event105443 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28264⟩⟩) 1 ⟨28259⟩ 105426

def event105444 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28264⟩⟩) (.sum [.predecessor 0 105442 .coefficient, .predecessor 1 105443 .coefficient])

def exact105445RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6699⟩⟩, ⟨.program ⟨214⟩, ⟨28258⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6726⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16168⟩⟩], [⟨.program ⟨214⟩, ⟨24278⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17652⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact105445RawTermsValid :
    exact105445RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event105445 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28264⟩⟩) exact105445RawTerms .large 105444 .exactZero (none)

def event105446 : Event := .preFoldPolynomial 105445 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6699⟩⟩, ⟨.program ⟨214⟩, ⟨28258⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6726⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16168⟩⟩], [⟨.program ⟨214⟩, ⟨24278⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17652⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩] .exactZero none

def exact105447RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6699⟩⟩, ⟨.program ⟨214⟩, ⟨28258⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6726⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16168⟩⟩], [⟨.program ⟨214⟩, ⟨24278⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17652⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

def event105447 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨28264⟩⟩) 105446 exact105447RawTerms .large 105444 .exactZero (none)

def event105448 : Event := .specializationComputed (⟨.program ⟨214⟩, ⟨16169⟩⟩) ⟨⟨139⟩, ⟨47⟩, ⟨109⟩⟩ ⟨105314, 105448⟩

def event105449 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨21608⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21605⟩⟩]⟩) (1) 0 2 (.universal 105448 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21605⟩⟩]⟩) (none) 105447)

def event105450 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21608⟩⟩, .relation 105449 1, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6726⟩⟩]⟩, (1)⟩)

def event105451 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21608⟩⟩, .relation 105449 0, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6699⟩⟩, ⟨.program ⟨214⟩, ⟨28258⟩⟩]⟩, (-1)⟩)

def event105452 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21608⟩⟩, .relation 105449 2, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16168⟩⟩], [⟨.program ⟨214⟩, ⟨24278⟩⟩]⟩, (1)⟩)

def event105453 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21608⟩⟩, .relation 105449 3, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨17652⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def exact105454RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6699⟩⟩, ⟨.program ⟨214⟩, ⟨28258⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6726⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16168⟩⟩], [⟨.program ⟨214⟩, ⟨24278⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨17652⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact105454RawTermsValid :
    exact105454RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event105454 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21608⟩⟩) exact105454RawTerms .large 105310 (.finite 1811303510016) (some (105312))

def event105455 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28261⟩⟩) 0 ⟨21608⟩ 105454

def event105456 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28261⟩⟩) 1 ⟨28260⟩ 105300

def event105457 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28261⟩⟩) (.sum [.predecessor 0 105455 .coefficient, .predecessor 1 105456 .coefficient])

def event105458 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28261⟩⟩, .operator (⟨105454, 0⟩, ⟨105300, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6699⟩⟩, ⟨.program ⟨214⟩, ⟨28258⟩⟩]⟩, (1)⟩)

def event105459 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28261⟩⟩, .operator (⟨105454, 2⟩, ⟨105300, 1⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16168⟩⟩], [⟨.program ⟨214⟩, ⟨24278⟩⟩]⟩, (-1)⟩)

def event105460 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28261⟩⟩) (.sum [.result 105454 .summary, .result 105300 .summary])

def exact105461RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6726⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨17652⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact105461RawTermsValid :
    exact105461RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event105461 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28261⟩⟩) exact105461RawTerms .large 105457 (.finite 1292180536164689260544) (some (105460))

def event105462 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28262⟩⟩) 0 ⟨28261⟩ 105461

def event105463 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28262⟩⟩) 1 ⟨6682⟩ 5679

def event105464 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28262⟩⟩) (.product (.predecessor 0 105462 .coefficient) (.predecessor 1 105463 .coefficient) (⟨false, false, none, none, none⟩))

def event105465 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28262⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨6681⟩⟩]⟩) [⟨.result 5675 .coefficient, false, none⟩])

def event105466 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28262⟩⟩) (.product (.result 105461 .summary) (.transfer 105465) (⟨false, false, none, none, none⟩))

def event105467 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28262⟩⟩, .operator (⟨105461, 0⟩, ⟨5679, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6726⟩⟩, ⟨.program ⟨214⟩, ⟨6681⟩⟩]⟩, (1)⟩)

def event105468 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28262⟩⟩, .operator (⟨105461, 1⟩, ⟨5679, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨17652⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨6681⟩⟩]⟩, (-1)⟩)

def event105469 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨28262⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨17652⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨6681⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨6681⟩⟩) ⟨6612⟩ 5672)

def event105470 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28262⟩⟩, .relation 105469 0, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6502⟩⟩, ⟨.program ⟨214⟩, ⟨17652⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def exact105471RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6726⟩⟩, ⟨.program ⟨214⟩, ⟨6681⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6502⟩⟩, ⟨.program ⟨214⟩, ⟨17652⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact105471RawTermsValid :
    exact105471RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event105471 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28262⟩⟩) exact105471RawTerms .large 105464 (.finite 4742323242612988221224648704) (some (105466))

def eventLeaf6576 : Array AnnotatedEvent := #[
  { event := event105216
    frameStart := 105168 },
  { event := event105217
    frameStart := 105168 },
  { event := event105218
    frameStart := 105168 },
  { event := event105219
    frameStart := 105168 },
  { event := event105220
    frameStart := 105168 },
  { event := event105221
    frameStart := 105168 },
  { event := event105222
    frameStart := 105168 },
  { event := event105223
    frameStart := 105168 },
  { event := event105224
    frameStart := 105168 },
  { event := event105225
    frameStart := 105168 },
  { event := event105226
    frameStart := 105168 },
  { event := event105227
    frameStart := 105168 },
  { event := event105228
    frameStart := 105168 },
  { event := event105229
    frameStart := 105168 },
  { event := event105230
    frameStart := 105168 },
  { event := event105231
    frameStart := 105168 }
]

def eventLeaf6577 : Array AnnotatedEvent := #[
  { event := event105232
    frameStart := 105168 },
  { event := event105233
    frameStart := 105168 },
  { event := event105234
    frameStart := 105168 },
  { event := event105235
    frameStart := 105168 },
  { event := event105236
    frameStart := 105168 },
  { event := event105237
    frameStart := 105168 },
  { event := event105238
    frameStart := 105168 },
  { event := event105239
    frameStart := 105168 },
  { event := event105240
    frameStart := 105168 },
  { event := event105241
    frameStart := 105168 },
  { event := event105242
    frameStart := 105168 },
  { event := event105243
    frameStart := 105168 },
  { event := event105244
    frameStart := 105168 },
  { event := event105245
    frameStart := 105168 },
  { event := event105246
    frameStart := 105168 },
  { event := event105247
    frameStart := 105168 }
]

def eventLeaf6578 : Array AnnotatedEvent := #[
  { event := event105248
    frameStart := 105168 },
  { event := event105249
    frameStart := 105168 },
  { event := event105250
    frameStart := 105168 },
  { event := event105251
    frameStart := 105168 },
  { event := event105252
    frameStart := 105168 },
  { event := event105253
    frameStart := 105168 },
  { event := event105254
    frameStart := 105168 },
  { event := event105255
    frameStart := 105168 },
  { event := event105256
    frameStart := 105168 },
  { event := event105257
    frameStart := 105168 },
  { event := event105258
    frameStart := 105168 },
  { event := event105259
    frameStart := 105168 },
  { event := event105260
    frameStart := 0 },
  { event := event105261
    frameStart := 0 },
  { event := event105262
    frameStart := 0 },
  { event := event105263
    frameStart := 0 }
]

def eventLeaf6579 : Array AnnotatedEvent := #[
  { event := event105264
    frameStart := 0 },
  { event := event105265
    frameStart := 0 },
  { event := event105266
    frameStart := 0 },
  { event := event105267
    frameStart := 0 },
  { event := event105268
    frameStart := 0 },
  { event := event105269
    frameStart := 0 },
  { event := event105270
    frameStart := 0 },
  { event := event105271
    frameStart := 0 },
  { event := event105272
    frameStart := 0 },
  { event := event105273
    frameStart := 0 },
  { event := event105274
    frameStart := 0 },
  { event := event105275
    frameStart := 0 },
  { event := event105276
    frameStart := 0 },
  { event := event105277
    frameStart := 0 },
  { event := event105278
    frameStart := 0 },
  { event := event105279
    frameStart := 0 }
]

def eventLeaf6580 : Array AnnotatedEvent := #[
  { event := event105280
    frameStart := 0 },
  { event := event105281
    frameStart := 0 },
  { event := event105282
    frameStart := 0 },
  { event := event105283
    frameStart := 0 },
  { event := event105284
    frameStart := 0 },
  { event := event105285
    frameStart := 0 },
  { event := event105286
    frameStart := 0 },
  { event := event105287
    frameStart := 0 },
  { event := event105288
    frameStart := 0 },
  { event := event105289
    frameStart := 0 },
  { event := event105290
    frameStart := 0 },
  { event := event105291
    frameStart := 0 },
  { event := event105292
    frameStart := 0 },
  { event := event105293
    frameStart := 0 },
  { event := event105294
    frameStart := 0 },
  { event := event105295
    frameStart := 0 }
]

def eventLeaf6581 : Array AnnotatedEvent := #[
  { event := event105296
    frameStart := 0 },
  { event := event105297
    frameStart := 0 },
  { event := event105298
    frameStart := 0 },
  { event := event105299
    frameStart := 0 },
  { event := event105300
    frameStart := 0 },
  { event := event105301
    frameStart := 0 },
  { event := event105302
    frameStart := 0 },
  { event := event105303
    frameStart := 0 },
  { event := event105304
    frameStart := 0 },
  { event := event105305
    frameStart := 0 },
  { event := event105306
    frameStart := 0 },
  { event := event105307
    frameStart := 0 },
  { event := event105308
    frameStart := 0 },
  { event := event105309
    frameStart := 0 },
  { event := event105310
    frameStart := 0 },
  { event := event105311
    frameStart := 0 }
]

def eventLeaf6582 : Array AnnotatedEvent := #[
  { event := event105312
    frameStart := 0 },
  { event := event105313
    frameStart := 0 },
  { event := event105314
    frameStart := 105314 },
  { event := event105315
    frameStart := 105314 },
  { event := event105316
    frameStart := 105314 },
  { event := event105317
    frameStart := 105314 },
  { event := event105318
    frameStart := 105314 },
  { event := event105319
    frameStart := 105314 },
  { event := event105320
    frameStart := 105314 },
  { event := event105321
    frameStart := 105314 },
  { event := event105322
    frameStart := 105314 },
  { event := event105323
    frameStart := 105314 },
  { event := event105324
    frameStart := 105314 },
  { event := event105325
    frameStart := 105314 },
  { event := event105326
    frameStart := 105314 },
  { event := event105327
    frameStart := 105314 }
]

def eventLeaf6583 : Array AnnotatedEvent := #[
  { event := event105328
    frameStart := 105314 },
  { event := event105329
    frameStart := 105314 },
  { event := event105330
    frameStart := 105314 },
  { event := event105331
    frameStart := 105314 },
  { event := event105332
    frameStart := 105314 },
  { event := event105333
    frameStart := 105314 },
  { event := event105334
    frameStart := 105314 },
  { event := event105335
    frameStart := 105314 },
  { event := event105336
    frameStart := 105314 },
  { event := event105337
    frameStart := 105314 },
  { event := event105338
    frameStart := 105314 },
  { event := event105339
    frameStart := 105314 },
  { event := event105340
    frameStart := 105314 },
  { event := event105341
    frameStart := 105314 },
  { event := event105342
    frameStart := 105314 },
  { event := event105343
    frameStart := 105314 }
]

def eventLeaf6584 : Array AnnotatedEvent := #[
  { event := event105344
    frameStart := 105314 },
  { event := event105345
    frameStart := 105314 },
  { event := event105346
    frameStart := 105314 },
  { event := event105347
    frameStart := 105314 },
  { event := event105348
    frameStart := 105314 },
  { event := event105349
    frameStart := 105314 },
  { event := event105350
    frameStart := 105314 },
  { event := event105351
    frameStart := 105314 },
  { event := event105352
    frameStart := 105314 },
  { event := event105353
    frameStart := 105314 },
  { event := event105354
    frameStart := 105314 },
  { event := event105355
    frameStart := 105314 },
  { event := event105356
    frameStart := 105356 },
  { event := event105357
    frameStart := 105356 },
  { event := event105358
    frameStart := 105356 },
  { event := event105359
    frameStart := 105356 }
]

def eventLeaf6585 : Array AnnotatedEvent := #[
  { event := event105360
    frameStart := 105356 },
  { event := event105361
    frameStart := 105356 },
  { event := event105362
    frameStart := 105356 },
  { event := event105363
    frameStart := 105356 },
  { event := event105364
    frameStart := 105356 },
  { event := event105365
    frameStart := 105356 },
  { event := event105366
    frameStart := 105356 },
  { event := event105367
    frameStart := 105356 },
  { event := event105368
    frameStart := 105356 },
  { event := event105369
    frameStart := 105356 },
  { event := event105370
    frameStart := 105356 },
  { event := event105371
    frameStart := 105356 },
  { event := event105372
    frameStart := 105356 },
  { event := event105373
    frameStart := 105356 },
  { event := event105374
    frameStart := 105356 },
  { event := event105375
    frameStart := 105356 }
]

def eventLeaf6586 : Array AnnotatedEvent := #[
  { event := event105376
    frameStart := 105356 },
  { event := event105377
    frameStart := 105356 },
  { event := event105378
    frameStart := 105356 },
  { event := event105379
    frameStart := 105356 },
  { event := event105380
    frameStart := 105356 },
  { event := event105381
    frameStart := 105356 },
  { event := event105382
    frameStart := 105356 },
  { event := event105383
    frameStart := 105356 },
  { event := event105384
    frameStart := 105356 },
  { event := event105385
    frameStart := 105356 },
  { event := event105386
    frameStart := 105356 },
  { event := event105387
    frameStart := 105356 },
  { event := event105388
    frameStart := 105356 },
  { event := event105389
    frameStart := 105356 },
  { event := event105390
    frameStart := 105356 },
  { event := event105391
    frameStart := 105356 }
]

def eventLeaf6587 : Array AnnotatedEvent := #[
  { event := event105392
    frameStart := 105356 },
  { event := event105393
    frameStart := 105356 },
  { event := event105394
    frameStart := 105356 },
  { event := event105395
    frameStart := 105356 },
  { event := event105396
    frameStart := 105356 },
  { event := event105397
    frameStart := 105356 },
  { event := event105398
    frameStart := 105356 },
  { event := event105399
    frameStart := 105356 },
  { event := event105400
    frameStart := 105356 },
  { event := event105401
    frameStart := 105356 },
  { event := event105402
    frameStart := 105356 },
  { event := event105403
    frameStart := 105356 },
  { event := event105404
    frameStart := 105356 },
  { event := event105405
    frameStart := 105356 },
  { event := event105406
    frameStart := 105356 },
  { event := event105407
    frameStart := 105356 }
]

def eventLeaf6588 : Array AnnotatedEvent := #[
  { event := event105408
    frameStart := 105356 },
  { event := event105409
    frameStart := 105356 },
  { event := event105410
    frameStart := 105356 },
  { event := event105411
    frameStart := 105356 },
  { event := event105412
    frameStart := 105356 },
  { event := event105413
    frameStart := 105356 },
  { event := event105414
    frameStart := 105356 },
  { event := event105415
    frameStart := 105356 },
  { event := event105416
    frameStart := 105356 },
  { event := event105417
    frameStart := 105356 },
  { event := event105418
    frameStart := 105356 },
  { event := event105419
    frameStart := 105356 },
  { event := event105420
    frameStart := 105356 },
  { event := event105421
    frameStart := 105356 },
  { event := event105422
    frameStart := 105356 },
  { event := event105423
    frameStart := 105356 }
]

def eventLeaf6589 : Array AnnotatedEvent := #[
  { event := event105424
    frameStart := 105356 },
  { event := event105425
    frameStart := 105356 },
  { event := event105426
    frameStart := 105356 },
  { event := event105427
    frameStart := 105356 },
  { event := event105428
    frameStart := 105356 },
  { event := event105429
    frameStart := 105356 },
  { event := event105430
    frameStart := 105356 },
  { event := event105431
    frameStart := 105356 },
  { event := event105432
    frameStart := 105356 },
  { event := event105433
    frameStart := 105356 },
  { event := event105434
    frameStart := 105356 },
  { event := event105435
    frameStart := 105356 },
  { event := event105436
    frameStart := 105356 },
  { event := event105437
    frameStart := 105356 },
  { event := event105438
    frameStart := 105356 },
  { event := event105439
    frameStart := 105356 }
]

def eventLeaf6590 : Array AnnotatedEvent := #[
  { event := event105440
    frameStart := 105356 },
  { event := event105441
    frameStart := 105356 },
  { event := event105442
    frameStart := 105356 },
  { event := event105443
    frameStart := 105356 },
  { event := event105444
    frameStart := 105356 },
  { event := event105445
    frameStart := 105356 },
  { event := event105446
    frameStart := 105356 },
  { event := event105447
    frameStart := 105356 },
  { event := event105448
    frameStart := 0 },
  { event := event105449
    frameStart := 0 },
  { event := event105450
    frameStart := 0 },
  { event := event105451
    frameStart := 0 },
  { event := event105452
    frameStart := 0 },
  { event := event105453
    frameStart := 0 },
  { event := event105454
    frameStart := 0 },
  { event := event105455
    frameStart := 0 }
]

def eventLeaf6591 : Array AnnotatedEvent := #[
  { event := event105456
    frameStart := 0 },
  { event := event105457
    frameStart := 0 },
  { event := event105458
    frameStart := 0 },
  { event := event105459
    frameStart := 0 },
  { event := event105460
    frameStart := 0 },
  { event := event105461
    frameStart := 0 },
  { event := event105462
    frameStart := 0 },
  { event := event105463
    frameStart := 0 },
  { event := event105464
    frameStart := 0 },
  { event := event105465
    frameStart := 0 },
  { event := event105466
    frameStart := 0 },
  { event := event105467
    frameStart := 0 },
  { event := event105468
    frameStart := 0 },
  { event := event105469
    frameStart := 0 },
  { event := event105470
    frameStart := 0 },
  { event := event105471
    frameStart := 0 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events411
