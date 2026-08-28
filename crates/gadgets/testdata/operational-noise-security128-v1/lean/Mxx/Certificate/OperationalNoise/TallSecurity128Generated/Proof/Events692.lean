import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events692

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event177152 : Event := .predecessor (⟨.program ⟨257⟩, ⟨22754⟩⟩) 1 ⟨2370⟩ 4

def event177153 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨22754⟩⟩) (.scale (.predecessor 0 177151 .coefficient) (.value (.predecessor 1 177152 .coefficient)))

def exact177154RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨22752⟩⟩]⟩, (1)⟩]

theorem exact177154RawTermsValid :
    exact177154RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event177154 : Event := .resultExact (⟨.program ⟨257⟩, ⟨22754⟩⟩) exact177154RawTerms (.finite 5647228698) 177153 .exactZero (none)

def event177155 : Event := .predecessor (⟨.program ⟨257⟩, ⟨22755⟩⟩) 0 ⟨6466⟩ 163745

def event177156 : Event := .predecessor (⟨.program ⟨257⟩, ⟨22755⟩⟩) 1 ⟨22754⟩ 177154

def event177157 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨22755⟩⟩) (.product (.predecessor 0 177155 .coefficient) (.predecessor 1 177156 .coefficient) (⟨false, false, none, none, none⟩))

def event177158 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨22755⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨22752⟩⟩]⟩) [⟨.result 177150 .coefficient, false, none⟩])

def event177159 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨22755⟩⟩) (.product (.result 163745 .summary) (.transfer 177158) (⟨false, false, none, none, none⟩))

def event177160 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨22755⟩⟩, .operator (⟨163745, 0⟩, ⟨177154, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22752⟩⟩]⟩, (1)⟩)

def event177161 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨22753⟩⟩)

def event177162 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event177163 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event177164 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6449⟩⟩) (.authority (.operator))

def event177165 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨6449⟩⟩) (.finite 9)

def event177166 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event177167 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event177168 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event177169 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event177170 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 177169

def event177171 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 177167

def event177172 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 177170 .coefficient) (.value (.predecessor 1 177171 .coefficient)))

def event177173 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event177174 : Event := .predecessor (⟨.program ⟨257⟩, ⟨6451⟩⟩) 0 ⟨392⟩ 177173

def event177175 : Event := .predecessor (⟨.program ⟨257⟩, ⟨6451⟩⟩) 1 ⟨6449⟩ 177165

def event177176 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6451⟩⟩) (.sum [.predecessor 0 177174 .coefficient, .predecessor 1 177175 .coefficient])

def event177177 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨6451⟩⟩) (.finite 655349)

def event177178 : Event := .predecessor (⟨.program ⟨257⟩, ⟨6462⟩⟩) 0 ⟨6451⟩ 177177

def event177179 : Event := .predecessor (⟨.program ⟨257⟩, ⟨6462⟩⟩) 1 ⟨5426⟩ 177163

def event177180 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6462⟩⟩) (.identity (.predecessor 1 177179 .coefficient))

def event177181 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨6462⟩⟩) (.finite 655360)

def event177182 : Event := .predecessor (⟨.program ⟨257⟩, ⟨21590⟩⟩) 0 ⟨6462⟩ 177181

def event177183 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨21590⟩⟩) (.authority (.programFamilyFact))

def exact177184RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨21590⟩⟩], []⟩, (1)⟩]

theorem exact177184RawTermsValid :
    exact177184RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event177184 : Event := .resultExact (⟨.program ⟨257⟩, ⟨21590⟩⟩) exact177184RawTerms (.finite 4) 177183 .exactZero (none)

def event177185 : Event := .predecessor (⟨.program ⟨257⟩, ⟨21161⟩⟩) 0 ⟨6462⟩ 177181

def event177186 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨21161⟩⟩) (.authority (.programFamilyFact))

def exact177187RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨21161⟩⟩], []⟩, (1)⟩]

theorem exact177187RawTermsValid :
    exact177187RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event177187 : Event := .resultExact (⟨.program ⟨257⟩, ⟨21161⟩⟩) exact177187RawTerms (.finite 4) 177186 .exactZero (none)

def event177188 : Event := .predecessor (⟨.program ⟨257⟩, ⟨21591⟩⟩) 0 ⟨21161⟩ 177187

def event177189 : Event := .predecessor (⟨.program ⟨257⟩, ⟨21591⟩⟩) 1 ⟨21590⟩ 177184

def event177190 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨21591⟩⟩) (.product (.predecessor 0 177188 .coefficient) (.predecessor 1 177189 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event177191 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨21591⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨21161⟩⟩, ⟨.program ⟨257⟩, ⟨21590⟩⟩], []⟩) [⟨.result 177187 .coefficient, true, some 1⟩, ⟨.result 177184 .coefficient, true, some 1⟩])

def event177192 : Event := .survivorFold (1) 177191

def exact177193RawTerms : List Term := []

theorem exact177193RawTermsValid :
    exact177193RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event177193 : Event := .resultExact (⟨.program ⟨257⟩, ⟨21591⟩⟩) exact177193RawTerms (.finite 16) 177190 (.finite 16) (some (177191))

def event177194 : Event := .predecessor (⟨.program ⟨257⟩, ⟨21592⟩⟩) 0 ⟨21591⟩ 177193

def event177195 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨21592⟩⟩) (.identity (.predecessor 0 177194 .coefficient))

def event177196 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨21592⟩⟩) (.finite 16)

def event177197 : Event := .predecessor (⟨.program ⟨257⟩, ⟨21840⟩⟩) 0 ⟨21592⟩ 177196

def event177198 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨21840⟩⟩) (.authority (.programFamilyFact))

def exact177199RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨21840⟩⟩], []⟩, (1)⟩]

theorem exact177199RawTermsValid :
    exact177199RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event177199 : Event := .resultExact (⟨.program ⟨257⟩, ⟨21840⟩⟩) exact177199RawTerms (.finite 4) 177198 .exactZero (none)

def event177200 : Event := .predecessor (⟨.program ⟨257⟩, ⟨21841⟩⟩) 0 ⟨21840⟩ 177199

def event177201 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨21841⟩⟩) (.identity (.predecessor 0 177200 .coefficient))

def event177202 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨21841⟩⟩) (.finite 4)

def event177203 : Event := .predecessor (⟨.program ⟨257⟩, ⟨22752⟩⟩) 0 ⟨21841⟩ 177202

def event177204 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨22752⟩⟩) (.authority (.relationPreimageSource ⟨60⟩))

def exact177205RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨22752⟩⟩]⟩, (1)⟩]

theorem exact177205RawTermsValid :
    exact177205RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event177205 : Event := .resultExact (⟨.program ⟨257⟩, ⟨22752⟩⟩) exact177205RawTerms (.finite 5647228698) 177204 .exactZero (none)

def event177206 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35⟩⟩) (.authority (.operator))

def exact177207RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩]⟩, (1)⟩]

theorem exact177207RawTermsValid :
    exact177207RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event177207 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35⟩⟩) exact177207RawTerms .large 177206 .exactZero (none)

def event177208 : Event := .predecessor (⟨.program ⟨257⟩, ⟨22753⟩⟩) 0 ⟨35⟩ 177207

def event177209 : Event := .predecessor (⟨.program ⟨257⟩, ⟨22753⟩⟩) 1 ⟨22752⟩ 177205

def event177210 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨22753⟩⟩) (.product (.predecessor 0 177208 .coefficient) (.predecessor 1 177209 .coefficient) (⟨false, false, none, none, none⟩))

def event177211 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨22753⟩⟩, .operator (⟨177207, 0⟩, ⟨177205, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22752⟩⟩]⟩, (1)⟩)

def exact177212RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22752⟩⟩]⟩, (1)⟩]

theorem exact177212RawTermsValid :
    exact177212RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event177212 : Event := .resultExact (⟨.program ⟨257⟩, ⟨22753⟩⟩) exact177212RawTerms .large 177210 .exactZero (none)

def event177213 : Event := .preFoldPolynomial 177212 [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22752⟩⟩]⟩, (1)⟩] .exactZero none

def exact177214RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22752⟩⟩]⟩, (1)⟩]

def event177214 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨22753⟩⟩) 177213 exact177214RawTerms .large 177210 .exactZero (none)

def event177215 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨23995⟩⟩)

def event177216 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event177217 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event177218 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6449⟩⟩) (.authority (.operator))

def event177219 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨6449⟩⟩) (.finite 9)

def event177220 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event177221 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event177222 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event177223 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event177224 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 177223

def event177225 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 177221

def event177226 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 177224 .coefficient) (.value (.predecessor 1 177225 .coefficient)))

def event177227 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event177228 : Event := .predecessor (⟨.program ⟨257⟩, ⟨6451⟩⟩) 0 ⟨392⟩ 177227

def event177229 : Event := .predecessor (⟨.program ⟨257⟩, ⟨6451⟩⟩) 1 ⟨6449⟩ 177219

def event177230 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6451⟩⟩) (.sum [.predecessor 0 177228 .coefficient, .predecessor 1 177229 .coefficient])

def event177231 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨6451⟩⟩) (.finite 655349)

def event177232 : Event := .predecessor (⟨.program ⟨257⟩, ⟨6462⟩⟩) 0 ⟨6451⟩ 177231

def event177233 : Event := .predecessor (⟨.program ⟨257⟩, ⟨6462⟩⟩) 1 ⟨5426⟩ 177217

def event177234 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6462⟩⟩) (.identity (.predecessor 1 177233 .coefficient))

def event177235 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨6462⟩⟩) (.finite 655360)

def event177236 : Event := .predecessor (⟨.program ⟨257⟩, ⟨21590⟩⟩) 0 ⟨6462⟩ 177235

def event177237 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨21590⟩⟩) (.authority (.programFamilyFact))

def exact177238RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨21590⟩⟩], []⟩, (1)⟩]

theorem exact177238RawTermsValid :
    exact177238RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event177238 : Event := .resultExact (⟨.program ⟨257⟩, ⟨21590⟩⟩) exact177238RawTerms (.finite 4) 177237 .exactZero (none)

def event177239 : Event := .predecessor (⟨.program ⟨257⟩, ⟨21161⟩⟩) 0 ⟨6462⟩ 177235

def event177240 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨21161⟩⟩) (.authority (.programFamilyFact))

def exact177241RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨21161⟩⟩], []⟩, (1)⟩]

theorem exact177241RawTermsValid :
    exact177241RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event177241 : Event := .resultExact (⟨.program ⟨257⟩, ⟨21161⟩⟩) exact177241RawTerms (.finite 4) 177240 .exactZero (none)

def event177242 : Event := .predecessor (⟨.program ⟨257⟩, ⟨21591⟩⟩) 0 ⟨21161⟩ 177241

def event177243 : Event := .predecessor (⟨.program ⟨257⟩, ⟨21591⟩⟩) 1 ⟨21590⟩ 177238

def event177244 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨21591⟩⟩) (.product (.predecessor 0 177242 .coefficient) (.predecessor 1 177243 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event177245 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨21591⟩⟩, .operator (⟨177241, 0⟩, ⟨177238, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨21161⟩⟩, ⟨.program ⟨257⟩, ⟨21590⟩⟩], []⟩, (1)⟩)

def exact177246RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨21161⟩⟩, ⟨.program ⟨257⟩, ⟨21590⟩⟩], []⟩, (1)⟩]

theorem exact177246RawTermsValid :
    exact177246RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event177246 : Event := .resultExact (⟨.program ⟨257⟩, ⟨21591⟩⟩) exact177246RawTerms (.finite 16) 177244 .exactZero (none)

def event177247 : Event := .predecessor (⟨.program ⟨257⟩, ⟨21592⟩⟩) 0 ⟨21591⟩ 177246

def event177248 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨21592⟩⟩) (.identity (.predecessor 0 177247 .coefficient))

def event177249 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨21592⟩⟩) (.finite 16)

def event177250 : Event := .predecessor (⟨.program ⟨257⟩, ⟨21840⟩⟩) 0 ⟨21592⟩ 177249

def event177251 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨21840⟩⟩) (.authority (.programFamilyFact))

def exact177252RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨21840⟩⟩], []⟩, (1)⟩]

theorem exact177252RawTermsValid :
    exact177252RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event177252 : Event := .resultExact (⟨.program ⟨257⟩, ⟨21840⟩⟩) exact177252RawTerms (.finite 4) 177251 .exactZero (none)

def event177253 : Event := .predecessor (⟨.program ⟨257⟩, ⟨21841⟩⟩) 0 ⟨21840⟩ 177252

def event177254 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨21841⟩⟩) (.identity (.predecessor 0 177253 .coefficient))

def event177255 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨21841⟩⟩) (.finite 4)

def event177256 : Event := .predecessor (⟨.program ⟨257⟩, ⟨23115⟩⟩) 0 ⟨21841⟩ 177255

def event177257 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨23115⟩⟩) (.authority (.programFamilyFact))

def event177258 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨23115⟩⟩) (.finite 3720)

def event177259 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨7177⟩⟩) .missing

def event177260 : Event := .predecessor (⟨.program ⟨257⟩, ⟨23116⟩⟩) 0 ⟨7177⟩ 177259

def event177261 : Event := .predecessor (⟨.program ⟨257⟩, ⟨23116⟩⟩) 1 ⟨23115⟩ 177258

def event177262 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨23116⟩⟩) (.authority (.operator))

def exact177263RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨23116⟩⟩]⟩, (1)⟩]

theorem exact177263RawTermsValid :
    exact177263RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event177263 : Event := .resultExact (⟨.program ⟨257⟩, ⟨23116⟩⟩) exact177263RawTerms .large 177262 .exactZero (none)

def event177264 : Event := .predecessor (⟨.program ⟨257⟩, ⟨23989⟩⟩) 0 ⟨23116⟩ 177263

def event177265 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨23989⟩⟩) (.authority (.operator))

def exact177266RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨23989⟩⟩]⟩, (1)⟩]

theorem exact177266RawTermsValid :
    exact177266RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event177266 : Event := .resultExact (⟨.program ⟨257⟩, ⟨23989⟩⟩) exact177266RawTerms (.finite 8192) 177265 .exactZero (none)

def event177267 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨136⟩⟩) (.authority (.operator))

def event177268 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨136⟩⟩) .exactZero

def event177269 : Event := .predecessor (⟨.program ⟨257⟩, ⟨23302⟩⟩) 0 ⟨21841⟩ 177255

def event177270 : Event := .predecessor (⟨.program ⟨257⟩, ⟨23302⟩⟩) 1 ⟨136⟩ 177268

def event177271 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨23302⟩⟩) (.sum [.predecessor 0 177269 .coefficient, .predecessor 1 177270 .coefficient])

def event177272 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨23302⟩⟩) (.finite 4)

def event177273 : Event := .predecessor (⟨.program ⟨257⟩, ⟨23303⟩⟩) 0 ⟨23302⟩ 177272

def event177274 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨23303⟩⟩) (.identity (.predecessor 0 177273 .coefficient))

def exact177275RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨21840⟩⟩], []⟩, (1)⟩]

theorem exact177275RawTermsValid :
    exact177275RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event177275 : Event := .resultExact (⟨.program ⟨257⟩, ⟨23303⟩⟩) exact177275RawTerms (.finite 4) 177274 .exactZero (none)

def event177276 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6908⟩⟩) (.authority (.factStore))

def exact177277RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact177277RawTermsValid :
    exact177277RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event177277 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6908⟩⟩) exact177277RawTerms .large 177276 .exactZero (none)

def event177278 : Event := .predecessor (⟨.program ⟨257⟩, ⟨23304⟩⟩) 0 ⟨6908⟩ 177277

def event177279 : Event := .predecessor (⟨.program ⟨257⟩, ⟨23304⟩⟩) 1 ⟨23303⟩ 177275

def event177280 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨23304⟩⟩) (.product (.predecessor 0 177278 .coefficient) (.predecessor 1 177279 .coefficient) (⟨false, false, none, none, none⟩))

def event177281 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨23304⟩⟩, .operator (⟨177277, 0⟩, ⟨177275, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨21840⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact177282RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨21840⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact177282RawTermsValid :
    exact177282RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event177282 : Event := .resultExact (⟨.program ⟨257⟩, ⟨23304⟩⟩) exact177282RawTerms .large 177280 .exactZero (none)

def event177283 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7181⟩⟩) 0 ⟨7177⟩ 177259

def event177284 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7181⟩⟩) (.authority (.operator))

def exact177285RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7181⟩⟩]⟩, (1)⟩]

theorem exact177285RawTermsValid :
    exact177285RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event177285 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7181⟩⟩) exact177285RawTerms .large 177284 .exactZero (none)

def event177286 : Event := .predecessor (⟨.program ⟨257⟩, ⟨23305⟩⟩) 0 ⟨7181⟩ 177285

def event177287 : Event := .predecessor (⟨.program ⟨257⟩, ⟨23305⟩⟩) 1 ⟨23304⟩ 177282

def event177288 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨23305⟩⟩) (.sum [.predecessor 0 177286 .coefficient, .predecessor 1 177287 .coefficient])

def exact177289RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7181⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨21840⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact177289RawTermsValid :
    exact177289RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event177289 : Event := .resultExact (⟨.program ⟨257⟩, ⟨23305⟩⟩) exact177289RawTerms .large 177288 .exactZero (none)

def event177290 : Event := .predecessor (⟨.program ⟨257⟩, ⟨23990⟩⟩) 0 ⟨23305⟩ 177289

def event177291 : Event := .predecessor (⟨.program ⟨257⟩, ⟨23990⟩⟩) 1 ⟨23989⟩ 177266

def event177292 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨23990⟩⟩) (.product (.predecessor 0 177290 .coefficient) (.predecessor 1 177291 .coefficient) (⟨false, false, none, none, none⟩))

def event177293 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨23990⟩⟩, .operator (⟨177289, 0⟩, ⟨177266, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7181⟩⟩, ⟨.program ⟨257⟩, ⟨23989⟩⟩]⟩, (1)⟩)

def event177294 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨23990⟩⟩, .operator (⟨177289, 1⟩, ⟨177266, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨21840⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨23989⟩⟩]⟩, (-1)⟩)

def event177295 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨23990⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨21840⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨23989⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨23989⟩⟩) ⟨23116⟩ 177263)

def event177296 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨23990⟩⟩, .relation 177295 0, ⟨[⟨.program ⟨257⟩, ⟨21840⟩⟩], [⟨.program ⟨257⟩, ⟨23116⟩⟩]⟩, (-1)⟩)

def exact177297RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7181⟩⟩, ⟨.program ⟨257⟩, ⟨23989⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨21840⟩⟩], [⟨.program ⟨257⟩, ⟨23116⟩⟩]⟩, (-1)⟩]

theorem exact177297RawTermsValid :
    exact177297RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event177297 : Event := .resultExact (⟨.program ⟨257⟩, ⟨23990⟩⟩) exact177297RawTerms .large 177292 .exactZero (none)

def event177298 : Event := .predecessor (⟨.program ⟨257⟩, ⟨22157⟩⟩) 0 ⟨21841⟩ 177255

def event177299 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨22157⟩⟩) (.authority (.programFamilyFact))

def exact177300RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨22157⟩⟩], []⟩, (1)⟩]

theorem exact177300RawTermsValid :
    exact177300RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event177300 : Event := .resultExact (⟨.program ⟨257⟩, ⟨22157⟩⟩) exact177300RawTerms (.finite 4) 177299 .exactZero (none)

def event177301 : Event := .predecessor (⟨.program ⟨257⟩, ⟨22160⟩⟩) 0 ⟨6908⟩ 177277

def event177302 : Event := .predecessor (⟨.program ⟨257⟩, ⟨22160⟩⟩) 1 ⟨22157⟩ 177300

def event177303 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨22160⟩⟩) (.product (.predecessor 0 177301 .coefficient) (.predecessor 1 177302 .coefficient) (⟨false, true, none, none, some 1⟩))

def event177304 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨22160⟩⟩, .operator (⟨177277, 0⟩, ⟨177300, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨22157⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact177305RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨22157⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact177305RawTermsValid :
    exact177305RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event177305 : Event := .resultExact (⟨.program ⟨257⟩, ⟨22160⟩⟩) exact177305RawTerms .large 177303 .exactZero (none)

def event177306 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7201⟩⟩) 0 ⟨7177⟩ 177259

def event177307 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7201⟩⟩) (.authority (.operator))

def exact177308RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7201⟩⟩]⟩, (1)⟩]

theorem exact177308RawTermsValid :
    exact177308RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event177308 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7201⟩⟩) exact177308RawTerms .large 177307 .exactZero (none)

def event177309 : Event := .predecessor (⟨.program ⟨257⟩, ⟨22161⟩⟩) 0 ⟨7201⟩ 177308

def event177310 : Event := .predecessor (⟨.program ⟨257⟩, ⟨22161⟩⟩) 1 ⟨22160⟩ 177305

def event177311 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨22161⟩⟩) (.sum [.predecessor 0 177309 .coefficient, .predecessor 1 177310 .coefficient])

def exact177312RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7201⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨22157⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact177312RawTermsValid :
    exact177312RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event177312 : Event := .resultExact (⟨.program ⟨257⟩, ⟨22161⟩⟩) exact177312RawTerms .large 177311 .exactZero (none)

def event177313 : Event := .predecessor (⟨.program ⟨257⟩, ⟨23995⟩⟩) 0 ⟨22161⟩ 177312

def event177314 : Event := .predecessor (⟨.program ⟨257⟩, ⟨23995⟩⟩) 1 ⟨23990⟩ 177297

def event177315 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨23995⟩⟩) (.sum [.predecessor 0 177313 .coefficient, .predecessor 1 177314 .coefficient])

def exact177316RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7181⟩⟩, ⟨.program ⟨257⟩, ⟨23989⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7201⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨21840⟩⟩], [⟨.program ⟨257⟩, ⟨23116⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨22157⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact177316RawTermsValid :
    exact177316RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event177316 : Event := .resultExact (⟨.program ⟨257⟩, ⟨23995⟩⟩) exact177316RawTerms .large 177315 .exactZero (none)

def event177317 : Event := .preFoldPolynomial 177316 [⟨⟨[], [⟨.program ⟨257⟩, ⟨7181⟩⟩, ⟨.program ⟨257⟩, ⟨23989⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7201⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨21840⟩⟩], [⟨.program ⟨257⟩, ⟨23116⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨22157⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩] .exactZero none

def exact177318RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7181⟩⟩, ⟨.program ⟨257⟩, ⟨23989⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7201⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨21840⟩⟩], [⟨.program ⟨257⟩, ⟨23116⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨22157⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

def event177318 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨23995⟩⟩) 177317 exact177318RawTerms .large 177315 .exactZero (none)

def event177319 : Event := .specializationComputed (⟨.program ⟨257⟩, ⟨21841⟩⟩) ⟨⟨80⟩, ⟨60⟩, ⟨135⟩⟩ ⟨177161, 177319⟩

def event177320 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨22755⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22752⟩⟩]⟩) (1) 0 2 (.universal 177319 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22752⟩⟩]⟩) (none) 177318)

def event177321 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨22755⟩⟩, .relation 177320 1, ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩]⟩, (1)⟩)

def event177322 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨22755⟩⟩, .relation 177320 0, ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7181⟩⟩, ⟨.program ⟨257⟩, ⟨23989⟩⟩]⟩, (-1)⟩)

def event177323 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨22755⟩⟩, .relation 177320 2, ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨21840⟩⟩], [⟨.program ⟨257⟩, ⟨23116⟩⟩]⟩, (1)⟩)

def event177324 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨22755⟩⟩, .relation 177320 3, ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨22157⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact177325RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7181⟩⟩, ⟨.program ⟨257⟩, ⟨23989⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨21840⟩⟩], [⟨.program ⟨257⟩, ⟨23116⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨22157⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact177325RawTermsValid :
    exact177325RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event177325 : Event := .resultExact (⟨.program ⟨257⟩, ⟨22755⟩⟩) exact177325RawTerms .large 177157 (.finite 202072841853861888) (some (177159))

def event177326 : Event := .predecessor (⟨.program ⟨257⟩, ⟨23992⟩⟩) 0 ⟨22755⟩ 177325

def event177327 : Event := .predecessor (⟨.program ⟨257⟩, ⟨23992⟩⟩) 1 ⟨23991⟩ 177147

def event177328 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨23992⟩⟩) (.sum [.predecessor 0 177326 .coefficient, .predecessor 1 177327 .coefficient])

def event177329 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨23992⟩⟩, .operator (⟨177325, 0⟩, ⟨177147, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7181⟩⟩, ⟨.program ⟨257⟩, ⟨23989⟩⟩]⟩, (1)⟩)

def event177330 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨23992⟩⟩, .operator (⟨177325, 2⟩, ⟨177147, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨21840⟩⟩], [⟨.program ⟨257⟩, ⟨23116⟩⟩]⟩, (-1)⟩)

def event177331 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨23992⟩⟩) (.sum [.result 177325 .summary, .result 177147 .summary])

def exact177332RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨22157⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact177332RawTermsValid :
    exact177332RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event177332 : Event := .resultExact (⟨.program ⟨257⟩, ⟨23992⟩⟩) exact177332RawTerms .large 177328 (.finite 32189003662929394266751515230208) (some (177331))

def event177333 : Event := .predecessor (⟨.program ⟨257⟩, ⟨23993⟩⟩) 0 ⟨23992⟩ 177332

def event177334 : Event := .predecessor (⟨.program ⟨257⟩, ⟨23993⟩⟩) 1 ⟨7156⟩ 15842

def event177335 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨23993⟩⟩) (.product (.predecessor 0 177333 .coefficient) (.predecessor 1 177334 .coefficient) (⟨false, false, none, none, none⟩))

def event177336 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨23993⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩) [⟨.result 15838 .coefficient, false, none⟩])

def event177337 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨23993⟩⟩) (.product (.result 177332 .summary) (.transfer 177336) (⟨false, false, none, none, none⟩))

def event177338 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨23993⟩⟩, .operator (⟨177332, 0⟩, ⟨15842, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩, (1)⟩)

def event177339 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨23993⟩⟩, .operator (⟨177332, 1⟩, ⟨15842, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨22157⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩, (-1)⟩)

def event177340 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨23993⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨22157⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨7155⟩⟩) ⟨7043⟩ 15835)

def event177341 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨23993⟩⟩, .relation 177340 0, ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22157⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact177342RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22157⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact177342RawTermsValid :
    exact177342RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event177342 : Event := .resultExact (⟨.program ⟨257⟩, ⟨23993⟩⟩) exact177342RawTerms .large 177335 (.finite 345626795057764889831969145180473178193920) (some (177337))

def event177343 : Event := .predecessor (⟨.program ⟨257⟩, ⟨19896⟩⟩) 0 ⟨7177⟩ 15500

def event177344 : Event := .predecessor (⟨.program ⟨257⟩, ⟨19896⟩⟩) 1 ⟨19895⟩ 171359

def event177345 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨19896⟩⟩) (.authority (.operator))

def exact177346RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨19896⟩⟩]⟩, (1)⟩]

theorem exact177346RawTermsValid :
    exact177346RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event177346 : Event := .resultExact (⟨.program ⟨257⟩, ⟨19896⟩⟩) exact177346RawTerms .large 177345 .exactZero (none)

def event177347 : Event := .predecessor (⟨.program ⟨257⟩, ⟨20769⟩⟩) 0 ⟨19896⟩ 177346

def event177348 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨20769⟩⟩) (.authority (.operator))

def exact177349RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨20769⟩⟩]⟩, (1)⟩]

theorem exact177349RawTermsValid :
    exact177349RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event177349 : Event := .resultExact (⟨.program ⟨257⟩, ⟨20769⟩⟩) exact177349RawTerms (.finite 8192) 177348 .exactZero (none)

def event177350 : Event := .predecessor (⟨.program ⟨257⟩, ⟨20771⟩⟩) 0 ⟨20265⟩ 171643

def event177351 : Event := .predecessor (⟨.program ⟨257⟩, ⟨20771⟩⟩) 1 ⟨20769⟩ 177349

def event177352 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨20771⟩⟩) (.product (.predecessor 0 177350 .coefficient) (.predecessor 1 177351 .coefficient) (⟨false, false, none, none, none⟩))

def event177353 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨20771⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨20769⟩⟩]⟩) [⟨.result 177349 .coefficient, false, none⟩])

def event177354 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨20771⟩⟩) (.product (.result 171643 .summary) (.transfer 177353) (⟨false, false, none, none, none⟩))

def event177355 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨20771⟩⟩, .operator (⟨171643, 0⟩, ⟨177349, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7180⟩⟩, ⟨.program ⟨257⟩, ⟨20769⟩⟩]⟩, (1)⟩)

def event177356 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨20771⟩⟩, .operator (⟨171643, 1⟩, ⟨177349, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨18620⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨20769⟩⟩]⟩, (-1)⟩)

def event177357 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨20771⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨18620⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨20769⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨20769⟩⟩) ⟨19896⟩ 177346)

def event177358 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨20771⟩⟩, .relation 177357 0, ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨18620⟩⟩], [⟨.program ⟨257⟩, ⟨19896⟩⟩]⟩, (-1)⟩)

def exact177359RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7180⟩⟩, ⟨.program ⟨257⟩, ⟨20769⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨18620⟩⟩], [⟨.program ⟨257⟩, ⟨19896⟩⟩]⟩, (-1)⟩]

theorem exact177359RawTermsValid :
    exact177359RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event177359 : Event := .resultExact (⟨.program ⟨257⟩, ⟨20771⟩⟩) exact177359RawTerms .large 177352 (.finite 32188905437706348505289216491520) (some (177354))

def event177360 : Event := .predecessor (⟨.program ⟨257⟩, ⟨19532⟩⟩) 0 ⟨18621⟩ 7959

def event177361 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨19532⟩⟩) (.authority (.relationPreimageSource ⟨58⟩))

def exact177362RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨19532⟩⟩]⟩, (1)⟩]

theorem exact177362RawTermsValid :
    exact177362RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event177362 : Event := .resultExact (⟨.program ⟨257⟩, ⟨19532⟩⟩) exact177362RawTerms (.finite 5647228698) 177361 .exactZero (none)

def event177363 : Event := .predecessor (⟨.program ⟨257⟩, ⟨19534⟩⟩) 0 ⟨19532⟩ 177362

def event177364 : Event := .predecessor (⟨.program ⟨257⟩, ⟨19534⟩⟩) 1 ⟨2370⟩ 4

def event177365 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨19534⟩⟩) (.scale (.predecessor 0 177363 .coefficient) (.value (.predecessor 1 177364 .coefficient)))

def exact177366RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨19532⟩⟩]⟩, (1)⟩]

theorem exact177366RawTermsValid :
    exact177366RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event177366 : Event := .resultExact (⟨.program ⟨257⟩, ⟨19534⟩⟩) exact177366RawTerms (.finite 5647228698) 177365 .exactZero (none)

def event177367 : Event := .predecessor (⟨.program ⟨257⟩, ⟨19535⟩⟩) 0 ⟨6466⟩ 163745

def event177368 : Event := .predecessor (⟨.program ⟨257⟩, ⟨19535⟩⟩) 1 ⟨19534⟩ 177366

def event177369 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨19535⟩⟩) (.product (.predecessor 0 177367 .coefficient) (.predecessor 1 177368 .coefficient) (⟨false, false, none, none, none⟩))

def event177370 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨19535⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨19532⟩⟩]⟩) [⟨.result 177362 .coefficient, false, none⟩])

def event177371 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨19535⟩⟩) (.product (.result 163745 .summary) (.transfer 177370) (⟨false, false, none, none, none⟩))

def event177372 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨19535⟩⟩, .operator (⟨163745, 0⟩, ⟨177366, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19532⟩⟩]⟩, (1)⟩)

def event177373 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨19533⟩⟩)

def event177374 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event177375 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event177376 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6449⟩⟩) (.authority (.operator))

def event177377 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨6449⟩⟩) (.finite 9)

def event177378 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event177379 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event177380 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event177381 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event177382 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 177381

def event177383 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 177379

def event177384 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 177382 .coefficient) (.value (.predecessor 1 177383 .coefficient)))

def event177385 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event177386 : Event := .predecessor (⟨.program ⟨257⟩, ⟨6451⟩⟩) 0 ⟨392⟩ 177385

def event177387 : Event := .predecessor (⟨.program ⟨257⟩, ⟨6451⟩⟩) 1 ⟨6449⟩ 177377

def event177388 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6451⟩⟩) (.sum [.predecessor 0 177386 .coefficient, .predecessor 1 177387 .coefficient])

def event177389 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨6451⟩⟩) (.finite 655349)

def event177390 : Event := .predecessor (⟨.program ⟨257⟩, ⟨6462⟩⟩) 0 ⟨6451⟩ 177389

def event177391 : Event := .predecessor (⟨.program ⟨257⟩, ⟨6462⟩⟩) 1 ⟨5426⟩ 177375

def event177392 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6462⟩⟩) (.identity (.predecessor 1 177391 .coefficient))

def event177393 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨6462⟩⟩) (.finite 655360)

def event177394 : Event := .predecessor (⟨.program ⟨257⟩, ⟨18370⟩⟩) 0 ⟨6462⟩ 177393

def event177395 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨18370⟩⟩) (.authority (.programFamilyFact))

def exact177396RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨18370⟩⟩], []⟩, (1)⟩]

theorem exact177396RawTermsValid :
    exact177396RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event177396 : Event := .resultExact (⟨.program ⟨257⟩, ⟨18370⟩⟩) exact177396RawTerms (.finite 3) 177395 .exactZero (none)

def event177397 : Event := .predecessor (⟨.program ⟨257⟩, ⟨12741⟩⟩) 0 ⟨6462⟩ 177393

def event177398 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨12741⟩⟩) (.authority (.programFamilyFact))

def exact177399RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨12741⟩⟩], []⟩, (1)⟩]

theorem exact177399RawTermsValid :
    exact177399RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event177399 : Event := .resultExact (⟨.program ⟨257⟩, ⟨12741⟩⟩) exact177399RawTerms (.finite 3) 177398 .exactZero (none)

def event177400 : Event := .predecessor (⟨.program ⟨257⟩, ⟨18371⟩⟩) 0 ⟨12741⟩ 177399

def event177401 : Event := .predecessor (⟨.program ⟨257⟩, ⟨18371⟩⟩) 1 ⟨18370⟩ 177396

def event177402 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨18371⟩⟩) (.product (.predecessor 0 177400 .coefficient) (.predecessor 1 177401 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event177403 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨18371⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨12741⟩⟩, ⟨.program ⟨257⟩, ⟨18370⟩⟩], []⟩) [⟨.result 177399 .coefficient, true, some 1⟩, ⟨.result 177396 .coefficient, true, some 1⟩])

def event177404 : Event := .survivorFold (1) 177403

def exact177405RawTerms : List Term := []

theorem exact177405RawTermsValid :
    exact177405RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event177405 : Event := .resultExact (⟨.program ⟨257⟩, ⟨18371⟩⟩) exact177405RawTerms (.finite 9) 177402 (.finite 9) (some (177403))

def event177406 : Event := .predecessor (⟨.program ⟨257⟩, ⟨18372⟩⟩) 0 ⟨18371⟩ 177405

def event177407 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨18372⟩⟩) (.identity (.predecessor 0 177406 .coefficient))

def eventLeaf11072 : Array AnnotatedEvent := #[
  { event := event177152
    frameStart := 0 },
  { event := event177153
    frameStart := 0 },
  { event := event177154
    frameStart := 0 },
  { event := event177155
    frameStart := 0 },
  { event := event177156
    frameStart := 0 },
  { event := event177157
    frameStart := 0 },
  { event := event177158
    frameStart := 0 },
  { event := event177159
    frameStart := 0 },
  { event := event177160
    frameStart := 0 },
  { event := event177161
    frameStart := 177161 },
  { event := event177162
    frameStart := 177161 },
  { event := event177163
    frameStart := 177161 },
  { event := event177164
    frameStart := 177161 },
  { event := event177165
    frameStart := 177161 },
  { event := event177166
    frameStart := 177161 },
  { event := event177167
    frameStart := 177161 }
]

def eventLeaf11073 : Array AnnotatedEvent := #[
  { event := event177168
    frameStart := 177161 },
  { event := event177169
    frameStart := 177161 },
  { event := event177170
    frameStart := 177161 },
  { event := event177171
    frameStart := 177161 },
  { event := event177172
    frameStart := 177161 },
  { event := event177173
    frameStart := 177161 },
  { event := event177174
    frameStart := 177161 },
  { event := event177175
    frameStart := 177161 },
  { event := event177176
    frameStart := 177161 },
  { event := event177177
    frameStart := 177161 },
  { event := event177178
    frameStart := 177161 },
  { event := event177179
    frameStart := 177161 },
  { event := event177180
    frameStart := 177161 },
  { event := event177181
    frameStart := 177161 },
  { event := event177182
    frameStart := 177161 },
  { event := event177183
    frameStart := 177161 }
]

def eventLeaf11074 : Array AnnotatedEvent := #[
  { event := event177184
    frameStart := 177161 },
  { event := event177185
    frameStart := 177161 },
  { event := event177186
    frameStart := 177161 },
  { event := event177187
    frameStart := 177161 },
  { event := event177188
    frameStart := 177161 },
  { event := event177189
    frameStart := 177161 },
  { event := event177190
    frameStart := 177161 },
  { event := event177191
    frameStart := 177161 },
  { event := event177192
    frameStart := 177161 },
  { event := event177193
    frameStart := 177161 },
  { event := event177194
    frameStart := 177161 },
  { event := event177195
    frameStart := 177161 },
  { event := event177196
    frameStart := 177161 },
  { event := event177197
    frameStart := 177161 },
  { event := event177198
    frameStart := 177161 },
  { event := event177199
    frameStart := 177161 }
]

def eventLeaf11075 : Array AnnotatedEvent := #[
  { event := event177200
    frameStart := 177161 },
  { event := event177201
    frameStart := 177161 },
  { event := event177202
    frameStart := 177161 },
  { event := event177203
    frameStart := 177161 },
  { event := event177204
    frameStart := 177161 },
  { event := event177205
    frameStart := 177161 },
  { event := event177206
    frameStart := 177161 },
  { event := event177207
    frameStart := 177161 },
  { event := event177208
    frameStart := 177161 },
  { event := event177209
    frameStart := 177161 },
  { event := event177210
    frameStart := 177161 },
  { event := event177211
    frameStart := 177161 },
  { event := event177212
    frameStart := 177161 },
  { event := event177213
    frameStart := 177161 },
  { event := event177214
    frameStart := 177161 },
  { event := event177215
    frameStart := 177215 }
]

def eventLeaf11076 : Array AnnotatedEvent := #[
  { event := event177216
    frameStart := 177215 },
  { event := event177217
    frameStart := 177215 },
  { event := event177218
    frameStart := 177215 },
  { event := event177219
    frameStart := 177215 },
  { event := event177220
    frameStart := 177215 },
  { event := event177221
    frameStart := 177215 },
  { event := event177222
    frameStart := 177215 },
  { event := event177223
    frameStart := 177215 },
  { event := event177224
    frameStart := 177215 },
  { event := event177225
    frameStart := 177215 },
  { event := event177226
    frameStart := 177215 },
  { event := event177227
    frameStart := 177215 },
  { event := event177228
    frameStart := 177215 },
  { event := event177229
    frameStart := 177215 },
  { event := event177230
    frameStart := 177215 },
  { event := event177231
    frameStart := 177215 }
]

def eventLeaf11077 : Array AnnotatedEvent := #[
  { event := event177232
    frameStart := 177215 },
  { event := event177233
    frameStart := 177215 },
  { event := event177234
    frameStart := 177215 },
  { event := event177235
    frameStart := 177215 },
  { event := event177236
    frameStart := 177215 },
  { event := event177237
    frameStart := 177215 },
  { event := event177238
    frameStart := 177215 },
  { event := event177239
    frameStart := 177215 },
  { event := event177240
    frameStart := 177215 },
  { event := event177241
    frameStart := 177215 },
  { event := event177242
    frameStart := 177215 },
  { event := event177243
    frameStart := 177215 },
  { event := event177244
    frameStart := 177215 },
  { event := event177245
    frameStart := 177215 },
  { event := event177246
    frameStart := 177215 },
  { event := event177247
    frameStart := 177215 }
]

def eventLeaf11078 : Array AnnotatedEvent := #[
  { event := event177248
    frameStart := 177215 },
  { event := event177249
    frameStart := 177215 },
  { event := event177250
    frameStart := 177215 },
  { event := event177251
    frameStart := 177215 },
  { event := event177252
    frameStart := 177215 },
  { event := event177253
    frameStart := 177215 },
  { event := event177254
    frameStart := 177215 },
  { event := event177255
    frameStart := 177215 },
  { event := event177256
    frameStart := 177215 },
  { event := event177257
    frameStart := 177215 },
  { event := event177258
    frameStart := 177215 },
  { event := event177259
    frameStart := 177215 },
  { event := event177260
    frameStart := 177215 },
  { event := event177261
    frameStart := 177215 },
  { event := event177262
    frameStart := 177215 },
  { event := event177263
    frameStart := 177215 }
]

def eventLeaf11079 : Array AnnotatedEvent := #[
  { event := event177264
    frameStart := 177215 },
  { event := event177265
    frameStart := 177215 },
  { event := event177266
    frameStart := 177215 },
  { event := event177267
    frameStart := 177215 },
  { event := event177268
    frameStart := 177215 },
  { event := event177269
    frameStart := 177215 },
  { event := event177270
    frameStart := 177215 },
  { event := event177271
    frameStart := 177215 },
  { event := event177272
    frameStart := 177215 },
  { event := event177273
    frameStart := 177215 },
  { event := event177274
    frameStart := 177215 },
  { event := event177275
    frameStart := 177215 },
  { event := event177276
    frameStart := 177215 },
  { event := event177277
    frameStart := 177215 },
  { event := event177278
    frameStart := 177215 },
  { event := event177279
    frameStart := 177215 }
]

def eventLeaf11080 : Array AnnotatedEvent := #[
  { event := event177280
    frameStart := 177215 },
  { event := event177281
    frameStart := 177215 },
  { event := event177282
    frameStart := 177215 },
  { event := event177283
    frameStart := 177215 },
  { event := event177284
    frameStart := 177215 },
  { event := event177285
    frameStart := 177215 },
  { event := event177286
    frameStart := 177215 },
  { event := event177287
    frameStart := 177215 },
  { event := event177288
    frameStart := 177215 },
  { event := event177289
    frameStart := 177215 },
  { event := event177290
    frameStart := 177215 },
  { event := event177291
    frameStart := 177215 },
  { event := event177292
    frameStart := 177215 },
  { event := event177293
    frameStart := 177215 },
  { event := event177294
    frameStart := 177215 },
  { event := event177295
    frameStart := 177215 }
]

def eventLeaf11081 : Array AnnotatedEvent := #[
  { event := event177296
    frameStart := 177215 },
  { event := event177297
    frameStart := 177215 },
  { event := event177298
    frameStart := 177215 },
  { event := event177299
    frameStart := 177215 },
  { event := event177300
    frameStart := 177215 },
  { event := event177301
    frameStart := 177215 },
  { event := event177302
    frameStart := 177215 },
  { event := event177303
    frameStart := 177215 },
  { event := event177304
    frameStart := 177215 },
  { event := event177305
    frameStart := 177215 },
  { event := event177306
    frameStart := 177215 },
  { event := event177307
    frameStart := 177215 },
  { event := event177308
    frameStart := 177215 },
  { event := event177309
    frameStart := 177215 },
  { event := event177310
    frameStart := 177215 },
  { event := event177311
    frameStart := 177215 }
]

def eventLeaf11082 : Array AnnotatedEvent := #[
  { event := event177312
    frameStart := 177215 },
  { event := event177313
    frameStart := 177215 },
  { event := event177314
    frameStart := 177215 },
  { event := event177315
    frameStart := 177215 },
  { event := event177316
    frameStart := 177215 },
  { event := event177317
    frameStart := 177215 },
  { event := event177318
    frameStart := 177215 },
  { event := event177319
    frameStart := 0 },
  { event := event177320
    frameStart := 0 },
  { event := event177321
    frameStart := 0 },
  { event := event177322
    frameStart := 0 },
  { event := event177323
    frameStart := 0 },
  { event := event177324
    frameStart := 0 },
  { event := event177325
    frameStart := 0 },
  { event := event177326
    frameStart := 0 },
  { event := event177327
    frameStart := 0 }
]

def eventLeaf11083 : Array AnnotatedEvent := #[
  { event := event177328
    frameStart := 0 },
  { event := event177329
    frameStart := 0 },
  { event := event177330
    frameStart := 0 },
  { event := event177331
    frameStart := 0 },
  { event := event177332
    frameStart := 0 },
  { event := event177333
    frameStart := 0 },
  { event := event177334
    frameStart := 0 },
  { event := event177335
    frameStart := 0 },
  { event := event177336
    frameStart := 0 },
  { event := event177337
    frameStart := 0 },
  { event := event177338
    frameStart := 0 },
  { event := event177339
    frameStart := 0 },
  { event := event177340
    frameStart := 0 },
  { event := event177341
    frameStart := 0 },
  { event := event177342
    frameStart := 0 },
  { event := event177343
    frameStart := 0 }
]

def eventLeaf11084 : Array AnnotatedEvent := #[
  { event := event177344
    frameStart := 0 },
  { event := event177345
    frameStart := 0 },
  { event := event177346
    frameStart := 0 },
  { event := event177347
    frameStart := 0 },
  { event := event177348
    frameStart := 0 },
  { event := event177349
    frameStart := 0 },
  { event := event177350
    frameStart := 0 },
  { event := event177351
    frameStart := 0 },
  { event := event177352
    frameStart := 0 },
  { event := event177353
    frameStart := 0 },
  { event := event177354
    frameStart := 0 },
  { event := event177355
    frameStart := 0 },
  { event := event177356
    frameStart := 0 },
  { event := event177357
    frameStart := 0 },
  { event := event177358
    frameStart := 0 },
  { event := event177359
    frameStart := 0 }
]

def eventLeaf11085 : Array AnnotatedEvent := #[
  { event := event177360
    frameStart := 0 },
  { event := event177361
    frameStart := 0 },
  { event := event177362
    frameStart := 0 },
  { event := event177363
    frameStart := 0 },
  { event := event177364
    frameStart := 0 },
  { event := event177365
    frameStart := 0 },
  { event := event177366
    frameStart := 0 },
  { event := event177367
    frameStart := 0 },
  { event := event177368
    frameStart := 0 },
  { event := event177369
    frameStart := 0 },
  { event := event177370
    frameStart := 0 },
  { event := event177371
    frameStart := 0 },
  { event := event177372
    frameStart := 0 },
  { event := event177373
    frameStart := 177373 },
  { event := event177374
    frameStart := 177373 },
  { event := event177375
    frameStart := 177373 }
]

def eventLeaf11086 : Array AnnotatedEvent := #[
  { event := event177376
    frameStart := 177373 },
  { event := event177377
    frameStart := 177373 },
  { event := event177378
    frameStart := 177373 },
  { event := event177379
    frameStart := 177373 },
  { event := event177380
    frameStart := 177373 },
  { event := event177381
    frameStart := 177373 },
  { event := event177382
    frameStart := 177373 },
  { event := event177383
    frameStart := 177373 },
  { event := event177384
    frameStart := 177373 },
  { event := event177385
    frameStart := 177373 },
  { event := event177386
    frameStart := 177373 },
  { event := event177387
    frameStart := 177373 },
  { event := event177388
    frameStart := 177373 },
  { event := event177389
    frameStart := 177373 },
  { event := event177390
    frameStart := 177373 },
  { event := event177391
    frameStart := 177373 }
]

def eventLeaf11087 : Array AnnotatedEvent := #[
  { event := event177392
    frameStart := 177373 },
  { event := event177393
    frameStart := 177373 },
  { event := event177394
    frameStart := 177373 },
  { event := event177395
    frameStart := 177373 },
  { event := event177396
    frameStart := 177373 },
  { event := event177397
    frameStart := 177373 },
  { event := event177398
    frameStart := 177373 },
  { event := event177399
    frameStart := 177373 },
  { event := event177400
    frameStart := 177373 },
  { event := event177401
    frameStart := 177373 },
  { event := event177402
    frameStart := 177373 },
  { event := event177403
    frameStart := 177373 },
  { event := event177404
    frameStart := 177373 },
  { event := event177405
    frameStart := 177373 },
  { event := event177406
    frameStart := 177373 },
  { event := event177407
    frameStart := 177373 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events692
