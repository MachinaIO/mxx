import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events024

open Mxx.Certificate.OperationalNoise
open CertificateABI

def exact6144RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨6503⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact6144RawTermsValid :
    exact6144RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event6144 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6613⟩⟩) exact6144RawTerms .large 6142 .exactZero (none)

def event6145 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6683⟩⟩) 0 ⟨6613⟩ 6144

def event6146 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6683⟩⟩) (.authority (.operator))

def exact6147RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6683⟩⟩]⟩, (1)⟩]

theorem exact6147RawTermsValid :
    exact6147RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event6147 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6683⟩⟩) exact6147RawTerms (.finite 8192) 6146 .exactZero (none)

def event6148 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6684⟩⟩) 0 ⟨6683⟩ 6147

def event6149 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6684⟩⟩) 1 ⟨2348⟩ 4

def event6150 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6684⟩⟩) (.scale (.predecessor 0 6148 .coefficient) (.value (.predecessor 1 6149 .coefficient)))

def exact6151RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6683⟩⟩]⟩, (1)⟩]

theorem exact6151RawTermsValid :
    exact6151RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event6151 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6684⟩⟩) exact6151RawTerms (.finite 8192) 6150 .exactZero (none)

def event6152 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6756⟩⟩) 0 ⟨6689⟩ 5477

def event6153 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6756⟩⟩) (.authority (.operator))

def exact6154RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6756⟩⟩]⟩, (1)⟩]

theorem exact6154RawTermsValid :
    exact6154RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event6154 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6756⟩⟩) exact6154RawTerms .large 6153 .exactZero (none)

def event6155 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7829⟩⟩) 0 ⟨6756⟩ 6154

def event6156 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7829⟩⟩) (.authority (.operator))

def exact6157RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨7829⟩⟩]⟩, (1)⟩]

theorem exact6157RawTermsValid :
    exact6157RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event6157 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7829⟩⟩) exact6157RawTerms (.finite 8192) 6156 .exactZero (none)

def event6158 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7830⟩⟩) 0 ⟨7829⟩ 6157

def event6159 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7830⟩⟩) 1 ⟨2348⟩ 4

def event6160 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7830⟩⟩) (.scale (.predecessor 0 6158 .coefficient) (.value (.predecessor 1 6159 .coefficient)))

def exact6161RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨7829⟩⟩]⟩, (1)⟩]

theorem exact6161RawTermsValid :
    exact6161RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event6161 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7830⟩⟩) exact6161RawTerms (.finite 8192) 6160 .exactZero (none)

def event6162 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6755⟩⟩) 0 ⟨6689⟩ 5477

def event6163 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6755⟩⟩) (.authority (.operator))

def exact6164RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6755⟩⟩]⟩, (1)⟩]

theorem exact6164RawTermsValid :
    exact6164RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event6164 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6755⟩⟩) exact6164RawTerms .large 6163 .exactZero (none)

def event6165 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7892⟩⟩) 0 ⟨6755⟩ 6164

def event6166 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7892⟩⟩) 1 ⟨7886⟩ 5961

def event6167 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7892⟩⟩) (.product (.predecessor 0 6165 .coefficient) (.predecessor 1 6166 .coefficient) (⟨false, false, none, none, none⟩))

def event6168 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7892⟩⟩, .operator (⟨6164, 0⟩, ⟨5961, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6755⟩⟩, ⟨.program ⟨214⟩, ⟨7885⟩⟩]⟩, (1)⟩)

def exact6169RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6755⟩⟩, ⟨.program ⟨214⟩, ⟨7885⟩⟩]⟩, (1)⟩]

theorem exact6169RawTermsValid :
    exact6169RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event6169 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7892⟩⟩) exact6169RawTerms .large 6167 .exactZero (none)

def event6170 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7916⟩⟩) 0 ⟨7892⟩ 6169

def event6171 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7916⟩⟩) 1 ⟨7830⟩ 6161

def event6172 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7916⟩⟩) (.product (.predecessor 0 6170 .coefficient) (.predecessor 1 6171 .coefficient) (⟨false, false, none, none, none⟩))

def event6173 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7916⟩⟩, .operator (⟨6169, 0⟩, ⟨6161, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6755⟩⟩, ⟨.program ⟨214⟩, ⟨7885⟩⟩, ⟨.program ⟨214⟩, ⟨7829⟩⟩]⟩, (1)⟩)

def exact6174RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6755⟩⟩, ⟨.program ⟨214⟩, ⟨7885⟩⟩, ⟨.program ⟨214⟩, ⟨7829⟩⟩]⟩, (1)⟩]

theorem exact6174RawTermsValid :
    exact6174RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event6174 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7916⟩⟩) exact6174RawTerms .large 6172 .exactZero (none)

def event6175 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7922⟩⟩) 0 ⟨7916⟩ 6174

def event6176 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7922⟩⟩) 1 ⟨6684⟩ 6151

def event6177 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7922⟩⟩) (.product (.predecessor 0 6175 .coefficient) (.predecessor 1 6176 .coefficient) (⟨false, false, none, none, none⟩))

def event6178 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7922⟩⟩, .operator (⟨6174, 0⟩, ⟨6151, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6755⟩⟩, ⟨.program ⟨214⟩, ⟨7885⟩⟩, ⟨.program ⟨214⟩, ⟨7829⟩⟩, ⟨.program ⟨214⟩, ⟨6683⟩⟩]⟩, (1)⟩)

def exact6179RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6755⟩⟩, ⟨.program ⟨214⟩, ⟨7885⟩⟩, ⟨.program ⟨214⟩, ⟨7829⟩⟩, ⟨.program ⟨214⟩, ⟨6683⟩⟩]⟩, (1)⟩]

theorem exact6179RawTermsValid :
    exact6179RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event6179 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7922⟩⟩) exact6179RawTerms .large 6177 .exactZero (none)

def event6180 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6591⟩⟩) 0 ⟨6544⟩ 2

def event6181 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6591⟩⟩) 1 ⟨6384⟩ 4563

def event6182 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6591⟩⟩) (.product (.predecessor 0 6180 .coefficient) (.predecessor 1 6181 .coefficient) (⟨false, true, none, none, some 1⟩))

def event6183 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨6591⟩⟩, .operator (⟨2, 0⟩, ⟨4563, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨6384⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact6184RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨6384⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact6184RawTermsValid :
    exact6184RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event6184 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6591⟩⟩) exact6184RawTerms .large 6182 .exactZero (none)

def event6185 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6639⟩⟩) 0 ⟨6591⟩ 6184

def event6186 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6639⟩⟩) (.authority (.operator))

def exact6187RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩]

theorem exact6187RawTermsValid :
    exact6187RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event6187 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6639⟩⟩) exact6187RawTerms (.finite 8192) 6186 .exactZero (none)

def event6188 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6640⟩⟩) 0 ⟨6639⟩ 6187

def event6189 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6640⟩⟩) 1 ⟨2348⟩ 4

def event6190 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6640⟩⟩) (.scale (.predecessor 0 6188 .coefficient) (.value (.predecessor 1 6189 .coefficient)))

def exact6191RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩]

theorem exact6191RawTermsValid :
    exact6191RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event6191 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6640⟩⟩) exact6191RawTerms (.finite 8192) 6190 .exactZero (none)

def event6192 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7796⟩⟩) 0 ⟨7795⟩ 5954

def event6193 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7796⟩⟩) 1 ⟨6640⟩ 6191

def event6194 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7796⟩⟩) (.product (.predecessor 0 6192 .coefficient) (.predecessor 1 6193 .coefficient) (⟨false, false, none, none, none⟩))

def event6195 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7796⟩⟩, .operator (⟨5954, 18⟩, ⟨6191, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6744⟩⟩, ⟨.program ⟨214⟩, ⟨6651⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (-1)⟩)

def event6196 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7796⟩⟩, .operator (⟨5954, 17⟩, ⟨6191, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6742⟩⟩, ⟨.program ⟨214⟩, ⟨6657⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩)

def event6197 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7796⟩⟩, .operator (⟨5954, 16⟩, ⟨6191, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6740⟩⟩, ⟨.program ⟨214⟩, ⟨6659⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩)

def event6198 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7796⟩⟩, .operator (⟨5954, 15⟩, ⟨6191, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6738⟩⟩, ⟨.program ⟨214⟩, ⟨6661⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩)

def event6199 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7796⟩⟩, .operator (⟨5954, 14⟩, ⟨6191, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6736⟩⟩, ⟨.program ⟨214⟩, ⟨6665⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩)

def event6200 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7796⟩⟩, .operator (⟨5954, 13⟩, ⟨6191, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6734⟩⟩, ⟨.program ⟨214⟩, ⟨6667⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩)

def event6201 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7796⟩⟩, .operator (⟨5954, 12⟩, ⟨6191, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6732⟩⟩, ⟨.program ⟨214⟩, ⟨6669⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩)

def event6202 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7796⟩⟩, .operator (⟨5954, 11⟩, ⟨6191, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6730⟩⟩, ⟨.program ⟨214⟩, ⟨6673⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩)

def event6203 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7796⟩⟩, .operator (⟨5954, 10⟩, ⟨6191, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6728⟩⟩, ⟨.program ⟨214⟩, ⟨6677⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩)

def event6204 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7796⟩⟩, .operator (⟨5954, 9⟩, ⟨6191, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6726⟩⟩, ⟨.program ⟨214⟩, ⟨6681⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩)

def event6205 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7796⟩⟩, .operator (⟨5954, 8⟩, ⟨6191, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6724⟩⟩, ⟨.program ⟨214⟩, ⟨6637⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩)

def event6206 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7796⟩⟩, .operator (⟨5954, 7⟩, ⟨6191, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6722⟩⟩, ⟨.program ⟨214⟩, ⟨6641⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩)

def event6207 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7796⟩⟩, .operator (⟨5954, 6⟩, ⟨6191, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6720⟩⟩, ⟨.program ⟨214⟩, ⟨6643⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩)

def event6208 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7796⟩⟩, .operator (⟨5954, 5⟩, ⟨6191, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6718⟩⟩, ⟨.program ⟨214⟩, ⟨6647⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩)

def event6209 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7796⟩⟩, .operator (⟨5954, 4⟩, ⟨6191, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6716⟩⟩, ⟨.program ⟨214⟩, ⟨6649⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩)

def event6210 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7796⟩⟩, .operator (⟨5954, 3⟩, ⟨6191, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6714⟩⟩, ⟨.program ⟨214⟩, ⟨6655⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩)

def event6211 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7796⟩⟩, .operator (⟨5954, 2⟩, ⟨6191, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6712⟩⟩, ⟨.program ⟨214⟩, ⟨6663⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩)

def event6212 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7796⟩⟩, .operator (⟨5954, 1⟩, ⟨6191, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6710⟩⟩, ⟨.program ⟨214⟩, ⟨6671⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩)

def event6213 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7796⟩⟩, .operator (⟨5954, 0⟩, ⟨6191, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6708⟩⟩, ⟨.program ⟨214⟩, ⟨6679⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩)

def exact6214RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6708⟩⟩, ⟨.program ⟨214⟩, ⟨6679⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6710⟩⟩, ⟨.program ⟨214⟩, ⟨6671⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6712⟩⟩, ⟨.program ⟨214⟩, ⟨6663⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6714⟩⟩, ⟨.program ⟨214⟩, ⟨6655⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6716⟩⟩, ⟨.program ⟨214⟩, ⟨6649⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6718⟩⟩, ⟨.program ⟨214⟩, ⟨6647⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6720⟩⟩, ⟨.program ⟨214⟩, ⟨6643⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6722⟩⟩, ⟨.program ⟨214⟩, ⟨6641⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6724⟩⟩, ⟨.program ⟨214⟩, ⟨6637⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6726⟩⟩, ⟨.program ⟨214⟩, ⟨6681⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6728⟩⟩, ⟨.program ⟨214⟩, ⟨6677⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6730⟩⟩, ⟨.program ⟨214⟩, ⟨6673⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6732⟩⟩, ⟨.program ⟨214⟩, ⟨6669⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6734⟩⟩, ⟨.program ⟨214⟩, ⟨6667⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6736⟩⟩, ⟨.program ⟨214⟩, ⟨6665⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6738⟩⟩, ⟨.program ⟨214⟩, ⟨6661⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6740⟩⟩, ⟨.program ⟨214⟩, ⟨6659⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6742⟩⟩, ⟨.program ⟨214⟩, ⟨6657⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6744⟩⟩, ⟨.program ⟨214⟩, ⟨6651⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (-1)⟩]

theorem exact6214RawTermsValid :
    exact6214RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event6214 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7796⟩⟩) exact6214RawTerms .large 6194 .exactZero (none)

def event6215 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7797⟩⟩) 0 ⟨7650⟩ 5878

def event6216 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7797⟩⟩) 1 ⟨7796⟩ 6214

def event6217 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7797⟩⟩) (.sum [.predecessor 0 6215 .coefficient, .predecessor 1 6216 .coefficient])

def exact6218RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6708⟩⟩, ⟨.program ⟨214⟩, ⟨6679⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6710⟩⟩, ⟨.program ⟨214⟩, ⟨6671⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6712⟩⟩, ⟨.program ⟨214⟩, ⟨6663⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6714⟩⟩, ⟨.program ⟨214⟩, ⟨6655⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6716⟩⟩, ⟨.program ⟨214⟩, ⟨6649⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6718⟩⟩, ⟨.program ⟨214⟩, ⟨6647⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6720⟩⟩, ⟨.program ⟨214⟩, ⟨6643⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6722⟩⟩, ⟨.program ⟨214⟩, ⟨6641⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6724⟩⟩, ⟨.program ⟨214⟩, ⟨6637⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6726⟩⟩, ⟨.program ⟨214⟩, ⟨6681⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6728⟩⟩, ⟨.program ⟨214⟩, ⟨6677⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6730⟩⟩, ⟨.program ⟨214⟩, ⟨6673⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6732⟩⟩, ⟨.program ⟨214⟩, ⟨6669⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6734⟩⟩, ⟨.program ⟨214⟩, ⟨6667⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6736⟩⟩, ⟨.program ⟨214⟩, ⟨6665⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6738⟩⟩, ⟨.program ⟨214⟩, ⟨6661⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6740⟩⟩, ⟨.program ⟨214⟩, ⟨6659⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6742⟩⟩, ⟨.program ⟨214⟩, ⟨6657⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6744⟩⟩, ⟨.program ⟨214⟩, ⟨6651⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (-1)⟩]

theorem exact6218RawTermsValid :
    exact6218RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event6218 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7797⟩⟩) exact6218RawTerms .large 6217 .exactZero (none)

def event6219 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7923⟩⟩) 0 ⟨7797⟩ 6218

def event6220 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7923⟩⟩) 1 ⟨7922⟩ 6179

def event6221 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7923⟩⟩) (.sum [.predecessor 0 6219 .coefficient, .predecessor 1 6220 .coefficient])

def exact6222RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6708⟩⟩, ⟨.program ⟨214⟩, ⟨6679⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6710⟩⟩, ⟨.program ⟨214⟩, ⟨6671⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6712⟩⟩, ⟨.program ⟨214⟩, ⟨6663⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6714⟩⟩, ⟨.program ⟨214⟩, ⟨6655⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6716⟩⟩, ⟨.program ⟨214⟩, ⟨6649⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6718⟩⟩, ⟨.program ⟨214⟩, ⟨6647⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6720⟩⟩, ⟨.program ⟨214⟩, ⟨6643⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6722⟩⟩, ⟨.program ⟨214⟩, ⟨6641⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6724⟩⟩, ⟨.program ⟨214⟩, ⟨6637⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6726⟩⟩, ⟨.program ⟨214⟩, ⟨6681⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6728⟩⟩, ⟨.program ⟨214⟩, ⟨6677⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6730⟩⟩, ⟨.program ⟨214⟩, ⟨6673⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6732⟩⟩, ⟨.program ⟨214⟩, ⟨6669⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6734⟩⟩, ⟨.program ⟨214⟩, ⟨6667⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6736⟩⟩, ⟨.program ⟨214⟩, ⟨6665⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6738⟩⟩, ⟨.program ⟨214⟩, ⟨6661⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6740⟩⟩, ⟨.program ⟨214⟩, ⟨6659⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6742⟩⟩, ⟨.program ⟨214⟩, ⟨6657⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6744⟩⟩, ⟨.program ⟨214⟩, ⟨6651⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6755⟩⟩, ⟨.program ⟨214⟩, ⟨7885⟩⟩, ⟨.program ⟨214⟩, ⟨7829⟩⟩, ⟨.program ⟨214⟩, ⟨6683⟩⟩]⟩, (1)⟩]

theorem exact6222RawTermsValid :
    exact6222RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event6222 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7923⟩⟩) exact6222RawTerms .large 6221 .exactZero (none)

def event6223 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7924⟩⟩) 0 ⟨7923⟩ 6222

def event6224 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7924⟩⟩) 1 ⟨7921⟩ 6139

def event6225 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7924⟩⟩) (.sum [.predecessor 0 6223 .coefficient, .predecessor 1 6224 .coefficient])

def exact6226RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6708⟩⟩, ⟨.program ⟨214⟩, ⟨6679⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6710⟩⟩, ⟨.program ⟨214⟩, ⟨6671⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6712⟩⟩, ⟨.program ⟨214⟩, ⟨6663⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6714⟩⟩, ⟨.program ⟨214⟩, ⟨6655⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6716⟩⟩, ⟨.program ⟨214⟩, ⟨6649⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6718⟩⟩, ⟨.program ⟨214⟩, ⟨6647⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6720⟩⟩, ⟨.program ⟨214⟩, ⟨6643⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6722⟩⟩, ⟨.program ⟨214⟩, ⟨6641⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6724⟩⟩, ⟨.program ⟨214⟩, ⟨6637⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6726⟩⟩, ⟨.program ⟨214⟩, ⟨6681⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6728⟩⟩, ⟨.program ⟨214⟩, ⟨6677⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6730⟩⟩, ⟨.program ⟨214⟩, ⟨6673⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6732⟩⟩, ⟨.program ⟨214⟩, ⟨6669⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6734⟩⟩, ⟨.program ⟨214⟩, ⟨6667⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6736⟩⟩, ⟨.program ⟨214⟩, ⟨6665⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6738⟩⟩, ⟨.program ⟨214⟩, ⟨6661⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6740⟩⟩, ⟨.program ⟨214⟩, ⟨6659⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6742⟩⟩, ⟨.program ⟨214⟩, ⟨6657⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6744⟩⟩, ⟨.program ⟨214⟩, ⟨6651⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6753⟩⟩, ⟨.program ⟨214⟩, ⟨7885⟩⟩, ⟨.program ⟨214⟩, ⟨7827⟩⟩, ⟨.program ⟨214⟩, ⟨6685⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6755⟩⟩, ⟨.program ⟨214⟩, ⟨7885⟩⟩, ⟨.program ⟨214⟩, ⟨7829⟩⟩, ⟨.program ⟨214⟩, ⟨6683⟩⟩]⟩, (1)⟩]

theorem exact6226RawTermsValid :
    exact6226RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event6226 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7924⟩⟩) exact6226RawTerms .large 6225 .exactZero (none)

def event6227 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7925⟩⟩) 0 ⟨7924⟩ 6226

def event6228 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7925⟩⟩) 1 ⟨7920⟩ 6099

def event6229 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7925⟩⟩) (.sum [.predecessor 0 6227 .coefficient, .predecessor 1 6228 .coefficient])

def exact6230RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6708⟩⟩, ⟨.program ⟨214⟩, ⟨6679⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6710⟩⟩, ⟨.program ⟨214⟩, ⟨6671⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6712⟩⟩, ⟨.program ⟨214⟩, ⟨6663⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6714⟩⟩, ⟨.program ⟨214⟩, ⟨6655⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6716⟩⟩, ⟨.program ⟨214⟩, ⟨6649⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6718⟩⟩, ⟨.program ⟨214⟩, ⟨6647⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6720⟩⟩, ⟨.program ⟨214⟩, ⟨6643⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6722⟩⟩, ⟨.program ⟨214⟩, ⟨6641⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6724⟩⟩, ⟨.program ⟨214⟩, ⟨6637⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6726⟩⟩, ⟨.program ⟨214⟩, ⟨6681⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6728⟩⟩, ⟨.program ⟨214⟩, ⟨6677⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6730⟩⟩, ⟨.program ⟨214⟩, ⟨6673⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6732⟩⟩, ⟨.program ⟨214⟩, ⟨6669⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6734⟩⟩, ⟨.program ⟨214⟩, ⟨6667⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6736⟩⟩, ⟨.program ⟨214⟩, ⟨6665⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6738⟩⟩, ⟨.program ⟨214⟩, ⟨6661⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6740⟩⟩, ⟨.program ⟨214⟩, ⟨6659⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6742⟩⟩, ⟨.program ⟨214⟩, ⟨6657⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6744⟩⟩, ⟨.program ⟨214⟩, ⟨6651⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6751⟩⟩, ⟨.program ⟨214⟩, ⟨7885⟩⟩, ⟨.program ⟨214⟩, ⟨7825⟩⟩, ⟨.program ⟨214⟩, ⟨6675⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6753⟩⟩, ⟨.program ⟨214⟩, ⟨7885⟩⟩, ⟨.program ⟨214⟩, ⟨7827⟩⟩, ⟨.program ⟨214⟩, ⟨6685⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6755⟩⟩, ⟨.program ⟨214⟩, ⟨7885⟩⟩, ⟨.program ⟨214⟩, ⟨7829⟩⟩, ⟨.program ⟨214⟩, ⟨6683⟩⟩]⟩, (1)⟩]

theorem exact6230RawTermsValid :
    exact6230RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event6230 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7925⟩⟩) exact6230RawTerms .large 6229 .exactZero (none)

def event6231 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7926⟩⟩) 0 ⟨7925⟩ 6230

def event6232 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7926⟩⟩) 1 ⟨7919⟩ 6059

def event6233 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7926⟩⟩) (.sum [.predecessor 0 6231 .coefficient, .predecessor 1 6232 .coefficient])

def exact6234RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6708⟩⟩, ⟨.program ⟨214⟩, ⟨6679⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6710⟩⟩, ⟨.program ⟨214⟩, ⟨6671⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6712⟩⟩, ⟨.program ⟨214⟩, ⟨6663⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6714⟩⟩, ⟨.program ⟨214⟩, ⟨6655⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6716⟩⟩, ⟨.program ⟨214⟩, ⟨6649⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6718⟩⟩, ⟨.program ⟨214⟩, ⟨6647⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6720⟩⟩, ⟨.program ⟨214⟩, ⟨6643⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6722⟩⟩, ⟨.program ⟨214⟩, ⟨6641⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6724⟩⟩, ⟨.program ⟨214⟩, ⟨6637⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6726⟩⟩, ⟨.program ⟨214⟩, ⟨6681⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6728⟩⟩, ⟨.program ⟨214⟩, ⟨6677⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6730⟩⟩, ⟨.program ⟨214⟩, ⟨6673⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6732⟩⟩, ⟨.program ⟨214⟩, ⟨6669⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6734⟩⟩, ⟨.program ⟨214⟩, ⟨6667⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6736⟩⟩, ⟨.program ⟨214⟩, ⟨6665⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6738⟩⟩, ⟨.program ⟨214⟩, ⟨6661⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6740⟩⟩, ⟨.program ⟨214⟩, ⟨6659⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6742⟩⟩, ⟨.program ⟨214⟩, ⟨6657⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6744⟩⟩, ⟨.program ⟨214⟩, ⟨6651⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6749⟩⟩, ⟨.program ⟨214⟩, ⟨7885⟩⟩, ⟨.program ⟨214⟩, ⟨7823⟩⟩, ⟨.program ⟨214⟩, ⟨6653⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6751⟩⟩, ⟨.program ⟨214⟩, ⟨7885⟩⟩, ⟨.program ⟨214⟩, ⟨7825⟩⟩, ⟨.program ⟨214⟩, ⟨6675⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6753⟩⟩, ⟨.program ⟨214⟩, ⟨7885⟩⟩, ⟨.program ⟨214⟩, ⟨7827⟩⟩, ⟨.program ⟨214⟩, ⟨6685⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6755⟩⟩, ⟨.program ⟨214⟩, ⟨7885⟩⟩, ⟨.program ⟨214⟩, ⟨7829⟩⟩, ⟨.program ⟨214⟩, ⟨6683⟩⟩]⟩, (1)⟩]

theorem exact6234RawTermsValid :
    exact6234RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event6234 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7926⟩⟩) exact6234RawTerms .large 6233 .exactZero (none)

def event6235 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7927⟩⟩) 0 ⟨7926⟩ 6234

def event6236 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7927⟩⟩) 1 ⟨7918⟩ 6019

def event6237 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7927⟩⟩) (.sum [.predecessor 0 6235 .coefficient, .predecessor 1 6236 .coefficient])

def exact6238RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6708⟩⟩, ⟨.program ⟨214⟩, ⟨6679⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6710⟩⟩, ⟨.program ⟨214⟩, ⟨6671⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6712⟩⟩, ⟨.program ⟨214⟩, ⟨6663⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6714⟩⟩, ⟨.program ⟨214⟩, ⟨6655⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6716⟩⟩, ⟨.program ⟨214⟩, ⟨6649⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6718⟩⟩, ⟨.program ⟨214⟩, ⟨6647⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6720⟩⟩, ⟨.program ⟨214⟩, ⟨6643⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6722⟩⟩, ⟨.program ⟨214⟩, ⟨6641⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6724⟩⟩, ⟨.program ⟨214⟩, ⟨6637⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6726⟩⟩, ⟨.program ⟨214⟩, ⟨6681⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6728⟩⟩, ⟨.program ⟨214⟩, ⟨6677⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6730⟩⟩, ⟨.program ⟨214⟩, ⟨6673⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6732⟩⟩, ⟨.program ⟨214⟩, ⟨6669⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6734⟩⟩, ⟨.program ⟨214⟩, ⟨6667⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6736⟩⟩, ⟨.program ⟨214⟩, ⟨6665⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6738⟩⟩, ⟨.program ⟨214⟩, ⟨6661⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6740⟩⟩, ⟨.program ⟨214⟩, ⟨6659⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6742⟩⟩, ⟨.program ⟨214⟩, ⟨6657⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6744⟩⟩, ⟨.program ⟨214⟩, ⟨6651⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6747⟩⟩, ⟨.program ⟨214⟩, ⟨7885⟩⟩, ⟨.program ⟨214⟩, ⟨7821⟩⟩, ⟨.program ⟨214⟩, ⟨6687⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6749⟩⟩, ⟨.program ⟨214⟩, ⟨7885⟩⟩, ⟨.program ⟨214⟩, ⟨7823⟩⟩, ⟨.program ⟨214⟩, ⟨6653⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6751⟩⟩, ⟨.program ⟨214⟩, ⟨7885⟩⟩, ⟨.program ⟨214⟩, ⟨7825⟩⟩, ⟨.program ⟨214⟩, ⟨6675⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6753⟩⟩, ⟨.program ⟨214⟩, ⟨7885⟩⟩, ⟨.program ⟨214⟩, ⟨7827⟩⟩, ⟨.program ⟨214⟩, ⟨6685⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6755⟩⟩, ⟨.program ⟨214⟩, ⟨7885⟩⟩, ⟨.program ⟨214⟩, ⟨7829⟩⟩, ⟨.program ⟨214⟩, ⟨6683⟩⟩]⟩, (1)⟩]

theorem exact6238RawTermsValid :
    exact6238RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event6238 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7927⟩⟩) exact6238RawTerms .large 6237 .exactZero (none)

def event6239 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7928⟩⟩) 0 ⟨7927⟩ 6238

def event6240 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7928⟩⟩) 1 ⟨7917⟩ 5979

def event6241 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7928⟩⟩) (.sum [.predecessor 0 6239 .coefficient, .predecessor 1 6240 .coefficient])

def exact6242RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6708⟩⟩, ⟨.program ⟨214⟩, ⟨6679⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6710⟩⟩, ⟨.program ⟨214⟩, ⟨6671⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6712⟩⟩, ⟨.program ⟨214⟩, ⟨6663⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6714⟩⟩, ⟨.program ⟨214⟩, ⟨6655⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6716⟩⟩, ⟨.program ⟨214⟩, ⟨6649⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6718⟩⟩, ⟨.program ⟨214⟩, ⟨6647⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6720⟩⟩, ⟨.program ⟨214⟩, ⟨6643⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6722⟩⟩, ⟨.program ⟨214⟩, ⟨6641⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6724⟩⟩, ⟨.program ⟨214⟩, ⟨6637⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6726⟩⟩, ⟨.program ⟨214⟩, ⟨6681⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6728⟩⟩, ⟨.program ⟨214⟩, ⟨6677⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6730⟩⟩, ⟨.program ⟨214⟩, ⟨6673⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6732⟩⟩, ⟨.program ⟨214⟩, ⟨6669⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6734⟩⟩, ⟨.program ⟨214⟩, ⟨6667⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6736⟩⟩, ⟨.program ⟨214⟩, ⟨6665⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6738⟩⟩, ⟨.program ⟨214⟩, ⟨6661⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6740⟩⟩, ⟨.program ⟨214⟩, ⟨6659⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6742⟩⟩, ⟨.program ⟨214⟩, ⟨6657⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6744⟩⟩, ⟨.program ⟨214⟩, ⟨6651⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6745⟩⟩, ⟨.program ⟨214⟩, ⟨7885⟩⟩, ⟨.program ⟨214⟩, ⟨7819⟩⟩, ⟨.program ⟨214⟩, ⟨6645⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6747⟩⟩, ⟨.program ⟨214⟩, ⟨7885⟩⟩, ⟨.program ⟨214⟩, ⟨7821⟩⟩, ⟨.program ⟨214⟩, ⟨6687⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6749⟩⟩, ⟨.program ⟨214⟩, ⟨7885⟩⟩, ⟨.program ⟨214⟩, ⟨7823⟩⟩, ⟨.program ⟨214⟩, ⟨6653⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6751⟩⟩, ⟨.program ⟨214⟩, ⟨7885⟩⟩, ⟨.program ⟨214⟩, ⟨7825⟩⟩, ⟨.program ⟨214⟩, ⟨6675⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6753⟩⟩, ⟨.program ⟨214⟩, ⟨7885⟩⟩, ⟨.program ⟨214⟩, ⟨7827⟩⟩, ⟨.program ⟨214⟩, ⟨6685⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6755⟩⟩, ⟨.program ⟨214⟩, ⟨7885⟩⟩, ⟨.program ⟨214⟩, ⟨7829⟩⟩, ⟨.program ⟨214⟩, ⟨6683⟩⟩]⟩, (1)⟩]

theorem exact6242RawTermsValid :
    exact6242RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event6242 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7928⟩⟩) exact6242RawTerms .large 6241 .exactZero (none)

def event6243 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7929⟩⟩) 0 ⟨5506⟩ 27

def event6244 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7929⟩⟩) 1 ⟨7928⟩ 6242

def event6245 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7929⟩⟩) (.product (.predecessor 0 6243 .coefficient) (.predecessor 1 6244 .coefficient) (⟨false, false, none, none, none⟩))

def event6246 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7929⟩⟩, .operator (⟨27, 0⟩, ⟨6242, 19⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6745⟩⟩, ⟨.program ⟨214⟩, ⟨7885⟩⟩, ⟨.program ⟨214⟩, ⟨7819⟩⟩, ⟨.program ⟨214⟩, ⟨6645⟩⟩]⟩, (1)⟩)

def event6247 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7929⟩⟩, .operator (⟨27, 0⟩, ⟨6242, 20⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6747⟩⟩, ⟨.program ⟨214⟩, ⟨7885⟩⟩, ⟨.program ⟨214⟩, ⟨7821⟩⟩, ⟨.program ⟨214⟩, ⟨6687⟩⟩]⟩, (1)⟩)

def event6248 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7929⟩⟩, .operator (⟨27, 0⟩, ⟨6242, 21⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6749⟩⟩, ⟨.program ⟨214⟩, ⟨7885⟩⟩, ⟨.program ⟨214⟩, ⟨7823⟩⟩, ⟨.program ⟨214⟩, ⟨6653⟩⟩]⟩, (1)⟩)

def event6249 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7929⟩⟩, .operator (⟨27, 0⟩, ⟨6242, 22⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6751⟩⟩, ⟨.program ⟨214⟩, ⟨7885⟩⟩, ⟨.program ⟨214⟩, ⟨7825⟩⟩, ⟨.program ⟨214⟩, ⟨6675⟩⟩]⟩, (1)⟩)

def event6250 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7929⟩⟩, .operator (⟨27, 0⟩, ⟨6242, 23⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6753⟩⟩, ⟨.program ⟨214⟩, ⟨7885⟩⟩, ⟨.program ⟨214⟩, ⟨7827⟩⟩, ⟨.program ⟨214⟩, ⟨6685⟩⟩]⟩, (1)⟩)

def event6251 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7929⟩⟩, .operator (⟨27, 0⟩, ⟨6242, 24⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6755⟩⟩, ⟨.program ⟨214⟩, ⟨7885⟩⟩, ⟨.program ⟨214⟩, ⟨7829⟩⟩, ⟨.program ⟨214⟩, ⟨6683⟩⟩]⟩, (1)⟩)

def event6252 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7929⟩⟩, .operator (⟨27, 0⟩, ⟨6242, 18⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6744⟩⟩, ⟨.program ⟨214⟩, ⟨6651⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (-1)⟩)

def event6253 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7929⟩⟩, .operator (⟨27, 0⟩, ⟨6242, 17⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6742⟩⟩, ⟨.program ⟨214⟩, ⟨6657⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩)

def event6254 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7929⟩⟩, .operator (⟨27, 0⟩, ⟨6242, 16⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6740⟩⟩, ⟨.program ⟨214⟩, ⟨6659⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩)

def event6255 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7929⟩⟩, .operator (⟨27, 0⟩, ⟨6242, 15⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6738⟩⟩, ⟨.program ⟨214⟩, ⟨6661⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩)

def event6256 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7929⟩⟩, .operator (⟨27, 0⟩, ⟨6242, 14⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6736⟩⟩, ⟨.program ⟨214⟩, ⟨6665⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩)

def event6257 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7929⟩⟩, .operator (⟨27, 0⟩, ⟨6242, 13⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6734⟩⟩, ⟨.program ⟨214⟩, ⟨6667⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩)

def event6258 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7929⟩⟩, .operator (⟨27, 0⟩, ⟨6242, 12⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6732⟩⟩, ⟨.program ⟨214⟩, ⟨6669⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩)

def event6259 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7929⟩⟩, .operator (⟨27, 0⟩, ⟨6242, 11⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6730⟩⟩, ⟨.program ⟨214⟩, ⟨6673⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩)

def event6260 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7929⟩⟩, .operator (⟨27, 0⟩, ⟨6242, 10⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6728⟩⟩, ⟨.program ⟨214⟩, ⟨6677⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩)

def event6261 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7929⟩⟩, .operator (⟨27, 0⟩, ⟨6242, 9⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6726⟩⟩, ⟨.program ⟨214⟩, ⟨6681⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩)

def event6262 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7929⟩⟩, .operator (⟨27, 0⟩, ⟨6242, 8⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6724⟩⟩, ⟨.program ⟨214⟩, ⟨6637⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩)

def event6263 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7929⟩⟩, .operator (⟨27, 0⟩, ⟨6242, 7⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6722⟩⟩, ⟨.program ⟨214⟩, ⟨6641⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩)

def event6264 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7929⟩⟩, .operator (⟨27, 0⟩, ⟨6242, 6⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6720⟩⟩, ⟨.program ⟨214⟩, ⟨6643⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩)

def event6265 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7929⟩⟩, .operator (⟨27, 0⟩, ⟨6242, 5⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6718⟩⟩, ⟨.program ⟨214⟩, ⟨6647⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩)

def event6266 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7929⟩⟩, .operator (⟨27, 0⟩, ⟨6242, 4⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6716⟩⟩, ⟨.program ⟨214⟩, ⟨6649⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩)

def event6267 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7929⟩⟩, .operator (⟨27, 0⟩, ⟨6242, 3⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6714⟩⟩, ⟨.program ⟨214⟩, ⟨6655⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩)

def event6268 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7929⟩⟩, .operator (⟨27, 0⟩, ⟨6242, 2⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6712⟩⟩, ⟨.program ⟨214⟩, ⟨6663⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩)

def event6269 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7929⟩⟩, .operator (⟨27, 0⟩, ⟨6242, 1⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6710⟩⟩, ⟨.program ⟨214⟩, ⟨6671⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩)

def event6270 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7929⟩⟩, .operator (⟨27, 0⟩, ⟨6242, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6708⟩⟩, ⟨.program ⟨214⟩, ⟨6679⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩)

def exact6271RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6708⟩⟩, ⟨.program ⟨214⟩, ⟨6679⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6710⟩⟩, ⟨.program ⟨214⟩, ⟨6671⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6712⟩⟩, ⟨.program ⟨214⟩, ⟨6663⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6714⟩⟩, ⟨.program ⟨214⟩, ⟨6655⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6716⟩⟩, ⟨.program ⟨214⟩, ⟨6649⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6718⟩⟩, ⟨.program ⟨214⟩, ⟨6647⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6720⟩⟩, ⟨.program ⟨214⟩, ⟨6643⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6722⟩⟩, ⟨.program ⟨214⟩, ⟨6641⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6724⟩⟩, ⟨.program ⟨214⟩, ⟨6637⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6726⟩⟩, ⟨.program ⟨214⟩, ⟨6681⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6728⟩⟩, ⟨.program ⟨214⟩, ⟨6677⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6730⟩⟩, ⟨.program ⟨214⟩, ⟨6673⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6732⟩⟩, ⟨.program ⟨214⟩, ⟨6669⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6734⟩⟩, ⟨.program ⟨214⟩, ⟨6667⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6736⟩⟩, ⟨.program ⟨214⟩, ⟨6665⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6738⟩⟩, ⟨.program ⟨214⟩, ⟨6661⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6740⟩⟩, ⟨.program ⟨214⟩, ⟨6659⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6742⟩⟩, ⟨.program ⟨214⟩, ⟨6657⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6744⟩⟩, ⟨.program ⟨214⟩, ⟨6651⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6745⟩⟩, ⟨.program ⟨214⟩, ⟨7885⟩⟩, ⟨.program ⟨214⟩, ⟨7819⟩⟩, ⟨.program ⟨214⟩, ⟨6645⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6747⟩⟩, ⟨.program ⟨214⟩, ⟨7885⟩⟩, ⟨.program ⟨214⟩, ⟨7821⟩⟩, ⟨.program ⟨214⟩, ⟨6687⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6749⟩⟩, ⟨.program ⟨214⟩, ⟨7885⟩⟩, ⟨.program ⟨214⟩, ⟨7823⟩⟩, ⟨.program ⟨214⟩, ⟨6653⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6751⟩⟩, ⟨.program ⟨214⟩, ⟨7885⟩⟩, ⟨.program ⟨214⟩, ⟨7825⟩⟩, ⟨.program ⟨214⟩, ⟨6675⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6753⟩⟩, ⟨.program ⟨214⟩, ⟨7885⟩⟩, ⟨.program ⟨214⟩, ⟨7827⟩⟩, ⟨.program ⟨214⟩, ⟨6685⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6755⟩⟩, ⟨.program ⟨214⟩, ⟨7885⟩⟩, ⟨.program ⟨214⟩, ⟨7829⟩⟩, ⟨.program ⟨214⟩, ⟨6683⟩⟩]⟩, (1)⟩]

theorem exact6271RawTermsValid :
    exact6271RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event6271 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7929⟩⟩) exact6271RawTerms .large 6245 .exactZero (none)

def event6272 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18909⟩⟩) 0 ⟨7929⟩ 6271

def event6273 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18909⟩⟩) 1 ⟨18907⟩ 5464

def event6274 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨18909⟩⟩) (.sum [.predecessor 0 6272 .coefficient, .predecessor 1 6273 .coefficient])

def exact6275RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6708⟩⟩, ⟨.program ⟨214⟩, ⟨6679⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6710⟩⟩, ⟨.program ⟨214⟩, ⟨6671⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6712⟩⟩, ⟨.program ⟨214⟩, ⟨6663⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6714⟩⟩, ⟨.program ⟨214⟩, ⟨6655⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6716⟩⟩, ⟨.program ⟨214⟩, ⟨6649⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6718⟩⟩, ⟨.program ⟨214⟩, ⟨6647⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6720⟩⟩, ⟨.program ⟨214⟩, ⟨6643⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6722⟩⟩, ⟨.program ⟨214⟩, ⟨6641⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6724⟩⟩, ⟨.program ⟨214⟩, ⟨6637⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6726⟩⟩, ⟨.program ⟨214⟩, ⟨6681⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6728⟩⟩, ⟨.program ⟨214⟩, ⟨6677⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6730⟩⟩, ⟨.program ⟨214⟩, ⟨6673⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6732⟩⟩, ⟨.program ⟨214⟩, ⟨6669⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6734⟩⟩, ⟨.program ⟨214⟩, ⟨6667⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6736⟩⟩, ⟨.program ⟨214⟩, ⟨6665⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6738⟩⟩, ⟨.program ⟨214⟩, ⟨6661⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6740⟩⟩, ⟨.program ⟨214⟩, ⟨6659⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6742⟩⟩, ⟨.program ⟨214⟩, ⟨6657⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6744⟩⟩, ⟨.program ⟨214⟩, ⟨6651⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6745⟩⟩, ⟨.program ⟨214⟩, ⟨7885⟩⟩, ⟨.program ⟨214⟩, ⟨7819⟩⟩, ⟨.program ⟨214⟩, ⟨6645⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6747⟩⟩, ⟨.program ⟨214⟩, ⟨7885⟩⟩, ⟨.program ⟨214⟩, ⟨7821⟩⟩, ⟨.program ⟨214⟩, ⟨6687⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6749⟩⟩, ⟨.program ⟨214⟩, ⟨7885⟩⟩, ⟨.program ⟨214⟩, ⟨7823⟩⟩, ⟨.program ⟨214⟩, ⟨6653⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6751⟩⟩, ⟨.program ⟨214⟩, ⟨7885⟩⟩, ⟨.program ⟨214⟩, ⟨7825⟩⟩, ⟨.program ⟨214⟩, ⟨6675⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6753⟩⟩, ⟨.program ⟨214⟩, ⟨7885⟩⟩, ⟨.program ⟨214⟩, ⟨7827⟩⟩, ⟨.program ⟨214⟩, ⟨6685⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6755⟩⟩, ⟨.program ⟨214⟩, ⟨7885⟩⟩, ⟨.program ⟨214⟩, ⟨7829⟩⟩, ⟨.program ⟨214⟩, ⟨6683⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6383⟩⟩, ⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨18016⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6383⟩⟩, ⟨.program ⟨214⟩, ⟨6396⟩⟩, ⟨.program ⟨214⟩, ⟨18063⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6383⟩⟩, ⟨.program ⟨214⟩, ⟨6425⟩⟩, ⟨.program ⟨214⟩, ⟨18049⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6383⟩⟩, ⟨.program ⟨214⟩, ⟨6493⟩⟩, ⟨.program ⟨214⟩, ⟨18042⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6383⟩⟩, ⟨.program ⟨214⟩, ⟨6503⟩⟩, ⟨.program ⟨214⟩, ⟨18035⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6383⟩⟩, ⟨.program ⟨214⟩, ⟨6542⟩⟩, ⟨.program ⟨214⟩, ⟨18028⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6383⟩⟩, ⟨.program ⟨214⟩, ⟨6543⟩⟩, ⟨.program ⟨214⟩, ⟨18056⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6387⟩⟩, ⟨.program ⟨214⟩, ⟨17155⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6391⟩⟩, ⟨.program ⟨214⟩, ⟨17211⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6398⟩⟩, ⟨.program ⟨214⟩, ⟨17428⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6407⟩⟩, ⟨.program ⟨214⟩, ⟨17792⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6410⟩⟩, ⟨.program ⟨214⟩, ⟨18485⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6427⟩⟩, ⟨.program ⟨214⟩, ⟨15503⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6435⟩⟩, ⟨.program ⟨214⟩, ⟨18114⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6437⟩⟩, ⟨.program ⟨214⟩, ⟨16917⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6449⟩⟩, ⟨.program ⟨214⟩, ⟨17484⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6452⟩⟩, ⟨.program ⟨214⟩, ⟨15195⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6459⟩⟩, ⟨.program ⟨214⟩, ⟨17708⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6467⟩⟩, ⟨.program ⟨214⟩, ⟨17939⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6473⟩⟩, ⟨.program ⟨214⟩, ⟨17540⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6475⟩⟩, ⟨.program ⟨214⟩, ⟨15034⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6490⟩⟩, ⟨.program ⟨214⟩, ⟨18792⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6494⟩⟩, ⟨.program ⟨214⟩, ⟨17596⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6495⟩⟩, ⟨.program ⟨214⟩, ⟨14873⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6502⟩⟩, ⟨.program ⟨214⟩, ⟨17652⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6387⟩⟩, ⟨.program ⟨214⟩, ⟨6396⟩⟩, ⟨.program ⟨214⟩, ⟨17181⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6387⟩⟩, ⟨.program ⟨214⟩, ⟨6425⟩⟩, ⟨.program ⟨214⟩, ⟨17173⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6387⟩⟩, ⟨.program ⟨214⟩, ⟨6493⟩⟩, ⟨.program ⟨214⟩, ⟨17169⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6387⟩⟩, ⟨.program ⟨214⟩, ⟨6503⟩⟩, ⟨.program ⟨214⟩, ⟨17165⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6387⟩⟩, ⟨.program ⟨214⟩, ⟨6542⟩⟩, ⟨.program ⟨214⟩, ⟨17161⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6387⟩⟩, ⟨.program ⟨214⟩, ⟨6543⟩⟩, ⟨.program ⟨214⟩, ⟨17177⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6391⟩⟩, ⟨.program ⟨214⟩, ⟨6396⟩⟩, ⟨.program ⟨214⟩, ⟨17237⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6391⟩⟩, ⟨.program ⟨214⟩, ⟨6425⟩⟩, ⟨.program ⟨214⟩, ⟨17229⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6391⟩⟩, ⟨.program ⟨214⟩, ⟨6493⟩⟩, ⟨.program ⟨214⟩, ⟨17225⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6391⟩⟩, ⟨.program ⟨214⟩, ⟨6503⟩⟩, ⟨.program ⟨214⟩, ⟨17221⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6391⟩⟩, ⟨.program ⟨214⟩, ⟨6542⟩⟩, ⟨.program ⟨214⟩, ⟨17217⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6391⟩⟩, ⟨.program ⟨214⟩, ⟨6543⟩⟩, ⟨.program ⟨214⟩, ⟨17233⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6396⟩⟩, ⟨.program ⟨214⟩, ⟨6398⟩⟩, ⟨.program ⟨214⟩, ⟨17454⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6396⟩⟩, ⟨.program ⟨214⟩, ⟨6407⟩⟩, ⟨.program ⟨214⟩, ⟨17846⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6396⟩⟩, ⟨.program ⟨214⟩, ⟨6410⟩⟩, ⟨.program ⟨214⟩, ⟨18511⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6396⟩⟩, ⟨.program ⟨214⟩, ⟨6427⟩⟩, ⟨.program ⟨214⟩, ⟨15536⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6396⟩⟩, ⟨.program ⟨214⟩, ⟨6435⟩⟩, ⟨.program ⟨214⟩, ⟨18140⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6396⟩⟩, ⟨.program ⟨214⟩, ⟨6437⟩⟩, ⟨.program ⟨214⟩, ⟨16943⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6396⟩⟩, ⟨.program ⟨214⟩, ⟨6449⟩⟩, ⟨.program ⟨214⟩, ⟨17510⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6396⟩⟩, ⟨.program ⟨214⟩, ⟨6452⟩⟩, ⟨.program ⟨214⟩, ⟨15228⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6396⟩⟩, ⟨.program ⟨214⟩, ⟨6459⟩⟩, ⟨.program ⟨214⟩, ⟨17734⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6396⟩⟩, ⟨.program ⟨214⟩, ⟨6467⟩⟩, ⟨.program ⟨214⟩, ⟨17965⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6396⟩⟩, ⟨.program ⟨214⟩, ⟨6473⟩⟩, ⟨.program ⟨214⟩, ⟨17566⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6396⟩⟩, ⟨.program ⟨214⟩, ⟨6475⟩⟩, ⟨.program ⟨214⟩, ⟨15067⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6396⟩⟩, ⟨.program ⟨214⟩, ⟨6490⟩⟩, ⟨.program ⟨214⟩, ⟨18893⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6396⟩⟩, ⟨.program ⟨214⟩, ⟨6494⟩⟩, ⟨.program ⟨214⟩, ⟨17622⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6396⟩⟩, ⟨.program ⟨214⟩, ⟨6495⟩⟩, ⟨.program ⟨214⟩, ⟨14906⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6396⟩⟩, ⟨.program ⟨214⟩, ⟨6502⟩⟩, ⟨.program ⟨214⟩, ⟨17678⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6398⟩⟩, ⟨.program ⟨214⟩, ⟨6425⟩⟩, ⟨.program ⟨214⟩, ⟨17446⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6398⟩⟩, ⟨.program ⟨214⟩, ⟨6493⟩⟩, ⟨.program ⟨214⟩, ⟨17442⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6398⟩⟩, ⟨.program ⟨214⟩, ⟨6503⟩⟩, ⟨.program ⟨214⟩, ⟨17438⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6398⟩⟩, ⟨.program ⟨214⟩, ⟨6542⟩⟩, ⟨.program ⟨214⟩, ⟨17434⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6398⟩⟩, ⟨.program ⟨214⟩, ⟨6543⟩⟩, ⟨.program ⟨214⟩, ⟨17450⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6407⟩⟩, ⟨.program ⟨214⟩, ⟨6425⟩⟩, ⟨.program ⟨214⟩, ⟨17830⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6407⟩⟩, ⟨.program ⟨214⟩, ⟨6493⟩⟩, ⟨.program ⟨214⟩, ⟨17822⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6407⟩⟩, ⟨.program ⟨214⟩, ⟨6503⟩⟩, ⟨.program ⟨214⟩, ⟨17814⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6407⟩⟩, ⟨.program ⟨214⟩, ⟨6542⟩⟩, ⟨.program ⟨214⟩, ⟨17806⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6407⟩⟩, ⟨.program ⟨214⟩, ⟨6543⟩⟩, ⟨.program ⟨214⟩, ⟨17838⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6410⟩⟩, ⟨.program ⟨214⟩, ⟨6425⟩⟩, ⟨.program ⟨214⟩, ⟨18503⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6410⟩⟩, ⟨.program ⟨214⟩, ⟨6493⟩⟩, ⟨.program ⟨214⟩, ⟨18499⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6410⟩⟩, ⟨.program ⟨214⟩, ⟨6503⟩⟩, ⟨.program ⟨214⟩, ⟨18495⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6410⟩⟩, ⟨.program ⟨214⟩, ⟨6542⟩⟩, ⟨.program ⟨214⟩, ⟨18491⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6410⟩⟩, ⟨.program ⟨214⟩, ⟨6543⟩⟩, ⟨.program ⟨214⟩, ⟨18507⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6425⟩⟩, ⟨.program ⟨214⟩, ⟨6427⟩⟩, ⟨.program ⟨214⟩, ⟨15526⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6425⟩⟩, ⟨.program ⟨214⟩, ⟨6435⟩⟩, ⟨.program ⟨214⟩, ⟨18132⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6425⟩⟩, ⟨.program ⟨214⟩, ⟨6437⟩⟩, ⟨.program ⟨214⟩, ⟨16935⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6425⟩⟩, ⟨.program ⟨214⟩, ⟨6449⟩⟩, ⟨.program ⟨214⟩, ⟨17502⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6425⟩⟩, ⟨.program ⟨214⟩, ⟨6452⟩⟩, ⟨.program ⟨214⟩, ⟨15218⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6425⟩⟩, ⟨.program ⟨214⟩, ⟨6459⟩⟩, ⟨.program ⟨214⟩, ⟨17726⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6425⟩⟩, ⟨.program ⟨214⟩, ⟨6467⟩⟩, ⟨.program ⟨214⟩, ⟨17957⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6425⟩⟩, ⟨.program ⟨214⟩, ⟨6473⟩⟩, ⟨.program ⟨214⟩, ⟨17558⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6425⟩⟩, ⟨.program ⟨214⟩, ⟨6475⟩⟩, ⟨.program ⟨214⟩, ⟨15057⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6425⟩⟩, ⟨.program ⟨214⟩, ⟨6490⟩⟩, ⟨.program ⟨214⟩, ⟨18863⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6425⟩⟩, ⟨.program ⟨214⟩, ⟨6494⟩⟩, ⟨.program ⟨214⟩, ⟨17614⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6425⟩⟩, ⟨.program ⟨214⟩, ⟨6495⟩⟩, ⟨.program ⟨214⟩, ⟨14896⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6425⟩⟩, ⟨.program ⟨214⟩, ⟨6502⟩⟩, ⟨.program ⟨214⟩, ⟨17670⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6427⟩⟩, ⟨.program ⟨214⟩, ⟨6493⟩⟩, ⟨.program ⟨214⟩, ⟨15521⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6427⟩⟩, ⟨.program ⟨214⟩, ⟨6503⟩⟩, ⟨.program ⟨214⟩, ⟨15516⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6427⟩⟩, ⟨.program ⟨214⟩, ⟨6542⟩⟩, ⟨.program ⟨214⟩, ⟨15511⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6427⟩⟩, ⟨.program ⟨214⟩, ⟨6543⟩⟩, ⟨.program ⟨214⟩, ⟨15531⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6435⟩⟩, ⟨.program ⟨214⟩, ⟨6493⟩⟩, ⟨.program ⟨214⟩, ⟨18128⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6435⟩⟩, ⟨.program ⟨214⟩, ⟨6503⟩⟩, ⟨.program ⟨214⟩, ⟨18124⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6435⟩⟩, ⟨.program ⟨214⟩, ⟨6542⟩⟩, ⟨.program ⟨214⟩, ⟨18120⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6435⟩⟩, ⟨.program ⟨214⟩, ⟨6543⟩⟩, ⟨.program ⟨214⟩, ⟨18136⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6437⟩⟩, ⟨.program ⟨214⟩, ⟨6493⟩⟩, ⟨.program ⟨214⟩, ⟨16931⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6437⟩⟩, ⟨.program ⟨214⟩, ⟨6503⟩⟩, ⟨.program ⟨214⟩, ⟨16927⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6437⟩⟩, ⟨.program ⟨214⟩, ⟨6542⟩⟩, ⟨.program ⟨214⟩, ⟨16923⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6437⟩⟩, ⟨.program ⟨214⟩, ⟨6543⟩⟩, ⟨.program ⟨214⟩, ⟨16939⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6449⟩⟩, ⟨.program ⟨214⟩, ⟨6493⟩⟩, ⟨.program ⟨214⟩, ⟨17498⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6449⟩⟩, ⟨.program ⟨214⟩, ⟨6503⟩⟩, ⟨.program ⟨214⟩, ⟨17494⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6449⟩⟩, ⟨.program ⟨214⟩, ⟨6542⟩⟩, ⟨.program ⟨214⟩, ⟨17490⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6449⟩⟩, ⟨.program ⟨214⟩, ⟨6543⟩⟩, ⟨.program ⟨214⟩, ⟨17506⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6452⟩⟩, ⟨.program ⟨214⟩, ⟨6493⟩⟩, ⟨.program ⟨214⟩, ⟨15213⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6452⟩⟩, ⟨.program ⟨214⟩, ⟨6503⟩⟩, ⟨.program ⟨214⟩, ⟨15208⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6452⟩⟩, ⟨.program ⟨214⟩, ⟨6542⟩⟩, ⟨.program ⟨214⟩, ⟨15203⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6452⟩⟩, ⟨.program ⟨214⟩, ⟨6543⟩⟩, ⟨.program ⟨214⟩, ⟨15223⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6459⟩⟩, ⟨.program ⟨214⟩, ⟨6493⟩⟩, ⟨.program ⟨214⟩, ⟨17722⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6459⟩⟩, ⟨.program ⟨214⟩, ⟨6503⟩⟩, ⟨.program ⟨214⟩, ⟨17718⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6459⟩⟩, ⟨.program ⟨214⟩, ⟨6542⟩⟩, ⟨.program ⟨214⟩, ⟨17714⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6459⟩⟩, ⟨.program ⟨214⟩, ⟨6543⟩⟩, ⟨.program ⟨214⟩, ⟨17730⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6467⟩⟩, ⟨.program ⟨214⟩, ⟨6493⟩⟩, ⟨.program ⟨214⟩, ⟨17953⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6467⟩⟩, ⟨.program ⟨214⟩, ⟨6503⟩⟩, ⟨.program ⟨214⟩, ⟨17949⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6467⟩⟩, ⟨.program ⟨214⟩, ⟨6542⟩⟩, ⟨.program ⟨214⟩, ⟨17945⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6467⟩⟩, ⟨.program ⟨214⟩, ⟨6543⟩⟩, ⟨.program ⟨214⟩, ⟨17961⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6473⟩⟩, ⟨.program ⟨214⟩, ⟨6493⟩⟩, ⟨.program ⟨214⟩, ⟨17554⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6473⟩⟩, ⟨.program ⟨214⟩, ⟨6503⟩⟩, ⟨.program ⟨214⟩, ⟨17550⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6473⟩⟩, ⟨.program ⟨214⟩, ⟨6542⟩⟩, ⟨.program ⟨214⟩, ⟨17546⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6473⟩⟩, ⟨.program ⟨214⟩, ⟨6543⟩⟩, ⟨.program ⟨214⟩, ⟨17562⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6475⟩⟩, ⟨.program ⟨214⟩, ⟨6493⟩⟩, ⟨.program ⟨214⟩, ⟨15052⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6475⟩⟩, ⟨.program ⟨214⟩, ⟨6503⟩⟩, ⟨.program ⟨214⟩, ⟨15047⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6475⟩⟩, ⟨.program ⟨214⟩, ⟨6542⟩⟩, ⟨.program ⟨214⟩, ⟨15042⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6475⟩⟩, ⟨.program ⟨214⟩, ⟨6543⟩⟩, ⟨.program ⟨214⟩, ⟨15062⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6490⟩⟩, ⟨.program ⟨214⟩, ⟨6493⟩⟩, ⟨.program ⟨214⟩, ⟨18848⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6490⟩⟩, ⟨.program ⟨214⟩, ⟨6503⟩⟩, ⟨.program ⟨214⟩, ⟨18832⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6490⟩⟩, ⟨.program ⟨214⟩, ⟨6542⟩⟩, ⟨.program ⟨214⟩, ⟨18818⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6490⟩⟩, ⟨.program ⟨214⟩, ⟨6543⟩⟩, ⟨.program ⟨214⟩, ⟨18878⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6493⟩⟩, ⟨.program ⟨214⟩, ⟨6494⟩⟩, ⟨.program ⟨214⟩, ⟨17610⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6493⟩⟩, ⟨.program ⟨214⟩, ⟨6495⟩⟩, ⟨.program ⟨214⟩, ⟨14891⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6493⟩⟩, ⟨.program ⟨214⟩, ⟨6502⟩⟩, ⟨.program ⟨214⟩, ⟨17666⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6494⟩⟩, ⟨.program ⟨214⟩, ⟨6503⟩⟩, ⟨.program ⟨214⟩, ⟨17606⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6494⟩⟩, ⟨.program ⟨214⟩, ⟨6542⟩⟩, ⟨.program ⟨214⟩, ⟨17602⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6494⟩⟩, ⟨.program ⟨214⟩, ⟨6543⟩⟩, ⟨.program ⟨214⟩, ⟨17618⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6495⟩⟩, ⟨.program ⟨214⟩, ⟨6503⟩⟩, ⟨.program ⟨214⟩, ⟨14886⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6495⟩⟩, ⟨.program ⟨214⟩, ⟨6542⟩⟩, ⟨.program ⟨214⟩, ⟨14881⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6495⟩⟩, ⟨.program ⟨214⟩, ⟨6543⟩⟩, ⟨.program ⟨214⟩, ⟨14901⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6502⟩⟩, ⟨.program ⟨214⟩, ⟨6503⟩⟩, ⟨.program ⟨214⟩, ⟨17662⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6502⟩⟩, ⟨.program ⟨214⟩, ⟨6542⟩⟩, ⟨.program ⟨214⟩, ⟨17658⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6502⟩⟩, ⟨.program ⟨214⟩, ⟨6543⟩⟩, ⟨.program ⟨214⟩, ⟨17674⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact6275RawTermsValid :
    exact6275RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event6275 : Event := .resultExact (⟨.program ⟨214⟩, ⟨18909⟩⟩) exact6275RawTerms .large 6274 .exactZero (none)

def event6276 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5⟩⟩) (.authority (.operator))

def exact6277RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨5⟩⟩]⟩, (1)⟩]

theorem exact6277RawTermsValid :
    exact6277RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event6277 : Event := .resultExact (⟨.program ⟨214⟩, ⟨5⟩⟩) exact6277RawTerms (.finite 26) 6276 .exactZero (none)

def event6278 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨807⟩⟩) (.authority (.operator))

def event6279 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨807⟩⟩) (.finite 218)

def event6280 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5561⟩⟩) 0 ⟨5560⟩ 48

def event6281 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5561⟩⟩) 1 ⟨807⟩ 6279

def event6282 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5561⟩⟩) (.sum [.predecessor 0 6280 .coefficient, .predecessor 1 6281 .coefficient])

def event6283 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5561⟩⟩) (.finite 442)

def event6284 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5617⟩⟩) 0 ⟨5561⟩ 6283

def event6285 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5617⟩⟩) 1 ⟨961⟩ 38

def event6286 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5617⟩⟩) (.identity (.predecessor 1 6285 .coefficient))

def event6287 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5617⟩⟩) (.finite 224)

def event6288 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5618⟩⟩) 0 ⟨5617⟩ 6287

def event6289 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5618⟩⟩) 1 ⟨2348⟩ 4

def event6290 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5618⟩⟩) (.sum [.predecessor 0 6288 .coefficient, .predecessor 1 6289 .coefficient])

def event6291 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5618⟩⟩) (.finite 225)

def event6292 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5619⟩⟩) 0 ⟨0⟩ 20

def event6293 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5619⟩⟩) 1 ⟨5617⟩ 6287

def event6294 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5619⟩⟩) 2 ⟨5618⟩ 6291

def event6295 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5619⟩⟩) 3 ⟨110⟩ 6

def event6296 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5619⟩⟩) 4 ⟨2348⟩ 4

def event6297 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5619⟩⟩) (.identity (.predecessor 0 6292 .coefficient))

def exact6298RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨5506⟩⟩]⟩, (1)⟩]

theorem exact6298RawTermsValid :
    exact6298RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event6298 : Event := .resultExact (⟨.program ⟨214⟩, ⟨5619⟩⟩) exact6298RawTerms (.finite 1) 6297 .exactZero (none)

def event6299 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6583⟩⟩) 0 ⟨5619⟩ 6298

def event6300 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6583⟩⟩) 1 ⟨6544⟩ 2

def event6301 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6583⟩⟩) (.product (.predecessor 0 6299 .coefficient) (.predecessor 1 6300 .coefficient) (⟨false, false, none, none, none⟩))

def event6302 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨6583⟩⟩, .operator (⟨6298, 0⟩, ⟨2, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact6303RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact6303RawTermsValid :
    exact6303RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event6303 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6583⟩⟩) exact6303RawTerms .large 6301 .exactZero (none)

def event6304 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5562⟩⟩) 0 ⟨5560⟩ 48

def event6305 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5562⟩⟩) 1 ⟨2348⟩ 4

def event6306 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5562⟩⟩) (.sum [.predecessor 0 6304 .coefficient, .predecessor 1 6305 .coefficient])

def event6307 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5562⟩⟩) (.finite 225)

def event6308 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5563⟩⟩) 0 ⟨0⟩ 20

def event6309 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5563⟩⟩) 1 ⟨5560⟩ 48

def event6310 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5563⟩⟩) 2 ⟨5562⟩ 6307

def event6311 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5563⟩⟩) 3 ⟨110⟩ 6

def event6312 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5563⟩⟩) 4 ⟨2348⟩ 4

def event6313 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5563⟩⟩) (.identity (.predecessor 0 6308 .coefficient))

def exact6314RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨5519⟩⟩]⟩, (1)⟩]

theorem exact6314RawTermsValid :
    exact6314RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event6314 : Event := .resultExact (⟨.program ⟨214⟩, ⟨5563⟩⟩) exact6314RawTerms (.finite 1) 6313 .exactZero (none)

def event6315 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7365⟩⟩) 0 ⟨5563⟩ 6314

def event6316 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7365⟩⟩) 1 ⟨6746⟩ 5480

def event6317 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7365⟩⟩) (.product (.predecessor 0 6315 .coefficient) (.predecessor 1 6316 .coefficient) (⟨false, false, none, none, none⟩))

def event6318 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7365⟩⟩, .operator (⟨6314, 0⟩, ⟨5480, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6746⟩⟩]⟩, (1)⟩)

def exact6319RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6746⟩⟩]⟩, (1)⟩]

theorem exact6319RawTermsValid :
    exact6319RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event6319 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7365⟩⟩) exact6319RawTerms .large 6317 .exactZero (none)

def event6320 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7767⟩⟩) 0 ⟨7365⟩ 6319

def event6321 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7767⟩⟩) 1 ⟨6583⟩ 6303

def event6322 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7767⟩⟩) (.sum [.predecessor 0 6320 .coefficient, .predecessor 1 6321 .coefficient])

def exact6323RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6746⟩⟩]⟩, (1)⟩]

theorem exact6323RawTermsValid :
    exact6323RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event6323 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7767⟩⟩) exact6323RawTerms .large 6322 .exactZero (none)

def event6324 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7768⟩⟩) 0 ⟨7767⟩ 6323

def event6325 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7768⟩⟩) 1 ⟨5⟩ 6277

def event6326 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7768⟩⟩) (.sum [.predecessor 0 6324 .coefficient, .predecessor 1 6325 .coefficient])

def event6327 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7768⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨5⟩⟩]⟩) [⟨.result 6277 .coefficient, false, none⟩])

def event6328 : Event := .survivorFold (1) 6327

def exact6329RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6746⟩⟩]⟩, (1)⟩]

theorem exact6329RawTermsValid :
    exact6329RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event6329 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7768⟩⟩) exact6329RawTerms .large 6326 (.finite 26) (some (6327))

def event6330 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18906⟩⟩) 0 ⟨7768⟩ 6329

def event6331 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18906⟩⟩) 1 ⟨18903⟩ 804

def event6332 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨18906⟩⟩) (.product (.predecessor 0 6330 .coefficient) (.predecessor 1 6331 .coefficient) (⟨false, false, none, none, none⟩))

def event6333 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨18906⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨6410⟩⟩, ⟨.program ⟨214⟩, ⟨18511⟩⟩], []⟩) [⟨.result 36 .coefficient, true, some 1⟩, ⟨.result 536 .coefficient, true, some 1⟩])

def event6334 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨18906⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨6435⟩⟩, ⟨.program ⟨214⟩, ⟨18140⟩⟩], []⟩) [⟨.result 543 .coefficient, true, some 1⟩, ⟨.result 546 .coefficient, true, some 1⟩])

def event6335 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨18906⟩⟩) (.sum [.transfer 6333, .transfer 6334])

def event6336 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨18906⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨6437⟩⟩, ⟨.program ⟨214⟩, ⟨16943⟩⟩], []⟩) [⟨.result 553 .coefficient, true, some 1⟩, ⟨.result 556 .coefficient, true, some 1⟩])

def event6337 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨18906⟩⟩) (.sum [.transfer 6335, .transfer 6336])

def event6338 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨18906⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨6449⟩⟩, ⟨.program ⟨214⟩, ⟨17510⟩⟩], []⟩) [⟨.result 563 .coefficient, true, some 1⟩, ⟨.result 566 .coefficient, true, some 1⟩])

def event6339 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨18906⟩⟩) (.sum [.transfer 6337, .transfer 6338])

def event6340 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨18906⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨6459⟩⟩, ⟨.program ⟨214⟩, ⟨17734⟩⟩], []⟩) [⟨.result 573 .coefficient, true, some 1⟩, ⟨.result 576 .coefficient, true, some 1⟩])

def event6341 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨18906⟩⟩) (.sum [.transfer 6339, .transfer 6340])

def event6342 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨18906⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨6467⟩⟩, ⟨.program ⟨214⟩, ⟨17965⟩⟩], []⟩) [⟨.result 583 .coefficient, true, some 1⟩, ⟨.result 586 .coefficient, true, some 1⟩])

def event6343 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨18906⟩⟩) (.sum [.transfer 6341, .transfer 6342])

def event6344 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨18906⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨6473⟩⟩, ⟨.program ⟨214⟩, ⟨17566⟩⟩], []⟩) [⟨.result 593 .coefficient, true, some 1⟩, ⟨.result 596 .coefficient, true, some 1⟩])

def event6345 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨18906⟩⟩) (.sum [.transfer 6343, .transfer 6344])

def event6346 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨18906⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨6490⟩⟩, ⟨.program ⟨214⟩, ⟨18893⟩⟩], []⟩) [⟨.result 603 .coefficient, true, some 1⟩, ⟨.result 606 .coefficient, true, some 1⟩])

def event6347 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨18906⟩⟩) (.sum [.transfer 6345, .transfer 6346])

def event6348 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨18906⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨6494⟩⟩, ⟨.program ⟨214⟩, ⟨17622⟩⟩], []⟩) [⟨.result 613 .coefficient, true, some 1⟩, ⟨.result 616 .coefficient, true, some 1⟩])

def event6349 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨18906⟩⟩) (.sum [.transfer 6347, .transfer 6348])

def event6350 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨18906⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨6502⟩⟩, ⟨.program ⟨214⟩, ⟨17678⟩⟩], []⟩) [⟨.result 623 .coefficient, true, some 1⟩, ⟨.result 626 .coefficient, true, some 1⟩])

def event6351 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨18906⟩⟩) (.sum [.transfer 6349, .transfer 6350])

def event6352 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨18906⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨6383⟩⟩, ⟨.program ⟨214⟩, ⟨18063⟩⟩], []⟩) [⟨.result 633 .coefficient, true, some 1⟩, ⟨.result 636 .coefficient, true, some 1⟩])

def event6353 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨18906⟩⟩) (.sum [.transfer 6351, .transfer 6352])

def event6354 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨18906⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨6387⟩⟩, ⟨.program ⟨214⟩, ⟨17181⟩⟩], []⟩) [⟨.result 643 .coefficient, true, some 1⟩, ⟨.result 646 .coefficient, true, some 1⟩])

def event6355 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨18906⟩⟩) (.sum [.transfer 6353, .transfer 6354])

def event6356 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨18906⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨6391⟩⟩, ⟨.program ⟨214⟩, ⟨17237⟩⟩], []⟩) [⟨.result 653 .coefficient, true, some 1⟩, ⟨.result 656 .coefficient, true, some 1⟩])

def event6357 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨18906⟩⟩) (.sum [.transfer 6355, .transfer 6356])

def event6358 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨18906⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨6398⟩⟩, ⟨.program ⟨214⟩, ⟨17454⟩⟩], []⟩) [⟨.result 663 .coefficient, true, some 1⟩, ⟨.result 666 .coefficient, true, some 1⟩])

def event6359 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨18906⟩⟩) (.sum [.transfer 6357, .transfer 6358])

def event6360 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨18906⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨6407⟩⟩, ⟨.program ⟨214⟩, ⟨17846⟩⟩], []⟩) [⟨.result 673 .coefficient, true, some 1⟩, ⟨.result 676 .coefficient, true, some 1⟩])

def event6361 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨18906⟩⟩) (.sum [.transfer 6359, .transfer 6360])

def event6362 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨18906⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨6427⟩⟩, ⟨.program ⟨214⟩, ⟨15536⟩⟩], []⟩) [⟨.result 683 .coefficient, true, some 1⟩, ⟨.result 686 .coefficient, true, some 1⟩])

def event6363 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨18906⟩⟩) (.sum [.transfer 6361, .transfer 6362])

def event6364 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨18906⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨6452⟩⟩, ⟨.program ⟨214⟩, ⟨15228⟩⟩], []⟩) [⟨.result 693 .coefficient, true, some 1⟩, ⟨.result 696 .coefficient, true, some 1⟩])

def event6365 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨18906⟩⟩) (.sum [.transfer 6363, .transfer 6364])

def event6366 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨18906⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨6475⟩⟩, ⟨.program ⟨214⟩, ⟨15067⟩⟩], []⟩) [⟨.result 703 .coefficient, true, some 1⟩, ⟨.result 706 .coefficient, true, some 1⟩])

def event6367 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨18906⟩⟩) (.sum [.transfer 6365, .transfer 6366])

def event6368 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨18906⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨6495⟩⟩, ⟨.program ⟨214⟩, ⟨14906⟩⟩], []⟩) [⟨.result 713 .coefficient, true, some 1⟩, ⟨.result 716 .coefficient, true, some 1⟩])

def event6369 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨18906⟩⟩) (.sum [.transfer 6367, .transfer 6368])

def event6370 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨18906⟩⟩) (.product (.result 6329 .summary) (.transfer 6369) (⟨false, false, none, none, none⟩))

def event6371 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18906⟩⟩, .operator (⟨6329, 0⟩, ⟨804, 5⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6410⟩⟩, ⟨.program ⟨214⟩, ⟨18511⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def event6372 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18906⟩⟩, .operator (⟨6329, 0⟩, ⟨804, 7⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6435⟩⟩, ⟨.program ⟨214⟩, ⟨18140⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def event6373 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18906⟩⟩, .operator (⟨6329, 0⟩, ⟨804, 8⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6437⟩⟩, ⟨.program ⟨214⟩, ⟨16943⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def event6374 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18906⟩⟩, .operator (⟨6329, 0⟩, ⟨804, 9⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6449⟩⟩, ⟨.program ⟨214⟩, ⟨17510⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def event6375 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18906⟩⟩, .operator (⟨6329, 0⟩, ⟨804, 11⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6459⟩⟩, ⟨.program ⟨214⟩, ⟨17734⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def event6376 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18906⟩⟩, .operator (⟨6329, 0⟩, ⟨804, 12⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6467⟩⟩, ⟨.program ⟨214⟩, ⟨17965⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def event6377 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18906⟩⟩, .operator (⟨6329, 0⟩, ⟨804, 13⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6473⟩⟩, ⟨.program ⟨214⟩, ⟨17566⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def event6378 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18906⟩⟩, .operator (⟨6329, 0⟩, ⟨804, 15⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6490⟩⟩, ⟨.program ⟨214⟩, ⟨18893⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def event6379 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18906⟩⟩, .operator (⟨6329, 0⟩, ⟨804, 16⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6494⟩⟩, ⟨.program ⟨214⟩, ⟨17622⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def event6380 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18906⟩⟩, .operator (⟨6329, 0⟩, ⟨804, 18⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6502⟩⟩, ⟨.program ⟨214⟩, ⟨17678⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def event6381 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18906⟩⟩, .operator (⟨6329, 0⟩, ⟨804, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6383⟩⟩, ⟨.program ⟨214⟩, ⟨18063⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def event6382 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18906⟩⟩, .operator (⟨6329, 0⟩, ⟨804, 1⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6387⟩⟩, ⟨.program ⟨214⟩, ⟨17181⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def event6383 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18906⟩⟩, .operator (⟨6329, 0⟩, ⟨804, 2⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6391⟩⟩, ⟨.program ⟨214⟩, ⟨17237⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def event6384 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18906⟩⟩, .operator (⟨6329, 0⟩, ⟨804, 3⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6398⟩⟩, ⟨.program ⟨214⟩, ⟨17454⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def event6385 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18906⟩⟩, .operator (⟨6329, 0⟩, ⟨804, 4⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6407⟩⟩, ⟨.program ⟨214⟩, ⟨17846⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def event6386 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18906⟩⟩, .operator (⟨6329, 0⟩, ⟨804, 6⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6427⟩⟩, ⟨.program ⟨214⟩, ⟨15536⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def event6387 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18906⟩⟩, .operator (⟨6329, 0⟩, ⟨804, 10⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6452⟩⟩, ⟨.program ⟨214⟩, ⟨15228⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def event6388 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18906⟩⟩, .operator (⟨6329, 0⟩, ⟨804, 14⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6475⟩⟩, ⟨.program ⟨214⟩, ⟨15067⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def event6389 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18906⟩⟩, .operator (⟨6329, 0⟩, ⟨804, 17⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6495⟩⟩, ⟨.program ⟨214⟩, ⟨14906⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def event6390 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18906⟩⟩, .operator (⟨6329, 1⟩, ⟨804, 5⟩), ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6410⟩⟩, ⟨.program ⟨214⟩, ⟨18511⟩⟩], [⟨.program ⟨214⟩, ⟨6746⟩⟩]⟩, (-1)⟩)

def event6391 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18906⟩⟩, .operator (⟨6329, 1⟩, ⟨804, 7⟩), ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6435⟩⟩, ⟨.program ⟨214⟩, ⟨18140⟩⟩], [⟨.program ⟨214⟩, ⟨6746⟩⟩]⟩, (1)⟩)

def event6392 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18906⟩⟩, .operator (⟨6329, 1⟩, ⟨804, 8⟩), ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6437⟩⟩, ⟨.program ⟨214⟩, ⟨16943⟩⟩], [⟨.program ⟨214⟩, ⟨6746⟩⟩]⟩, (1)⟩)

def event6393 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18906⟩⟩, .operator (⟨6329, 1⟩, ⟨804, 9⟩), ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6449⟩⟩, ⟨.program ⟨214⟩, ⟨17510⟩⟩], [⟨.program ⟨214⟩, ⟨6746⟩⟩]⟩, (1)⟩)

def event6394 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18906⟩⟩, .operator (⟨6329, 1⟩, ⟨804, 11⟩), ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6459⟩⟩, ⟨.program ⟨214⟩, ⟨17734⟩⟩], [⟨.program ⟨214⟩, ⟨6746⟩⟩]⟩, (1)⟩)

def event6395 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18906⟩⟩, .operator (⟨6329, 1⟩, ⟨804, 12⟩), ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6467⟩⟩, ⟨.program ⟨214⟩, ⟨17965⟩⟩], [⟨.program ⟨214⟩, ⟨6746⟩⟩]⟩, (1)⟩)

def event6396 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18906⟩⟩, .operator (⟨6329, 1⟩, ⟨804, 13⟩), ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6473⟩⟩, ⟨.program ⟨214⟩, ⟨17566⟩⟩], [⟨.program ⟨214⟩, ⟨6746⟩⟩]⟩, (1)⟩)

def event6397 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18906⟩⟩, .operator (⟨6329, 1⟩, ⟨804, 15⟩), ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6490⟩⟩, ⟨.program ⟨214⟩, ⟨18893⟩⟩], [⟨.program ⟨214⟩, ⟨6746⟩⟩]⟩, (1)⟩)

def event6398 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18906⟩⟩, .operator (⟨6329, 1⟩, ⟨804, 16⟩), ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6494⟩⟩, ⟨.program ⟨214⟩, ⟨17622⟩⟩], [⟨.program ⟨214⟩, ⟨6746⟩⟩]⟩, (1)⟩)

def event6399 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18906⟩⟩, .operator (⟨6329, 1⟩, ⟨804, 18⟩), ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6502⟩⟩, ⟨.program ⟨214⟩, ⟨17678⟩⟩], [⟨.program ⟨214⟩, ⟨6746⟩⟩]⟩, (1)⟩)

def eventLeaf384 : Array AnnotatedEvent := #[
  { event := event6144
    frameStart := 0 },
  { event := event6145
    frameStart := 0 },
  { event := event6146
    frameStart := 0 },
  { event := event6147
    frameStart := 0 },
  { event := event6148
    frameStart := 0 },
  { event := event6149
    frameStart := 0 },
  { event := event6150
    frameStart := 0 },
  { event := event6151
    frameStart := 0 },
  { event := event6152
    frameStart := 0 },
  { event := event6153
    frameStart := 0 },
  { event := event6154
    frameStart := 0 },
  { event := event6155
    frameStart := 0 },
  { event := event6156
    frameStart := 0 },
  { event := event6157
    frameStart := 0 },
  { event := event6158
    frameStart := 0 },
  { event := event6159
    frameStart := 0 }
]

def eventLeaf385 : Array AnnotatedEvent := #[
  { event := event6160
    frameStart := 0 },
  { event := event6161
    frameStart := 0 },
  { event := event6162
    frameStart := 0 },
  { event := event6163
    frameStart := 0 },
  { event := event6164
    frameStart := 0 },
  { event := event6165
    frameStart := 0 },
  { event := event6166
    frameStart := 0 },
  { event := event6167
    frameStart := 0 },
  { event := event6168
    frameStart := 0 },
  { event := event6169
    frameStart := 0 },
  { event := event6170
    frameStart := 0 },
  { event := event6171
    frameStart := 0 },
  { event := event6172
    frameStart := 0 },
  { event := event6173
    frameStart := 0 },
  { event := event6174
    frameStart := 0 },
  { event := event6175
    frameStart := 0 }
]

def eventLeaf386 : Array AnnotatedEvent := #[
  { event := event6176
    frameStart := 0 },
  { event := event6177
    frameStart := 0 },
  { event := event6178
    frameStart := 0 },
  { event := event6179
    frameStart := 0 },
  { event := event6180
    frameStart := 0 },
  { event := event6181
    frameStart := 0 },
  { event := event6182
    frameStart := 0 },
  { event := event6183
    frameStart := 0 },
  { event := event6184
    frameStart := 0 },
  { event := event6185
    frameStart := 0 },
  { event := event6186
    frameStart := 0 },
  { event := event6187
    frameStart := 0 },
  { event := event6188
    frameStart := 0 },
  { event := event6189
    frameStart := 0 },
  { event := event6190
    frameStart := 0 },
  { event := event6191
    frameStart := 0 }
]

def eventLeaf387 : Array AnnotatedEvent := #[
  { event := event6192
    frameStart := 0 },
  { event := event6193
    frameStart := 0 },
  { event := event6194
    frameStart := 0 },
  { event := event6195
    frameStart := 0 },
  { event := event6196
    frameStart := 0 },
  { event := event6197
    frameStart := 0 },
  { event := event6198
    frameStart := 0 },
  { event := event6199
    frameStart := 0 },
  { event := event6200
    frameStart := 0 },
  { event := event6201
    frameStart := 0 },
  { event := event6202
    frameStart := 0 },
  { event := event6203
    frameStart := 0 },
  { event := event6204
    frameStart := 0 },
  { event := event6205
    frameStart := 0 },
  { event := event6206
    frameStart := 0 },
  { event := event6207
    frameStart := 0 }
]

def eventLeaf388 : Array AnnotatedEvent := #[
  { event := event6208
    frameStart := 0 },
  { event := event6209
    frameStart := 0 },
  { event := event6210
    frameStart := 0 },
  { event := event6211
    frameStart := 0 },
  { event := event6212
    frameStart := 0 },
  { event := event6213
    frameStart := 0 },
  { event := event6214
    frameStart := 0 },
  { event := event6215
    frameStart := 0 },
  { event := event6216
    frameStart := 0 },
  { event := event6217
    frameStart := 0 },
  { event := event6218
    frameStart := 0 },
  { event := event6219
    frameStart := 0 },
  { event := event6220
    frameStart := 0 },
  { event := event6221
    frameStart := 0 },
  { event := event6222
    frameStart := 0 },
  { event := event6223
    frameStart := 0 }
]

def eventLeaf389 : Array AnnotatedEvent := #[
  { event := event6224
    frameStart := 0 },
  { event := event6225
    frameStart := 0 },
  { event := event6226
    frameStart := 0 },
  { event := event6227
    frameStart := 0 },
  { event := event6228
    frameStart := 0 },
  { event := event6229
    frameStart := 0 },
  { event := event6230
    frameStart := 0 },
  { event := event6231
    frameStart := 0 },
  { event := event6232
    frameStart := 0 },
  { event := event6233
    frameStart := 0 },
  { event := event6234
    frameStart := 0 },
  { event := event6235
    frameStart := 0 },
  { event := event6236
    frameStart := 0 },
  { event := event6237
    frameStart := 0 },
  { event := event6238
    frameStart := 0 },
  { event := event6239
    frameStart := 0 }
]

def eventLeaf390 : Array AnnotatedEvent := #[
  { event := event6240
    frameStart := 0 },
  { event := event6241
    frameStart := 0 },
  { event := event6242
    frameStart := 0 },
  { event := event6243
    frameStart := 0 },
  { event := event6244
    frameStart := 0 },
  { event := event6245
    frameStart := 0 },
  { event := event6246
    frameStart := 0 },
  { event := event6247
    frameStart := 0 },
  { event := event6248
    frameStart := 0 },
  { event := event6249
    frameStart := 0 },
  { event := event6250
    frameStart := 0 },
  { event := event6251
    frameStart := 0 },
  { event := event6252
    frameStart := 0 },
  { event := event6253
    frameStart := 0 },
  { event := event6254
    frameStart := 0 },
  { event := event6255
    frameStart := 0 }
]

def eventLeaf391 : Array AnnotatedEvent := #[
  { event := event6256
    frameStart := 0 },
  { event := event6257
    frameStart := 0 },
  { event := event6258
    frameStart := 0 },
  { event := event6259
    frameStart := 0 },
  { event := event6260
    frameStart := 0 },
  { event := event6261
    frameStart := 0 },
  { event := event6262
    frameStart := 0 },
  { event := event6263
    frameStart := 0 },
  { event := event6264
    frameStart := 0 },
  { event := event6265
    frameStart := 0 },
  { event := event6266
    frameStart := 0 },
  { event := event6267
    frameStart := 0 },
  { event := event6268
    frameStart := 0 },
  { event := event6269
    frameStart := 0 },
  { event := event6270
    frameStart := 0 },
  { event := event6271
    frameStart := 0 }
]

def eventLeaf392 : Array AnnotatedEvent := #[
  { event := event6272
    frameStart := 0 },
  { event := event6273
    frameStart := 0 },
  { event := event6274
    frameStart := 0 },
  { event := event6275
    frameStart := 0 },
  { event := event6276
    frameStart := 0 },
  { event := event6277
    frameStart := 0 },
  { event := event6278
    frameStart := 0 },
  { event := event6279
    frameStart := 0 },
  { event := event6280
    frameStart := 0 },
  { event := event6281
    frameStart := 0 },
  { event := event6282
    frameStart := 0 },
  { event := event6283
    frameStart := 0 },
  { event := event6284
    frameStart := 0 },
  { event := event6285
    frameStart := 0 },
  { event := event6286
    frameStart := 0 },
  { event := event6287
    frameStart := 0 }
]

def eventLeaf393 : Array AnnotatedEvent := #[
  { event := event6288
    frameStart := 0 },
  { event := event6289
    frameStart := 0 },
  { event := event6290
    frameStart := 0 },
  { event := event6291
    frameStart := 0 },
  { event := event6292
    frameStart := 0 },
  { event := event6293
    frameStart := 0 },
  { event := event6294
    frameStart := 0 },
  { event := event6295
    frameStart := 0 },
  { event := event6296
    frameStart := 0 },
  { event := event6297
    frameStart := 0 },
  { event := event6298
    frameStart := 0 },
  { event := event6299
    frameStart := 0 },
  { event := event6300
    frameStart := 0 },
  { event := event6301
    frameStart := 0 },
  { event := event6302
    frameStart := 0 },
  { event := event6303
    frameStart := 0 }
]

def eventLeaf394 : Array AnnotatedEvent := #[
  { event := event6304
    frameStart := 0 },
  { event := event6305
    frameStart := 0 },
  { event := event6306
    frameStart := 0 },
  { event := event6307
    frameStart := 0 },
  { event := event6308
    frameStart := 0 },
  { event := event6309
    frameStart := 0 },
  { event := event6310
    frameStart := 0 },
  { event := event6311
    frameStart := 0 },
  { event := event6312
    frameStart := 0 },
  { event := event6313
    frameStart := 0 },
  { event := event6314
    frameStart := 0 },
  { event := event6315
    frameStart := 0 },
  { event := event6316
    frameStart := 0 },
  { event := event6317
    frameStart := 0 },
  { event := event6318
    frameStart := 0 },
  { event := event6319
    frameStart := 0 }
]

def eventLeaf395 : Array AnnotatedEvent := #[
  { event := event6320
    frameStart := 0 },
  { event := event6321
    frameStart := 0 },
  { event := event6322
    frameStart := 0 },
  { event := event6323
    frameStart := 0 },
  { event := event6324
    frameStart := 0 },
  { event := event6325
    frameStart := 0 },
  { event := event6326
    frameStart := 0 },
  { event := event6327
    frameStart := 0 },
  { event := event6328
    frameStart := 0 },
  { event := event6329
    frameStart := 0 },
  { event := event6330
    frameStart := 0 },
  { event := event6331
    frameStart := 0 },
  { event := event6332
    frameStart := 0 },
  { event := event6333
    frameStart := 0 },
  { event := event6334
    frameStart := 0 },
  { event := event6335
    frameStart := 0 }
]

def eventLeaf396 : Array AnnotatedEvent := #[
  { event := event6336
    frameStart := 0 },
  { event := event6337
    frameStart := 0 },
  { event := event6338
    frameStart := 0 },
  { event := event6339
    frameStart := 0 },
  { event := event6340
    frameStart := 0 },
  { event := event6341
    frameStart := 0 },
  { event := event6342
    frameStart := 0 },
  { event := event6343
    frameStart := 0 },
  { event := event6344
    frameStart := 0 },
  { event := event6345
    frameStart := 0 },
  { event := event6346
    frameStart := 0 },
  { event := event6347
    frameStart := 0 },
  { event := event6348
    frameStart := 0 },
  { event := event6349
    frameStart := 0 },
  { event := event6350
    frameStart := 0 },
  { event := event6351
    frameStart := 0 }
]

def eventLeaf397 : Array AnnotatedEvent := #[
  { event := event6352
    frameStart := 0 },
  { event := event6353
    frameStart := 0 },
  { event := event6354
    frameStart := 0 },
  { event := event6355
    frameStart := 0 },
  { event := event6356
    frameStart := 0 },
  { event := event6357
    frameStart := 0 },
  { event := event6358
    frameStart := 0 },
  { event := event6359
    frameStart := 0 },
  { event := event6360
    frameStart := 0 },
  { event := event6361
    frameStart := 0 },
  { event := event6362
    frameStart := 0 },
  { event := event6363
    frameStart := 0 },
  { event := event6364
    frameStart := 0 },
  { event := event6365
    frameStart := 0 },
  { event := event6366
    frameStart := 0 },
  { event := event6367
    frameStart := 0 }
]

def eventLeaf398 : Array AnnotatedEvent := #[
  { event := event6368
    frameStart := 0 },
  { event := event6369
    frameStart := 0 },
  { event := event6370
    frameStart := 0 },
  { event := event6371
    frameStart := 0 },
  { event := event6372
    frameStart := 0 },
  { event := event6373
    frameStart := 0 },
  { event := event6374
    frameStart := 0 },
  { event := event6375
    frameStart := 0 },
  { event := event6376
    frameStart := 0 },
  { event := event6377
    frameStart := 0 },
  { event := event6378
    frameStart := 0 },
  { event := event6379
    frameStart := 0 },
  { event := event6380
    frameStart := 0 },
  { event := event6381
    frameStart := 0 },
  { event := event6382
    frameStart := 0 },
  { event := event6383
    frameStart := 0 }
]

def eventLeaf399 : Array AnnotatedEvent := #[
  { event := event6384
    frameStart := 0 },
  { event := event6385
    frameStart := 0 },
  { event := event6386
    frameStart := 0 },
  { event := event6387
    frameStart := 0 },
  { event := event6388
    frameStart := 0 },
  { event := event6389
    frameStart := 0 },
  { event := event6390
    frameStart := 0 },
  { event := event6391
    frameStart := 0 },
  { event := event6392
    frameStart := 0 },
  { event := event6393
    frameStart := 0 },
  { event := event6394
    frameStart := 0 },
  { event := event6395
    frameStart := 0 },
  { event := event6396
    frameStart := 0 },
  { event := event6397
    frameStart := 0 },
  { event := event6398
    frameStart := 0 },
  { event := event6399
    frameStart := 0 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events024
