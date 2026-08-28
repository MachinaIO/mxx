import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events274

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event70144 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11468⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨93⟩⟩]⟩) [⟨.result 11474 .coefficient, false, none⟩])

def event70145 : Event := .survivorFold (1) 70144

def exact70146RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6779⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨11465⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact70146RawTermsValid :
    exact70146RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70146 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11468⟩⟩) exact70146RawTerms .large 70143 (.finite 26) (some (70144))

def event70147 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14201⟩⟩) 0 ⟨11468⟩ 70146

def event70148 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14201⟩⟩) 1 ⟨14198⟩ 3319

def event70149 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14201⟩⟩) (.product (.predecessor 0 70147 .coefficient) (.predecessor 1 70148 .coefficient) (⟨false, true, none, none, some 1⟩))

def event70150 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14201⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨14198⟩⟩], []⟩) [⟨.result 3319 .coefficient, true, some 1⟩])

def event70151 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14201⟩⟩) (.product (.result 70146 .summary) (.transfer 70150) (⟨false, false, none, none, none⟩))

def event70152 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨14201⟩⟩, .operator (⟨70146, 1⟩, ⟨3319, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨11465⟩⟩, ⟨.program ⟨214⟩, ⟨14198⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def event70153 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨14201⟩⟩, .operator (⟨70146, 0⟩, ⟨3319, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨14198⟩⟩], [⟨.program ⟨214⟩, ⟨6779⟩⟩]⟩, (1)⟩)

def exact70154RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨11465⟩⟩, ⟨.program ⟨214⟩, ⟨14198⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨14198⟩⟩], [⟨.program ⟨214⟩, ⟨6779⟩⟩]⟩, (1)⟩]

theorem exact70154RawTermsValid :
    exact70154RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70154 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14201⟩⟩) exact70154RawTerms .large 70149 (.finite 14976) (some (70151))

def event70155 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14202⟩⟩) 0 ⟨14198⟩ 3319

def event70156 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14202⟩⟩) 1 ⟨6566⟩ 65295

def event70157 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14202⟩⟩) (.tensor (.predecessor 0 70155 .coefficient) (.predecessor 1 70156 .coefficient) true false)

def event70158 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨14202⟩⟩, .operator (⟨3319, 0⟩, ⟨65295, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨14198⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact70159RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨14198⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact70159RawTermsValid :
    exact70159RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70159 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14202⟩⟩) exact70159RawTerms .large 70157 .exactZero (none)

def event70160 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7177⟩⟩) 0 ⟨5533⟩ 65165

def event70161 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7177⟩⟩) 1 ⟨6759⟩ 11523

def event70162 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7177⟩⟩) (.product (.predecessor 0 70160 .coefficient) (.predecessor 1 70161 .coefficient) (⟨false, false, none, none, none⟩))

def event70163 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7177⟩⟩, .operator (⟨65165, 0⟩, ⟨11523, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6759⟩⟩]⟩, (1)⟩)

def exact70164RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6759⟩⟩]⟩, (1)⟩]

theorem exact70164RawTermsValid :
    exact70164RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70164 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7177⟩⟩) exact70164RawTerms .large 70162 .exactZero (none)

def event70165 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14203⟩⟩) 0 ⟨7177⟩ 70164

def event70166 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14203⟩⟩) 1 ⟨14202⟩ 70159

def event70167 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14203⟩⟩) (.sum [.predecessor 0 70165 .coefficient, .predecessor 1 70166 .coefficient])

def exact70168RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6759⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨14198⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact70168RawTermsValid :
    exact70168RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70168 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14203⟩⟩) exact70168RawTerms .large 70167 .exactZero (none)

def event70169 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14204⟩⟩) 0 ⟨14203⟩ 70168

def event70170 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14204⟩⟩) 1 ⟨73⟩ 11515

def event70171 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14204⟩⟩) (.sum [.predecessor 0 70169 .coefficient, .predecessor 1 70170 .coefficient])

def event70172 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14204⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨73⟩⟩]⟩) [⟨.result 11515 .coefficient, false, none⟩])

def event70173 : Event := .survivorFold (1) 70172

def exact70174RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6759⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨14198⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact70174RawTermsValid :
    exact70174RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70174 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14204⟩⟩) exact70174RawTerms .large 70171 (.finite 26) (some (70172))

def event70175 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14205⟩⟩) 0 ⟨14204⟩ 70174

def event70176 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14205⟩⟩) 1 ⟨7853⟩ 11512

def event70177 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14205⟩⟩) (.product (.predecessor 0 70175 .coefficient) (.predecessor 1 70176 .coefficient) (⟨false, false, none, none, none⟩))

def event70178 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14205⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨7852⟩⟩]⟩) [⟨.result 11508 .coefficient, false, none⟩])

def event70179 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14205⟩⟩) (.product (.result 70174 .summary) (.transfer 70178) (⟨false, false, none, none, none⟩))

def event70180 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨14205⟩⟩, .operator (⟨70174, 1⟩, ⟨11512, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨14198⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7852⟩⟩]⟩, (-1)⟩)

def event70181 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨14205⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨14198⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7852⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨7852⟩⟩) ⟨6779⟩ 11482)

def event70182 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨14205⟩⟩, .relation 70181 0, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨14198⟩⟩], [⟨.program ⟨214⟩, ⟨6779⟩⟩]⟩, (-1)⟩)

def event70183 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨14205⟩⟩, .operator (⟨70174, 0⟩, ⟨11512, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6759⟩⟩, ⟨.program ⟨214⟩, ⟨7852⟩⟩]⟩, (1)⟩)

def exact70184RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6759⟩⟩, ⟨.program ⟨214⟩, ⟨7852⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨14198⟩⟩], [⟨.program ⟨214⟩, ⟨6779⟩⟩]⟩, (-1)⟩]

theorem exact70184RawTermsValid :
    exact70184RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70184 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14205⟩⟩) exact70184RawTerms .large 70177 (.finite 95420416) (some (70179))

def event70185 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14206⟩⟩) 0 ⟨14205⟩ 70184

def event70186 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14206⟩⟩) 1 ⟨14201⟩ 70154

def event70187 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14206⟩⟩) (.sum [.predecessor 0 70185 .coefficient, .predecessor 1 70186 .coefficient])

def event70188 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨14206⟩⟩, .operator (⟨70184, 1⟩, ⟨70154, 1⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨14198⟩⟩], [⟨.program ⟨214⟩, ⟨6779⟩⟩]⟩, (1)⟩)

def event70189 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14206⟩⟩) (.sum [.result 70184 .summary, .result 70154 .summary])

def exact70190RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6759⟩⟩, ⟨.program ⟨214⟩, ⟨7852⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨11465⟩⟩, ⟨.program ⟨214⟩, ⟨14198⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact70190RawTermsValid :
    exact70190RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70190 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14206⟩⟩) exact70190RawTerms .large 70187 (.finite 95435392) (some (70189))

def event70191 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26062⟩⟩) 0 ⟨14206⟩ 70190

def event70192 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26062⟩⟩) 1 ⟨26061⟩ 70126

def event70193 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26062⟩⟩) (.product (.predecessor 0 70191 .coefficient) (.predecessor 1 70192 .coefficient) (⟨false, false, none, none, none⟩))

def event70194 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26062⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨26061⟩⟩]⟩) [⟨.result 70126 .coefficient, false, none⟩])

def event70195 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26062⟩⟩) (.product (.result 70190 .summary) (.transfer 70194) (⟨false, false, none, none, none⟩))

def event70196 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26062⟩⟩, .operator (⟨70190, 1⟩, ⟨70126, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨11465⟩⟩, ⟨.program ⟨214⟩, ⟨14198⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26061⟩⟩]⟩, (-1)⟩)

def event70197 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨26062⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨11465⟩⟩, ⟨.program ⟨214⟩, ⟨14198⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26061⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨26061⟩⟩) ⟨23582⟩ 70123)

def event70198 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26062⟩⟩, .relation 70197 0, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨11465⟩⟩, ⟨.program ⟨214⟩, ⟨14198⟩⟩], [⟨.program ⟨214⟩, ⟨23582⟩⟩]⟩, (-1)⟩)

def event70199 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26062⟩⟩, .operator (⟨70190, 0⟩, ⟨70126, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6759⟩⟩, ⟨.program ⟨214⟩, ⟨7852⟩⟩, ⟨.program ⟨214⟩, ⟨26061⟩⟩]⟩, (1)⟩)

def exact70200RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6759⟩⟩, ⟨.program ⟨214⟩, ⟨7852⟩⟩, ⟨.program ⟨214⟩, ⟨26061⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨11465⟩⟩, ⟨.program ⟨214⟩, ⟨14198⟩⟩], [⟨.program ⟨214⟩, ⟨23582⟩⟩]⟩, (-1)⟩]

theorem exact70200RawTermsValid :
    exact70200RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70200 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26062⟩⟩) exact70200RawTerms .large 70193 (.finite 350249415606272) (some (70195))

def event70201 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19524⟩⟩) 0 ⟨14200⟩ 3327

def event70202 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19524⟩⟩) (.authority (.relationPreimageSource ⟨15⟩))

def exact70203RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨19524⟩⟩]⟩, (1)⟩]

theorem exact70203RawTermsValid :
    exact70203RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70203 : Event := .resultExact (⟨.program ⟨214⟩, ⟨19524⟩⟩) exact70203RawTerms (.finite 136065468) 70202 .exactZero (none)

def event70204 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19526⟩⟩) 0 ⟨19524⟩ 70203

def event70205 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19526⟩⟩) 1 ⟨2348⟩ 4

def event70206 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19526⟩⟩) (.scale (.predecessor 0 70204 .coefficient) (.value (.predecessor 1 70205 .coefficient)))

def exact70207RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨19524⟩⟩]⟩, (1)⟩]

theorem exact70207RawTermsValid :
    exact70207RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70207 : Event := .resultExact (⟨.program ⟨214⟩, ⟨19526⟩⟩) exact70207RawTerms (.finite 136065468) 70206 .exactZero (none)

def event70208 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19527⟩⟩) 0 ⟨5535⟩ 65387

def event70209 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19527⟩⟩) 1 ⟨19526⟩ 70207

def event70210 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19527⟩⟩) (.product (.predecessor 0 70208 .coefficient) (.predecessor 1 70209 .coefficient) (⟨false, false, none, none, none⟩))

def event70211 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19527⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨19524⟩⟩]⟩) [⟨.result 70203 .coefficient, false, none⟩])

def event70212 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19527⟩⟩) (.product (.result 65387 .summary) (.transfer 70211) (⟨false, false, none, none, none⟩))

def event70213 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19527⟩⟩, .operator (⟨65387, 0⟩, ⟨70207, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19524⟩⟩]⟩, (1)⟩)

def event70214 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨19525⟩⟩)

def event70215 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event70216 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event70217 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨360⟩⟩) (.authority (.operator))

def event70218 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨360⟩⟩) (.finite 2)

def event70219 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event70220 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event70221 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event70222 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event70223 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 70222

def event70224 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 70220

def event70225 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 70223 .coefficient) (.value (.predecessor 1 70224 .coefficient)))

def event70226 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event70227 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5504⟩⟩) 0 ⟨5503⟩ 70226

def event70228 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5504⟩⟩) 1 ⟨360⟩ 70218

def event70229 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5504⟩⟩) (.sum [.predecessor 0 70227 .coefficient, .predecessor 1 70228 .coefficient])

def event70230 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5504⟩⟩) (.finite 219)

def event70231 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5530⟩⟩) 0 ⟨5504⟩ 70230

def event70232 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5530⟩⟩) 1 ⟨961⟩ 70216

def event70233 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5530⟩⟩) (.identity (.predecessor 1 70232 .coefficient))

def event70234 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5530⟩⟩) (.finite 224)

def event70235 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11465⟩⟩) 0 ⟨5530⟩ 70234

def event70236 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11465⟩⟩) (.authority (.programFamilyFact))

def exact70237RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11465⟩⟩], []⟩, (1)⟩]

theorem exact70237RawTermsValid :
    exact70237RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70237 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11465⟩⟩) exact70237RawTerms (.finite 18) 70236 .exactZero (none)

def event70238 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14198⟩⟩) 0 ⟨5530⟩ 70234

def event70239 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14198⟩⟩) (.authority (.programFamilyFact))

def exact70240RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨14198⟩⟩], []⟩, (1)⟩]

theorem exact70240RawTermsValid :
    exact70240RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70240 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14198⟩⟩) exact70240RawTerms (.finite 18) 70239 .exactZero (none)

def event70241 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14199⟩⟩) 0 ⟨14198⟩ 70240

def event70242 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14199⟩⟩) 1 ⟨11465⟩ 70237

def event70243 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14199⟩⟩) (.product (.predecessor 0 70241 .coefficient) (.predecessor 1 70242 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event70244 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14199⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨11465⟩⟩, ⟨.program ⟨214⟩, ⟨14198⟩⟩], []⟩) [⟨.result 70240 .coefficient, true, some 1⟩, ⟨.result 70237 .coefficient, true, some 1⟩])

def event70245 : Event := .survivorFold (1) 70244

def exact70246RawTerms : List Term := []

theorem exact70246RawTermsValid :
    exact70246RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70246 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14199⟩⟩) exact70246RawTerms (.finite 324) 70243 (.finite 324) (some (70244))

def event70247 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14200⟩⟩) 0 ⟨14199⟩ 70246

def event70248 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14200⟩⟩) (.identity (.predecessor 0 70247 .coefficient))

def event70249 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨14200⟩⟩) (.finite 324)

def event70250 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19524⟩⟩) 0 ⟨14200⟩ 70249

def event70251 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19524⟩⟩) (.authority (.relationPreimageSource ⟨15⟩))

def exact70252RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨19524⟩⟩]⟩, (1)⟩]

theorem exact70252RawTermsValid :
    exact70252RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70252 : Event := .resultExact (⟨.program ⟨214⟩, ⟨19524⟩⟩) exact70252RawTerms (.finite 136065468) 70251 .exactZero (none)

def event70253 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6⟩⟩) (.authority (.operator))

def exact70254RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩]⟩, (1)⟩]

theorem exact70254RawTermsValid :
    exact70254RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70254 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6⟩⟩) exact70254RawTerms .large 70253 .exactZero (none)

def event70255 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19525⟩⟩) 0 ⟨6⟩ 70254

def event70256 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19525⟩⟩) 1 ⟨19524⟩ 70252

def event70257 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19525⟩⟩) (.product (.predecessor 0 70255 .coefficient) (.predecessor 1 70256 .coefficient) (⟨false, false, none, none, none⟩))

def event70258 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19525⟩⟩, .operator (⟨70254, 0⟩, ⟨70252, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19524⟩⟩]⟩, (1)⟩)

def exact70259RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19524⟩⟩]⟩, (1)⟩]

theorem exact70259RawTermsValid :
    exact70259RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70259 : Event := .resultExact (⟨.program ⟨214⟩, ⟨19525⟩⟩) exact70259RawTerms .large 70257 .exactZero (none)

def event70260 : Event := .preFoldPolynomial 70259 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19524⟩⟩]⟩, (1)⟩] .exactZero none

def exact70261RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19524⟩⟩]⟩, (1)⟩]

def event70261 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨19525⟩⟩) 70260 exact70261RawTerms .large 70257 .exactZero (none)

def event70262 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨26065⟩⟩)

def event70263 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event70264 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event70265 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨360⟩⟩) (.authority (.operator))

def event70266 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨360⟩⟩) (.finite 2)

def event70267 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event70268 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event70269 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event70270 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event70271 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 70270

def event70272 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 70268

def event70273 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 70271 .coefficient) (.value (.predecessor 1 70272 .coefficient)))

def event70274 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event70275 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5504⟩⟩) 0 ⟨5503⟩ 70274

def event70276 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5504⟩⟩) 1 ⟨360⟩ 70266

def event70277 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5504⟩⟩) (.sum [.predecessor 0 70275 .coefficient, .predecessor 1 70276 .coefficient])

def event70278 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5504⟩⟩) (.finite 219)

def event70279 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5530⟩⟩) 0 ⟨5504⟩ 70278

def event70280 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5530⟩⟩) 1 ⟨961⟩ 70264

def event70281 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5530⟩⟩) (.identity (.predecessor 1 70280 .coefficient))

def event70282 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5530⟩⟩) (.finite 224)

def event70283 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11465⟩⟩) 0 ⟨5530⟩ 70282

def event70284 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11465⟩⟩) (.authority (.programFamilyFact))

def exact70285RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11465⟩⟩], []⟩, (1)⟩]

theorem exact70285RawTermsValid :
    exact70285RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70285 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11465⟩⟩) exact70285RawTerms (.finite 18) 70284 .exactZero (none)

def event70286 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14198⟩⟩) 0 ⟨5530⟩ 70282

def event70287 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14198⟩⟩) (.authority (.programFamilyFact))

def exact70288RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨14198⟩⟩], []⟩, (1)⟩]

theorem exact70288RawTermsValid :
    exact70288RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70288 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14198⟩⟩) exact70288RawTerms (.finite 18) 70287 .exactZero (none)

def event70289 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14199⟩⟩) 0 ⟨14198⟩ 70288

def event70290 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14199⟩⟩) 1 ⟨11465⟩ 70285

def event70291 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14199⟩⟩) (.product (.predecessor 0 70289 .coefficient) (.predecessor 1 70290 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event70292 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨14199⟩⟩, .operator (⟨70288, 0⟩, ⟨70285, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨11465⟩⟩, ⟨.program ⟨214⟩, ⟨14198⟩⟩], []⟩, (1)⟩)

def exact70293RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11465⟩⟩, ⟨.program ⟨214⟩, ⟨14198⟩⟩], []⟩, (1)⟩]

theorem exact70293RawTermsValid :
    exact70293RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70293 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14199⟩⟩) exact70293RawTerms (.finite 324) 70291 .exactZero (none)

def event70294 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14200⟩⟩) 0 ⟨14199⟩ 70293

def event70295 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14200⟩⟩) (.identity (.predecessor 0 70294 .coefficient))

def event70296 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨14200⟩⟩) (.finite 324)

def event70297 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23581⟩⟩) 0 ⟨14200⟩ 70296

def event70298 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23581⟩⟩) (.authority (.programFamilyFact))

def event70299 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨23581⟩⟩) (.finite 3720)

def event70300 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨6689⟩⟩) .missing

def event70301 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23582⟩⟩) 0 ⟨6689⟩ 70300

def event70302 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23582⟩⟩) 1 ⟨23581⟩ 70299

def event70303 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23582⟩⟩) (.authority (.operator))

def exact70304RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨23582⟩⟩]⟩, (1)⟩]

theorem exact70304RawTermsValid :
    exact70304RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70304 : Event := .resultExact (⟨.program ⟨214⟩, ⟨23582⟩⟩) exact70304RawTerms .large 70303 .exactZero (none)

def event70305 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26061⟩⟩) 0 ⟨23582⟩ 70304

def event70306 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26061⟩⟩) (.authority (.operator))

def exact70307RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨26061⟩⟩]⟩, (1)⟩]

theorem exact70307RawTermsValid :
    exact70307RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70307 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26061⟩⟩) exact70307RawTerms (.finite 8192) 70306 .exactZero (none)

def event70308 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨110⟩⟩) (.authority (.operator))

def event70309 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨110⟩⟩) .exactZero

def event70310 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14310⟩⟩) 0 ⟨14200⟩ 70296

def event70311 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14310⟩⟩) 1 ⟨110⟩ 70309

def event70312 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14310⟩⟩) (.sum [.predecessor 0 70310 .coefficient, .predecessor 1 70311 .coefficient])

def event70313 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨14310⟩⟩) (.finite 324)

def event70314 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14311⟩⟩) 0 ⟨14310⟩ 70313

def event70315 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14311⟩⟩) (.identity (.predecessor 0 70314 .coefficient))

def exact70316RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11465⟩⟩, ⟨.program ⟨214⟩, ⟨14198⟩⟩], []⟩, (1)⟩]

theorem exact70316RawTermsValid :
    exact70316RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70316 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14311⟩⟩) exact70316RawTerms (.finite 324) 70315 .exactZero (none)

def event70317 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6544⟩⟩) (.authority (.factStore))

def exact70318RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact70318RawTermsValid :
    exact70318RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70318 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6544⟩⟩) exact70318RawTerms .large 70317 .exactZero (none)

def event70319 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14312⟩⟩) 0 ⟨6544⟩ 70318

def event70320 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14312⟩⟩) 1 ⟨14311⟩ 70316

def event70321 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14312⟩⟩) (.product (.predecessor 0 70319 .coefficient) (.predecessor 1 70320 .coefficient) (⟨false, false, none, none, none⟩))

def event70322 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨14312⟩⟩, .operator (⟨70318, 0⟩, ⟨70316, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨11465⟩⟩, ⟨.program ⟨214⟩, ⟨14198⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact70323RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11465⟩⟩, ⟨.program ⟨214⟩, ⟨14198⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact70323RawTermsValid :
    exact70323RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70323 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14312⟩⟩) exact70323RawTerms .large 70321 .exactZero (none)

def event70324 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨2348⟩⟩) (.authority (.operator))

def event70325 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨2348⟩⟩) (.finite 1)

def event70326 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6757⟩⟩) 0 ⟨6689⟩ 70300

def event70327 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6757⟩⟩) (.authority (.operator))

def exact70328RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6757⟩⟩]⟩, (1)⟩]

theorem exact70328RawTermsValid :
    exact70328RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70328 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6757⟩⟩) exact70328RawTerms .large 70327 .exactZero (none)

def event70329 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6779⟩⟩) 0 ⟨6757⟩ 70328

def event70330 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6779⟩⟩) (.identity (.predecessor 0 70329 .coefficient))

def exact70331RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6779⟩⟩]⟩, (1)⟩]

theorem exact70331RawTermsValid :
    exact70331RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70331 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6779⟩⟩) exact70331RawTerms .large 70330 .exactZero (none)

def event70332 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7852⟩⟩) 0 ⟨6779⟩ 70331

def event70333 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7852⟩⟩) (.authority (.operator))

def exact70334RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨7852⟩⟩]⟩, (1)⟩]

theorem exact70334RawTermsValid :
    exact70334RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70334 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7852⟩⟩) exact70334RawTerms (.finite 8192) 70333 .exactZero (none)

def event70335 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7853⟩⟩) 0 ⟨7852⟩ 70334

def event70336 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7853⟩⟩) 1 ⟨2348⟩ 70325

def event70337 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7853⟩⟩) (.scale (.predecessor 0 70335 .coefficient) (.value (.predecessor 1 70336 .coefficient)))

def exact70338RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨7852⟩⟩]⟩, (1)⟩]

theorem exact70338RawTermsValid :
    exact70338RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70338 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7853⟩⟩) exact70338RawTerms (.finite 8192) 70337 .exactZero (none)

def event70339 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6759⟩⟩) 0 ⟨6757⟩ 70328

def event70340 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6759⟩⟩) (.identity (.predecessor 0 70339 .coefficient))

def exact70341RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6759⟩⟩]⟩, (1)⟩]

theorem exact70341RawTermsValid :
    exact70341RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70341 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6759⟩⟩) exact70341RawTerms .large 70340 .exactZero (none)

def event70342 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7854⟩⟩) 0 ⟨6759⟩ 70341

def event70343 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7854⟩⟩) 1 ⟨7853⟩ 70338

def event70344 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7854⟩⟩) (.product (.predecessor 0 70342 .coefficient) (.predecessor 1 70343 .coefficient) (⟨false, false, none, none, none⟩))

def event70345 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7854⟩⟩, .operator (⟨70341, 0⟩, ⟨70338, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6759⟩⟩, ⟨.program ⟨214⟩, ⟨7852⟩⟩]⟩, (1)⟩)

def exact70346RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6759⟩⟩, ⟨.program ⟨214⟩, ⟨7852⟩⟩]⟩, (1)⟩]

theorem exact70346RawTermsValid :
    exact70346RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70346 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7854⟩⟩) exact70346RawTerms .large 70344 .exactZero (none)

def event70347 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14313⟩⟩) 0 ⟨7854⟩ 70346

def event70348 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14313⟩⟩) 1 ⟨14312⟩ 70323

def event70349 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14313⟩⟩) (.sum [.predecessor 0 70347 .coefficient, .predecessor 1 70348 .coefficient])

def exact70350RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6759⟩⟩, ⟨.program ⟨214⟩, ⟨7852⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨11465⟩⟩, ⟨.program ⟨214⟩, ⟨14198⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact70350RawTermsValid :
    exact70350RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70350 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14313⟩⟩) exact70350RawTerms .large 70349 .exactZero (none)

def event70351 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26064⟩⟩) 0 ⟨14313⟩ 70350

def event70352 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26064⟩⟩) 1 ⟨26061⟩ 70307

def event70353 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26064⟩⟩) (.product (.predecessor 0 70351 .coefficient) (.predecessor 1 70352 .coefficient) (⟨false, false, none, none, none⟩))

def event70354 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26064⟩⟩, .operator (⟨70350, 0⟩, ⟨70307, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6759⟩⟩, ⟨.program ⟨214⟩, ⟨7852⟩⟩, ⟨.program ⟨214⟩, ⟨26061⟩⟩]⟩, (1)⟩)

def event70355 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26064⟩⟩, .operator (⟨70350, 1⟩, ⟨70307, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨11465⟩⟩, ⟨.program ⟨214⟩, ⟨14198⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26061⟩⟩]⟩, (-1)⟩)

def event70356 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨26064⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨11465⟩⟩, ⟨.program ⟨214⟩, ⟨14198⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26061⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨26061⟩⟩) ⟨23582⟩ 70304)

def event70357 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26064⟩⟩, .relation 70356 0, ⟨[⟨.program ⟨214⟩, ⟨11465⟩⟩, ⟨.program ⟨214⟩, ⟨14198⟩⟩], [⟨.program ⟨214⟩, ⟨23582⟩⟩]⟩, (-1)⟩)

def exact70358RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6759⟩⟩, ⟨.program ⟨214⟩, ⟨7852⟩⟩, ⟨.program ⟨214⟩, ⟨26061⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨11465⟩⟩, ⟨.program ⟨214⟩, ⟨14198⟩⟩], [⟨.program ⟨214⟩, ⟨23582⟩⟩]⟩, (-1)⟩]

theorem exact70358RawTermsValid :
    exact70358RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70358 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26064⟩⟩) exact70358RawTerms .large 70353 .exactZero (none)

def event70359 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15936⟩⟩) 0 ⟨14200⟩ 70296

def event70360 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15936⟩⟩) (.authority (.programFamilyFact))

def exact70361RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15936⟩⟩], []⟩, (1)⟩]

theorem exact70361RawTermsValid :
    exact70361RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70361 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15936⟩⟩) exact70361RawTerms (.finite 18) 70360 .exactZero (none)

def event70362 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15938⟩⟩) 0 ⟨6544⟩ 70318

def event70363 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15938⟩⟩) 1 ⟨15936⟩ 70361

def event70364 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15938⟩⟩) (.product (.predecessor 0 70362 .coefficient) (.predecessor 1 70363 .coefficient) (⟨false, true, none, none, some 1⟩))

def event70365 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨15938⟩⟩, .operator (⟨70318, 0⟩, ⟨70361, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨15936⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact70366RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15936⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact70366RawTermsValid :
    exact70366RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70366 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15938⟩⟩) exact70366RawTerms .large 70364 .exactZero (none)

def event70367 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6697⟩⟩) 0 ⟨6689⟩ 70300

def event70368 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6697⟩⟩) (.authority (.operator))

def exact70369RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6697⟩⟩]⟩, (1)⟩]

theorem exact70369RawTermsValid :
    exact70369RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70369 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6697⟩⟩) exact70369RawTerms .large 70368 .exactZero (none)

def event70370 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15939⟩⟩) 0 ⟨6697⟩ 70369

def event70371 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15939⟩⟩) 1 ⟨15938⟩ 70366

def event70372 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15939⟩⟩) (.sum [.predecessor 0 70370 .coefficient, .predecessor 1 70371 .coefficient])

def exact70373RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6697⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15936⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact70373RawTermsValid :
    exact70373RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70373 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15939⟩⟩) exact70373RawTerms .large 70372 .exactZero (none)

def event70374 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26065⟩⟩) 0 ⟨15939⟩ 70373

def event70375 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26065⟩⟩) 1 ⟨26064⟩ 70358

def event70376 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26065⟩⟩) (.sum [.predecessor 0 70374 .coefficient, .predecessor 1 70375 .coefficient])

def exact70377RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6697⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6759⟩⟩, ⟨.program ⟨214⟩, ⟨7852⟩⟩, ⟨.program ⟨214⟩, ⟨26061⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨11465⟩⟩, ⟨.program ⟨214⟩, ⟨14198⟩⟩], [⟨.program ⟨214⟩, ⟨23582⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15936⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact70377RawTermsValid :
    exact70377RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70377 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26065⟩⟩) exact70377RawTerms .large 70376 .exactZero (none)

def event70378 : Event := .preFoldPolynomial 70377 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6697⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6759⟩⟩, ⟨.program ⟨214⟩, ⟨7852⟩⟩, ⟨.program ⟨214⟩, ⟨26061⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨11465⟩⟩, ⟨.program ⟨214⟩, ⟨14198⟩⟩], [⟨.program ⟨214⟩, ⟨23582⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15936⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩] .exactZero none

def exact70379RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6697⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6759⟩⟩, ⟨.program ⟨214⟩, ⟨7852⟩⟩, ⟨.program ⟨214⟩, ⟨26061⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨11465⟩⟩, ⟨.program ⟨214⟩, ⟨14198⟩⟩], [⟨.program ⟨214⟩, ⟨23582⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15936⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

def event70379 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨26065⟩⟩) 70378 exact70379RawTerms .large 70376 .exactZero (none)

def event70380 : Event := .specializationComputed (⟨.program ⟨214⟩, ⟨14200⟩⟩) ⟨⟨110⟩, ⟨15⟩, ⟨109⟩⟩ ⟨70214, 70380⟩

def event70381 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨19527⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19524⟩⟩]⟩) (1) 0 2 (.universal 70380 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19524⟩⟩]⟩) (none) 70379)

def event70382 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19527⟩⟩, .relation 70381 0, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6697⟩⟩]⟩, (1)⟩)

def event70383 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19527⟩⟩, .relation 70381 1, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6759⟩⟩, ⟨.program ⟨214⟩, ⟨7852⟩⟩, ⟨.program ⟨214⟩, ⟨26061⟩⟩]⟩, (-1)⟩)

def event70384 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19527⟩⟩, .relation 70381 2, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨11465⟩⟩, ⟨.program ⟨214⟩, ⟨14198⟩⟩], [⟨.program ⟨214⟩, ⟨23582⟩⟩]⟩, (1)⟩)

def event70385 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19527⟩⟩, .relation 70381 3, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15936⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def exact70386RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6697⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6759⟩⟩, ⟨.program ⟨214⟩, ⟨7852⟩⟩, ⟨.program ⟨214⟩, ⟨26061⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨11465⟩⟩, ⟨.program ⟨214⟩, ⟨14198⟩⟩], [⟨.program ⟨214⟩, ⟨23582⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15936⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact70386RawTermsValid :
    exact70386RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70386 : Event := .resultExact (⟨.program ⟨214⟩, ⟨19527⟩⟩) exact70386RawTerms .large 70210 (.finite 1811303510016) (some (70212))

def event70387 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26063⟩⟩) 0 ⟨19527⟩ 70386

def event70388 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26063⟩⟩) 1 ⟨26062⟩ 70200

def event70389 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26063⟩⟩) (.sum [.predecessor 0 70387 .coefficient, .predecessor 1 70388 .coefficient])

def event70390 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26063⟩⟩, .operator (⟨70386, 2⟩, ⟨70200, 1⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨11465⟩⟩, ⟨.program ⟨214⟩, ⟨14198⟩⟩], [⟨.program ⟨214⟩, ⟨23582⟩⟩]⟩, (-1)⟩)

def event70391 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26063⟩⟩, .operator (⟨70386, 1⟩, ⟨70200, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6759⟩⟩, ⟨.program ⟨214⟩, ⟨7852⟩⟩, ⟨.program ⟨214⟩, ⟨26061⟩⟩]⟩, (1)⟩)

def event70392 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26063⟩⟩) (.sum [.result 70386 .summary, .result 70200 .summary])

def exact70393RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6697⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15936⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact70393RawTermsValid :
    exact70393RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70393 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26063⟩⟩) exact70393RawTerms .large 70389 (.finite 352060719116288) (some (70392))

def event70394 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27855⟩⟩) 0 ⟨26063⟩ 70393

def event70395 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27855⟩⟩) 1 ⟨27853⟩ 70116

def event70396 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27855⟩⟩) (.product (.predecessor 0 70394 .coefficient) (.predecessor 1 70395 .coefficient) (⟨false, false, none, none, none⟩))

def event70397 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27855⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨27853⟩⟩]⟩) [⟨.result 70116 .coefficient, false, none⟩])

def event70398 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27855⟩⟩) (.product (.result 70393 .summary) (.transfer 70397) (⟨false, false, none, none, none⟩))

def event70399 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27855⟩⟩, .operator (⟨70393, 0⟩, ⟨70116, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6697⟩⟩, ⟨.program ⟨214⟩, ⟨27853⟩⟩]⟩, (1)⟩)

def eventLeaf4384 : Array AnnotatedEvent := #[
  { event := event70144
    frameStart := 0 },
  { event := event70145
    frameStart := 0 },
  { event := event70146
    frameStart := 0 },
  { event := event70147
    frameStart := 0 },
  { event := event70148
    frameStart := 0 },
  { event := event70149
    frameStart := 0 },
  { event := event70150
    frameStart := 0 },
  { event := event70151
    frameStart := 0 },
  { event := event70152
    frameStart := 0 },
  { event := event70153
    frameStart := 0 },
  { event := event70154
    frameStart := 0 },
  { event := event70155
    frameStart := 0 },
  { event := event70156
    frameStart := 0 },
  { event := event70157
    frameStart := 0 },
  { event := event70158
    frameStart := 0 },
  { event := event70159
    frameStart := 0 }
]

def eventLeaf4385 : Array AnnotatedEvent := #[
  { event := event70160
    frameStart := 0 },
  { event := event70161
    frameStart := 0 },
  { event := event70162
    frameStart := 0 },
  { event := event70163
    frameStart := 0 },
  { event := event70164
    frameStart := 0 },
  { event := event70165
    frameStart := 0 },
  { event := event70166
    frameStart := 0 },
  { event := event70167
    frameStart := 0 },
  { event := event70168
    frameStart := 0 },
  { event := event70169
    frameStart := 0 },
  { event := event70170
    frameStart := 0 },
  { event := event70171
    frameStart := 0 },
  { event := event70172
    frameStart := 0 },
  { event := event70173
    frameStart := 0 },
  { event := event70174
    frameStart := 0 },
  { event := event70175
    frameStart := 0 }
]

def eventLeaf4386 : Array AnnotatedEvent := #[
  { event := event70176
    frameStart := 0 },
  { event := event70177
    frameStart := 0 },
  { event := event70178
    frameStart := 0 },
  { event := event70179
    frameStart := 0 },
  { event := event70180
    frameStart := 0 },
  { event := event70181
    frameStart := 0 },
  { event := event70182
    frameStart := 0 },
  { event := event70183
    frameStart := 0 },
  { event := event70184
    frameStart := 0 },
  { event := event70185
    frameStart := 0 },
  { event := event70186
    frameStart := 0 },
  { event := event70187
    frameStart := 0 },
  { event := event70188
    frameStart := 0 },
  { event := event70189
    frameStart := 0 },
  { event := event70190
    frameStart := 0 },
  { event := event70191
    frameStart := 0 }
]

def eventLeaf4387 : Array AnnotatedEvent := #[
  { event := event70192
    frameStart := 0 },
  { event := event70193
    frameStart := 0 },
  { event := event70194
    frameStart := 0 },
  { event := event70195
    frameStart := 0 },
  { event := event70196
    frameStart := 0 },
  { event := event70197
    frameStart := 0 },
  { event := event70198
    frameStart := 0 },
  { event := event70199
    frameStart := 0 },
  { event := event70200
    frameStart := 0 },
  { event := event70201
    frameStart := 0 },
  { event := event70202
    frameStart := 0 },
  { event := event70203
    frameStart := 0 },
  { event := event70204
    frameStart := 0 },
  { event := event70205
    frameStart := 0 },
  { event := event70206
    frameStart := 0 },
  { event := event70207
    frameStart := 0 }
]

def eventLeaf4388 : Array AnnotatedEvent := #[
  { event := event70208
    frameStart := 0 },
  { event := event70209
    frameStart := 0 },
  { event := event70210
    frameStart := 0 },
  { event := event70211
    frameStart := 0 },
  { event := event70212
    frameStart := 0 },
  { event := event70213
    frameStart := 0 },
  { event := event70214
    frameStart := 70214 },
  { event := event70215
    frameStart := 70214 },
  { event := event70216
    frameStart := 70214 },
  { event := event70217
    frameStart := 70214 },
  { event := event70218
    frameStart := 70214 },
  { event := event70219
    frameStart := 70214 },
  { event := event70220
    frameStart := 70214 },
  { event := event70221
    frameStart := 70214 },
  { event := event70222
    frameStart := 70214 },
  { event := event70223
    frameStart := 70214 }
]

def eventLeaf4389 : Array AnnotatedEvent := #[
  { event := event70224
    frameStart := 70214 },
  { event := event70225
    frameStart := 70214 },
  { event := event70226
    frameStart := 70214 },
  { event := event70227
    frameStart := 70214 },
  { event := event70228
    frameStart := 70214 },
  { event := event70229
    frameStart := 70214 },
  { event := event70230
    frameStart := 70214 },
  { event := event70231
    frameStart := 70214 },
  { event := event70232
    frameStart := 70214 },
  { event := event70233
    frameStart := 70214 },
  { event := event70234
    frameStart := 70214 },
  { event := event70235
    frameStart := 70214 },
  { event := event70236
    frameStart := 70214 },
  { event := event70237
    frameStart := 70214 },
  { event := event70238
    frameStart := 70214 },
  { event := event70239
    frameStart := 70214 }
]

def eventLeaf4390 : Array AnnotatedEvent := #[
  { event := event70240
    frameStart := 70214 },
  { event := event70241
    frameStart := 70214 },
  { event := event70242
    frameStart := 70214 },
  { event := event70243
    frameStart := 70214 },
  { event := event70244
    frameStart := 70214 },
  { event := event70245
    frameStart := 70214 },
  { event := event70246
    frameStart := 70214 },
  { event := event70247
    frameStart := 70214 },
  { event := event70248
    frameStart := 70214 },
  { event := event70249
    frameStart := 70214 },
  { event := event70250
    frameStart := 70214 },
  { event := event70251
    frameStart := 70214 },
  { event := event70252
    frameStart := 70214 },
  { event := event70253
    frameStart := 70214 },
  { event := event70254
    frameStart := 70214 },
  { event := event70255
    frameStart := 70214 }
]

def eventLeaf4391 : Array AnnotatedEvent := #[
  { event := event70256
    frameStart := 70214 },
  { event := event70257
    frameStart := 70214 },
  { event := event70258
    frameStart := 70214 },
  { event := event70259
    frameStart := 70214 },
  { event := event70260
    frameStart := 70214 },
  { event := event70261
    frameStart := 70214 },
  { event := event70262
    frameStart := 70262 },
  { event := event70263
    frameStart := 70262 },
  { event := event70264
    frameStart := 70262 },
  { event := event70265
    frameStart := 70262 },
  { event := event70266
    frameStart := 70262 },
  { event := event70267
    frameStart := 70262 },
  { event := event70268
    frameStart := 70262 },
  { event := event70269
    frameStart := 70262 },
  { event := event70270
    frameStart := 70262 },
  { event := event70271
    frameStart := 70262 }
]

def eventLeaf4392 : Array AnnotatedEvent := #[
  { event := event70272
    frameStart := 70262 },
  { event := event70273
    frameStart := 70262 },
  { event := event70274
    frameStart := 70262 },
  { event := event70275
    frameStart := 70262 },
  { event := event70276
    frameStart := 70262 },
  { event := event70277
    frameStart := 70262 },
  { event := event70278
    frameStart := 70262 },
  { event := event70279
    frameStart := 70262 },
  { event := event70280
    frameStart := 70262 },
  { event := event70281
    frameStart := 70262 },
  { event := event70282
    frameStart := 70262 },
  { event := event70283
    frameStart := 70262 },
  { event := event70284
    frameStart := 70262 },
  { event := event70285
    frameStart := 70262 },
  { event := event70286
    frameStart := 70262 },
  { event := event70287
    frameStart := 70262 }
]

def eventLeaf4393 : Array AnnotatedEvent := #[
  { event := event70288
    frameStart := 70262 },
  { event := event70289
    frameStart := 70262 },
  { event := event70290
    frameStart := 70262 },
  { event := event70291
    frameStart := 70262 },
  { event := event70292
    frameStart := 70262 },
  { event := event70293
    frameStart := 70262 },
  { event := event70294
    frameStart := 70262 },
  { event := event70295
    frameStart := 70262 },
  { event := event70296
    frameStart := 70262 },
  { event := event70297
    frameStart := 70262 },
  { event := event70298
    frameStart := 70262 },
  { event := event70299
    frameStart := 70262 },
  { event := event70300
    frameStart := 70262 },
  { event := event70301
    frameStart := 70262 },
  { event := event70302
    frameStart := 70262 },
  { event := event70303
    frameStart := 70262 }
]

def eventLeaf4394 : Array AnnotatedEvent := #[
  { event := event70304
    frameStart := 70262 },
  { event := event70305
    frameStart := 70262 },
  { event := event70306
    frameStart := 70262 },
  { event := event70307
    frameStart := 70262 },
  { event := event70308
    frameStart := 70262 },
  { event := event70309
    frameStart := 70262 },
  { event := event70310
    frameStart := 70262 },
  { event := event70311
    frameStart := 70262 },
  { event := event70312
    frameStart := 70262 },
  { event := event70313
    frameStart := 70262 },
  { event := event70314
    frameStart := 70262 },
  { event := event70315
    frameStart := 70262 },
  { event := event70316
    frameStart := 70262 },
  { event := event70317
    frameStart := 70262 },
  { event := event70318
    frameStart := 70262 },
  { event := event70319
    frameStart := 70262 }
]

def eventLeaf4395 : Array AnnotatedEvent := #[
  { event := event70320
    frameStart := 70262 },
  { event := event70321
    frameStart := 70262 },
  { event := event70322
    frameStart := 70262 },
  { event := event70323
    frameStart := 70262 },
  { event := event70324
    frameStart := 70262 },
  { event := event70325
    frameStart := 70262 },
  { event := event70326
    frameStart := 70262 },
  { event := event70327
    frameStart := 70262 },
  { event := event70328
    frameStart := 70262 },
  { event := event70329
    frameStart := 70262 },
  { event := event70330
    frameStart := 70262 },
  { event := event70331
    frameStart := 70262 },
  { event := event70332
    frameStart := 70262 },
  { event := event70333
    frameStart := 70262 },
  { event := event70334
    frameStart := 70262 },
  { event := event70335
    frameStart := 70262 }
]

def eventLeaf4396 : Array AnnotatedEvent := #[
  { event := event70336
    frameStart := 70262 },
  { event := event70337
    frameStart := 70262 },
  { event := event70338
    frameStart := 70262 },
  { event := event70339
    frameStart := 70262 },
  { event := event70340
    frameStart := 70262 },
  { event := event70341
    frameStart := 70262 },
  { event := event70342
    frameStart := 70262 },
  { event := event70343
    frameStart := 70262 },
  { event := event70344
    frameStart := 70262 },
  { event := event70345
    frameStart := 70262 },
  { event := event70346
    frameStart := 70262 },
  { event := event70347
    frameStart := 70262 },
  { event := event70348
    frameStart := 70262 },
  { event := event70349
    frameStart := 70262 },
  { event := event70350
    frameStart := 70262 },
  { event := event70351
    frameStart := 70262 }
]

def eventLeaf4397 : Array AnnotatedEvent := #[
  { event := event70352
    frameStart := 70262 },
  { event := event70353
    frameStart := 70262 },
  { event := event70354
    frameStart := 70262 },
  { event := event70355
    frameStart := 70262 },
  { event := event70356
    frameStart := 70262 },
  { event := event70357
    frameStart := 70262 },
  { event := event70358
    frameStart := 70262 },
  { event := event70359
    frameStart := 70262 },
  { event := event70360
    frameStart := 70262 },
  { event := event70361
    frameStart := 70262 },
  { event := event70362
    frameStart := 70262 },
  { event := event70363
    frameStart := 70262 },
  { event := event70364
    frameStart := 70262 },
  { event := event70365
    frameStart := 70262 },
  { event := event70366
    frameStart := 70262 },
  { event := event70367
    frameStart := 70262 }
]

def eventLeaf4398 : Array AnnotatedEvent := #[
  { event := event70368
    frameStart := 70262 },
  { event := event70369
    frameStart := 70262 },
  { event := event70370
    frameStart := 70262 },
  { event := event70371
    frameStart := 70262 },
  { event := event70372
    frameStart := 70262 },
  { event := event70373
    frameStart := 70262 },
  { event := event70374
    frameStart := 70262 },
  { event := event70375
    frameStart := 70262 },
  { event := event70376
    frameStart := 70262 },
  { event := event70377
    frameStart := 70262 },
  { event := event70378
    frameStart := 70262 },
  { event := event70379
    frameStart := 70262 },
  { event := event70380
    frameStart := 0 },
  { event := event70381
    frameStart := 0 },
  { event := event70382
    frameStart := 0 },
  { event := event70383
    frameStart := 0 }
]

def eventLeaf4399 : Array AnnotatedEvent := #[
  { event := event70384
    frameStart := 0 },
  { event := event70385
    frameStart := 0 },
  { event := event70386
    frameStart := 0 },
  { event := event70387
    frameStart := 0 },
  { event := event70388
    frameStart := 0 },
  { event := event70389
    frameStart := 0 },
  { event := event70390
    frameStart := 0 },
  { event := event70391
    frameStart := 0 },
  { event := event70392
    frameStart := 0 },
  { event := event70393
    frameStart := 0 },
  { event := event70394
    frameStart := 0 },
  { event := event70395
    frameStart := 0 },
  { event := event70396
    frameStart := 0 },
  { event := event70397
    frameStart := 0 },
  { event := event70398
    frameStart := 0 },
  { event := event70399
    frameStart := 0 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events274
