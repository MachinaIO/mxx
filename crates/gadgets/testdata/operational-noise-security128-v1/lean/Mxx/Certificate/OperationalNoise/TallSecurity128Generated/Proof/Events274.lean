import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events274

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event70144 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48002⟩⟩) (.authority (.programFamilyFact))

def exact70145RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨48002⟩⟩], []⟩, (1)⟩]

theorem exact70145RawTermsValid :
    exact70145RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70145 : Event := .resultExact (⟨.program ⟨257⟩, ⟨48002⟩⟩) exact70145RawTerms (.finite 60) 70144 .exactZero (none)

def event70146 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15186⟩⟩) 0 ⟨10749⟩ 70142

def event70147 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15186⟩⟩) (.authority (.programFamilyFact))

def exact70148RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15186⟩⟩], []⟩, (1)⟩]

theorem exact70148RawTermsValid :
    exact70148RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70148 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15186⟩⟩) exact70148RawTerms (.finite 60) 70147 .exactZero (none)

def event70149 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48003⟩⟩) 0 ⟨15186⟩ 70148

def event70150 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48003⟩⟩) 1 ⟨48002⟩ 70145

def event70151 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48003⟩⟩) (.product (.predecessor 0 70149 .coefficient) (.predecessor 1 70150 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event70152 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48003⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨15186⟩⟩, ⟨.program ⟨257⟩, ⟨48002⟩⟩], []⟩) [⟨.result 70148 .coefficient, true, some 1⟩, ⟨.result 70145 .coefficient, true, some 1⟩])

def event70153 : Event := .survivorFold (1) 70152

def exact70154RawTerms : List Term := []

theorem exact70154RawTermsValid :
    exact70154RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70154 : Event := .resultExact (⟨.program ⟨257⟩, ⟨48003⟩⟩) exact70154RawTerms (.finite 3600) 70151 (.finite 3600) (some (70152))

def event70155 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48004⟩⟩) 0 ⟨48003⟩ 70154

def event70156 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48004⟩⟩) (.identity (.predecessor 0 70155 .coefficient))

def event70157 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨48004⟩⟩) (.finite 3600)

def event70158 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48204⟩⟩) 0 ⟨48004⟩ 70157

def event70159 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48204⟩⟩) (.authority (.programFamilyFact))

def exact70160RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨48204⟩⟩], []⟩, (1)⟩]

theorem exact70160RawTermsValid :
    exact70160RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70160 : Event := .resultExact (⟨.program ⟨257⟩, ⟨48204⟩⟩) exact70160RawTerms (.finite 60) 70159 .exactZero (none)

def event70161 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48205⟩⟩) 0 ⟨48204⟩ 70160

def event70162 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48205⟩⟩) (.identity (.predecessor 0 70161 .coefficient))

def event70163 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨48205⟩⟩) (.finite 60)

def event70164 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48454⟩⟩) 0 ⟨48205⟩ 70163

def event70165 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48454⟩⟩) (.authority (.programFamilyFact))

def exact70166RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨48454⟩⟩], []⟩, (1)⟩]

theorem exact70166RawTermsValid :
    exact70166RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70166 : Event := .resultExact (⟨.program ⟨257⟩, ⟨48454⟩⟩) exact70166RawTerms (.finite 63) 70165 .exactZero (none)

def event70167 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45322⟩⟩) 0 ⟨10749⟩ 70142

def event70168 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45322⟩⟩) (.authority (.programFamilyFact))

def exact70169RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨45322⟩⟩], []⟩, (1)⟩]

theorem exact70169RawTermsValid :
    exact70169RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70169 : Event := .resultExact (⟨.program ⟨257⟩, ⟨45322⟩⟩) exact70169RawTerms (.finite 58) 70168 .exactZero (none)

def event70170 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14886⟩⟩) 0 ⟨10749⟩ 70142

def event70171 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14886⟩⟩) (.authority (.programFamilyFact))

def exact70172RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14886⟩⟩], []⟩, (1)⟩]

theorem exact70172RawTermsValid :
    exact70172RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70172 : Event := .resultExact (⟨.program ⟨257⟩, ⟨14886⟩⟩) exact70172RawTerms (.finite 58) 70171 .exactZero (none)

def event70173 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45323⟩⟩) 0 ⟨14886⟩ 70172

def event70174 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45323⟩⟩) 1 ⟨45322⟩ 70169

def event70175 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45323⟩⟩) (.product (.predecessor 0 70173 .coefficient) (.predecessor 1 70174 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event70176 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45323⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨14886⟩⟩, ⟨.program ⟨257⟩, ⟨45322⟩⟩], []⟩) [⟨.result 70172 .coefficient, true, some 1⟩, ⟨.result 70169 .coefficient, true, some 1⟩])

def event70177 : Event := .survivorFold (1) 70176

def exact70178RawTerms : List Term := []

theorem exact70178RawTermsValid :
    exact70178RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70178 : Event := .resultExact (⟨.program ⟨257⟩, ⟨45323⟩⟩) exact70178RawTerms (.finite 3364) 70175 (.finite 3364) (some (70176))

def event70179 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45324⟩⟩) 0 ⟨45323⟩ 70178

def event70180 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45324⟩⟩) (.identity (.predecessor 0 70179 .coefficient))

def event70181 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨45324⟩⟩) (.finite 3364)

def event70182 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45524⟩⟩) 0 ⟨45324⟩ 70181

def event70183 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45524⟩⟩) (.authority (.programFamilyFact))

def exact70184RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨45524⟩⟩], []⟩, (1)⟩]

theorem exact70184RawTermsValid :
    exact70184RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70184 : Event := .resultExact (⟨.program ⟨257⟩, ⟨45524⟩⟩) exact70184RawTerms (.finite 58) 70183 .exactZero (none)

def event70185 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45525⟩⟩) 0 ⟨45524⟩ 70184

def event70186 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45525⟩⟩) (.identity (.predecessor 0 70185 .coefficient))

def event70187 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨45525⟩⟩) (.finite 58)

def event70188 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45774⟩⟩) 0 ⟨45525⟩ 70187

def event70189 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45774⟩⟩) (.authority (.programFamilyFact))

def exact70190RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨45774⟩⟩], []⟩, (1)⟩]

theorem exact70190RawTermsValid :
    exact70190RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70190 : Event := .resultExact (⟨.program ⟨257⟩, ⟨45774⟩⟩) exact70190RawTerms (.finite 63) 70189 .exactZero (none)

def event70191 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42642⟩⟩) 0 ⟨10749⟩ 70142

def event70192 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42642⟩⟩) (.authority (.programFamilyFact))

def exact70193RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨42642⟩⟩], []⟩, (1)⟩]

theorem exact70193RawTermsValid :
    exact70193RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70193 : Event := .resultExact (⟨.program ⟨257⟩, ⟨42642⟩⟩) exact70193RawTerms (.finite 52) 70192 .exactZero (none)

def event70194 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14586⟩⟩) 0 ⟨10749⟩ 70142

def event70195 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14586⟩⟩) (.authority (.programFamilyFact))

def exact70196RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14586⟩⟩], []⟩, (1)⟩]

theorem exact70196RawTermsValid :
    exact70196RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70196 : Event := .resultExact (⟨.program ⟨257⟩, ⟨14586⟩⟩) exact70196RawTerms (.finite 52) 70195 .exactZero (none)

def event70197 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42643⟩⟩) 0 ⟨14586⟩ 70196

def event70198 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42643⟩⟩) 1 ⟨42642⟩ 70193

def event70199 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42643⟩⟩) (.product (.predecessor 0 70197 .coefficient) (.predecessor 1 70198 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event70200 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42643⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨14586⟩⟩, ⟨.program ⟨257⟩, ⟨42642⟩⟩], []⟩) [⟨.result 70196 .coefficient, true, some 1⟩, ⟨.result 70193 .coefficient, true, some 1⟩])

def event70201 : Event := .survivorFold (1) 70200

def exact70202RawTerms : List Term := []

theorem exact70202RawTermsValid :
    exact70202RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70202 : Event := .resultExact (⟨.program ⟨257⟩, ⟨42643⟩⟩) exact70202RawTerms (.finite 2704) 70199 (.finite 2704) (some (70200))

def event70203 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42644⟩⟩) 0 ⟨42643⟩ 70202

def event70204 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42644⟩⟩) (.identity (.predecessor 0 70203 .coefficient))

def event70205 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨42644⟩⟩) (.finite 2704)

def event70206 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42844⟩⟩) 0 ⟨42644⟩ 70205

def event70207 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42844⟩⟩) (.authority (.programFamilyFact))

def exact70208RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨42844⟩⟩], []⟩, (1)⟩]

theorem exact70208RawTermsValid :
    exact70208RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70208 : Event := .resultExact (⟨.program ⟨257⟩, ⟨42844⟩⟩) exact70208RawTerms (.finite 52) 70207 .exactZero (none)

def event70209 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42845⟩⟩) 0 ⟨42844⟩ 70208

def event70210 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42845⟩⟩) (.identity (.predecessor 0 70209 .coefficient))

def event70211 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨42845⟩⟩) (.finite 52)

def event70212 : Event := .predecessor (⟨.program ⟨257⟩, ⟨43090⟩⟩) 0 ⟨42845⟩ 70211

def event70213 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨43090⟩⟩) (.authority (.programFamilyFact))

def exact70214RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨43090⟩⟩], []⟩, (1)⟩]

theorem exact70214RawTermsValid :
    exact70214RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70214 : Event := .resultExact (⟨.program ⟨257⟩, ⟨43090⟩⟩) exact70214RawTerms (.finite 63) 70213 .exactZero (none)

def event70215 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39962⟩⟩) 0 ⟨10749⟩ 70142

def event70216 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39962⟩⟩) (.authority (.programFamilyFact))

def exact70217RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨39962⟩⟩], []⟩, (1)⟩]

theorem exact70217RawTermsValid :
    exact70217RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70217 : Event := .resultExact (⟨.program ⟨257⟩, ⟨39962⟩⟩) exact70217RawTerms (.finite 46) 70216 .exactZero (none)

def event70218 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14286⟩⟩) 0 ⟨10749⟩ 70142

def event70219 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14286⟩⟩) (.authority (.programFamilyFact))

def exact70220RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14286⟩⟩], []⟩, (1)⟩]

theorem exact70220RawTermsValid :
    exact70220RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70220 : Event := .resultExact (⟨.program ⟨257⟩, ⟨14286⟩⟩) exact70220RawTerms (.finite 46) 70219 .exactZero (none)

def event70221 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39963⟩⟩) 0 ⟨14286⟩ 70220

def event70222 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39963⟩⟩) 1 ⟨39962⟩ 70217

def event70223 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39963⟩⟩) (.product (.predecessor 0 70221 .coefficient) (.predecessor 1 70222 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event70224 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39963⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨14286⟩⟩, ⟨.program ⟨257⟩, ⟨39962⟩⟩], []⟩) [⟨.result 70220 .coefficient, true, some 1⟩, ⟨.result 70217 .coefficient, true, some 1⟩])

def event70225 : Event := .survivorFold (1) 70224

def exact70226RawTerms : List Term := []

theorem exact70226RawTermsValid :
    exact70226RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70226 : Event := .resultExact (⟨.program ⟨257⟩, ⟨39963⟩⟩) exact70226RawTerms (.finite 2116) 70223 (.finite 2116) (some (70224))

def event70227 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39964⟩⟩) 0 ⟨39963⟩ 70226

def event70228 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39964⟩⟩) (.identity (.predecessor 0 70227 .coefficient))

def event70229 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨39964⟩⟩) (.finite 2116)

def event70230 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40164⟩⟩) 0 ⟨39964⟩ 70229

def event70231 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨40164⟩⟩) (.authority (.programFamilyFact))

def exact70232RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨40164⟩⟩], []⟩, (1)⟩]

theorem exact70232RawTermsValid :
    exact70232RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70232 : Event := .resultExact (⟨.program ⟨257⟩, ⟨40164⟩⟩) exact70232RawTerms (.finite 46) 70231 .exactZero (none)

def event70233 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40165⟩⟩) 0 ⟨40164⟩ 70232

def event70234 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨40165⟩⟩) (.identity (.predecessor 0 70233 .coefficient))

def event70235 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨40165⟩⟩) (.finite 46)

def event70236 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40410⟩⟩) 0 ⟨40165⟩ 70235

def event70237 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨40410⟩⟩) (.authority (.programFamilyFact))

def exact70238RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨40410⟩⟩], []⟩, (1)⟩]

theorem exact70238RawTermsValid :
    exact70238RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70238 : Event := .resultExact (⟨.program ⟨257⟩, ⟨40410⟩⟩) exact70238RawTerms (.finite 63) 70237 .exactZero (none)

def event70239 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37282⟩⟩) 0 ⟨10749⟩ 70142

def event70240 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37282⟩⟩) (.authority (.programFamilyFact))

def exact70241RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨37282⟩⟩], []⟩, (1)⟩]

theorem exact70241RawTermsValid :
    exact70241RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70241 : Event := .resultExact (⟨.program ⟨257⟩, ⟨37282⟩⟩) exact70241RawTerms (.finite 42) 70240 .exactZero (none)

def event70242 : Event := .predecessor (⟨.program ⟨257⟩, ⟨13986⟩⟩) 0 ⟨10749⟩ 70142

def event70243 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨13986⟩⟩) (.authority (.programFamilyFact))

def exact70244RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨13986⟩⟩], []⟩, (1)⟩]

theorem exact70244RawTermsValid :
    exact70244RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70244 : Event := .resultExact (⟨.program ⟨257⟩, ⟨13986⟩⟩) exact70244RawTerms (.finite 42) 70243 .exactZero (none)

def event70245 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37283⟩⟩) 0 ⟨13986⟩ 70244

def event70246 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37283⟩⟩) 1 ⟨37282⟩ 70241

def event70247 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37283⟩⟩) (.product (.predecessor 0 70245 .coefficient) (.predecessor 1 70246 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event70248 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37283⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨13986⟩⟩, ⟨.program ⟨257⟩, ⟨37282⟩⟩], []⟩) [⟨.result 70244 .coefficient, true, some 1⟩, ⟨.result 70241 .coefficient, true, some 1⟩])

def event70249 : Event := .survivorFold (1) 70248

def exact70250RawTerms : List Term := []

theorem exact70250RawTermsValid :
    exact70250RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70250 : Event := .resultExact (⟨.program ⟨257⟩, ⟨37283⟩⟩) exact70250RawTerms (.finite 1764) 70247 (.finite 1764) (some (70248))

def event70251 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37284⟩⟩) 0 ⟨37283⟩ 70250

def event70252 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37284⟩⟩) (.identity (.predecessor 0 70251 .coefficient))

def event70253 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨37284⟩⟩) (.finite 1764)

def event70254 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37484⟩⟩) 0 ⟨37284⟩ 70253

def event70255 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37484⟩⟩) (.authority (.programFamilyFact))

def exact70256RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨37484⟩⟩], []⟩, (1)⟩]

theorem exact70256RawTermsValid :
    exact70256RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70256 : Event := .resultExact (⟨.program ⟨257⟩, ⟨37484⟩⟩) exact70256RawTerms (.finite 42) 70255 .exactZero (none)

def event70257 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37485⟩⟩) 0 ⟨37484⟩ 70256

def event70258 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37485⟩⟩) (.identity (.predecessor 0 70257 .coefficient))

def event70259 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨37485⟩⟩) (.finite 42)

def event70260 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37734⟩⟩) 0 ⟨37485⟩ 70259

def event70261 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37734⟩⟩) (.authority (.programFamilyFact))

def exact70262RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨37734⟩⟩], []⟩, (1)⟩]

theorem exact70262RawTermsValid :
    exact70262RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70262 : Event := .resultExact (⟨.program ⟨257⟩, ⟨37734⟩⟩) exact70262RawTerms (.finite 63) 70261 .exactZero (none)

def event70263 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34602⟩⟩) 0 ⟨10749⟩ 70142

def event70264 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34602⟩⟩) (.authority (.programFamilyFact))

def exact70265RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨34602⟩⟩], []⟩, (1)⟩]

theorem exact70265RawTermsValid :
    exact70265RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70265 : Event := .resultExact (⟨.program ⟨257⟩, ⟨34602⟩⟩) exact70265RawTerms (.finite 40) 70264 .exactZero (none)

def event70266 : Event := .predecessor (⟨.program ⟨257⟩, ⟨13686⟩⟩) 0 ⟨10749⟩ 70142

def event70267 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨13686⟩⟩) (.authority (.programFamilyFact))

def exact70268RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨13686⟩⟩], []⟩, (1)⟩]

theorem exact70268RawTermsValid :
    exact70268RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70268 : Event := .resultExact (⟨.program ⟨257⟩, ⟨13686⟩⟩) exact70268RawTerms (.finite 40) 70267 .exactZero (none)

def event70269 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34603⟩⟩) 0 ⟨13686⟩ 70268

def event70270 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34603⟩⟩) 1 ⟨34602⟩ 70265

def event70271 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34603⟩⟩) (.product (.predecessor 0 70269 .coefficient) (.predecessor 1 70270 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event70272 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34603⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨13686⟩⟩, ⟨.program ⟨257⟩, ⟨34602⟩⟩], []⟩) [⟨.result 70268 .coefficient, true, some 1⟩, ⟨.result 70265 .coefficient, true, some 1⟩])

def event70273 : Event := .survivorFold (1) 70272

def exact70274RawTerms : List Term := []

theorem exact70274RawTermsValid :
    exact70274RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70274 : Event := .resultExact (⟨.program ⟨257⟩, ⟨34603⟩⟩) exact70274RawTerms (.finite 1600) 70271 (.finite 1600) (some (70272))

def event70275 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34604⟩⟩) 0 ⟨34603⟩ 70274

def event70276 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34604⟩⟩) (.identity (.predecessor 0 70275 .coefficient))

def event70277 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨34604⟩⟩) (.finite 1600)

def event70278 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34804⟩⟩) 0 ⟨34604⟩ 70277

def event70279 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34804⟩⟩) (.authority (.programFamilyFact))

def exact70280RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨34804⟩⟩], []⟩, (1)⟩]

theorem exact70280RawTermsValid :
    exact70280RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70280 : Event := .resultExact (⟨.program ⟨257⟩, ⟨34804⟩⟩) exact70280RawTerms (.finite 40) 70279 .exactZero (none)

def event70281 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34805⟩⟩) 0 ⟨34804⟩ 70280

def event70282 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34805⟩⟩) (.identity (.predecessor 0 70281 .coefficient))

def event70283 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨34805⟩⟩) (.finite 40)

def event70284 : Event := .predecessor (⟨.program ⟨257⟩, ⟨35054⟩⟩) 0 ⟨34805⟩ 70283

def event70285 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35054⟩⟩) (.authority (.programFamilyFact))

def exact70286RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨35054⟩⟩], []⟩, (1)⟩]

theorem exact70286RawTermsValid :
    exact70286RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70286 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35054⟩⟩) exact70286RawTerms (.finite 62) 70285 .exactZero (none)

def event70287 : Event := .predecessor (⟨.program ⟨257⟩, ⟨28942⟩⟩) 0 ⟨10749⟩ 70142

def event70288 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨28942⟩⟩) (.authority (.programFamilyFact))

def exact70289RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨28942⟩⟩], []⟩, (1)⟩]

theorem exact70289RawTermsValid :
    exact70289RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70289 : Event := .resultExact (⟨.program ⟨257⟩, ⟨28942⟩⟩) exact70289RawTerms (.finite 36) 70288 .exactZero (none)

def event70290 : Event := .predecessor (⟨.program ⟨257⟩, ⟨13386⟩⟩) 0 ⟨10749⟩ 70142

def event70291 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨13386⟩⟩) (.authority (.programFamilyFact))

def exact70292RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨13386⟩⟩], []⟩, (1)⟩]

theorem exact70292RawTermsValid :
    exact70292RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70292 : Event := .resultExact (⟨.program ⟨257⟩, ⟨13386⟩⟩) exact70292RawTerms (.finite 36) 70291 .exactZero (none)

def event70293 : Event := .predecessor (⟨.program ⟨257⟩, ⟨28943⟩⟩) 0 ⟨13386⟩ 70292

def event70294 : Event := .predecessor (⟨.program ⟨257⟩, ⟨28943⟩⟩) 1 ⟨28942⟩ 70289

def event70295 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨28943⟩⟩) (.product (.predecessor 0 70293 .coefficient) (.predecessor 1 70294 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event70296 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨28943⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨13386⟩⟩, ⟨.program ⟨257⟩, ⟨28942⟩⟩], []⟩) [⟨.result 70292 .coefficient, true, some 1⟩, ⟨.result 70289 .coefficient, true, some 1⟩])

def event70297 : Event := .survivorFold (1) 70296

def exact70298RawTerms : List Term := []

theorem exact70298RawTermsValid :
    exact70298RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70298 : Event := .resultExact (⟨.program ⟨257⟩, ⟨28943⟩⟩) exact70298RawTerms (.finite 1296) 70295 (.finite 1296) (some (70296))

def event70299 : Event := .predecessor (⟨.program ⟨257⟩, ⟨28944⟩⟩) 0 ⟨28943⟩ 70298

def event70300 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨28944⟩⟩) (.identity (.predecessor 0 70299 .coefficient))

def event70301 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨28944⟩⟩) (.finite 1296)

def event70302 : Event := .predecessor (⟨.program ⟨257⟩, ⟨29144⟩⟩) 0 ⟨28944⟩ 70301

def event70303 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨29144⟩⟩) (.authority (.programFamilyFact))

def exact70304RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨29144⟩⟩], []⟩, (1)⟩]

theorem exact70304RawTermsValid :
    exact70304RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70304 : Event := .resultExact (⟨.program ⟨257⟩, ⟨29144⟩⟩) exact70304RawTerms (.finite 36) 70303 .exactZero (none)

def event70305 : Event := .predecessor (⟨.program ⟨257⟩, ⟨29145⟩⟩) 0 ⟨29144⟩ 70304

def event70306 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨29145⟩⟩) (.identity (.predecessor 0 70305 .coefficient))

def event70307 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨29145⟩⟩) (.finite 36)

def event70308 : Event := .predecessor (⟨.program ⟨257⟩, ⟨29390⟩⟩) 0 ⟨29145⟩ 70307

def event70309 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨29390⟩⟩) (.authority (.programFamilyFact))

def exact70310RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨29390⟩⟩], []⟩, (1)⟩]

theorem exact70310RawTermsValid :
    exact70310RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70310 : Event := .resultExact (⟨.program ⟨257⟩, ⟨29390⟩⟩) exact70310RawTerms (.finite 62) 70309 .exactZero (none)

def event70311 : Event := .predecessor (⟨.program ⟨257⟩, ⟨26262⟩⟩) 0 ⟨10749⟩ 70142

def event70312 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨26262⟩⟩) (.authority (.programFamilyFact))

def exact70313RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨26262⟩⟩], []⟩, (1)⟩]

theorem exact70313RawTermsValid :
    exact70313RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70313 : Event := .resultExact (⟨.program ⟨257⟩, ⟨26262⟩⟩) exact70313RawTerms (.finite 30) 70312 .exactZero (none)

def event70314 : Event := .predecessor (⟨.program ⟨257⟩, ⟨13086⟩⟩) 0 ⟨10749⟩ 70142

def event70315 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨13086⟩⟩) (.authority (.programFamilyFact))

def exact70316RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨13086⟩⟩], []⟩, (1)⟩]

theorem exact70316RawTermsValid :
    exact70316RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70316 : Event := .resultExact (⟨.program ⟨257⟩, ⟨13086⟩⟩) exact70316RawTerms (.finite 30) 70315 .exactZero (none)

def event70317 : Event := .predecessor (⟨.program ⟨257⟩, ⟨26263⟩⟩) 0 ⟨13086⟩ 70316

def event70318 : Event := .predecessor (⟨.program ⟨257⟩, ⟨26263⟩⟩) 1 ⟨26262⟩ 70313

def event70319 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨26263⟩⟩) (.product (.predecessor 0 70317 .coefficient) (.predecessor 1 70318 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event70320 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨26263⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨13086⟩⟩, ⟨.program ⟨257⟩, ⟨26262⟩⟩], []⟩) [⟨.result 70316 .coefficient, true, some 1⟩, ⟨.result 70313 .coefficient, true, some 1⟩])

def event70321 : Event := .survivorFold (1) 70320

def exact70322RawTerms : List Term := []

theorem exact70322RawTermsValid :
    exact70322RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70322 : Event := .resultExact (⟨.program ⟨257⟩, ⟨26263⟩⟩) exact70322RawTerms (.finite 900) 70319 (.finite 900) (some (70320))

def event70323 : Event := .predecessor (⟨.program ⟨257⟩, ⟨26264⟩⟩) 0 ⟨26263⟩ 70322

def event70324 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨26264⟩⟩) (.identity (.predecessor 0 70323 .coefficient))

def event70325 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨26264⟩⟩) (.finite 900)

def event70326 : Event := .predecessor (⟨.program ⟨257⟩, ⟨26464⟩⟩) 0 ⟨26264⟩ 70325

def event70327 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨26464⟩⟩) (.authority (.programFamilyFact))

def exact70328RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨26464⟩⟩], []⟩, (1)⟩]

theorem exact70328RawTermsValid :
    exact70328RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70328 : Event := .resultExact (⟨.program ⟨257⟩, ⟨26464⟩⟩) exact70328RawTerms (.finite 30) 70327 .exactZero (none)

def event70329 : Event := .predecessor (⟨.program ⟨257⟩, ⟨26465⟩⟩) 0 ⟨26464⟩ 70328

def event70330 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨26465⟩⟩) (.identity (.predecessor 0 70329 .coefficient))

def event70331 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨26465⟩⟩) (.finite 30)

def event70332 : Event := .predecessor (⟨.program ⟨257⟩, ⟨26710⟩⟩) 0 ⟨26465⟩ 70331

def event70333 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨26710⟩⟩) (.authority (.programFamilyFact))

def exact70334RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨26710⟩⟩], []⟩, (1)⟩]

theorem exact70334RawTermsValid :
    exact70334RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70334 : Event := .resultExact (⟨.program ⟨257⟩, ⟨26710⟩⟩) exact70334RawTerms (.finite 62) 70333 .exactZero (none)

def event70335 : Event := .predecessor (⟨.program ⟨257⟩, ⟨25814⟩⟩) 0 ⟨10749⟩ 70142

def event70336 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨25814⟩⟩) (.authority (.programFamilyFact))

def exact70337RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨25814⟩⟩], []⟩, (1)⟩]

theorem exact70337RawTermsValid :
    exact70337RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70337 : Event := .resultExact (⟨.program ⟨257⟩, ⟨25814⟩⟩) exact70337RawTerms (.finite 28) 70336 .exactZero (none)

def event70338 : Event := .predecessor (⟨.program ⟨257⟩, ⟨65634⟩⟩) 0 ⟨10749⟩ 70142

def event70339 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨65634⟩⟩) (.authority (.programFamilyFact))

def exact70340RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨65634⟩⟩], []⟩, (1)⟩]

theorem exact70340RawTermsValid :
    exact70340RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70340 : Event := .resultExact (⟨.program ⟨257⟩, ⟨65634⟩⟩) exact70340RawTerms (.finite 28) 70339 .exactZero (none)

def event70341 : Event := .predecessor (⟨.program ⟨257⟩, ⟨65635⟩⟩) 0 ⟨65634⟩ 70340

def event70342 : Event := .predecessor (⟨.program ⟨257⟩, ⟨65635⟩⟩) 1 ⟨25814⟩ 70337

def event70343 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨65635⟩⟩) (.product (.predecessor 0 70341 .coefficient) (.predecessor 1 70342 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event70344 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨65635⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨25814⟩⟩, ⟨.program ⟨257⟩, ⟨65634⟩⟩], []⟩) [⟨.result 70340 .coefficient, true, some 1⟩, ⟨.result 70337 .coefficient, true, some 1⟩])

def event70345 : Event := .survivorFold (1) 70344

def exact70346RawTerms : List Term := []

theorem exact70346RawTermsValid :
    exact70346RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70346 : Event := .resultExact (⟨.program ⟨257⟩, ⟨65635⟩⟩) exact70346RawTerms (.finite 784) 70343 (.finite 784) (some (70344))

def event70347 : Event := .predecessor (⟨.program ⟨257⟩, ⟨65636⟩⟩) 0 ⟨65635⟩ 70346

def event70348 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨65636⟩⟩) (.identity (.predecessor 0 70347 .coefficient))

def event70349 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨65636⟩⟩) (.finite 784)

def event70350 : Event := .predecessor (⟨.program ⟨257⟩, ⟨65844⟩⟩) 0 ⟨65636⟩ 70349

def event70351 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨65844⟩⟩) (.authority (.programFamilyFact))

def exact70352RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨65844⟩⟩], []⟩, (1)⟩]

theorem exact70352RawTermsValid :
    exact70352RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70352 : Event := .resultExact (⟨.program ⟨257⟩, ⟨65844⟩⟩) exact70352RawTerms (.finite 28) 70351 .exactZero (none)

def event70353 : Event := .predecessor (⟨.program ⟨257⟩, ⟨65845⟩⟩) 0 ⟨65844⟩ 70352

def event70354 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨65845⟩⟩) (.identity (.predecessor 0 70353 .coefficient))

def event70355 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨65845⟩⟩) (.finite 28)

def event70356 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67091⟩⟩) 0 ⟨65845⟩ 70355

def event70357 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67091⟩⟩) (.authority (.programFamilyFact))

def exact70358RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨67091⟩⟩], []⟩, (1)⟩]

theorem exact70358RawTermsValid :
    exact70358RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70358 : Event := .resultExact (⟨.program ⟨257⟩, ⟨67091⟩⟩) exact70358RawTerms (.finite 62) 70357 .exactZero (none)

def event70359 : Event := .predecessor (⟨.program ⟨257⟩, ⟨25574⟩⟩) 0 ⟨10749⟩ 70142

def event70360 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨25574⟩⟩) (.authority (.programFamilyFact))

def exact70361RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨25574⟩⟩], []⟩, (1)⟩]

theorem exact70361RawTermsValid :
    exact70361RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70361 : Event := .resultExact (⟨.program ⟨257⟩, ⟨25574⟩⟩) exact70361RawTerms (.finite 22) 70360 .exactZero (none)

def event70362 : Event := .predecessor (⟨.program ⟨257⟩, ⟨62654⟩⟩) 0 ⟨10749⟩ 70142

def event70363 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨62654⟩⟩) (.authority (.programFamilyFact))

def exact70364RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨62654⟩⟩], []⟩, (1)⟩]

theorem exact70364RawTermsValid :
    exact70364RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70364 : Event := .resultExact (⟨.program ⟨257⟩, ⟨62654⟩⟩) exact70364RawTerms (.finite 22) 70363 .exactZero (none)

def event70365 : Event := .predecessor (⟨.program ⟨257⟩, ⟨62655⟩⟩) 0 ⟨62654⟩ 70364

def event70366 : Event := .predecessor (⟨.program ⟨257⟩, ⟨62655⟩⟩) 1 ⟨25574⟩ 70361

def event70367 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨62655⟩⟩) (.product (.predecessor 0 70365 .coefficient) (.predecessor 1 70366 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event70368 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨62655⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨25574⟩⟩, ⟨.program ⟨257⟩, ⟨62654⟩⟩], []⟩) [⟨.result 70364 .coefficient, true, some 1⟩, ⟨.result 70361 .coefficient, true, some 1⟩])

def event70369 : Event := .survivorFold (1) 70368

def exact70370RawTerms : List Term := []

theorem exact70370RawTermsValid :
    exact70370RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70370 : Event := .resultExact (⟨.program ⟨257⟩, ⟨62655⟩⟩) exact70370RawTerms (.finite 484) 70367 (.finite 484) (some (70368))

def event70371 : Event := .predecessor (⟨.program ⟨257⟩, ⟨62656⟩⟩) 0 ⟨62655⟩ 70370

def event70372 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨62656⟩⟩) (.identity (.predecessor 0 70371 .coefficient))

def event70373 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨62656⟩⟩) (.finite 484)

def event70374 : Event := .predecessor (⟨.program ⟨257⟩, ⟨62864⟩⟩) 0 ⟨62656⟩ 70373

def event70375 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨62864⟩⟩) (.authority (.programFamilyFact))

def exact70376RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨62864⟩⟩], []⟩, (1)⟩]

theorem exact70376RawTermsValid :
    exact70376RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70376 : Event := .resultExact (⟨.program ⟨257⟩, ⟨62864⟩⟩) exact70376RawTerms (.finite 22) 70375 .exactZero (none)

def event70377 : Event := .predecessor (⟨.program ⟨257⟩, ⟨62865⟩⟩) 0 ⟨62864⟩ 70376

def event70378 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨62865⟩⟩) (.identity (.predecessor 0 70377 .coefficient))

def event70379 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨62865⟩⟩) (.finite 22)

def event70380 : Event := .predecessor (⟨.program ⟨257⟩, ⟨63214⟩⟩) 0 ⟨62865⟩ 70379

def event70381 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨63214⟩⟩) (.authority (.programFamilyFact))

def exact70382RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨63214⟩⟩], []⟩, (1)⟩]

theorem exact70382RawTermsValid :
    exact70382RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70382 : Event := .resultExact (⟨.program ⟨257⟩, ⟨63214⟩⟩) exact70382RawTerms (.finite 61) 70381 .exactZero (none)

def event70383 : Event := .predecessor (⟨.program ⟨257⟩, ⟨25334⟩⟩) 0 ⟨10749⟩ 70142

def event70384 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨25334⟩⟩) (.authority (.programFamilyFact))

def exact70385RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨25334⟩⟩], []⟩, (1)⟩]

theorem exact70385RawTermsValid :
    exact70385RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70385 : Event := .resultExact (⟨.program ⟨257⟩, ⟨25334⟩⟩) exact70385RawTerms (.finite 18) 70384 .exactZero (none)

def event70386 : Event := .predecessor (⟨.program ⟨257⟩, ⟨59674⟩⟩) 0 ⟨10749⟩ 70142

def event70387 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨59674⟩⟩) (.authority (.programFamilyFact))

def exact70388RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨59674⟩⟩], []⟩, (1)⟩]

theorem exact70388RawTermsValid :
    exact70388RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70388 : Event := .resultExact (⟨.program ⟨257⟩, ⟨59674⟩⟩) exact70388RawTerms (.finite 18) 70387 .exactZero (none)

def event70389 : Event := .predecessor (⟨.program ⟨257⟩, ⟨59675⟩⟩) 0 ⟨59674⟩ 70388

def event70390 : Event := .predecessor (⟨.program ⟨257⟩, ⟨59675⟩⟩) 1 ⟨25334⟩ 70385

def event70391 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨59675⟩⟩) (.product (.predecessor 0 70389 .coefficient) (.predecessor 1 70390 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event70392 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨59675⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨25334⟩⟩, ⟨.program ⟨257⟩, ⟨59674⟩⟩], []⟩) [⟨.result 70388 .coefficient, true, some 1⟩, ⟨.result 70385 .coefficient, true, some 1⟩])

def event70393 : Event := .survivorFold (1) 70392

def exact70394RawTerms : List Term := []

theorem exact70394RawTermsValid :
    exact70394RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70394 : Event := .resultExact (⟨.program ⟨257⟩, ⟨59675⟩⟩) exact70394RawTerms (.finite 324) 70391 (.finite 324) (some (70392))

def event70395 : Event := .predecessor (⟨.program ⟨257⟩, ⟨59676⟩⟩) 0 ⟨59675⟩ 70394

def event70396 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨59676⟩⟩) (.identity (.predecessor 0 70395 .coefficient))

def event70397 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨59676⟩⟩) (.finite 324)

def event70398 : Event := .predecessor (⟨.program ⟨257⟩, ⟨59884⟩⟩) 0 ⟨59676⟩ 70397

def event70399 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨59884⟩⟩) (.authority (.programFamilyFact))

def eventLeaf4384 : Array AnnotatedEvent := #[
  { event := event70144
    frameStart := 70122 },
  { event := event70145
    frameStart := 70122 },
  { event := event70146
    frameStart := 70122 },
  { event := event70147
    frameStart := 70122 },
  { event := event70148
    frameStart := 70122 },
  { event := event70149
    frameStart := 70122 },
  { event := event70150
    frameStart := 70122 },
  { event := event70151
    frameStart := 70122 },
  { event := event70152
    frameStart := 70122 },
  { event := event70153
    frameStart := 70122 },
  { event := event70154
    frameStart := 70122 },
  { event := event70155
    frameStart := 70122 },
  { event := event70156
    frameStart := 70122 },
  { event := event70157
    frameStart := 70122 },
  { event := event70158
    frameStart := 70122 },
  { event := event70159
    frameStart := 70122 }
]

def eventLeaf4385 : Array AnnotatedEvent := #[
  { event := event70160
    frameStart := 70122 },
  { event := event70161
    frameStart := 70122 },
  { event := event70162
    frameStart := 70122 },
  { event := event70163
    frameStart := 70122 },
  { event := event70164
    frameStart := 70122 },
  { event := event70165
    frameStart := 70122 },
  { event := event70166
    frameStart := 70122 },
  { event := event70167
    frameStart := 70122 },
  { event := event70168
    frameStart := 70122 },
  { event := event70169
    frameStart := 70122 },
  { event := event70170
    frameStart := 70122 },
  { event := event70171
    frameStart := 70122 },
  { event := event70172
    frameStart := 70122 },
  { event := event70173
    frameStart := 70122 },
  { event := event70174
    frameStart := 70122 },
  { event := event70175
    frameStart := 70122 }
]

def eventLeaf4386 : Array AnnotatedEvent := #[
  { event := event70176
    frameStart := 70122 },
  { event := event70177
    frameStart := 70122 },
  { event := event70178
    frameStart := 70122 },
  { event := event70179
    frameStart := 70122 },
  { event := event70180
    frameStart := 70122 },
  { event := event70181
    frameStart := 70122 },
  { event := event70182
    frameStart := 70122 },
  { event := event70183
    frameStart := 70122 },
  { event := event70184
    frameStart := 70122 },
  { event := event70185
    frameStart := 70122 },
  { event := event70186
    frameStart := 70122 },
  { event := event70187
    frameStart := 70122 },
  { event := event70188
    frameStart := 70122 },
  { event := event70189
    frameStart := 70122 },
  { event := event70190
    frameStart := 70122 },
  { event := event70191
    frameStart := 70122 }
]

def eventLeaf4387 : Array AnnotatedEvent := #[
  { event := event70192
    frameStart := 70122 },
  { event := event70193
    frameStart := 70122 },
  { event := event70194
    frameStart := 70122 },
  { event := event70195
    frameStart := 70122 },
  { event := event70196
    frameStart := 70122 },
  { event := event70197
    frameStart := 70122 },
  { event := event70198
    frameStart := 70122 },
  { event := event70199
    frameStart := 70122 },
  { event := event70200
    frameStart := 70122 },
  { event := event70201
    frameStart := 70122 },
  { event := event70202
    frameStart := 70122 },
  { event := event70203
    frameStart := 70122 },
  { event := event70204
    frameStart := 70122 },
  { event := event70205
    frameStart := 70122 },
  { event := event70206
    frameStart := 70122 },
  { event := event70207
    frameStart := 70122 }
]

def eventLeaf4388 : Array AnnotatedEvent := #[
  { event := event70208
    frameStart := 70122 },
  { event := event70209
    frameStart := 70122 },
  { event := event70210
    frameStart := 70122 },
  { event := event70211
    frameStart := 70122 },
  { event := event70212
    frameStart := 70122 },
  { event := event70213
    frameStart := 70122 },
  { event := event70214
    frameStart := 70122 },
  { event := event70215
    frameStart := 70122 },
  { event := event70216
    frameStart := 70122 },
  { event := event70217
    frameStart := 70122 },
  { event := event70218
    frameStart := 70122 },
  { event := event70219
    frameStart := 70122 },
  { event := event70220
    frameStart := 70122 },
  { event := event70221
    frameStart := 70122 },
  { event := event70222
    frameStart := 70122 },
  { event := event70223
    frameStart := 70122 }
]

def eventLeaf4389 : Array AnnotatedEvent := #[
  { event := event70224
    frameStart := 70122 },
  { event := event70225
    frameStart := 70122 },
  { event := event70226
    frameStart := 70122 },
  { event := event70227
    frameStart := 70122 },
  { event := event70228
    frameStart := 70122 },
  { event := event70229
    frameStart := 70122 },
  { event := event70230
    frameStart := 70122 },
  { event := event70231
    frameStart := 70122 },
  { event := event70232
    frameStart := 70122 },
  { event := event70233
    frameStart := 70122 },
  { event := event70234
    frameStart := 70122 },
  { event := event70235
    frameStart := 70122 },
  { event := event70236
    frameStart := 70122 },
  { event := event70237
    frameStart := 70122 },
  { event := event70238
    frameStart := 70122 },
  { event := event70239
    frameStart := 70122 }
]

def eventLeaf4390 : Array AnnotatedEvent := #[
  { event := event70240
    frameStart := 70122 },
  { event := event70241
    frameStart := 70122 },
  { event := event70242
    frameStart := 70122 },
  { event := event70243
    frameStart := 70122 },
  { event := event70244
    frameStart := 70122 },
  { event := event70245
    frameStart := 70122 },
  { event := event70246
    frameStart := 70122 },
  { event := event70247
    frameStart := 70122 },
  { event := event70248
    frameStart := 70122 },
  { event := event70249
    frameStart := 70122 },
  { event := event70250
    frameStart := 70122 },
  { event := event70251
    frameStart := 70122 },
  { event := event70252
    frameStart := 70122 },
  { event := event70253
    frameStart := 70122 },
  { event := event70254
    frameStart := 70122 },
  { event := event70255
    frameStart := 70122 }
]

def eventLeaf4391 : Array AnnotatedEvent := #[
  { event := event70256
    frameStart := 70122 },
  { event := event70257
    frameStart := 70122 },
  { event := event70258
    frameStart := 70122 },
  { event := event70259
    frameStart := 70122 },
  { event := event70260
    frameStart := 70122 },
  { event := event70261
    frameStart := 70122 },
  { event := event70262
    frameStart := 70122 },
  { event := event70263
    frameStart := 70122 },
  { event := event70264
    frameStart := 70122 },
  { event := event70265
    frameStart := 70122 },
  { event := event70266
    frameStart := 70122 },
  { event := event70267
    frameStart := 70122 },
  { event := event70268
    frameStart := 70122 },
  { event := event70269
    frameStart := 70122 },
  { event := event70270
    frameStart := 70122 },
  { event := event70271
    frameStart := 70122 }
]

def eventLeaf4392 : Array AnnotatedEvent := #[
  { event := event70272
    frameStart := 70122 },
  { event := event70273
    frameStart := 70122 },
  { event := event70274
    frameStart := 70122 },
  { event := event70275
    frameStart := 70122 },
  { event := event70276
    frameStart := 70122 },
  { event := event70277
    frameStart := 70122 },
  { event := event70278
    frameStart := 70122 },
  { event := event70279
    frameStart := 70122 },
  { event := event70280
    frameStart := 70122 },
  { event := event70281
    frameStart := 70122 },
  { event := event70282
    frameStart := 70122 },
  { event := event70283
    frameStart := 70122 },
  { event := event70284
    frameStart := 70122 },
  { event := event70285
    frameStart := 70122 },
  { event := event70286
    frameStart := 70122 },
  { event := event70287
    frameStart := 70122 }
]

def eventLeaf4393 : Array AnnotatedEvent := #[
  { event := event70288
    frameStart := 70122 },
  { event := event70289
    frameStart := 70122 },
  { event := event70290
    frameStart := 70122 },
  { event := event70291
    frameStart := 70122 },
  { event := event70292
    frameStart := 70122 },
  { event := event70293
    frameStart := 70122 },
  { event := event70294
    frameStart := 70122 },
  { event := event70295
    frameStart := 70122 },
  { event := event70296
    frameStart := 70122 },
  { event := event70297
    frameStart := 70122 },
  { event := event70298
    frameStart := 70122 },
  { event := event70299
    frameStart := 70122 },
  { event := event70300
    frameStart := 70122 },
  { event := event70301
    frameStart := 70122 },
  { event := event70302
    frameStart := 70122 },
  { event := event70303
    frameStart := 70122 }
]

def eventLeaf4394 : Array AnnotatedEvent := #[
  { event := event70304
    frameStart := 70122 },
  { event := event70305
    frameStart := 70122 },
  { event := event70306
    frameStart := 70122 },
  { event := event70307
    frameStart := 70122 },
  { event := event70308
    frameStart := 70122 },
  { event := event70309
    frameStart := 70122 },
  { event := event70310
    frameStart := 70122 },
  { event := event70311
    frameStart := 70122 },
  { event := event70312
    frameStart := 70122 },
  { event := event70313
    frameStart := 70122 },
  { event := event70314
    frameStart := 70122 },
  { event := event70315
    frameStart := 70122 },
  { event := event70316
    frameStart := 70122 },
  { event := event70317
    frameStart := 70122 },
  { event := event70318
    frameStart := 70122 },
  { event := event70319
    frameStart := 70122 }
]

def eventLeaf4395 : Array AnnotatedEvent := #[
  { event := event70320
    frameStart := 70122 },
  { event := event70321
    frameStart := 70122 },
  { event := event70322
    frameStart := 70122 },
  { event := event70323
    frameStart := 70122 },
  { event := event70324
    frameStart := 70122 },
  { event := event70325
    frameStart := 70122 },
  { event := event70326
    frameStart := 70122 },
  { event := event70327
    frameStart := 70122 },
  { event := event70328
    frameStart := 70122 },
  { event := event70329
    frameStart := 70122 },
  { event := event70330
    frameStart := 70122 },
  { event := event70331
    frameStart := 70122 },
  { event := event70332
    frameStart := 70122 },
  { event := event70333
    frameStart := 70122 },
  { event := event70334
    frameStart := 70122 },
  { event := event70335
    frameStart := 70122 }
]

def eventLeaf4396 : Array AnnotatedEvent := #[
  { event := event70336
    frameStart := 70122 },
  { event := event70337
    frameStart := 70122 },
  { event := event70338
    frameStart := 70122 },
  { event := event70339
    frameStart := 70122 },
  { event := event70340
    frameStart := 70122 },
  { event := event70341
    frameStart := 70122 },
  { event := event70342
    frameStart := 70122 },
  { event := event70343
    frameStart := 70122 },
  { event := event70344
    frameStart := 70122 },
  { event := event70345
    frameStart := 70122 },
  { event := event70346
    frameStart := 70122 },
  { event := event70347
    frameStart := 70122 },
  { event := event70348
    frameStart := 70122 },
  { event := event70349
    frameStart := 70122 },
  { event := event70350
    frameStart := 70122 },
  { event := event70351
    frameStart := 70122 }
]

def eventLeaf4397 : Array AnnotatedEvent := #[
  { event := event70352
    frameStart := 70122 },
  { event := event70353
    frameStart := 70122 },
  { event := event70354
    frameStart := 70122 },
  { event := event70355
    frameStart := 70122 },
  { event := event70356
    frameStart := 70122 },
  { event := event70357
    frameStart := 70122 },
  { event := event70358
    frameStart := 70122 },
  { event := event70359
    frameStart := 70122 },
  { event := event70360
    frameStart := 70122 },
  { event := event70361
    frameStart := 70122 },
  { event := event70362
    frameStart := 70122 },
  { event := event70363
    frameStart := 70122 },
  { event := event70364
    frameStart := 70122 },
  { event := event70365
    frameStart := 70122 },
  { event := event70366
    frameStart := 70122 },
  { event := event70367
    frameStart := 70122 }
]

def eventLeaf4398 : Array AnnotatedEvent := #[
  { event := event70368
    frameStart := 70122 },
  { event := event70369
    frameStart := 70122 },
  { event := event70370
    frameStart := 70122 },
  { event := event70371
    frameStart := 70122 },
  { event := event70372
    frameStart := 70122 },
  { event := event70373
    frameStart := 70122 },
  { event := event70374
    frameStart := 70122 },
  { event := event70375
    frameStart := 70122 },
  { event := event70376
    frameStart := 70122 },
  { event := event70377
    frameStart := 70122 },
  { event := event70378
    frameStart := 70122 },
  { event := event70379
    frameStart := 70122 },
  { event := event70380
    frameStart := 70122 },
  { event := event70381
    frameStart := 70122 },
  { event := event70382
    frameStart := 70122 },
  { event := event70383
    frameStart := 70122 }
]

def eventLeaf4399 : Array AnnotatedEvent := #[
  { event := event70384
    frameStart := 70122 },
  { event := event70385
    frameStart := 70122 },
  { event := event70386
    frameStart := 70122 },
  { event := event70387
    frameStart := 70122 },
  { event := event70388
    frameStart := 70122 },
  { event := event70389
    frameStart := 70122 },
  { event := event70390
    frameStart := 70122 },
  { event := event70391
    frameStart := 70122 },
  { event := event70392
    frameStart := 70122 },
  { event := event70393
    frameStart := 70122 },
  { event := event70394
    frameStart := 70122 },
  { event := event70395
    frameStart := 70122 },
  { event := event70396
    frameStart := 70122 },
  { event := event70397
    frameStart := 70122 },
  { event := event70398
    frameStart := 70122 },
  { event := event70399
    frameStart := 70122 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events274
