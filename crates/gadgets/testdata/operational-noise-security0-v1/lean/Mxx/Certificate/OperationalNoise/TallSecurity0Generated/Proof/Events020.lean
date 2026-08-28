import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events020

open Mxx.Certificate.OperationalNoise
open CertificateABI

def exact5120RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨6494⟩⟩, ⟨.program ⟨214⟩, ⟨17596⟩⟩], []⟩, (1)⟩]

theorem exact5120RawTermsValid :
    exact5120RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event5120 : Event := .resultExact (⟨.program ⟨214⟩, ⟨17597⟩⟩) exact5120RawTerms (.finite 227009770373045750290200) 5118 .exactZero (none)

def event5121 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17652⟩⟩) 0 ⟨16169⟩ 4767

def event5122 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨17652⟩⟩) (.authority (.programFamilyFact))

def exact5123RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨17652⟩⟩], []⟩, (1)⟩]

theorem exact5123RawTermsValid :
    exact5123RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event5123 : Event := .resultExact (⟨.program ⟨214⟩, ⟨17652⟩⟩) exact5123RawTerms (.finite 28) 5122 .exactZero (none)

def event5124 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17653⟩⟩) 0 ⟨17652⟩ 5123

def event5125 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17653⟩⟩) 1 ⟨6502⟩ 623

def event5126 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨17653⟩⟩) (.product (.predecessor 0 5124 .coefficient) (.predecessor 1 5125 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event5127 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨17653⟩⟩, .operator (⟨5123, 0⟩, ⟨623, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨6502⟩⟩, ⟨.program ⟨214⟩, ⟨17652⟩⟩], []⟩, (1)⟩)

def exact5128RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨6502⟩⟩, ⟨.program ⟨214⟩, ⟨17652⟩⟩], []⟩, (1)⟩]

theorem exact5128RawTermsValid :
    exact5128RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event5128 : Event := .resultExact (⟨.program ⟨214⟩, ⟨17653⟩⟩) exact5128RawTerms (.finite 226487908831958288795280) 5126 .exactZero (none)

def event5129 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18016⟩⟩) 0 ⟨16050⟩ 4790

def event5130 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨18016⟩⟩) (.authority (.programFamilyFact))

def exact5131RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨18016⟩⟩], []⟩, (1)⟩]

theorem exact5131RawTermsValid :
    exact5131RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event5131 : Event := .resultExact (⟨.program ⟨214⟩, ⟨18016⟩⟩) exact5131RawTerms (.finite 22) 5130 .exactZero (none)

def event5132 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18017⟩⟩) 0 ⟨18016⟩ 5131

def event5133 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18017⟩⟩) 1 ⟨6383⟩ 633

def event5134 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨18017⟩⟩) (.product (.predecessor 0 5132 .coefficient) (.predecessor 1 5133 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event5135 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18017⟩⟩, .operator (⟨5131, 0⟩, ⟨633, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨6383⟩⟩, ⟨.program ⟨214⟩, ⟨18016⟩⟩], []⟩, (1)⟩)

def exact5136RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨6383⟩⟩, ⟨.program ⟨214⟩, ⟨18016⟩⟩], []⟩, (1)⟩]

theorem exact5136RawTermsValid :
    exact5136RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event5136 : Event := .resultExact (⟨.program ⟨214⟩, ⟨18017⟩⟩) exact5136RawTerms (.finite 224377773035387248837560) 5134 .exactZero (none)

def event5137 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17155⟩⟩) 0 ⟨15931⟩ 4813

def event5138 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨17155⟩⟩) (.authority (.programFamilyFact))

def exact5139RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨17155⟩⟩], []⟩, (1)⟩]

theorem exact5139RawTermsValid :
    exact5139RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event5139 : Event := .resultExact (⟨.program ⟨214⟩, ⟨17155⟩⟩) exact5139RawTerms (.finite 18) 5138 .exactZero (none)

def event5140 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17156⟩⟩) 0 ⟨17155⟩ 5139

def event5141 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17156⟩⟩) 1 ⟨6387⟩ 643

def event5142 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨17156⟩⟩) (.product (.predecessor 0 5140 .coefficient) (.predecessor 1 5141 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event5143 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨17156⟩⟩, .operator (⟨5139, 0⟩, ⟨643, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨6387⟩⟩, ⟨.program ⟨214⟩, ⟨17155⟩⟩], []⟩, (1)⟩)

def exact5144RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨6387⟩⟩, ⟨.program ⟨214⟩, ⟨17155⟩⟩], []⟩, (1)⟩]

theorem exact5144RawTermsValid :
    exact5144RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event5144 : Event := .resultExact (⟨.program ⟨214⟩, ⟨17156⟩⟩) exact5144RawTerms (.finite 222230617312560576599880) 5142 .exactZero (none)

def event5145 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17211⟩⟩) 0 ⟨15812⟩ 4836

def event5146 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨17211⟩⟩) (.authority (.programFamilyFact))

def exact5147RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨17211⟩⟩], []⟩, (1)⟩]

theorem exact5147RawTermsValid :
    exact5147RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event5147 : Event := .resultExact (⟨.program ⟨214⟩, ⟨17211⟩⟩) exact5147RawTerms (.finite 16) 5146 .exactZero (none)

def event5148 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17212⟩⟩) 0 ⟨17211⟩ 5147

def event5149 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17212⟩⟩) 1 ⟨6391⟩ 653

def event5150 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨17212⟩⟩) (.product (.predecessor 0 5148 .coefficient) (.predecessor 1 5149 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event5151 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨17212⟩⟩, .operator (⟨5147, 0⟩, ⟨653, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨6391⟩⟩, ⟨.program ⟨214⟩, ⟨17211⟩⟩], []⟩, (1)⟩)

def exact5152RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨6391⟩⟩, ⟨.program ⟨214⟩, ⟨17211⟩⟩], []⟩, (1)⟩]

theorem exact5152RawTermsValid :
    exact5152RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event5152 : Event := .resultExact (⟨.program ⟨214⟩, ⟨17212⟩⟩) exact5152RawTerms (.finite 220778129617707239497920) 5150 .exactZero (none)

def event5153 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17428⟩⟩) 0 ⟨15693⟩ 4859

def event5154 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨17428⟩⟩) (.authority (.programFamilyFact))

def exact5155RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨17428⟩⟩], []⟩, (1)⟩]

theorem exact5155RawTermsValid :
    exact5155RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event5155 : Event := .resultExact (⟨.program ⟨214⟩, ⟨17428⟩⟩) exact5155RawTerms (.finite 12) 5154 .exactZero (none)

def event5156 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17429⟩⟩) 0 ⟨17428⟩ 5155

def event5157 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17429⟩⟩) 1 ⟨6398⟩ 663

def event5158 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨17429⟩⟩) (.product (.predecessor 0 5156 .coefficient) (.predecessor 1 5157 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event5159 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨17429⟩⟩, .operator (⟨5155, 0⟩, ⟨663, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨6398⟩⟩, ⟨.program ⟨214⟩, ⟨17428⟩⟩], []⟩, (1)⟩)

def exact5160RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨6398⟩⟩, ⟨.program ⟨214⟩, ⟨17428⟩⟩], []⟩, (1)⟩]

theorem exact5160RawTermsValid :
    exact5160RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event5160 : Event := .resultExact (⟨.program ⟨214⟩, ⟨17429⟩⟩) exact5160RawTerms (.finite 216532396355828254122960) 5158 .exactZero (none)

def event5161 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17792⟩⟩) 0 ⟨15574⟩ 4882

def event5162 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨17792⟩⟩) (.authority (.programFamilyFact))

def exact5163RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨17792⟩⟩], []⟩, (1)⟩]

theorem exact5163RawTermsValid :
    exact5163RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event5163 : Event := .resultExact (⟨.program ⟨214⟩, ⟨17792⟩⟩) exact5163RawTerms (.finite 10) 5162 .exactZero (none)

def event5164 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17793⟩⟩) 0 ⟨17792⟩ 5163

def event5165 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17793⟩⟩) 1 ⟨6407⟩ 673

def event5166 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨17793⟩⟩) (.product (.predecessor 0 5164 .coefficient) (.predecessor 1 5165 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event5167 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨17793⟩⟩, .operator (⟨5163, 0⟩, ⟨673, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨6407⟩⟩, ⟨.program ⟨214⟩, ⟨17792⟩⟩], []⟩, (1)⟩)

def exact5168RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨6407⟩⟩, ⟨.program ⟨214⟩, ⟨17792⟩⟩], []⟩, (1)⟩]

theorem exact5168RawTermsValid :
    exact5168RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event5168 : Event := .resultExact (⟨.program ⟨214⟩, ⟨17793⟩⟩) exact5168RawTerms (.finite 213251602471649038151400) 5166 .exactZero (none)

def event5169 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15503⟩⟩) 0 ⟨15413⟩ 4905

def event5170 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15503⟩⟩) (.authority (.programFamilyFact))

def exact5171RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15503⟩⟩], []⟩, (1)⟩]

theorem exact5171RawTermsValid :
    exact5171RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event5171 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15503⟩⟩) exact5171RawTerms (.finite 6) 5170 .exactZero (none)

def event5172 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15504⟩⟩) 0 ⟨15503⟩ 5171

def event5173 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15504⟩⟩) 1 ⟨6427⟩ 683

def event5174 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15504⟩⟩) (.product (.predecessor 0 5172 .coefficient) (.predecessor 1 5173 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event5175 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨15504⟩⟩, .operator (⟨5171, 0⟩, ⟨683, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨6427⟩⟩, ⟨.program ⟨214⟩, ⟨15503⟩⟩], []⟩, (1)⟩)

def exact5176RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨6427⟩⟩, ⟨.program ⟨214⟩, ⟨15503⟩⟩], []⟩, (1)⟩]

theorem exact5176RawTermsValid :
    exact5176RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event5176 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15504⟩⟩) exact5176RawTerms (.finite 201065796616126235971320) 5174 .exactZero (none)

def event5177 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15195⟩⟩) 0 ⟨15105⟩ 4928

def event5178 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15195⟩⟩) (.authority (.programFamilyFact))

def exact5179RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15195⟩⟩], []⟩, (1)⟩]

theorem exact5179RawTermsValid :
    exact5179RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event5179 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15195⟩⟩) exact5179RawTerms (.finite 4) 5178 .exactZero (none)

def event5180 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15196⟩⟩) 0 ⟨15195⟩ 5179

def event5181 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15196⟩⟩) 1 ⟨6452⟩ 693

def event5182 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15196⟩⟩) (.product (.predecessor 0 5180 .coefficient) (.predecessor 1 5181 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event5183 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨15196⟩⟩, .operator (⟨5179, 0⟩, ⟨693, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨6452⟩⟩, ⟨.program ⟨214⟩, ⟨15195⟩⟩], []⟩, (1)⟩)

def exact5184RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨6452⟩⟩, ⟨.program ⟨214⟩, ⟨15195⟩⟩], []⟩, (1)⟩]

theorem exact5184RawTermsValid :
    exact5184RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event5184 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15196⟩⟩) exact5184RawTerms (.finite 187661410175051153573232) 5182 .exactZero (none)

def event5185 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15034⟩⟩) 0 ⟨14944⟩ 4951

def event5186 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15034⟩⟩) (.authority (.programFamilyFact))

def exact5187RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15034⟩⟩], []⟩, (1)⟩]

theorem exact5187RawTermsValid :
    exact5187RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event5187 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15034⟩⟩) exact5187RawTerms (.finite 3) 5186 .exactZero (none)

def event5188 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15035⟩⟩) 0 ⟨15034⟩ 5187

def event5189 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15035⟩⟩) 1 ⟨6475⟩ 703

def event5190 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15035⟩⟩) (.product (.predecessor 0 5188 .coefficient) (.predecessor 1 5189 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event5191 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨15035⟩⟩, .operator (⟨5187, 0⟩, ⟨703, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨6475⟩⟩, ⟨.program ⟨214⟩, ⟨15034⟩⟩], []⟩, (1)⟩)

def exact5192RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨6475⟩⟩, ⟨.program ⟨214⟩, ⟨15034⟩⟩], []⟩, (1)⟩]

theorem exact5192RawTermsValid :
    exact5192RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event5192 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15035⟩⟩) exact5192RawTerms (.finite 175932572039110456474905) 5190 .exactZero (none)

def event5193 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14873⟩⟩) 0 ⟨14783⟩ 4974

def event5194 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14873⟩⟩) (.authority (.programFamilyFact))

def exact5195RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨14873⟩⟩], []⟩, (1)⟩]

theorem exact5195RawTermsValid :
    exact5195RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event5195 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14873⟩⟩) exact5195RawTerms (.finite 2) 5194 .exactZero (none)

def event5196 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14874⟩⟩) 0 ⟨14873⟩ 5195

def event5197 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14874⟩⟩) 1 ⟨6495⟩ 713

def event5198 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14874⟩⟩) (.product (.predecessor 0 5196 .coefficient) (.predecessor 1 5197 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event5199 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨14874⟩⟩, .operator (⟨5195, 0⟩, ⟨713, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨6495⟩⟩, ⟨.program ⟨214⟩, ⟨14873⟩⟩], []⟩, (1)⟩)

def exact5200RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨6495⟩⟩, ⟨.program ⟨214⟩, ⟨14873⟩⟩], []⟩, (1)⟩]

theorem exact5200RawTermsValid :
    exact5200RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event5200 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14874⟩⟩) exact5200RawTerms (.finite 156384508479209294644360) 5198 .exactZero (none)

def event5201 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14875⟩⟩) 0 ⟨6379⟩ 728

def event5202 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14875⟩⟩) 1 ⟨14874⟩ 5200

def event5203 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14875⟩⟩) (.sum [.predecessor 0 5201 .coefficient, .predecessor 1 5202 .coefficient])

def exact5204RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨6495⟩⟩, ⟨.program ⟨214⟩, ⟨14873⟩⟩], []⟩, (1)⟩]

theorem exact5204RawTermsValid :
    exact5204RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event5204 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14875⟩⟩) exact5204RawTerms (.finite 156384508479209294644360) 5203 .exactZero (none)

def event5205 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15036⟩⟩) 0 ⟨14875⟩ 5204

def event5206 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15036⟩⟩) 1 ⟨15035⟩ 5192

def event5207 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15036⟩⟩) (.sum [.predecessor 0 5205 .coefficient, .predecessor 1 5206 .coefficient])

def exact5208RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨6475⟩⟩, ⟨.program ⟨214⟩, ⟨15034⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6495⟩⟩, ⟨.program ⟨214⟩, ⟨14873⟩⟩], []⟩, (1)⟩]

theorem exact5208RawTermsValid :
    exact5208RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event5208 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15036⟩⟩) exact5208RawTerms (.finite 332317080518319751119265) 5207 .exactZero (none)

def event5209 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15197⟩⟩) 0 ⟨15036⟩ 5208

def event5210 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15197⟩⟩) 1 ⟨15196⟩ 5184

def event5211 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15197⟩⟩) (.sum [.predecessor 0 5209 .coefficient, .predecessor 1 5210 .coefficient])

def exact5212RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨6452⟩⟩, ⟨.program ⟨214⟩, ⟨15195⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6475⟩⟩, ⟨.program ⟨214⟩, ⟨15034⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6495⟩⟩, ⟨.program ⟨214⟩, ⟨14873⟩⟩], []⟩, (1)⟩]

theorem exact5212RawTermsValid :
    exact5212RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event5212 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15197⟩⟩) exact5212RawTerms (.finite 519978490693370904692497) 5211 .exactZero (none)

def event5213 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15505⟩⟩) 0 ⟨15197⟩ 5212

def event5214 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15505⟩⟩) 1 ⟨15504⟩ 5176

def event5215 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15505⟩⟩) (.sum [.predecessor 0 5213 .coefficient, .predecessor 1 5214 .coefficient])

def exact5216RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨6427⟩⟩, ⟨.program ⟨214⟩, ⟨15503⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6452⟩⟩, ⟨.program ⟨214⟩, ⟨15195⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6475⟩⟩, ⟨.program ⟨214⟩, ⟨15034⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6495⟩⟩, ⟨.program ⟨214⟩, ⟨14873⟩⟩], []⟩, (1)⟩]

theorem exact5216RawTermsValid :
    exact5216RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event5216 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15505⟩⟩) exact5216RawTerms (.finite 721044287309497140663817) 5215 .exactZero (none)

def event5217 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17794⟩⟩) 0 ⟨15505⟩ 5216

def event5218 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17794⟩⟩) 1 ⟨17793⟩ 5168

def event5219 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨17794⟩⟩) (.sum [.predecessor 0 5217 .coefficient, .predecessor 1 5218 .coefficient])

def exact5220RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨6407⟩⟩, ⟨.program ⟨214⟩, ⟨17792⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6427⟩⟩, ⟨.program ⟨214⟩, ⟨15503⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6452⟩⟩, ⟨.program ⟨214⟩, ⟨15195⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6475⟩⟩, ⟨.program ⟨214⟩, ⟨15034⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6495⟩⟩, ⟨.program ⟨214⟩, ⟨14873⟩⟩], []⟩, (1)⟩]

theorem exact5220RawTermsValid :
    exact5220RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event5220 : Event := .resultExact (⟨.program ⟨214⟩, ⟨17794⟩⟩) exact5220RawTerms (.finite 934295889781146178815217) 5219 .exactZero (none)

def event5221 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17795⟩⟩) 0 ⟨17794⟩ 5220

def event5222 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17795⟩⟩) 1 ⟨17429⟩ 5160

def event5223 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨17795⟩⟩) (.sum [.predecessor 0 5221 .coefficient, .predecessor 1 5222 .coefficient])

def exact5224RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨6398⟩⟩, ⟨.program ⟨214⟩, ⟨17428⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6407⟩⟩, ⟨.program ⟨214⟩, ⟨17792⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6427⟩⟩, ⟨.program ⟨214⟩, ⟨15503⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6452⟩⟩, ⟨.program ⟨214⟩, ⟨15195⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6475⟩⟩, ⟨.program ⟨214⟩, ⟨15034⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6495⟩⟩, ⟨.program ⟨214⟩, ⟨14873⟩⟩], []⟩, (1)⟩]

theorem exact5224RawTermsValid :
    exact5224RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event5224 : Event := .resultExact (⟨.program ⟨214⟩, ⟨17795⟩⟩) exact5224RawTerms (.finite 1150828286136974432938177) 5223 .exactZero (none)

def event5225 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17796⟩⟩) 0 ⟨17795⟩ 5224

def event5226 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17796⟩⟩) 1 ⟨17212⟩ 5152

def event5227 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨17796⟩⟩) (.sum [.predecessor 0 5225 .coefficient, .predecessor 1 5226 .coefficient])

def exact5228RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨6391⟩⟩, ⟨.program ⟨214⟩, ⟨17211⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6398⟩⟩, ⟨.program ⟨214⟩, ⟨17428⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6407⟩⟩, ⟨.program ⟨214⟩, ⟨17792⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6427⟩⟩, ⟨.program ⟨214⟩, ⟨15503⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6452⟩⟩, ⟨.program ⟨214⟩, ⟨15195⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6475⟩⟩, ⟨.program ⟨214⟩, ⟨15034⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6495⟩⟩, ⟨.program ⟨214⟩, ⟨14873⟩⟩], []⟩, (1)⟩]

theorem exact5228RawTermsValid :
    exact5228RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event5228 : Event := .resultExact (⟨.program ⟨214⟩, ⟨17796⟩⟩) exact5228RawTerms (.finite 1371606415754681672436097) 5227 .exactZero (none)

def event5229 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17797⟩⟩) 0 ⟨17796⟩ 5228

def event5230 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17797⟩⟩) 1 ⟨17156⟩ 5144

def event5231 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨17797⟩⟩) (.sum [.predecessor 0 5229 .coefficient, .predecessor 1 5230 .coefficient])

def exact5232RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨6387⟩⟩, ⟨.program ⟨214⟩, ⟨17155⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6391⟩⟩, ⟨.program ⟨214⟩, ⟨17211⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6398⟩⟩, ⟨.program ⟨214⟩, ⟨17428⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6407⟩⟩, ⟨.program ⟨214⟩, ⟨17792⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6427⟩⟩, ⟨.program ⟨214⟩, ⟨15503⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6452⟩⟩, ⟨.program ⟨214⟩, ⟨15195⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6475⟩⟩, ⟨.program ⟨214⟩, ⟨15034⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6495⟩⟩, ⟨.program ⟨214⟩, ⟨14873⟩⟩], []⟩, (1)⟩]

theorem exact5232RawTermsValid :
    exact5232RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event5232 : Event := .resultExact (⟨.program ⟨214⟩, ⟨17797⟩⟩) exact5232RawTerms (.finite 1593837033067242249035977) 5231 .exactZero (none)

def event5233 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18018⟩⟩) 0 ⟨17797⟩ 5232

def event5234 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18018⟩⟩) 1 ⟨18017⟩ 5136

def event5235 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨18018⟩⟩) (.sum [.predecessor 0 5233 .coefficient, .predecessor 1 5234 .coefficient])

def exact5236RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨6383⟩⟩, ⟨.program ⟨214⟩, ⟨18016⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6387⟩⟩, ⟨.program ⟨214⟩, ⟨17155⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6391⟩⟩, ⟨.program ⟨214⟩, ⟨17211⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6398⟩⟩, ⟨.program ⟨214⟩, ⟨17428⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6407⟩⟩, ⟨.program ⟨214⟩, ⟨17792⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6427⟩⟩, ⟨.program ⟨214⟩, ⟨15503⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6452⟩⟩, ⟨.program ⟨214⟩, ⟨15195⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6475⟩⟩, ⟨.program ⟨214⟩, ⟨15034⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6495⟩⟩, ⟨.program ⟨214⟩, ⟨14873⟩⟩], []⟩, (1)⟩]

theorem exact5236RawTermsValid :
    exact5236RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event5236 : Event := .resultExact (⟨.program ⟨214⟩, ⟨18018⟩⟩) exact5236RawTerms (.finite 1818214806102629497873537) 5235 .exactZero (none)

def event5237 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18019⟩⟩) 0 ⟨18018⟩ 5236

def event5238 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18019⟩⟩) 1 ⟨17653⟩ 5128

def event5239 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨18019⟩⟩) (.sum [.predecessor 0 5237 .coefficient, .predecessor 1 5238 .coefficient])

def exact5240RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨6383⟩⟩, ⟨.program ⟨214⟩, ⟨18016⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6387⟩⟩, ⟨.program ⟨214⟩, ⟨17155⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6391⟩⟩, ⟨.program ⟨214⟩, ⟨17211⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6398⟩⟩, ⟨.program ⟨214⟩, ⟨17428⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6407⟩⟩, ⟨.program ⟨214⟩, ⟨17792⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6427⟩⟩, ⟨.program ⟨214⟩, ⟨15503⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6452⟩⟩, ⟨.program ⟨214⟩, ⟨15195⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6475⟩⟩, ⟨.program ⟨214⟩, ⟨15034⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6495⟩⟩, ⟨.program ⟨214⟩, ⟨14873⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6502⟩⟩, ⟨.program ⟨214⟩, ⟨17652⟩⟩], []⟩, (1)⟩]

theorem exact5240RawTermsValid :
    exact5240RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event5240 : Event := .resultExact (⟨.program ⟨214⟩, ⟨18019⟩⟩) exact5240RawTerms (.finite 2044702714934587786668817) 5239 .exactZero (none)

def event5241 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18020⟩⟩) 0 ⟨18019⟩ 5240

def event5242 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18020⟩⟩) 1 ⟨17597⟩ 5120

def event5243 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨18020⟩⟩) (.sum [.predecessor 0 5241 .coefficient, .predecessor 1 5242 .coefficient])

def exact5244RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨6383⟩⟩, ⟨.program ⟨214⟩, ⟨18016⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6387⟩⟩, ⟨.program ⟨214⟩, ⟨17155⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6391⟩⟩, ⟨.program ⟨214⟩, ⟨17211⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6398⟩⟩, ⟨.program ⟨214⟩, ⟨17428⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6407⟩⟩, ⟨.program ⟨214⟩, ⟨17792⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6427⟩⟩, ⟨.program ⟨214⟩, ⟨15503⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6452⟩⟩, ⟨.program ⟨214⟩, ⟨15195⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6475⟩⟩, ⟨.program ⟨214⟩, ⟨15034⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6494⟩⟩, ⟨.program ⟨214⟩, ⟨17596⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6495⟩⟩, ⟨.program ⟨214⟩, ⟨14873⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6502⟩⟩, ⟨.program ⟨214⟩, ⟨17652⟩⟩], []⟩, (1)⟩]

theorem exact5244RawTermsValid :
    exact5244RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event5244 : Event := .resultExact (⟨.program ⟨214⟩, ⟨18020⟩⟩) exact5244RawTerms (.finite 2271712485307633536959017) 5243 .exactZero (none)

def event5245 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18794⟩⟩) 0 ⟨18020⟩ 5244

def event5246 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18794⟩⟩) 1 ⟨18793⟩ 5112

def event5247 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨18794⟩⟩) (.sum [.predecessor 0 5245 .coefficient, .predecessor 1 5246 .coefficient])

def exact5248RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨6383⟩⟩, ⟨.program ⟨214⟩, ⟨18016⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6387⟩⟩, ⟨.program ⟨214⟩, ⟨17155⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6391⟩⟩, ⟨.program ⟨214⟩, ⟨17211⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6398⟩⟩, ⟨.program ⟨214⟩, ⟨17428⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6407⟩⟩, ⟨.program ⟨214⟩, ⟨17792⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6427⟩⟩, ⟨.program ⟨214⟩, ⟨15503⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6452⟩⟩, ⟨.program ⟨214⟩, ⟨15195⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6475⟩⟩, ⟨.program ⟨214⟩, ⟨15034⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6490⟩⟩, ⟨.program ⟨214⟩, ⟨18792⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6494⟩⟩, ⟨.program ⟨214⟩, ⟨17596⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6495⟩⟩, ⟨.program ⟨214⟩, ⟨14873⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6502⟩⟩, ⟨.program ⟨214⟩, ⟨17652⟩⟩], []⟩, (1)⟩]

theorem exact5248RawTermsValid :
    exact5248RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event5248 : Event := .resultExact (⟨.program ⟨214⟩, ⟨18794⟩⟩) exact5248RawTerms (.finite 2499949335520533588602137) 5247 .exactZero (none)

def event5249 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18795⟩⟩) 0 ⟨18794⟩ 5248

def event5250 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18795⟩⟩) 1 ⟨17541⟩ 5104

def event5251 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨18795⟩⟩) (.sum [.predecessor 0 5249 .coefficient, .predecessor 1 5250 .coefficient])

def exact5252RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨6383⟩⟩, ⟨.program ⟨214⟩, ⟨18016⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6387⟩⟩, ⟨.program ⟨214⟩, ⟨17155⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6391⟩⟩, ⟨.program ⟨214⟩, ⟨17211⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6398⟩⟩, ⟨.program ⟨214⟩, ⟨17428⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6407⟩⟩, ⟨.program ⟨214⟩, ⟨17792⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6427⟩⟩, ⟨.program ⟨214⟩, ⟨15503⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6452⟩⟩, ⟨.program ⟨214⟩, ⟨15195⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6473⟩⟩, ⟨.program ⟨214⟩, ⟨17540⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6475⟩⟩, ⟨.program ⟨214⟩, ⟨15034⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6490⟩⟩, ⟨.program ⟨214⟩, ⟨18792⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6494⟩⟩, ⟨.program ⟨214⟩, ⟨17596⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6495⟩⟩, ⟨.program ⟨214⟩, ⟨14873⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6502⟩⟩, ⟨.program ⟨214⟩, ⟨17652⟩⟩], []⟩, (1)⟩]

theorem exact5252RawTermsValid :
    exact5252RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event5252 : Event := .resultExact (⟨.program ⟨214⟩, ⟨18795⟩⟩) exact5252RawTerms (.finite 2728804713782791092959737) 5251 .exactZero (none)

def event5253 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18796⟩⟩) 0 ⟨18795⟩ 5252

def event5254 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18796⟩⟩) 1 ⟨17940⟩ 5096

def event5255 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨18796⟩⟩) (.sum [.predecessor 0 5253 .coefficient, .predecessor 1 5254 .coefficient])

def exact5256RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨6383⟩⟩, ⟨.program ⟨214⟩, ⟨18016⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6387⟩⟩, ⟨.program ⟨214⟩, ⟨17155⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6391⟩⟩, ⟨.program ⟨214⟩, ⟨17211⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6398⟩⟩, ⟨.program ⟨214⟩, ⟨17428⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6407⟩⟩, ⟨.program ⟨214⟩, ⟨17792⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6427⟩⟩, ⟨.program ⟨214⟩, ⟨15503⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6452⟩⟩, ⟨.program ⟨214⟩, ⟨15195⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6467⟩⟩, ⟨.program ⟨214⟩, ⟨17939⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6473⟩⟩, ⟨.program ⟨214⟩, ⟨17540⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6475⟩⟩, ⟨.program ⟨214⟩, ⟨15034⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6490⟩⟩, ⟨.program ⟨214⟩, ⟨18792⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6494⟩⟩, ⟨.program ⟨214⟩, ⟨17596⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6495⟩⟩, ⟨.program ⟨214⟩, ⟨14873⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6502⟩⟩, ⟨.program ⟨214⟩, ⟨17652⟩⟩], []⟩, (1)⟩]

theorem exact5256RawTermsValid :
    exact5256RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event5256 : Event := .resultExact (⟨.program ⟨214⟩, ⟨18796⟩⟩) exact5256RawTerms (.finite 2957926202950004710694497) 5255 .exactZero (none)

def event5257 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18797⟩⟩) 0 ⟨18796⟩ 5256

def event5258 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18797⟩⟩) 1 ⟨17709⟩ 5088

def event5259 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨18797⟩⟩) (.sum [.predecessor 0 5257 .coefficient, .predecessor 1 5258 .coefficient])

def exact5260RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨6383⟩⟩, ⟨.program ⟨214⟩, ⟨18016⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6387⟩⟩, ⟨.program ⟨214⟩, ⟨17155⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6391⟩⟩, ⟨.program ⟨214⟩, ⟨17211⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6398⟩⟩, ⟨.program ⟨214⟩, ⟨17428⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6407⟩⟩, ⟨.program ⟨214⟩, ⟨17792⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6427⟩⟩, ⟨.program ⟨214⟩, ⟨15503⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6452⟩⟩, ⟨.program ⟨214⟩, ⟨15195⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6459⟩⟩, ⟨.program ⟨214⟩, ⟨17708⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6467⟩⟩, ⟨.program ⟨214⟩, ⟨17939⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6473⟩⟩, ⟨.program ⟨214⟩, ⟨17540⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6475⟩⟩, ⟨.program ⟨214⟩, ⟨15034⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6490⟩⟩, ⟨.program ⟨214⟩, ⟨18792⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6494⟩⟩, ⟨.program ⟨214⟩, ⟨17596⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6495⟩⟩, ⟨.program ⟨214⟩, ⟨14873⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6502⟩⟩, ⟨.program ⟨214⟩, ⟨17652⟩⟩], []⟩, (1)⟩]

theorem exact5260RawTermsValid :
    exact5260RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event5260 : Event := .resultExact (⟨.program ⟨214⟩, ⟨18797⟩⟩) exact5260RawTerms (.finite 3187511970717354526236217) 5259 .exactZero (none)

def event5261 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18798⟩⟩) 0 ⟨18797⟩ 5260

def event5262 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18798⟩⟩) 1 ⟨17485⟩ 5080

def event5263 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨18798⟩⟩) (.sum [.predecessor 0 5261 .coefficient, .predecessor 1 5262 .coefficient])

def exact5264RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨6383⟩⟩, ⟨.program ⟨214⟩, ⟨18016⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6387⟩⟩, ⟨.program ⟨214⟩, ⟨17155⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6391⟩⟩, ⟨.program ⟨214⟩, ⟨17211⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6398⟩⟩, ⟨.program ⟨214⟩, ⟨17428⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6407⟩⟩, ⟨.program ⟨214⟩, ⟨17792⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6427⟩⟩, ⟨.program ⟨214⟩, ⟨15503⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6449⟩⟩, ⟨.program ⟨214⟩, ⟨17484⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6452⟩⟩, ⟨.program ⟨214⟩, ⟨15195⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6459⟩⟩, ⟨.program ⟨214⟩, ⟨17708⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6467⟩⟩, ⟨.program ⟨214⟩, ⟨17939⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6473⟩⟩, ⟨.program ⟨214⟩, ⟨17540⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6475⟩⟩, ⟨.program ⟨214⟩, ⟨15034⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6490⟩⟩, ⟨.program ⟨214⟩, ⟨18792⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6494⟩⟩, ⟨.program ⟨214⟩, ⟨17596⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6495⟩⟩, ⟨.program ⟨214⟩, ⟨14873⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6502⟩⟩, ⟨.program ⟨214⟩, ⟨17652⟩⟩], []⟩, (1)⟩]

theorem exact5264RawTermsValid :
    exact5264RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event5264 : Event := .resultExact (⟨.program ⟨214⟩, ⟨18798⟩⟩) exact5264RawTerms (.finite 3417662756781096507033577) 5263 .exactZero (none)

def event5265 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18799⟩⟩) 0 ⟨18798⟩ 5264

def event5266 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18799⟩⟩) 1 ⟨16918⟩ 5072

def event5267 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨18799⟩⟩) (.sum [.predecessor 0 5265 .coefficient, .predecessor 1 5266 .coefficient])

def exact5268RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨6383⟩⟩, ⟨.program ⟨214⟩, ⟨18016⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6387⟩⟩, ⟨.program ⟨214⟩, ⟨17155⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6391⟩⟩, ⟨.program ⟨214⟩, ⟨17211⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6398⟩⟩, ⟨.program ⟨214⟩, ⟨17428⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6407⟩⟩, ⟨.program ⟨214⟩, ⟨17792⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6427⟩⟩, ⟨.program ⟨214⟩, ⟨15503⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6437⟩⟩, ⟨.program ⟨214⟩, ⟨16917⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6449⟩⟩, ⟨.program ⟨214⟩, ⟨17484⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6452⟩⟩, ⟨.program ⟨214⟩, ⟨15195⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6459⟩⟩, ⟨.program ⟨214⟩, ⟨17708⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6467⟩⟩, ⟨.program ⟨214⟩, ⟨17939⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6473⟩⟩, ⟨.program ⟨214⟩, ⟨17540⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6475⟩⟩, ⟨.program ⟨214⟩, ⟨15034⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6490⟩⟩, ⟨.program ⟨214⟩, ⟨18792⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6494⟩⟩, ⟨.program ⟨214⟩, ⟨17596⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6495⟩⟩, ⟨.program ⟨214⟩, ⟨14873⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6502⟩⟩, ⟨.program ⟨214⟩, ⟨17652⟩⟩], []⟩, (1)⟩]

theorem exact5268RawTermsValid :
    exact5268RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event5268 : Event := .resultExact (⟨.program ⟨214⟩, ⟨18799⟩⟩) exact5268RawTerms (.finite 3648263642165693263543057) 5267 .exactZero (none)

def event5269 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18800⟩⟩) 0 ⟨18799⟩ 5268

def event5270 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18800⟩⟩) 1 ⟨18115⟩ 5064

def event5271 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨18800⟩⟩) (.sum [.predecessor 0 5269 .coefficient, .predecessor 1 5270 .coefficient])

def exact5272RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨6383⟩⟩, ⟨.program ⟨214⟩, ⟨18016⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6387⟩⟩, ⟨.program ⟨214⟩, ⟨17155⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6391⟩⟩, ⟨.program ⟨214⟩, ⟨17211⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6398⟩⟩, ⟨.program ⟨214⟩, ⟨17428⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6407⟩⟩, ⟨.program ⟨214⟩, ⟨17792⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6427⟩⟩, ⟨.program ⟨214⟩, ⟨15503⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6435⟩⟩, ⟨.program ⟨214⟩, ⟨18114⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6437⟩⟩, ⟨.program ⟨214⟩, ⟨16917⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6449⟩⟩, ⟨.program ⟨214⟩, ⟨17484⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6452⟩⟩, ⟨.program ⟨214⟩, ⟨15195⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6459⟩⟩, ⟨.program ⟨214⟩, ⟨17708⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6467⟩⟩, ⟨.program ⟨214⟩, ⟨17939⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6473⟩⟩, ⟨.program ⟨214⟩, ⟨17540⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6475⟩⟩, ⟨.program ⟨214⟩, ⟨15034⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6490⟩⟩, ⟨.program ⟨214⟩, ⟨18792⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6494⟩⟩, ⟨.program ⟨214⟩, ⟨17596⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6495⟩⟩, ⟨.program ⟨214⟩, ⟨14873⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6502⟩⟩, ⟨.program ⟨214⟩, ⟨17652⟩⟩], []⟩, (1)⟩]

theorem exact5272RawTermsValid :
    exact5272RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event5272 : Event := .resultExact (⟨.program ⟨214⟩, ⟨18800⟩⟩) exact5272RawTerms (.finite 3878994884184198780231457) 5271 .exactZero (none)

def event5273 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18802⟩⟩) 0 ⟨18800⟩ 5272

def event5274 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18802⟩⟩) 1 ⟨18486⟩ 5056

def event5275 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨18802⟩⟩) (.sum [.predecessor 0 5273 .coefficient, .predecessor 1 5274 .coefficient])

def exact5276RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨6383⟩⟩, ⟨.program ⟨214⟩, ⟨18016⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6387⟩⟩, ⟨.program ⟨214⟩, ⟨17155⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6391⟩⟩, ⟨.program ⟨214⟩, ⟨17211⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6398⟩⟩, ⟨.program ⟨214⟩, ⟨17428⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6407⟩⟩, ⟨.program ⟨214⟩, ⟨17792⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6410⟩⟩, ⟨.program ⟨214⟩, ⟨18485⟩⟩], []⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6427⟩⟩, ⟨.program ⟨214⟩, ⟨15503⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6435⟩⟩, ⟨.program ⟨214⟩, ⟨18114⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6437⟩⟩, ⟨.program ⟨214⟩, ⟨16917⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6449⟩⟩, ⟨.program ⟨214⟩, ⟨17484⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6452⟩⟩, ⟨.program ⟨214⟩, ⟨15195⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6459⟩⟩, ⟨.program ⟨214⟩, ⟨17708⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6467⟩⟩, ⟨.program ⟨214⟩, ⟨17939⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6473⟩⟩, ⟨.program ⟨214⟩, ⟨17540⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6475⟩⟩, ⟨.program ⟨214⟩, ⟨15034⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6490⟩⟩, ⟨.program ⟨214⟩, ⟨18792⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6494⟩⟩, ⟨.program ⟨214⟩, ⟨17596⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6495⟩⟩, ⟨.program ⟨214⟩, ⟨14873⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6502⟩⟩, ⟨.program ⟨214⟩, ⟨17652⟩⟩], []⟩, (1)⟩]

theorem exact5276RawTermsValid :
    exact5276RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event5276 : Event := .resultExact (⟨.program ⟨214⟩, ⟨18802⟩⟩) exact5276RawTerms (.finite 8101376613122849735629177) 5275 .exactZero (none)

def event5277 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18803⟩⟩) 0 ⟨18802⟩ 5276

def event5278 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18803⟩⟩) 1 ⟨6384⟩ 4563

def event5279 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨18803⟩⟩) (.product (.predecessor 0 5277 .coefficient) (.predecessor 1 5278 .coefficient) (⟨false, true, none, none, some 1⟩))

def event5280 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18803⟩⟩, .operator (⟨5276, 5⟩, ⟨4563, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6410⟩⟩, ⟨.program ⟨214⟩, ⟨18485⟩⟩], []⟩, (-1)⟩)

def event5281 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18803⟩⟩, .operator (⟨5276, 7⟩, ⟨4563, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6435⟩⟩, ⟨.program ⟨214⟩, ⟨18114⟩⟩], []⟩, (1)⟩)

def event5282 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18803⟩⟩, .operator (⟨5276, 8⟩, ⟨4563, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6437⟩⟩, ⟨.program ⟨214⟩, ⟨16917⟩⟩], []⟩, (1)⟩)

def event5283 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18803⟩⟩, .operator (⟨5276, 9⟩, ⟨4563, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6449⟩⟩, ⟨.program ⟨214⟩, ⟨17484⟩⟩], []⟩, (1)⟩)

def event5284 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18803⟩⟩, .operator (⟨5276, 11⟩, ⟨4563, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6459⟩⟩, ⟨.program ⟨214⟩, ⟨17708⟩⟩], []⟩, (1)⟩)

def event5285 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18803⟩⟩, .operator (⟨5276, 12⟩, ⟨4563, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6467⟩⟩, ⟨.program ⟨214⟩, ⟨17939⟩⟩], []⟩, (1)⟩)

def event5286 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18803⟩⟩, .operator (⟨5276, 13⟩, ⟨4563, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6473⟩⟩, ⟨.program ⟨214⟩, ⟨17540⟩⟩], []⟩, (1)⟩)

def event5287 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18803⟩⟩, .operator (⟨5276, 15⟩, ⟨4563, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6490⟩⟩, ⟨.program ⟨214⟩, ⟨18792⟩⟩], []⟩, (1)⟩)

def event5288 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18803⟩⟩, .operator (⟨5276, 16⟩, ⟨4563, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6494⟩⟩, ⟨.program ⟨214⟩, ⟨17596⟩⟩], []⟩, (1)⟩)

def event5289 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18803⟩⟩, .operator (⟨5276, 18⟩, ⟨4563, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6502⟩⟩, ⟨.program ⟨214⟩, ⟨17652⟩⟩], []⟩, (1)⟩)

def event5290 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18803⟩⟩, .operator (⟨5276, 0⟩, ⟨4563, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨6383⟩⟩, ⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨18016⟩⟩], []⟩, (1)⟩)

def event5291 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18803⟩⟩, .operator (⟨5276, 1⟩, ⟨4563, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6387⟩⟩, ⟨.program ⟨214⟩, ⟨17155⟩⟩], []⟩, (1)⟩)

def event5292 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18803⟩⟩, .operator (⟨5276, 2⟩, ⟨4563, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6391⟩⟩, ⟨.program ⟨214⟩, ⟨17211⟩⟩], []⟩, (1)⟩)

def event5293 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18803⟩⟩, .operator (⟨5276, 3⟩, ⟨4563, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6398⟩⟩, ⟨.program ⟨214⟩, ⟨17428⟩⟩], []⟩, (1)⟩)

def event5294 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18803⟩⟩, .operator (⟨5276, 4⟩, ⟨4563, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6407⟩⟩, ⟨.program ⟨214⟩, ⟨17792⟩⟩], []⟩, (1)⟩)

def event5295 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18803⟩⟩, .operator (⟨5276, 6⟩, ⟨4563, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6427⟩⟩, ⟨.program ⟨214⟩, ⟨15503⟩⟩], []⟩, (1)⟩)

def event5296 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18803⟩⟩, .operator (⟨5276, 10⟩, ⟨4563, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6452⟩⟩, ⟨.program ⟨214⟩, ⟨15195⟩⟩], []⟩, (1)⟩)

def event5297 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18803⟩⟩, .operator (⟨5276, 14⟩, ⟨4563, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6475⟩⟩, ⟨.program ⟨214⟩, ⟨15034⟩⟩], []⟩, (1)⟩)

def event5298 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18803⟩⟩, .operator (⟨5276, 17⟩, ⟨4563, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6495⟩⟩, ⟨.program ⟨214⟩, ⟨14873⟩⟩], []⟩, (1)⟩)

def exact5299RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨6383⟩⟩, ⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨18016⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6387⟩⟩, ⟨.program ⟨214⟩, ⟨17155⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6391⟩⟩, ⟨.program ⟨214⟩, ⟨17211⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6398⟩⟩, ⟨.program ⟨214⟩, ⟨17428⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6407⟩⟩, ⟨.program ⟨214⟩, ⟨17792⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6410⟩⟩, ⟨.program ⟨214⟩, ⟨18485⟩⟩], []⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6427⟩⟩, ⟨.program ⟨214⟩, ⟨15503⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6435⟩⟩, ⟨.program ⟨214⟩, ⟨18114⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6437⟩⟩, ⟨.program ⟨214⟩, ⟨16917⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6449⟩⟩, ⟨.program ⟨214⟩, ⟨17484⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6452⟩⟩, ⟨.program ⟨214⟩, ⟨15195⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6459⟩⟩, ⟨.program ⟨214⟩, ⟨17708⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6467⟩⟩, ⟨.program ⟨214⟩, ⟨17939⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6473⟩⟩, ⟨.program ⟨214⟩, ⟨17540⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6475⟩⟩, ⟨.program ⟨214⟩, ⟨15034⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6490⟩⟩, ⟨.program ⟨214⟩, ⟨18792⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6494⟩⟩, ⟨.program ⟨214⟩, ⟨17596⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6495⟩⟩, ⟨.program ⟨214⟩, ⟨14873⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6502⟩⟩, ⟨.program ⟨214⟩, ⟨17652⟩⟩], []⟩, (1)⟩]

theorem exact5299RawTermsValid :
    exact5299RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event5299 : Event := .resultExact (⟨.program ⟨214⟩, ⟨18803⟩⟩) exact5299RawTerms (.finite 2777451680365593313469174004953636154164704869736134852198215577047367856749580419392) 5279 .exactZero (none)

def event5300 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18804⟩⟩) 0 ⟨6379⟩ 728

def event5301 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18804⟩⟩) 1 ⟨18803⟩ 5299

def event5302 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨18804⟩⟩) (.sum [.predecessor 0 5300 .coefficient, .predecessor 1 5301 .coefficient])

def exact5303RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨6383⟩⟩, ⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨18016⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6387⟩⟩, ⟨.program ⟨214⟩, ⟨17155⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6391⟩⟩, ⟨.program ⟨214⟩, ⟨17211⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6398⟩⟩, ⟨.program ⟨214⟩, ⟨17428⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6407⟩⟩, ⟨.program ⟨214⟩, ⟨17792⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6410⟩⟩, ⟨.program ⟨214⟩, ⟨18485⟩⟩], []⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6427⟩⟩, ⟨.program ⟨214⟩, ⟨15503⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6435⟩⟩, ⟨.program ⟨214⟩, ⟨18114⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6437⟩⟩, ⟨.program ⟨214⟩, ⟨16917⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6449⟩⟩, ⟨.program ⟨214⟩, ⟨17484⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6452⟩⟩, ⟨.program ⟨214⟩, ⟨15195⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6459⟩⟩, ⟨.program ⟨214⟩, ⟨17708⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6467⟩⟩, ⟨.program ⟨214⟩, ⟨17939⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6473⟩⟩, ⟨.program ⟨214⟩, ⟨17540⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6475⟩⟩, ⟨.program ⟨214⟩, ⟨15034⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6490⟩⟩, ⟨.program ⟨214⟩, ⟨18792⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6494⟩⟩, ⟨.program ⟨214⟩, ⟨17596⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6495⟩⟩, ⟨.program ⟨214⟩, ⟨14873⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6502⟩⟩, ⟨.program ⟨214⟩, ⟨17652⟩⟩], []⟩, (1)⟩]

theorem exact5303RawTermsValid :
    exact5303RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event5303 : Event := .resultExact (⟨.program ⟨214⟩, ⟨18804⟩⟩) exact5303RawTerms (.finite 2777451680365593313469174004953636154164704869736134852198215577047367856749580419392) 5302 .exactZero (none)

def event5304 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18844⟩⟩) 0 ⟨18804⟩ 5303

def event5305 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18844⟩⟩) 1 ⟨18843⟩ 4561

def event5306 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨18844⟩⟩) (.sum [.predecessor 0 5304 .coefficient, .predecessor 1 5305 .coefficient])

def exact5307RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨6383⟩⟩, ⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨18016⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6383⟩⟩, ⟨.program ⟨214⟩, ⟨6503⟩⟩, ⟨.program ⟨214⟩, ⟨18035⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6387⟩⟩, ⟨.program ⟨214⟩, ⟨17155⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6391⟩⟩, ⟨.program ⟨214⟩, ⟨17211⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6398⟩⟩, ⟨.program ⟨214⟩, ⟨17428⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6407⟩⟩, ⟨.program ⟨214⟩, ⟨17792⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6410⟩⟩, ⟨.program ⟨214⟩, ⟨18485⟩⟩], []⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6427⟩⟩, ⟨.program ⟨214⟩, ⟨15503⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6435⟩⟩, ⟨.program ⟨214⟩, ⟨18114⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6437⟩⟩, ⟨.program ⟨214⟩, ⟨16917⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6449⟩⟩, ⟨.program ⟨214⟩, ⟨17484⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6452⟩⟩, ⟨.program ⟨214⟩, ⟨15195⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6459⟩⟩, ⟨.program ⟨214⟩, ⟨17708⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6467⟩⟩, ⟨.program ⟨214⟩, ⟨17939⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6473⟩⟩, ⟨.program ⟨214⟩, ⟨17540⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6475⟩⟩, ⟨.program ⟨214⟩, ⟨15034⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6490⟩⟩, ⟨.program ⟨214⟩, ⟨18792⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6494⟩⟩, ⟨.program ⟨214⟩, ⟨17596⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6495⟩⟩, ⟨.program ⟨214⟩, ⟨14873⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6502⟩⟩, ⟨.program ⟨214⟩, ⟨17652⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6387⟩⟩, ⟨.program ⟨214⟩, ⟨6503⟩⟩, ⟨.program ⟨214⟩, ⟨17165⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6391⟩⟩, ⟨.program ⟨214⟩, ⟨6503⟩⟩, ⟨.program ⟨214⟩, ⟨17221⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6398⟩⟩, ⟨.program ⟨214⟩, ⟨6503⟩⟩, ⟨.program ⟨214⟩, ⟨17438⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6407⟩⟩, ⟨.program ⟨214⟩, ⟨6503⟩⟩, ⟨.program ⟨214⟩, ⟨17814⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6410⟩⟩, ⟨.program ⟨214⟩, ⟨6503⟩⟩, ⟨.program ⟨214⟩, ⟨18495⟩⟩], []⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6427⟩⟩, ⟨.program ⟨214⟩, ⟨6503⟩⟩, ⟨.program ⟨214⟩, ⟨15516⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6435⟩⟩, ⟨.program ⟨214⟩, ⟨6503⟩⟩, ⟨.program ⟨214⟩, ⟨18124⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6437⟩⟩, ⟨.program ⟨214⟩, ⟨6503⟩⟩, ⟨.program ⟨214⟩, ⟨16927⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6449⟩⟩, ⟨.program ⟨214⟩, ⟨6503⟩⟩, ⟨.program ⟨214⟩, ⟨17494⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6452⟩⟩, ⟨.program ⟨214⟩, ⟨6503⟩⟩, ⟨.program ⟨214⟩, ⟨15208⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6459⟩⟩, ⟨.program ⟨214⟩, ⟨6503⟩⟩, ⟨.program ⟨214⟩, ⟨17718⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6467⟩⟩, ⟨.program ⟨214⟩, ⟨6503⟩⟩, ⟨.program ⟨214⟩, ⟨17949⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6473⟩⟩, ⟨.program ⟨214⟩, ⟨6503⟩⟩, ⟨.program ⟨214⟩, ⟨17550⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6475⟩⟩, ⟨.program ⟨214⟩, ⟨6503⟩⟩, ⟨.program ⟨214⟩, ⟨15047⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6490⟩⟩, ⟨.program ⟨214⟩, ⟨6503⟩⟩, ⟨.program ⟨214⟩, ⟨18832⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6494⟩⟩, ⟨.program ⟨214⟩, ⟨6503⟩⟩, ⟨.program ⟨214⟩, ⟨17606⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6495⟩⟩, ⟨.program ⟨214⟩, ⟨6503⟩⟩, ⟨.program ⟨214⟩, ⟨14886⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6502⟩⟩, ⟨.program ⟨214⟩, ⟨6503⟩⟩, ⟨.program ⟨214⟩, ⟨17662⟩⟩], []⟩, (1)⟩]

theorem exact5307RawTermsValid :
    exact5307RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event5307 : Event := .resultExact (⟨.program ⟨214⟩, ⟨18844⟩⟩) exact5307RawTerms (.finite 6899444407929433029479311655944131635462950979646278142325202208416514303764052172448) 5306 .exactZero (none)

def event5308 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18845⟩⟩) 0 ⟨18844⟩ 5307

def event5309 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18845⟩⟩) 1 ⟨18829⟩ 3819

def event5310 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨18845⟩⟩) (.sum [.predecessor 0 5308 .coefficient, .predecessor 1 5309 .coefficient])

def exact5311RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨6383⟩⟩, ⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨18016⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6383⟩⟩, ⟨.program ⟨214⟩, ⟨6503⟩⟩, ⟨.program ⟨214⟩, ⟨18035⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6383⟩⟩, ⟨.program ⟨214⟩, ⟨6542⟩⟩, ⟨.program ⟨214⟩, ⟨18028⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6387⟩⟩, ⟨.program ⟨214⟩, ⟨17155⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6391⟩⟩, ⟨.program ⟨214⟩, ⟨17211⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6398⟩⟩, ⟨.program ⟨214⟩, ⟨17428⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6407⟩⟩, ⟨.program ⟨214⟩, ⟨17792⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6410⟩⟩, ⟨.program ⟨214⟩, ⟨18485⟩⟩], []⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6427⟩⟩, ⟨.program ⟨214⟩, ⟨15503⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6435⟩⟩, ⟨.program ⟨214⟩, ⟨18114⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6437⟩⟩, ⟨.program ⟨214⟩, ⟨16917⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6449⟩⟩, ⟨.program ⟨214⟩, ⟨17484⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6452⟩⟩, ⟨.program ⟨214⟩, ⟨15195⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6459⟩⟩, ⟨.program ⟨214⟩, ⟨17708⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6467⟩⟩, ⟨.program ⟨214⟩, ⟨17939⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6473⟩⟩, ⟨.program ⟨214⟩, ⟨17540⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6475⟩⟩, ⟨.program ⟨214⟩, ⟨15034⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6490⟩⟩, ⟨.program ⟨214⟩, ⟨18792⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6494⟩⟩, ⟨.program ⟨214⟩, ⟨17596⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6495⟩⟩, ⟨.program ⟨214⟩, ⟨14873⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6502⟩⟩, ⟨.program ⟨214⟩, ⟨17652⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6387⟩⟩, ⟨.program ⟨214⟩, ⟨6503⟩⟩, ⟨.program ⟨214⟩, ⟨17165⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6387⟩⟩, ⟨.program ⟨214⟩, ⟨6542⟩⟩, ⟨.program ⟨214⟩, ⟨17161⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6391⟩⟩, ⟨.program ⟨214⟩, ⟨6503⟩⟩, ⟨.program ⟨214⟩, ⟨17221⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6391⟩⟩, ⟨.program ⟨214⟩, ⟨6542⟩⟩, ⟨.program ⟨214⟩, ⟨17217⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6398⟩⟩, ⟨.program ⟨214⟩, ⟨6503⟩⟩, ⟨.program ⟨214⟩, ⟨17438⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6398⟩⟩, ⟨.program ⟨214⟩, ⟨6542⟩⟩, ⟨.program ⟨214⟩, ⟨17434⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6407⟩⟩, ⟨.program ⟨214⟩, ⟨6503⟩⟩, ⟨.program ⟨214⟩, ⟨17814⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6407⟩⟩, ⟨.program ⟨214⟩, ⟨6542⟩⟩, ⟨.program ⟨214⟩, ⟨17806⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6410⟩⟩, ⟨.program ⟨214⟩, ⟨6503⟩⟩, ⟨.program ⟨214⟩, ⟨18495⟩⟩], []⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6410⟩⟩, ⟨.program ⟨214⟩, ⟨6542⟩⟩, ⟨.program ⟨214⟩, ⟨18491⟩⟩], []⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6427⟩⟩, ⟨.program ⟨214⟩, ⟨6503⟩⟩, ⟨.program ⟨214⟩, ⟨15516⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6427⟩⟩, ⟨.program ⟨214⟩, ⟨6542⟩⟩, ⟨.program ⟨214⟩, ⟨15511⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6435⟩⟩, ⟨.program ⟨214⟩, ⟨6503⟩⟩, ⟨.program ⟨214⟩, ⟨18124⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6435⟩⟩, ⟨.program ⟨214⟩, ⟨6542⟩⟩, ⟨.program ⟨214⟩, ⟨18120⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6437⟩⟩, ⟨.program ⟨214⟩, ⟨6503⟩⟩, ⟨.program ⟨214⟩, ⟨16927⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6437⟩⟩, ⟨.program ⟨214⟩, ⟨6542⟩⟩, ⟨.program ⟨214⟩, ⟨16923⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6449⟩⟩, ⟨.program ⟨214⟩, ⟨6503⟩⟩, ⟨.program ⟨214⟩, ⟨17494⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6449⟩⟩, ⟨.program ⟨214⟩, ⟨6542⟩⟩, ⟨.program ⟨214⟩, ⟨17490⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6452⟩⟩, ⟨.program ⟨214⟩, ⟨6503⟩⟩, ⟨.program ⟨214⟩, ⟨15208⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6452⟩⟩, ⟨.program ⟨214⟩, ⟨6542⟩⟩, ⟨.program ⟨214⟩, ⟨15203⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6459⟩⟩, ⟨.program ⟨214⟩, ⟨6503⟩⟩, ⟨.program ⟨214⟩, ⟨17718⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6459⟩⟩, ⟨.program ⟨214⟩, ⟨6542⟩⟩, ⟨.program ⟨214⟩, ⟨17714⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6467⟩⟩, ⟨.program ⟨214⟩, ⟨6503⟩⟩, ⟨.program ⟨214⟩, ⟨17949⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6467⟩⟩, ⟨.program ⟨214⟩, ⟨6542⟩⟩, ⟨.program ⟨214⟩, ⟨17945⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6473⟩⟩, ⟨.program ⟨214⟩, ⟨6503⟩⟩, ⟨.program ⟨214⟩, ⟨17550⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6473⟩⟩, ⟨.program ⟨214⟩, ⟨6542⟩⟩, ⟨.program ⟨214⟩, ⟨17546⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6475⟩⟩, ⟨.program ⟨214⟩, ⟨6503⟩⟩, ⟨.program ⟨214⟩, ⟨15047⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6475⟩⟩, ⟨.program ⟨214⟩, ⟨6542⟩⟩, ⟨.program ⟨214⟩, ⟨15042⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6490⟩⟩, ⟨.program ⟨214⟩, ⟨6503⟩⟩, ⟨.program ⟨214⟩, ⟨18832⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6490⟩⟩, ⟨.program ⟨214⟩, ⟨6542⟩⟩, ⟨.program ⟨214⟩, ⟨18818⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6494⟩⟩, ⟨.program ⟨214⟩, ⟨6503⟩⟩, ⟨.program ⟨214⟩, ⟨17606⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6494⟩⟩, ⟨.program ⟨214⟩, ⟨6542⟩⟩, ⟨.program ⟨214⟩, ⟨17602⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6495⟩⟩, ⟨.program ⟨214⟩, ⟨6503⟩⟩, ⟨.program ⟨214⟩, ⟨14886⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6495⟩⟩, ⟨.program ⟨214⟩, ⟨6542⟩⟩, ⟨.program ⟨214⟩, ⟨14881⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6502⟩⟩, ⟨.program ⟨214⟩, ⟨6503⟩⟩, ⟨.program ⟨214⟩, ⟨17662⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6502⟩⟩, ⟨.program ⟨214⟩, ⟨6542⟩⟩, ⟨.program ⟨214⟩, ⟨17658⟩⟩], []⟩, (1)⟩]

theorem exact5311RawTermsValid :
    exact5311RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event5311 : Event := .resultExact (⟨.program ⟨214⟩, ⟨18845⟩⟩) exact5311RawTerms (.finite 9327185996870120055146642810718267612163083545646509660275303823152964232244344818048) 5310 .exactZero (none)

def event5312 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18860⟩⟩) 0 ⟨18845⟩ 5311

def event5313 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18860⟩⟩) 1 ⟨18859⟩ 3071

def event5314 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨18860⟩⟩) (.sum [.predecessor 0 5312 .coefficient, .predecessor 1 5313 .coefficient])

def exact5315RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨6383⟩⟩, ⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨18016⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6383⟩⟩, ⟨.program ⟨214⟩, ⟨6493⟩⟩, ⟨.program ⟨214⟩, ⟨18042⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6383⟩⟩, ⟨.program ⟨214⟩, ⟨6503⟩⟩, ⟨.program ⟨214⟩, ⟨18035⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6383⟩⟩, ⟨.program ⟨214⟩, ⟨6542⟩⟩, ⟨.program ⟨214⟩, ⟨18028⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6387⟩⟩, ⟨.program ⟨214⟩, ⟨17155⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6391⟩⟩, ⟨.program ⟨214⟩, ⟨17211⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6398⟩⟩, ⟨.program ⟨214⟩, ⟨17428⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6407⟩⟩, ⟨.program ⟨214⟩, ⟨17792⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6410⟩⟩, ⟨.program ⟨214⟩, ⟨18485⟩⟩], []⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6427⟩⟩, ⟨.program ⟨214⟩, ⟨15503⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6435⟩⟩, ⟨.program ⟨214⟩, ⟨18114⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6437⟩⟩, ⟨.program ⟨214⟩, ⟨16917⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6449⟩⟩, ⟨.program ⟨214⟩, ⟨17484⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6452⟩⟩, ⟨.program ⟨214⟩, ⟨15195⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6459⟩⟩, ⟨.program ⟨214⟩, ⟨17708⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6467⟩⟩, ⟨.program ⟨214⟩, ⟨17939⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6473⟩⟩, ⟨.program ⟨214⟩, ⟨17540⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6475⟩⟩, ⟨.program ⟨214⟩, ⟨15034⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6490⟩⟩, ⟨.program ⟨214⟩, ⟨18792⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6494⟩⟩, ⟨.program ⟨214⟩, ⟨17596⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6495⟩⟩, ⟨.program ⟨214⟩, ⟨14873⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6502⟩⟩, ⟨.program ⟨214⟩, ⟨17652⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6387⟩⟩, ⟨.program ⟨214⟩, ⟨6493⟩⟩, ⟨.program ⟨214⟩, ⟨17169⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6387⟩⟩, ⟨.program ⟨214⟩, ⟨6503⟩⟩, ⟨.program ⟨214⟩, ⟨17165⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6387⟩⟩, ⟨.program ⟨214⟩, ⟨6542⟩⟩, ⟨.program ⟨214⟩, ⟨17161⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6391⟩⟩, ⟨.program ⟨214⟩, ⟨6493⟩⟩, ⟨.program ⟨214⟩, ⟨17225⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6391⟩⟩, ⟨.program ⟨214⟩, ⟨6503⟩⟩, ⟨.program ⟨214⟩, ⟨17221⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6391⟩⟩, ⟨.program ⟨214⟩, ⟨6542⟩⟩, ⟨.program ⟨214⟩, ⟨17217⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6398⟩⟩, ⟨.program ⟨214⟩, ⟨6493⟩⟩, ⟨.program ⟨214⟩, ⟨17442⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6398⟩⟩, ⟨.program ⟨214⟩, ⟨6503⟩⟩, ⟨.program ⟨214⟩, ⟨17438⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6398⟩⟩, ⟨.program ⟨214⟩, ⟨6542⟩⟩, ⟨.program ⟨214⟩, ⟨17434⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6407⟩⟩, ⟨.program ⟨214⟩, ⟨6493⟩⟩, ⟨.program ⟨214⟩, ⟨17822⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6407⟩⟩, ⟨.program ⟨214⟩, ⟨6503⟩⟩, ⟨.program ⟨214⟩, ⟨17814⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6407⟩⟩, ⟨.program ⟨214⟩, ⟨6542⟩⟩, ⟨.program ⟨214⟩, ⟨17806⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6410⟩⟩, ⟨.program ⟨214⟩, ⟨6493⟩⟩, ⟨.program ⟨214⟩, ⟨18499⟩⟩], []⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6410⟩⟩, ⟨.program ⟨214⟩, ⟨6503⟩⟩, ⟨.program ⟨214⟩, ⟨18495⟩⟩], []⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6410⟩⟩, ⟨.program ⟨214⟩, ⟨6542⟩⟩, ⟨.program ⟨214⟩, ⟨18491⟩⟩], []⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6427⟩⟩, ⟨.program ⟨214⟩, ⟨6493⟩⟩, ⟨.program ⟨214⟩, ⟨15521⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6427⟩⟩, ⟨.program ⟨214⟩, ⟨6503⟩⟩, ⟨.program ⟨214⟩, ⟨15516⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6427⟩⟩, ⟨.program ⟨214⟩, ⟨6542⟩⟩, ⟨.program ⟨214⟩, ⟨15511⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6435⟩⟩, ⟨.program ⟨214⟩, ⟨6493⟩⟩, ⟨.program ⟨214⟩, ⟨18128⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6435⟩⟩, ⟨.program ⟨214⟩, ⟨6503⟩⟩, ⟨.program ⟨214⟩, ⟨18124⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6435⟩⟩, ⟨.program ⟨214⟩, ⟨6542⟩⟩, ⟨.program ⟨214⟩, ⟨18120⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6437⟩⟩, ⟨.program ⟨214⟩, ⟨6493⟩⟩, ⟨.program ⟨214⟩, ⟨16931⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6437⟩⟩, ⟨.program ⟨214⟩, ⟨6503⟩⟩, ⟨.program ⟨214⟩, ⟨16927⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6437⟩⟩, ⟨.program ⟨214⟩, ⟨6542⟩⟩, ⟨.program ⟨214⟩, ⟨16923⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6449⟩⟩, ⟨.program ⟨214⟩, ⟨6493⟩⟩, ⟨.program ⟨214⟩, ⟨17498⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6449⟩⟩, ⟨.program ⟨214⟩, ⟨6503⟩⟩, ⟨.program ⟨214⟩, ⟨17494⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6449⟩⟩, ⟨.program ⟨214⟩, ⟨6542⟩⟩, ⟨.program ⟨214⟩, ⟨17490⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6452⟩⟩, ⟨.program ⟨214⟩, ⟨6493⟩⟩, ⟨.program ⟨214⟩, ⟨15213⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6452⟩⟩, ⟨.program ⟨214⟩, ⟨6503⟩⟩, ⟨.program ⟨214⟩, ⟨15208⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6452⟩⟩, ⟨.program ⟨214⟩, ⟨6542⟩⟩, ⟨.program ⟨214⟩, ⟨15203⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6459⟩⟩, ⟨.program ⟨214⟩, ⟨6493⟩⟩, ⟨.program ⟨214⟩, ⟨17722⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6459⟩⟩, ⟨.program ⟨214⟩, ⟨6503⟩⟩, ⟨.program ⟨214⟩, ⟨17718⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6459⟩⟩, ⟨.program ⟨214⟩, ⟨6542⟩⟩, ⟨.program ⟨214⟩, ⟨17714⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6467⟩⟩, ⟨.program ⟨214⟩, ⟨6493⟩⟩, ⟨.program ⟨214⟩, ⟨17953⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6467⟩⟩, ⟨.program ⟨214⟩, ⟨6503⟩⟩, ⟨.program ⟨214⟩, ⟨17949⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6467⟩⟩, ⟨.program ⟨214⟩, ⟨6542⟩⟩, ⟨.program ⟨214⟩, ⟨17945⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6473⟩⟩, ⟨.program ⟨214⟩, ⟨6493⟩⟩, ⟨.program ⟨214⟩, ⟨17554⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6473⟩⟩, ⟨.program ⟨214⟩, ⟨6503⟩⟩, ⟨.program ⟨214⟩, ⟨17550⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6473⟩⟩, ⟨.program ⟨214⟩, ⟨6542⟩⟩, ⟨.program ⟨214⟩, ⟨17546⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6475⟩⟩, ⟨.program ⟨214⟩, ⟨6493⟩⟩, ⟨.program ⟨214⟩, ⟨15052⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6475⟩⟩, ⟨.program ⟨214⟩, ⟨6503⟩⟩, ⟨.program ⟨214⟩, ⟨15047⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6475⟩⟩, ⟨.program ⟨214⟩, ⟨6542⟩⟩, ⟨.program ⟨214⟩, ⟨15042⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6490⟩⟩, ⟨.program ⟨214⟩, ⟨6493⟩⟩, ⟨.program ⟨214⟩, ⟨18848⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6490⟩⟩, ⟨.program ⟨214⟩, ⟨6503⟩⟩, ⟨.program ⟨214⟩, ⟨18832⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6490⟩⟩, ⟨.program ⟨214⟩, ⟨6542⟩⟩, ⟨.program ⟨214⟩, ⟨18818⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6493⟩⟩, ⟨.program ⟨214⟩, ⟨6494⟩⟩, ⟨.program ⟨214⟩, ⟨17610⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6493⟩⟩, ⟨.program ⟨214⟩, ⟨6495⟩⟩, ⟨.program ⟨214⟩, ⟨14891⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6493⟩⟩, ⟨.program ⟨214⟩, ⟨6502⟩⟩, ⟨.program ⟨214⟩, ⟨17666⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6494⟩⟩, ⟨.program ⟨214⟩, ⟨6503⟩⟩, ⟨.program ⟨214⟩, ⟨17606⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6494⟩⟩, ⟨.program ⟨214⟩, ⟨6542⟩⟩, ⟨.program ⟨214⟩, ⟨17602⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6495⟩⟩, ⟨.program ⟨214⟩, ⟨6503⟩⟩, ⟨.program ⟨214⟩, ⟨14886⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6495⟩⟩, ⟨.program ⟨214⟩, ⟨6542⟩⟩, ⟨.program ⟨214⟩, ⟨14881⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6502⟩⟩, ⟨.program ⟨214⟩, ⟨6503⟩⟩, ⟨.program ⟨214⟩, ⟨17662⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6502⟩⟩, ⟨.program ⟨214⟩, ⟨6542⟩⟩, ⟨.program ⟨214⟩, ⟨17658⟩⟩], []⟩, (1)⟩]

theorem exact5315RawTermsValid :
    exact5315RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event5315 : Event := .resultExact (⟨.program ⟨214⟩, ⟨18860⟩⟩) exact5315RawTerms (.finite 16040835534369575115870578816901818037093284701769979658184651157159460229638776911040) 5314 .exactZero (none)

def event5316 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18875⟩⟩) 0 ⟨18860⟩ 5315

def event5317 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18875⟩⟩) 1 ⟨18874⟩ 2323

def event5318 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨18875⟩⟩) (.sum [.predecessor 0 5316 .coefficient, .predecessor 1 5317 .coefficient])

def exact5319RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨6383⟩⟩, ⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨18016⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6383⟩⟩, ⟨.program ⟨214⟩, ⟨6425⟩⟩, ⟨.program ⟨214⟩, ⟨18049⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6383⟩⟩, ⟨.program ⟨214⟩, ⟨6493⟩⟩, ⟨.program ⟨214⟩, ⟨18042⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6383⟩⟩, ⟨.program ⟨214⟩, ⟨6503⟩⟩, ⟨.program ⟨214⟩, ⟨18035⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6383⟩⟩, ⟨.program ⟨214⟩, ⟨6542⟩⟩, ⟨.program ⟨214⟩, ⟨18028⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6387⟩⟩, ⟨.program ⟨214⟩, ⟨17155⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6391⟩⟩, ⟨.program ⟨214⟩, ⟨17211⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6398⟩⟩, ⟨.program ⟨214⟩, ⟨17428⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6407⟩⟩, ⟨.program ⟨214⟩, ⟨17792⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6410⟩⟩, ⟨.program ⟨214⟩, ⟨18485⟩⟩], []⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6427⟩⟩, ⟨.program ⟨214⟩, ⟨15503⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6435⟩⟩, ⟨.program ⟨214⟩, ⟨18114⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6437⟩⟩, ⟨.program ⟨214⟩, ⟨16917⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6449⟩⟩, ⟨.program ⟨214⟩, ⟨17484⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6452⟩⟩, ⟨.program ⟨214⟩, ⟨15195⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6459⟩⟩, ⟨.program ⟨214⟩, ⟨17708⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6467⟩⟩, ⟨.program ⟨214⟩, ⟨17939⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6473⟩⟩, ⟨.program ⟨214⟩, ⟨17540⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6475⟩⟩, ⟨.program ⟨214⟩, ⟨15034⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6490⟩⟩, ⟨.program ⟨214⟩, ⟨18792⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6494⟩⟩, ⟨.program ⟨214⟩, ⟨17596⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6495⟩⟩, ⟨.program ⟨214⟩, ⟨14873⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6502⟩⟩, ⟨.program ⟨214⟩, ⟨17652⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6387⟩⟩, ⟨.program ⟨214⟩, ⟨6425⟩⟩, ⟨.program ⟨214⟩, ⟨17173⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6387⟩⟩, ⟨.program ⟨214⟩, ⟨6493⟩⟩, ⟨.program ⟨214⟩, ⟨17169⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6387⟩⟩, ⟨.program ⟨214⟩, ⟨6503⟩⟩, ⟨.program ⟨214⟩, ⟨17165⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6387⟩⟩, ⟨.program ⟨214⟩, ⟨6542⟩⟩, ⟨.program ⟨214⟩, ⟨17161⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6391⟩⟩, ⟨.program ⟨214⟩, ⟨6425⟩⟩, ⟨.program ⟨214⟩, ⟨17229⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6391⟩⟩, ⟨.program ⟨214⟩, ⟨6493⟩⟩, ⟨.program ⟨214⟩, ⟨17225⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6391⟩⟩, ⟨.program ⟨214⟩, ⟨6503⟩⟩, ⟨.program ⟨214⟩, ⟨17221⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6391⟩⟩, ⟨.program ⟨214⟩, ⟨6542⟩⟩, ⟨.program ⟨214⟩, ⟨17217⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6398⟩⟩, ⟨.program ⟨214⟩, ⟨6425⟩⟩, ⟨.program ⟨214⟩, ⟨17446⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6398⟩⟩, ⟨.program ⟨214⟩, ⟨6493⟩⟩, ⟨.program ⟨214⟩, ⟨17442⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6398⟩⟩, ⟨.program ⟨214⟩, ⟨6503⟩⟩, ⟨.program ⟨214⟩, ⟨17438⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6398⟩⟩, ⟨.program ⟨214⟩, ⟨6542⟩⟩, ⟨.program ⟨214⟩, ⟨17434⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6407⟩⟩, ⟨.program ⟨214⟩, ⟨6425⟩⟩, ⟨.program ⟨214⟩, ⟨17830⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6407⟩⟩, ⟨.program ⟨214⟩, ⟨6493⟩⟩, ⟨.program ⟨214⟩, ⟨17822⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6407⟩⟩, ⟨.program ⟨214⟩, ⟨6503⟩⟩, ⟨.program ⟨214⟩, ⟨17814⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6407⟩⟩, ⟨.program ⟨214⟩, ⟨6542⟩⟩, ⟨.program ⟨214⟩, ⟨17806⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6410⟩⟩, ⟨.program ⟨214⟩, ⟨6425⟩⟩, ⟨.program ⟨214⟩, ⟨18503⟩⟩], []⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6410⟩⟩, ⟨.program ⟨214⟩, ⟨6493⟩⟩, ⟨.program ⟨214⟩, ⟨18499⟩⟩], []⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6410⟩⟩, ⟨.program ⟨214⟩, ⟨6503⟩⟩, ⟨.program ⟨214⟩, ⟨18495⟩⟩], []⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6410⟩⟩, ⟨.program ⟨214⟩, ⟨6542⟩⟩, ⟨.program ⟨214⟩, ⟨18491⟩⟩], []⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6425⟩⟩, ⟨.program ⟨214⟩, ⟨6427⟩⟩, ⟨.program ⟨214⟩, ⟨15526⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6425⟩⟩, ⟨.program ⟨214⟩, ⟨6435⟩⟩, ⟨.program ⟨214⟩, ⟨18132⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6425⟩⟩, ⟨.program ⟨214⟩, ⟨6437⟩⟩, ⟨.program ⟨214⟩, ⟨16935⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6425⟩⟩, ⟨.program ⟨214⟩, ⟨6449⟩⟩, ⟨.program ⟨214⟩, ⟨17502⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6425⟩⟩, ⟨.program ⟨214⟩, ⟨6452⟩⟩, ⟨.program ⟨214⟩, ⟨15218⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6425⟩⟩, ⟨.program ⟨214⟩, ⟨6459⟩⟩, ⟨.program ⟨214⟩, ⟨17726⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6425⟩⟩, ⟨.program ⟨214⟩, ⟨6467⟩⟩, ⟨.program ⟨214⟩, ⟨17957⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6425⟩⟩, ⟨.program ⟨214⟩, ⟨6473⟩⟩, ⟨.program ⟨214⟩, ⟨17558⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6425⟩⟩, ⟨.program ⟨214⟩, ⟨6475⟩⟩, ⟨.program ⟨214⟩, ⟨15057⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6425⟩⟩, ⟨.program ⟨214⟩, ⟨6490⟩⟩, ⟨.program ⟨214⟩, ⟨18863⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6425⟩⟩, ⟨.program ⟨214⟩, ⟨6494⟩⟩, ⟨.program ⟨214⟩, ⟨17614⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6425⟩⟩, ⟨.program ⟨214⟩, ⟨6495⟩⟩, ⟨.program ⟨214⟩, ⟨14896⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6425⟩⟩, ⟨.program ⟨214⟩, ⟨6502⟩⟩, ⟨.program ⟨214⟩, ⟨17670⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6427⟩⟩, ⟨.program ⟨214⟩, ⟨6493⟩⟩, ⟨.program ⟨214⟩, ⟨15521⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6427⟩⟩, ⟨.program ⟨214⟩, ⟨6503⟩⟩, ⟨.program ⟨214⟩, ⟨15516⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6427⟩⟩, ⟨.program ⟨214⟩, ⟨6542⟩⟩, ⟨.program ⟨214⟩, ⟨15511⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6435⟩⟩, ⟨.program ⟨214⟩, ⟨6493⟩⟩, ⟨.program ⟨214⟩, ⟨18128⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6435⟩⟩, ⟨.program ⟨214⟩, ⟨6503⟩⟩, ⟨.program ⟨214⟩, ⟨18124⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6435⟩⟩, ⟨.program ⟨214⟩, ⟨6542⟩⟩, ⟨.program ⟨214⟩, ⟨18120⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6437⟩⟩, ⟨.program ⟨214⟩, ⟨6493⟩⟩, ⟨.program ⟨214⟩, ⟨16931⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6437⟩⟩, ⟨.program ⟨214⟩, ⟨6503⟩⟩, ⟨.program ⟨214⟩, ⟨16927⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6437⟩⟩, ⟨.program ⟨214⟩, ⟨6542⟩⟩, ⟨.program ⟨214⟩, ⟨16923⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6449⟩⟩, ⟨.program ⟨214⟩, ⟨6493⟩⟩, ⟨.program ⟨214⟩, ⟨17498⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6449⟩⟩, ⟨.program ⟨214⟩, ⟨6503⟩⟩, ⟨.program ⟨214⟩, ⟨17494⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6449⟩⟩, ⟨.program ⟨214⟩, ⟨6542⟩⟩, ⟨.program ⟨214⟩, ⟨17490⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6452⟩⟩, ⟨.program ⟨214⟩, ⟨6493⟩⟩, ⟨.program ⟨214⟩, ⟨15213⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6452⟩⟩, ⟨.program ⟨214⟩, ⟨6503⟩⟩, ⟨.program ⟨214⟩, ⟨15208⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6452⟩⟩, ⟨.program ⟨214⟩, ⟨6542⟩⟩, ⟨.program ⟨214⟩, ⟨15203⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6459⟩⟩, ⟨.program ⟨214⟩, ⟨6493⟩⟩, ⟨.program ⟨214⟩, ⟨17722⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6459⟩⟩, ⟨.program ⟨214⟩, ⟨6503⟩⟩, ⟨.program ⟨214⟩, ⟨17718⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6459⟩⟩, ⟨.program ⟨214⟩, ⟨6542⟩⟩, ⟨.program ⟨214⟩, ⟨17714⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6467⟩⟩, ⟨.program ⟨214⟩, ⟨6493⟩⟩, ⟨.program ⟨214⟩, ⟨17953⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6467⟩⟩, ⟨.program ⟨214⟩, ⟨6503⟩⟩, ⟨.program ⟨214⟩, ⟨17949⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6467⟩⟩, ⟨.program ⟨214⟩, ⟨6542⟩⟩, ⟨.program ⟨214⟩, ⟨17945⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6473⟩⟩, ⟨.program ⟨214⟩, ⟨6493⟩⟩, ⟨.program ⟨214⟩, ⟨17554⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6473⟩⟩, ⟨.program ⟨214⟩, ⟨6503⟩⟩, ⟨.program ⟨214⟩, ⟨17550⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6473⟩⟩, ⟨.program ⟨214⟩, ⟨6542⟩⟩, ⟨.program ⟨214⟩, ⟨17546⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6475⟩⟩, ⟨.program ⟨214⟩, ⟨6493⟩⟩, ⟨.program ⟨214⟩, ⟨15052⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6475⟩⟩, ⟨.program ⟨214⟩, ⟨6503⟩⟩, ⟨.program ⟨214⟩, ⟨15047⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6475⟩⟩, ⟨.program ⟨214⟩, ⟨6542⟩⟩, ⟨.program ⟨214⟩, ⟨15042⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6490⟩⟩, ⟨.program ⟨214⟩, ⟨6493⟩⟩, ⟨.program ⟨214⟩, ⟨18848⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6490⟩⟩, ⟨.program ⟨214⟩, ⟨6503⟩⟩, ⟨.program ⟨214⟩, ⟨18832⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6490⟩⟩, ⟨.program ⟨214⟩, ⟨6542⟩⟩, ⟨.program ⟨214⟩, ⟨18818⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6493⟩⟩, ⟨.program ⟨214⟩, ⟨6494⟩⟩, ⟨.program ⟨214⟩, ⟨17610⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6493⟩⟩, ⟨.program ⟨214⟩, ⟨6495⟩⟩, ⟨.program ⟨214⟩, ⟨14891⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6493⟩⟩, ⟨.program ⟨214⟩, ⟨6502⟩⟩, ⟨.program ⟨214⟩, ⟨17666⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6494⟩⟩, ⟨.program ⟨214⟩, ⟨6503⟩⟩, ⟨.program ⟨214⟩, ⟨17606⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6494⟩⟩, ⟨.program ⟨214⟩, ⟨6542⟩⟩, ⟨.program ⟨214⟩, ⟨17602⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6495⟩⟩, ⟨.program ⟨214⟩, ⟨6503⟩⟩, ⟨.program ⟨214⟩, ⟨14886⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6495⟩⟩, ⟨.program ⟨214⟩, ⟨6542⟩⟩, ⟨.program ⟨214⟩, ⟨14881⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6502⟩⟩, ⟨.program ⟨214⟩, ⟨6503⟩⟩, ⟨.program ⟨214⟩, ⟨17662⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6502⟩⟩, ⟨.program ⟨214⟩, ⟨6542⟩⟩, ⟨.program ⟨214⟩, ⟨17658⟩⟩], []⟩, (1)⟩]

theorem exact5319RawTermsValid :
    exact5319RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event5319 : Event := .resultExact (⟨.program ⟨214⟩, ⟨18875⟩⟩) exact5319RawTerms (.finite 24371812142836559120194421704797166357550217236588177560936837874198303656513935588064) 5318 .exactZero (none)

def event5320 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18890⟩⟩) 0 ⟨18875⟩ 5319

def event5321 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18890⟩⟩) 1 ⟨18889⟩ 1575

def event5322 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨18890⟩⟩) (.sum [.predecessor 0 5320 .coefficient, .predecessor 1 5321 .coefficient])

def exact5323RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨6383⟩⟩, ⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨18016⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6383⟩⟩, ⟨.program ⟨214⟩, ⟨6425⟩⟩, ⟨.program ⟨214⟩, ⟨18049⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6383⟩⟩, ⟨.program ⟨214⟩, ⟨6493⟩⟩, ⟨.program ⟨214⟩, ⟨18042⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6383⟩⟩, ⟨.program ⟨214⟩, ⟨6503⟩⟩, ⟨.program ⟨214⟩, ⟨18035⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6383⟩⟩, ⟨.program ⟨214⟩, ⟨6542⟩⟩, ⟨.program ⟨214⟩, ⟨18028⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6383⟩⟩, ⟨.program ⟨214⟩, ⟨6543⟩⟩, ⟨.program ⟨214⟩, ⟨18056⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6387⟩⟩, ⟨.program ⟨214⟩, ⟨17155⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6391⟩⟩, ⟨.program ⟨214⟩, ⟨17211⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6398⟩⟩, ⟨.program ⟨214⟩, ⟨17428⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6407⟩⟩, ⟨.program ⟨214⟩, ⟨17792⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6410⟩⟩, ⟨.program ⟨214⟩, ⟨18485⟩⟩], []⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6427⟩⟩, ⟨.program ⟨214⟩, ⟨15503⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6435⟩⟩, ⟨.program ⟨214⟩, ⟨18114⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6437⟩⟩, ⟨.program ⟨214⟩, ⟨16917⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6449⟩⟩, ⟨.program ⟨214⟩, ⟨17484⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6452⟩⟩, ⟨.program ⟨214⟩, ⟨15195⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6459⟩⟩, ⟨.program ⟨214⟩, ⟨17708⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6467⟩⟩, ⟨.program ⟨214⟩, ⟨17939⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6473⟩⟩, ⟨.program ⟨214⟩, ⟨17540⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6475⟩⟩, ⟨.program ⟨214⟩, ⟨15034⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6490⟩⟩, ⟨.program ⟨214⟩, ⟨18792⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6494⟩⟩, ⟨.program ⟨214⟩, ⟨17596⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6495⟩⟩, ⟨.program ⟨214⟩, ⟨14873⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6502⟩⟩, ⟨.program ⟨214⟩, ⟨17652⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6387⟩⟩, ⟨.program ⟨214⟩, ⟨6425⟩⟩, ⟨.program ⟨214⟩, ⟨17173⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6387⟩⟩, ⟨.program ⟨214⟩, ⟨6493⟩⟩, ⟨.program ⟨214⟩, ⟨17169⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6387⟩⟩, ⟨.program ⟨214⟩, ⟨6503⟩⟩, ⟨.program ⟨214⟩, ⟨17165⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6387⟩⟩, ⟨.program ⟨214⟩, ⟨6542⟩⟩, ⟨.program ⟨214⟩, ⟨17161⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6387⟩⟩, ⟨.program ⟨214⟩, ⟨6543⟩⟩, ⟨.program ⟨214⟩, ⟨17177⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6391⟩⟩, ⟨.program ⟨214⟩, ⟨6425⟩⟩, ⟨.program ⟨214⟩, ⟨17229⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6391⟩⟩, ⟨.program ⟨214⟩, ⟨6493⟩⟩, ⟨.program ⟨214⟩, ⟨17225⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6391⟩⟩, ⟨.program ⟨214⟩, ⟨6503⟩⟩, ⟨.program ⟨214⟩, ⟨17221⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6391⟩⟩, ⟨.program ⟨214⟩, ⟨6542⟩⟩, ⟨.program ⟨214⟩, ⟨17217⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6391⟩⟩, ⟨.program ⟨214⟩, ⟨6543⟩⟩, ⟨.program ⟨214⟩, ⟨17233⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6398⟩⟩, ⟨.program ⟨214⟩, ⟨6425⟩⟩, ⟨.program ⟨214⟩, ⟨17446⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6398⟩⟩, ⟨.program ⟨214⟩, ⟨6493⟩⟩, ⟨.program ⟨214⟩, ⟨17442⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6398⟩⟩, ⟨.program ⟨214⟩, ⟨6503⟩⟩, ⟨.program ⟨214⟩, ⟨17438⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6398⟩⟩, ⟨.program ⟨214⟩, ⟨6542⟩⟩, ⟨.program ⟨214⟩, ⟨17434⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6398⟩⟩, ⟨.program ⟨214⟩, ⟨6543⟩⟩, ⟨.program ⟨214⟩, ⟨17450⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6407⟩⟩, ⟨.program ⟨214⟩, ⟨6425⟩⟩, ⟨.program ⟨214⟩, ⟨17830⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6407⟩⟩, ⟨.program ⟨214⟩, ⟨6493⟩⟩, ⟨.program ⟨214⟩, ⟨17822⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6407⟩⟩, ⟨.program ⟨214⟩, ⟨6503⟩⟩, ⟨.program ⟨214⟩, ⟨17814⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6407⟩⟩, ⟨.program ⟨214⟩, ⟨6542⟩⟩, ⟨.program ⟨214⟩, ⟨17806⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6407⟩⟩, ⟨.program ⟨214⟩, ⟨6543⟩⟩, ⟨.program ⟨214⟩, ⟨17838⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6410⟩⟩, ⟨.program ⟨214⟩, ⟨6425⟩⟩, ⟨.program ⟨214⟩, ⟨18503⟩⟩], []⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6410⟩⟩, ⟨.program ⟨214⟩, ⟨6493⟩⟩, ⟨.program ⟨214⟩, ⟨18499⟩⟩], []⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6410⟩⟩, ⟨.program ⟨214⟩, ⟨6503⟩⟩, ⟨.program ⟨214⟩, ⟨18495⟩⟩], []⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6410⟩⟩, ⟨.program ⟨214⟩, ⟨6542⟩⟩, ⟨.program ⟨214⟩, ⟨18491⟩⟩], []⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6410⟩⟩, ⟨.program ⟨214⟩, ⟨6543⟩⟩, ⟨.program ⟨214⟩, ⟨18507⟩⟩], []⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6425⟩⟩, ⟨.program ⟨214⟩, ⟨6427⟩⟩, ⟨.program ⟨214⟩, ⟨15526⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6425⟩⟩, ⟨.program ⟨214⟩, ⟨6435⟩⟩, ⟨.program ⟨214⟩, ⟨18132⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6425⟩⟩, ⟨.program ⟨214⟩, ⟨6437⟩⟩, ⟨.program ⟨214⟩, ⟨16935⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6425⟩⟩, ⟨.program ⟨214⟩, ⟨6449⟩⟩, ⟨.program ⟨214⟩, ⟨17502⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6425⟩⟩, ⟨.program ⟨214⟩, ⟨6452⟩⟩, ⟨.program ⟨214⟩, ⟨15218⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6425⟩⟩, ⟨.program ⟨214⟩, ⟨6459⟩⟩, ⟨.program ⟨214⟩, ⟨17726⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6425⟩⟩, ⟨.program ⟨214⟩, ⟨6467⟩⟩, ⟨.program ⟨214⟩, ⟨17957⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6425⟩⟩, ⟨.program ⟨214⟩, ⟨6473⟩⟩, ⟨.program ⟨214⟩, ⟨17558⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6425⟩⟩, ⟨.program ⟨214⟩, ⟨6475⟩⟩, ⟨.program ⟨214⟩, ⟨15057⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6425⟩⟩, ⟨.program ⟨214⟩, ⟨6490⟩⟩, ⟨.program ⟨214⟩, ⟨18863⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6425⟩⟩, ⟨.program ⟨214⟩, ⟨6494⟩⟩, ⟨.program ⟨214⟩, ⟨17614⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6425⟩⟩, ⟨.program ⟨214⟩, ⟨6495⟩⟩, ⟨.program ⟨214⟩, ⟨14896⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6425⟩⟩, ⟨.program ⟨214⟩, ⟨6502⟩⟩, ⟨.program ⟨214⟩, ⟨17670⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6427⟩⟩, ⟨.program ⟨214⟩, ⟨6493⟩⟩, ⟨.program ⟨214⟩, ⟨15521⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6427⟩⟩, ⟨.program ⟨214⟩, ⟨6503⟩⟩, ⟨.program ⟨214⟩, ⟨15516⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6427⟩⟩, ⟨.program ⟨214⟩, ⟨6542⟩⟩, ⟨.program ⟨214⟩, ⟨15511⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6427⟩⟩, ⟨.program ⟨214⟩, ⟨6543⟩⟩, ⟨.program ⟨214⟩, ⟨15531⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6435⟩⟩, ⟨.program ⟨214⟩, ⟨6493⟩⟩, ⟨.program ⟨214⟩, ⟨18128⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6435⟩⟩, ⟨.program ⟨214⟩, ⟨6503⟩⟩, ⟨.program ⟨214⟩, ⟨18124⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6435⟩⟩, ⟨.program ⟨214⟩, ⟨6542⟩⟩, ⟨.program ⟨214⟩, ⟨18120⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6435⟩⟩, ⟨.program ⟨214⟩, ⟨6543⟩⟩, ⟨.program ⟨214⟩, ⟨18136⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6437⟩⟩, ⟨.program ⟨214⟩, ⟨6493⟩⟩, ⟨.program ⟨214⟩, ⟨16931⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6437⟩⟩, ⟨.program ⟨214⟩, ⟨6503⟩⟩, ⟨.program ⟨214⟩, ⟨16927⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6437⟩⟩, ⟨.program ⟨214⟩, ⟨6542⟩⟩, ⟨.program ⟨214⟩, ⟨16923⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6437⟩⟩, ⟨.program ⟨214⟩, ⟨6543⟩⟩, ⟨.program ⟨214⟩, ⟨16939⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6449⟩⟩, ⟨.program ⟨214⟩, ⟨6493⟩⟩, ⟨.program ⟨214⟩, ⟨17498⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6449⟩⟩, ⟨.program ⟨214⟩, ⟨6503⟩⟩, ⟨.program ⟨214⟩, ⟨17494⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6449⟩⟩, ⟨.program ⟨214⟩, ⟨6542⟩⟩, ⟨.program ⟨214⟩, ⟨17490⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6449⟩⟩, ⟨.program ⟨214⟩, ⟨6543⟩⟩, ⟨.program ⟨214⟩, ⟨17506⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6452⟩⟩, ⟨.program ⟨214⟩, ⟨6493⟩⟩, ⟨.program ⟨214⟩, ⟨15213⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6452⟩⟩, ⟨.program ⟨214⟩, ⟨6503⟩⟩, ⟨.program ⟨214⟩, ⟨15208⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6452⟩⟩, ⟨.program ⟨214⟩, ⟨6542⟩⟩, ⟨.program ⟨214⟩, ⟨15203⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6452⟩⟩, ⟨.program ⟨214⟩, ⟨6543⟩⟩, ⟨.program ⟨214⟩, ⟨15223⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6459⟩⟩, ⟨.program ⟨214⟩, ⟨6493⟩⟩, ⟨.program ⟨214⟩, ⟨17722⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6459⟩⟩, ⟨.program ⟨214⟩, ⟨6503⟩⟩, ⟨.program ⟨214⟩, ⟨17718⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6459⟩⟩, ⟨.program ⟨214⟩, ⟨6542⟩⟩, ⟨.program ⟨214⟩, ⟨17714⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6459⟩⟩, ⟨.program ⟨214⟩, ⟨6543⟩⟩, ⟨.program ⟨214⟩, ⟨17730⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6467⟩⟩, ⟨.program ⟨214⟩, ⟨6493⟩⟩, ⟨.program ⟨214⟩, ⟨17953⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6467⟩⟩, ⟨.program ⟨214⟩, ⟨6503⟩⟩, ⟨.program ⟨214⟩, ⟨17949⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6467⟩⟩, ⟨.program ⟨214⟩, ⟨6542⟩⟩, ⟨.program ⟨214⟩, ⟨17945⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6467⟩⟩, ⟨.program ⟨214⟩, ⟨6543⟩⟩, ⟨.program ⟨214⟩, ⟨17961⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6473⟩⟩, ⟨.program ⟨214⟩, ⟨6493⟩⟩, ⟨.program ⟨214⟩, ⟨17554⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6473⟩⟩, ⟨.program ⟨214⟩, ⟨6503⟩⟩, ⟨.program ⟨214⟩, ⟨17550⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6473⟩⟩, ⟨.program ⟨214⟩, ⟨6542⟩⟩, ⟨.program ⟨214⟩, ⟨17546⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6473⟩⟩, ⟨.program ⟨214⟩, ⟨6543⟩⟩, ⟨.program ⟨214⟩, ⟨17562⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6475⟩⟩, ⟨.program ⟨214⟩, ⟨6493⟩⟩, ⟨.program ⟨214⟩, ⟨15052⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6475⟩⟩, ⟨.program ⟨214⟩, ⟨6503⟩⟩, ⟨.program ⟨214⟩, ⟨15047⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6475⟩⟩, ⟨.program ⟨214⟩, ⟨6542⟩⟩, ⟨.program ⟨214⟩, ⟨15042⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6475⟩⟩, ⟨.program ⟨214⟩, ⟨6543⟩⟩, ⟨.program ⟨214⟩, ⟨15062⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6490⟩⟩, ⟨.program ⟨214⟩, ⟨6493⟩⟩, ⟨.program ⟨214⟩, ⟨18848⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6490⟩⟩, ⟨.program ⟨214⟩, ⟨6503⟩⟩, ⟨.program ⟨214⟩, ⟨18832⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6490⟩⟩, ⟨.program ⟨214⟩, ⟨6542⟩⟩, ⟨.program ⟨214⟩, ⟨18818⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6490⟩⟩, ⟨.program ⟨214⟩, ⟨6543⟩⟩, ⟨.program ⟨214⟩, ⟨18878⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6493⟩⟩, ⟨.program ⟨214⟩, ⟨6494⟩⟩, ⟨.program ⟨214⟩, ⟨17610⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6493⟩⟩, ⟨.program ⟨214⟩, ⟨6495⟩⟩, ⟨.program ⟨214⟩, ⟨14891⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6493⟩⟩, ⟨.program ⟨214⟩, ⟨6502⟩⟩, ⟨.program ⟨214⟩, ⟨17666⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6494⟩⟩, ⟨.program ⟨214⟩, ⟨6503⟩⟩, ⟨.program ⟨214⟩, ⟨17606⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6494⟩⟩, ⟨.program ⟨214⟩, ⟨6542⟩⟩, ⟨.program ⟨214⟩, ⟨17602⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6494⟩⟩, ⟨.program ⟨214⟩, ⟨6543⟩⟩, ⟨.program ⟨214⟩, ⟨17618⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6495⟩⟩, ⟨.program ⟨214⟩, ⟨6503⟩⟩, ⟨.program ⟨214⟩, ⟨14886⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6495⟩⟩, ⟨.program ⟨214⟩, ⟨6542⟩⟩, ⟨.program ⟨214⟩, ⟨14881⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6495⟩⟩, ⟨.program ⟨214⟩, ⟨6543⟩⟩, ⟨.program ⟨214⟩, ⟨14901⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6502⟩⟩, ⟨.program ⟨214⟩, ⟨6503⟩⟩, ⟨.program ⟨214⟩, ⟨17662⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6502⟩⟩, ⟨.program ⟨214⟩, ⟨6542⟩⟩, ⟨.program ⟨214⟩, ⟨17658⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6502⟩⟩, ⟨.program ⟨214⟩, ⟨6543⟩⟩, ⟨.program ⟨214⟩, ⟨17674⟩⟩], []⟩, (1)⟩]

theorem exact5323RawTermsValid :
    exact5323RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event5323 : Event := .resultExact (⟨.program ⟨214⟩, ⟨18890⟩⟩) exact5323RawTerms (.finite 26793426257238540784008937317309993308450690962494122927993413505751711810951227632256) 5322 .exactZero (none)

def event5324 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18905⟩⟩) 0 ⟨18890⟩ 5323

def event5325 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18905⟩⟩) 1 ⟨18904⟩ 827

def event5326 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨18905⟩⟩) (.sum [.predecessor 0 5324 .coefficient, .predecessor 1 5325 .coefficient])

def exact5327RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨6383⟩⟩, ⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨18016⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6383⟩⟩, ⟨.program ⟨214⟩, ⟨6396⟩⟩, ⟨.program ⟨214⟩, ⟨18063⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6383⟩⟩, ⟨.program ⟨214⟩, ⟨6425⟩⟩, ⟨.program ⟨214⟩, ⟨18049⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6383⟩⟩, ⟨.program ⟨214⟩, ⟨6493⟩⟩, ⟨.program ⟨214⟩, ⟨18042⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6383⟩⟩, ⟨.program ⟨214⟩, ⟨6503⟩⟩, ⟨.program ⟨214⟩, ⟨18035⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6383⟩⟩, ⟨.program ⟨214⟩, ⟨6542⟩⟩, ⟨.program ⟨214⟩, ⟨18028⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6383⟩⟩, ⟨.program ⟨214⟩, ⟨6543⟩⟩, ⟨.program ⟨214⟩, ⟨18056⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6387⟩⟩, ⟨.program ⟨214⟩, ⟨17155⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6391⟩⟩, ⟨.program ⟨214⟩, ⟨17211⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6398⟩⟩, ⟨.program ⟨214⟩, ⟨17428⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6407⟩⟩, ⟨.program ⟨214⟩, ⟨17792⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6410⟩⟩, ⟨.program ⟨214⟩, ⟨18485⟩⟩], []⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6427⟩⟩, ⟨.program ⟨214⟩, ⟨15503⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6435⟩⟩, ⟨.program ⟨214⟩, ⟨18114⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6437⟩⟩, ⟨.program ⟨214⟩, ⟨16917⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6449⟩⟩, ⟨.program ⟨214⟩, ⟨17484⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6452⟩⟩, ⟨.program ⟨214⟩, ⟨15195⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6459⟩⟩, ⟨.program ⟨214⟩, ⟨17708⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6467⟩⟩, ⟨.program ⟨214⟩, ⟨17939⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6473⟩⟩, ⟨.program ⟨214⟩, ⟨17540⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6475⟩⟩, ⟨.program ⟨214⟩, ⟨15034⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6490⟩⟩, ⟨.program ⟨214⟩, ⟨18792⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6494⟩⟩, ⟨.program ⟨214⟩, ⟨17596⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6495⟩⟩, ⟨.program ⟨214⟩, ⟨14873⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6384⟩⟩, ⟨.program ⟨214⟩, ⟨6502⟩⟩, ⟨.program ⟨214⟩, ⟨17652⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6387⟩⟩, ⟨.program ⟨214⟩, ⟨6396⟩⟩, ⟨.program ⟨214⟩, ⟨17181⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6387⟩⟩, ⟨.program ⟨214⟩, ⟨6425⟩⟩, ⟨.program ⟨214⟩, ⟨17173⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6387⟩⟩, ⟨.program ⟨214⟩, ⟨6493⟩⟩, ⟨.program ⟨214⟩, ⟨17169⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6387⟩⟩, ⟨.program ⟨214⟩, ⟨6503⟩⟩, ⟨.program ⟨214⟩, ⟨17165⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6387⟩⟩, ⟨.program ⟨214⟩, ⟨6542⟩⟩, ⟨.program ⟨214⟩, ⟨17161⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6387⟩⟩, ⟨.program ⟨214⟩, ⟨6543⟩⟩, ⟨.program ⟨214⟩, ⟨17177⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6391⟩⟩, ⟨.program ⟨214⟩, ⟨6396⟩⟩, ⟨.program ⟨214⟩, ⟨17237⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6391⟩⟩, ⟨.program ⟨214⟩, ⟨6425⟩⟩, ⟨.program ⟨214⟩, ⟨17229⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6391⟩⟩, ⟨.program ⟨214⟩, ⟨6493⟩⟩, ⟨.program ⟨214⟩, ⟨17225⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6391⟩⟩, ⟨.program ⟨214⟩, ⟨6503⟩⟩, ⟨.program ⟨214⟩, ⟨17221⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6391⟩⟩, ⟨.program ⟨214⟩, ⟨6542⟩⟩, ⟨.program ⟨214⟩, ⟨17217⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6391⟩⟩, ⟨.program ⟨214⟩, ⟨6543⟩⟩, ⟨.program ⟨214⟩, ⟨17233⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6396⟩⟩, ⟨.program ⟨214⟩, ⟨6398⟩⟩, ⟨.program ⟨214⟩, ⟨17454⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6396⟩⟩, ⟨.program ⟨214⟩, ⟨6407⟩⟩, ⟨.program ⟨214⟩, ⟨17846⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6396⟩⟩, ⟨.program ⟨214⟩, ⟨6410⟩⟩, ⟨.program ⟨214⟩, ⟨18511⟩⟩], []⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6396⟩⟩, ⟨.program ⟨214⟩, ⟨6427⟩⟩, ⟨.program ⟨214⟩, ⟨15536⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6396⟩⟩, ⟨.program ⟨214⟩, ⟨6435⟩⟩, ⟨.program ⟨214⟩, ⟨18140⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6396⟩⟩, ⟨.program ⟨214⟩, ⟨6437⟩⟩, ⟨.program ⟨214⟩, ⟨16943⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6396⟩⟩, ⟨.program ⟨214⟩, ⟨6449⟩⟩, ⟨.program ⟨214⟩, ⟨17510⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6396⟩⟩, ⟨.program ⟨214⟩, ⟨6452⟩⟩, ⟨.program ⟨214⟩, ⟨15228⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6396⟩⟩, ⟨.program ⟨214⟩, ⟨6459⟩⟩, ⟨.program ⟨214⟩, ⟨17734⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6396⟩⟩, ⟨.program ⟨214⟩, ⟨6467⟩⟩, ⟨.program ⟨214⟩, ⟨17965⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6396⟩⟩, ⟨.program ⟨214⟩, ⟨6473⟩⟩, ⟨.program ⟨214⟩, ⟨17566⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6396⟩⟩, ⟨.program ⟨214⟩, ⟨6475⟩⟩, ⟨.program ⟨214⟩, ⟨15067⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6396⟩⟩, ⟨.program ⟨214⟩, ⟨6490⟩⟩, ⟨.program ⟨214⟩, ⟨18893⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6396⟩⟩, ⟨.program ⟨214⟩, ⟨6494⟩⟩, ⟨.program ⟨214⟩, ⟨17622⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6396⟩⟩, ⟨.program ⟨214⟩, ⟨6495⟩⟩, ⟨.program ⟨214⟩, ⟨14906⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6396⟩⟩, ⟨.program ⟨214⟩, ⟨6502⟩⟩, ⟨.program ⟨214⟩, ⟨17678⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6398⟩⟩, ⟨.program ⟨214⟩, ⟨6425⟩⟩, ⟨.program ⟨214⟩, ⟨17446⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6398⟩⟩, ⟨.program ⟨214⟩, ⟨6493⟩⟩, ⟨.program ⟨214⟩, ⟨17442⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6398⟩⟩, ⟨.program ⟨214⟩, ⟨6503⟩⟩, ⟨.program ⟨214⟩, ⟨17438⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6398⟩⟩, ⟨.program ⟨214⟩, ⟨6542⟩⟩, ⟨.program ⟨214⟩, ⟨17434⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6398⟩⟩, ⟨.program ⟨214⟩, ⟨6543⟩⟩, ⟨.program ⟨214⟩, ⟨17450⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6407⟩⟩, ⟨.program ⟨214⟩, ⟨6425⟩⟩, ⟨.program ⟨214⟩, ⟨17830⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6407⟩⟩, ⟨.program ⟨214⟩, ⟨6493⟩⟩, ⟨.program ⟨214⟩, ⟨17822⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6407⟩⟩, ⟨.program ⟨214⟩, ⟨6503⟩⟩, ⟨.program ⟨214⟩, ⟨17814⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6407⟩⟩, ⟨.program ⟨214⟩, ⟨6542⟩⟩, ⟨.program ⟨214⟩, ⟨17806⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6407⟩⟩, ⟨.program ⟨214⟩, ⟨6543⟩⟩, ⟨.program ⟨214⟩, ⟨17838⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6410⟩⟩, ⟨.program ⟨214⟩, ⟨6425⟩⟩, ⟨.program ⟨214⟩, ⟨18503⟩⟩], []⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6410⟩⟩, ⟨.program ⟨214⟩, ⟨6493⟩⟩, ⟨.program ⟨214⟩, ⟨18499⟩⟩], []⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6410⟩⟩, ⟨.program ⟨214⟩, ⟨6503⟩⟩, ⟨.program ⟨214⟩, ⟨18495⟩⟩], []⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6410⟩⟩, ⟨.program ⟨214⟩, ⟨6542⟩⟩, ⟨.program ⟨214⟩, ⟨18491⟩⟩], []⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6410⟩⟩, ⟨.program ⟨214⟩, ⟨6543⟩⟩, ⟨.program ⟨214⟩, ⟨18507⟩⟩], []⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6425⟩⟩, ⟨.program ⟨214⟩, ⟨6427⟩⟩, ⟨.program ⟨214⟩, ⟨15526⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6425⟩⟩, ⟨.program ⟨214⟩, ⟨6435⟩⟩, ⟨.program ⟨214⟩, ⟨18132⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6425⟩⟩, ⟨.program ⟨214⟩, ⟨6437⟩⟩, ⟨.program ⟨214⟩, ⟨16935⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6425⟩⟩, ⟨.program ⟨214⟩, ⟨6449⟩⟩, ⟨.program ⟨214⟩, ⟨17502⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6425⟩⟩, ⟨.program ⟨214⟩, ⟨6452⟩⟩, ⟨.program ⟨214⟩, ⟨15218⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6425⟩⟩, ⟨.program ⟨214⟩, ⟨6459⟩⟩, ⟨.program ⟨214⟩, ⟨17726⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6425⟩⟩, ⟨.program ⟨214⟩, ⟨6467⟩⟩, ⟨.program ⟨214⟩, ⟨17957⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6425⟩⟩, ⟨.program ⟨214⟩, ⟨6473⟩⟩, ⟨.program ⟨214⟩, ⟨17558⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6425⟩⟩, ⟨.program ⟨214⟩, ⟨6475⟩⟩, ⟨.program ⟨214⟩, ⟨15057⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6425⟩⟩, ⟨.program ⟨214⟩, ⟨6490⟩⟩, ⟨.program ⟨214⟩, ⟨18863⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6425⟩⟩, ⟨.program ⟨214⟩, ⟨6494⟩⟩, ⟨.program ⟨214⟩, ⟨17614⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6425⟩⟩, ⟨.program ⟨214⟩, ⟨6495⟩⟩, ⟨.program ⟨214⟩, ⟨14896⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6425⟩⟩, ⟨.program ⟨214⟩, ⟨6502⟩⟩, ⟨.program ⟨214⟩, ⟨17670⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6427⟩⟩, ⟨.program ⟨214⟩, ⟨6493⟩⟩, ⟨.program ⟨214⟩, ⟨15521⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6427⟩⟩, ⟨.program ⟨214⟩, ⟨6503⟩⟩, ⟨.program ⟨214⟩, ⟨15516⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6427⟩⟩, ⟨.program ⟨214⟩, ⟨6542⟩⟩, ⟨.program ⟨214⟩, ⟨15511⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6427⟩⟩, ⟨.program ⟨214⟩, ⟨6543⟩⟩, ⟨.program ⟨214⟩, ⟨15531⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6435⟩⟩, ⟨.program ⟨214⟩, ⟨6493⟩⟩, ⟨.program ⟨214⟩, ⟨18128⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6435⟩⟩, ⟨.program ⟨214⟩, ⟨6503⟩⟩, ⟨.program ⟨214⟩, ⟨18124⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6435⟩⟩, ⟨.program ⟨214⟩, ⟨6542⟩⟩, ⟨.program ⟨214⟩, ⟨18120⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6435⟩⟩, ⟨.program ⟨214⟩, ⟨6543⟩⟩, ⟨.program ⟨214⟩, ⟨18136⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6437⟩⟩, ⟨.program ⟨214⟩, ⟨6493⟩⟩, ⟨.program ⟨214⟩, ⟨16931⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6437⟩⟩, ⟨.program ⟨214⟩, ⟨6503⟩⟩, ⟨.program ⟨214⟩, ⟨16927⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6437⟩⟩, ⟨.program ⟨214⟩, ⟨6542⟩⟩, ⟨.program ⟨214⟩, ⟨16923⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6437⟩⟩, ⟨.program ⟨214⟩, ⟨6543⟩⟩, ⟨.program ⟨214⟩, ⟨16939⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6449⟩⟩, ⟨.program ⟨214⟩, ⟨6493⟩⟩, ⟨.program ⟨214⟩, ⟨17498⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6449⟩⟩, ⟨.program ⟨214⟩, ⟨6503⟩⟩, ⟨.program ⟨214⟩, ⟨17494⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6449⟩⟩, ⟨.program ⟨214⟩, ⟨6542⟩⟩, ⟨.program ⟨214⟩, ⟨17490⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6449⟩⟩, ⟨.program ⟨214⟩, ⟨6543⟩⟩, ⟨.program ⟨214⟩, ⟨17506⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6452⟩⟩, ⟨.program ⟨214⟩, ⟨6493⟩⟩, ⟨.program ⟨214⟩, ⟨15213⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6452⟩⟩, ⟨.program ⟨214⟩, ⟨6503⟩⟩, ⟨.program ⟨214⟩, ⟨15208⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6452⟩⟩, ⟨.program ⟨214⟩, ⟨6542⟩⟩, ⟨.program ⟨214⟩, ⟨15203⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6452⟩⟩, ⟨.program ⟨214⟩, ⟨6543⟩⟩, ⟨.program ⟨214⟩, ⟨15223⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6459⟩⟩, ⟨.program ⟨214⟩, ⟨6493⟩⟩, ⟨.program ⟨214⟩, ⟨17722⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6459⟩⟩, ⟨.program ⟨214⟩, ⟨6503⟩⟩, ⟨.program ⟨214⟩, ⟨17718⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6459⟩⟩, ⟨.program ⟨214⟩, ⟨6542⟩⟩, ⟨.program ⟨214⟩, ⟨17714⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6459⟩⟩, ⟨.program ⟨214⟩, ⟨6543⟩⟩, ⟨.program ⟨214⟩, ⟨17730⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6467⟩⟩, ⟨.program ⟨214⟩, ⟨6493⟩⟩, ⟨.program ⟨214⟩, ⟨17953⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6467⟩⟩, ⟨.program ⟨214⟩, ⟨6503⟩⟩, ⟨.program ⟨214⟩, ⟨17949⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6467⟩⟩, ⟨.program ⟨214⟩, ⟨6542⟩⟩, ⟨.program ⟨214⟩, ⟨17945⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6467⟩⟩, ⟨.program ⟨214⟩, ⟨6543⟩⟩, ⟨.program ⟨214⟩, ⟨17961⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6473⟩⟩, ⟨.program ⟨214⟩, ⟨6493⟩⟩, ⟨.program ⟨214⟩, ⟨17554⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6473⟩⟩, ⟨.program ⟨214⟩, ⟨6503⟩⟩, ⟨.program ⟨214⟩, ⟨17550⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6473⟩⟩, ⟨.program ⟨214⟩, ⟨6542⟩⟩, ⟨.program ⟨214⟩, ⟨17546⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6473⟩⟩, ⟨.program ⟨214⟩, ⟨6543⟩⟩, ⟨.program ⟨214⟩, ⟨17562⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6475⟩⟩, ⟨.program ⟨214⟩, ⟨6493⟩⟩, ⟨.program ⟨214⟩, ⟨15052⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6475⟩⟩, ⟨.program ⟨214⟩, ⟨6503⟩⟩, ⟨.program ⟨214⟩, ⟨15047⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6475⟩⟩, ⟨.program ⟨214⟩, ⟨6542⟩⟩, ⟨.program ⟨214⟩, ⟨15042⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6475⟩⟩, ⟨.program ⟨214⟩, ⟨6543⟩⟩, ⟨.program ⟨214⟩, ⟨15062⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6490⟩⟩, ⟨.program ⟨214⟩, ⟨6493⟩⟩, ⟨.program ⟨214⟩, ⟨18848⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6490⟩⟩, ⟨.program ⟨214⟩, ⟨6503⟩⟩, ⟨.program ⟨214⟩, ⟨18832⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6490⟩⟩, ⟨.program ⟨214⟩, ⟨6542⟩⟩, ⟨.program ⟨214⟩, ⟨18818⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6490⟩⟩, ⟨.program ⟨214⟩, ⟨6543⟩⟩, ⟨.program ⟨214⟩, ⟨18878⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6493⟩⟩, ⟨.program ⟨214⟩, ⟨6494⟩⟩, ⟨.program ⟨214⟩, ⟨17610⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6493⟩⟩, ⟨.program ⟨214⟩, ⟨6495⟩⟩, ⟨.program ⟨214⟩, ⟨14891⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6493⟩⟩, ⟨.program ⟨214⟩, ⟨6502⟩⟩, ⟨.program ⟨214⟩, ⟨17666⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6494⟩⟩, ⟨.program ⟨214⟩, ⟨6503⟩⟩, ⟨.program ⟨214⟩, ⟨17606⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6494⟩⟩, ⟨.program ⟨214⟩, ⟨6542⟩⟩, ⟨.program ⟨214⟩, ⟨17602⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6494⟩⟩, ⟨.program ⟨214⟩, ⟨6543⟩⟩, ⟨.program ⟨214⟩, ⟨17618⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6495⟩⟩, ⟨.program ⟨214⟩, ⟨6503⟩⟩, ⟨.program ⟨214⟩, ⟨14886⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6495⟩⟩, ⟨.program ⟨214⟩, ⟨6542⟩⟩, ⟨.program ⟨214⟩, ⟨14881⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6495⟩⟩, ⟨.program ⟨214⟩, ⟨6543⟩⟩, ⟨.program ⟨214⟩, ⟨14901⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6502⟩⟩, ⟨.program ⟨214⟩, ⟨6503⟩⟩, ⟨.program ⟨214⟩, ⟨17662⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6502⟩⟩, ⟨.program ⟨214⟩, ⟨6542⟩⟩, ⟨.program ⟨214⟩, ⟨17658⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨6502⟩⟩, ⟨.program ⟨214⟩, ⟨6543⟩⟩, ⟨.program ⟨214⟩, ⟨17674⟩⟩], []⟩, (1)⟩]

theorem exact5327RawTermsValid :
    exact5327RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event5327 : Event := .resultExact (⟨.program ⟨214⟩, ⟨18905⟩⟩) exact5327RawTerms (.finite 31369995936811926932431840848922017667658107115879580346086281667382620985844517205344) 5326 .exactZero (none)

def event5328 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18907⟩⟩) 0 ⟨18905⟩ 5327

def event5329 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18907⟩⟩) 1 ⟨6564⟩ 32

def event5330 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨18907⟩⟩) (.product (.predecessor 0 5328 .coefficient) (.predecessor 1 5329 .coefficient) (⟨false, false, none, none, none⟩))

def event5331 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18907⟩⟩, .operator (⟨5327, 39⟩, ⟨32, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6396⟩⟩, ⟨.program ⟨214⟩, ⟨6410⟩⟩, ⟨.program ⟨214⟩, ⟨18511⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def event5332 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18907⟩⟩, .operator (⟨5327, 41⟩, ⟨32, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6396⟩⟩, ⟨.program ⟨214⟩, ⟨6435⟩⟩, ⟨.program ⟨214⟩, ⟨18140⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def event5333 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18907⟩⟩, .operator (⟨5327, 42⟩, ⟨32, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6396⟩⟩, ⟨.program ⟨214⟩, ⟨6437⟩⟩, ⟨.program ⟨214⟩, ⟨16943⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def event5334 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18907⟩⟩, .operator (⟨5327, 43⟩, ⟨32, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6396⟩⟩, ⟨.program ⟨214⟩, ⟨6449⟩⟩, ⟨.program ⟨214⟩, ⟨17510⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def event5335 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18907⟩⟩, .operator (⟨5327, 45⟩, ⟨32, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6396⟩⟩, ⟨.program ⟨214⟩, ⟨6459⟩⟩, ⟨.program ⟨214⟩, ⟨17734⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def event5336 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18907⟩⟩, .operator (⟨5327, 46⟩, ⟨32, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6396⟩⟩, ⟨.program ⟨214⟩, ⟨6467⟩⟩, ⟨.program ⟨214⟩, ⟨17965⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def event5337 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18907⟩⟩, .operator (⟨5327, 47⟩, ⟨32, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6396⟩⟩, ⟨.program ⟨214⟩, ⟨6473⟩⟩, ⟨.program ⟨214⟩, ⟨17566⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def event5338 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18907⟩⟩, .operator (⟨5327, 49⟩, ⟨32, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6396⟩⟩, ⟨.program ⟨214⟩, ⟨6490⟩⟩, ⟨.program ⟨214⟩, ⟨18893⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def event5339 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18907⟩⟩, .operator (⟨5327, 50⟩, ⟨32, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6396⟩⟩, ⟨.program ⟨214⟩, ⟨6494⟩⟩, ⟨.program ⟨214⟩, ⟨17622⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def event5340 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18907⟩⟩, .operator (⟨5327, 52⟩, ⟨32, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6396⟩⟩, ⟨.program ⟨214⟩, ⟨6502⟩⟩, ⟨.program ⟨214⟩, ⟨17678⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def event5341 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18907⟩⟩, .operator (⟨5327, 1⟩, ⟨32, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6383⟩⟩, ⟨.program ⟨214⟩, ⟨6396⟩⟩, ⟨.program ⟨214⟩, ⟨18063⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def event5342 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18907⟩⟩, .operator (⟨5327, 25⟩, ⟨32, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6387⟩⟩, ⟨.program ⟨214⟩, ⟨6396⟩⟩, ⟨.program ⟨214⟩, ⟨17181⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def event5343 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18907⟩⟩, .operator (⟨5327, 31⟩, ⟨32, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6391⟩⟩, ⟨.program ⟨214⟩, ⟨6396⟩⟩, ⟨.program ⟨214⟩, ⟨17237⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def event5344 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18907⟩⟩, .operator (⟨5327, 37⟩, ⟨32, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6396⟩⟩, ⟨.program ⟨214⟩, ⟨6398⟩⟩, ⟨.program ⟨214⟩, ⟨17454⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def event5345 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18907⟩⟩, .operator (⟨5327, 38⟩, ⟨32, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6396⟩⟩, ⟨.program ⟨214⟩, ⟨6407⟩⟩, ⟨.program ⟨214⟩, ⟨17846⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def event5346 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18907⟩⟩, .operator (⟨5327, 40⟩, ⟨32, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6396⟩⟩, ⟨.program ⟨214⟩, ⟨6427⟩⟩, ⟨.program ⟨214⟩, ⟨15536⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def event5347 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18907⟩⟩, .operator (⟨5327, 44⟩, ⟨32, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6396⟩⟩, ⟨.program ⟨214⟩, ⟨6452⟩⟩, ⟨.program ⟨214⟩, ⟨15228⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def event5348 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18907⟩⟩, .operator (⟨5327, 48⟩, ⟨32, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6396⟩⟩, ⟨.program ⟨214⟩, ⟨6475⟩⟩, ⟨.program ⟨214⟩, ⟨15067⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def event5349 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18907⟩⟩, .operator (⟨5327, 51⟩, ⟨32, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6396⟩⟩, ⟨.program ⟨214⟩, ⟨6495⟩⟩, ⟨.program ⟨214⟩, ⟨14906⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def event5350 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18907⟩⟩, .operator (⟨5327, 67⟩, ⟨32, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6410⟩⟩, ⟨.program ⟨214⟩, ⟨6543⟩⟩, ⟨.program ⟨214⟩, ⟨18507⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def event5351 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18907⟩⟩, .operator (⟨5327, 88⟩, ⟨32, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6435⟩⟩, ⟨.program ⟨214⟩, ⟨6543⟩⟩, ⟨.program ⟨214⟩, ⟨18136⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def event5352 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18907⟩⟩, .operator (⟨5327, 92⟩, ⟨32, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6437⟩⟩, ⟨.program ⟨214⟩, ⟨6543⟩⟩, ⟨.program ⟨214⟩, ⟨16939⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def event5353 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18907⟩⟩, .operator (⟨5327, 96⟩, ⟨32, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6449⟩⟩, ⟨.program ⟨214⟩, ⟨6543⟩⟩, ⟨.program ⟨214⟩, ⟨17506⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def event5354 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18907⟩⟩, .operator (⟨5327, 104⟩, ⟨32, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6459⟩⟩, ⟨.program ⟨214⟩, ⟨6543⟩⟩, ⟨.program ⟨214⟩, ⟨17730⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def event5355 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18907⟩⟩, .operator (⟨5327, 108⟩, ⟨32, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6467⟩⟩, ⟨.program ⟨214⟩, ⟨6543⟩⟩, ⟨.program ⟨214⟩, ⟨17961⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def event5356 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18907⟩⟩, .operator (⟨5327, 112⟩, ⟨32, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6473⟩⟩, ⟨.program ⟨214⟩, ⟨6543⟩⟩, ⟨.program ⟨214⟩, ⟨17562⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def event5357 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18907⟩⟩, .operator (⟨5327, 120⟩, ⟨32, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6490⟩⟩, ⟨.program ⟨214⟩, ⟨6543⟩⟩, ⟨.program ⟨214⟩, ⟨18878⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def event5358 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18907⟩⟩, .operator (⟨5327, 126⟩, ⟨32, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6494⟩⟩, ⟨.program ⟨214⟩, ⟨6543⟩⟩, ⟨.program ⟨214⟩, ⟨17618⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def event5359 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18907⟩⟩, .operator (⟨5327, 132⟩, ⟨32, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6502⟩⟩, ⟨.program ⟨214⟩, ⟨6543⟩⟩, ⟨.program ⟨214⟩, ⟨17674⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def event5360 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18907⟩⟩, .operator (⟨5327, 6⟩, ⟨32, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6383⟩⟩, ⟨.program ⟨214⟩, ⟨6543⟩⟩, ⟨.program ⟨214⟩, ⟨18056⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def event5361 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18907⟩⟩, .operator (⟨5327, 30⟩, ⟨32, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6387⟩⟩, ⟨.program ⟨214⟩, ⟨6543⟩⟩, ⟨.program ⟨214⟩, ⟨17177⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def event5362 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18907⟩⟩, .operator (⟨5327, 36⟩, ⟨32, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6391⟩⟩, ⟨.program ⟨214⟩, ⟨6543⟩⟩, ⟨.program ⟨214⟩, ⟨17233⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def event5363 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18907⟩⟩, .operator (⟨5327, 57⟩, ⟨32, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6398⟩⟩, ⟨.program ⟨214⟩, ⟨6543⟩⟩, ⟨.program ⟨214⟩, ⟨17450⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def event5364 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18907⟩⟩, .operator (⟨5327, 62⟩, ⟨32, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6407⟩⟩, ⟨.program ⟨214⟩, ⟨6543⟩⟩, ⟨.program ⟨214⟩, ⟨17838⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def event5365 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18907⟩⟩, .operator (⟨5327, 84⟩, ⟨32, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6427⟩⟩, ⟨.program ⟨214⟩, ⟨6543⟩⟩, ⟨.program ⟨214⟩, ⟨15531⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def event5366 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18907⟩⟩, .operator (⟨5327, 100⟩, ⟨32, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6452⟩⟩, ⟨.program ⟨214⟩, ⟨6543⟩⟩, ⟨.program ⟨214⟩, ⟨15223⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def event5367 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18907⟩⟩, .operator (⟨5327, 116⟩, ⟨32, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6475⟩⟩, ⟨.program ⟨214⟩, ⟨6543⟩⟩, ⟨.program ⟨214⟩, ⟨15062⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def event5368 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18907⟩⟩, .operator (⟨5327, 129⟩, ⟨32, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6495⟩⟩, ⟨.program ⟨214⟩, ⟨6543⟩⟩, ⟨.program ⟨214⟩, ⟨14901⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def event5369 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18907⟩⟩, .operator (⟨5327, 63⟩, ⟨32, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6410⟩⟩, ⟨.program ⟨214⟩, ⟨6425⟩⟩, ⟨.program ⟨214⟩, ⟨18503⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def event5370 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18907⟩⟩, .operator (⟨5327, 69⟩, ⟨32, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6425⟩⟩, ⟨.program ⟨214⟩, ⟨6435⟩⟩, ⟨.program ⟨214⟩, ⟨18132⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def event5371 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18907⟩⟩, .operator (⟨5327, 70⟩, ⟨32, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6425⟩⟩, ⟨.program ⟨214⟩, ⟨6437⟩⟩, ⟨.program ⟨214⟩, ⟨16935⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def event5372 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18907⟩⟩, .operator (⟨5327, 71⟩, ⟨32, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6425⟩⟩, ⟨.program ⟨214⟩, ⟨6449⟩⟩, ⟨.program ⟨214⟩, ⟨17502⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def event5373 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18907⟩⟩, .operator (⟨5327, 73⟩, ⟨32, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6425⟩⟩, ⟨.program ⟨214⟩, ⟨6459⟩⟩, ⟨.program ⟨214⟩, ⟨17726⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def event5374 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18907⟩⟩, .operator (⟨5327, 74⟩, ⟨32, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6425⟩⟩, ⟨.program ⟨214⟩, ⟨6467⟩⟩, ⟨.program ⟨214⟩, ⟨17957⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def event5375 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18907⟩⟩, .operator (⟨5327, 75⟩, ⟨32, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6425⟩⟩, ⟨.program ⟨214⟩, ⟨6473⟩⟩, ⟨.program ⟨214⟩, ⟨17558⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def eventLeaf320 : Array AnnotatedEvent := #[
  { event := event5120
    frameStart := 0 },
  { event := event5121
    frameStart := 0 },
  { event := event5122
    frameStart := 0 },
  { event := event5123
    frameStart := 0 },
  { event := event5124
    frameStart := 0 },
  { event := event5125
    frameStart := 0 },
  { event := event5126
    frameStart := 0 },
  { event := event5127
    frameStart := 0 },
  { event := event5128
    frameStart := 0 },
  { event := event5129
    frameStart := 0 },
  { event := event5130
    frameStart := 0 },
  { event := event5131
    frameStart := 0 },
  { event := event5132
    frameStart := 0 },
  { event := event5133
    frameStart := 0 },
  { event := event5134
    frameStart := 0 },
  { event := event5135
    frameStart := 0 }
]

def eventLeaf321 : Array AnnotatedEvent := #[
  { event := event5136
    frameStart := 0 },
  { event := event5137
    frameStart := 0 },
  { event := event5138
    frameStart := 0 },
  { event := event5139
    frameStart := 0 },
  { event := event5140
    frameStart := 0 },
  { event := event5141
    frameStart := 0 },
  { event := event5142
    frameStart := 0 },
  { event := event5143
    frameStart := 0 },
  { event := event5144
    frameStart := 0 },
  { event := event5145
    frameStart := 0 },
  { event := event5146
    frameStart := 0 },
  { event := event5147
    frameStart := 0 },
  { event := event5148
    frameStart := 0 },
  { event := event5149
    frameStart := 0 },
  { event := event5150
    frameStart := 0 },
  { event := event5151
    frameStart := 0 }
]

def eventLeaf322 : Array AnnotatedEvent := #[
  { event := event5152
    frameStart := 0 },
  { event := event5153
    frameStart := 0 },
  { event := event5154
    frameStart := 0 },
  { event := event5155
    frameStart := 0 },
  { event := event5156
    frameStart := 0 },
  { event := event5157
    frameStart := 0 },
  { event := event5158
    frameStart := 0 },
  { event := event5159
    frameStart := 0 },
  { event := event5160
    frameStart := 0 },
  { event := event5161
    frameStart := 0 },
  { event := event5162
    frameStart := 0 },
  { event := event5163
    frameStart := 0 },
  { event := event5164
    frameStart := 0 },
  { event := event5165
    frameStart := 0 },
  { event := event5166
    frameStart := 0 },
  { event := event5167
    frameStart := 0 }
]

def eventLeaf323 : Array AnnotatedEvent := #[
  { event := event5168
    frameStart := 0 },
  { event := event5169
    frameStart := 0 },
  { event := event5170
    frameStart := 0 },
  { event := event5171
    frameStart := 0 },
  { event := event5172
    frameStart := 0 },
  { event := event5173
    frameStart := 0 },
  { event := event5174
    frameStart := 0 },
  { event := event5175
    frameStart := 0 },
  { event := event5176
    frameStart := 0 },
  { event := event5177
    frameStart := 0 },
  { event := event5178
    frameStart := 0 },
  { event := event5179
    frameStart := 0 },
  { event := event5180
    frameStart := 0 },
  { event := event5181
    frameStart := 0 },
  { event := event5182
    frameStart := 0 },
  { event := event5183
    frameStart := 0 }
]

def eventLeaf324 : Array AnnotatedEvent := #[
  { event := event5184
    frameStart := 0 },
  { event := event5185
    frameStart := 0 },
  { event := event5186
    frameStart := 0 },
  { event := event5187
    frameStart := 0 },
  { event := event5188
    frameStart := 0 },
  { event := event5189
    frameStart := 0 },
  { event := event5190
    frameStart := 0 },
  { event := event5191
    frameStart := 0 },
  { event := event5192
    frameStart := 0 },
  { event := event5193
    frameStart := 0 },
  { event := event5194
    frameStart := 0 },
  { event := event5195
    frameStart := 0 },
  { event := event5196
    frameStart := 0 },
  { event := event5197
    frameStart := 0 },
  { event := event5198
    frameStart := 0 },
  { event := event5199
    frameStart := 0 }
]

def eventLeaf325 : Array AnnotatedEvent := #[
  { event := event5200
    frameStart := 0 },
  { event := event5201
    frameStart := 0 },
  { event := event5202
    frameStart := 0 },
  { event := event5203
    frameStart := 0 },
  { event := event5204
    frameStart := 0 },
  { event := event5205
    frameStart := 0 },
  { event := event5206
    frameStart := 0 },
  { event := event5207
    frameStart := 0 },
  { event := event5208
    frameStart := 0 },
  { event := event5209
    frameStart := 0 },
  { event := event5210
    frameStart := 0 },
  { event := event5211
    frameStart := 0 },
  { event := event5212
    frameStart := 0 },
  { event := event5213
    frameStart := 0 },
  { event := event5214
    frameStart := 0 },
  { event := event5215
    frameStart := 0 }
]

def eventLeaf326 : Array AnnotatedEvent := #[
  { event := event5216
    frameStart := 0 },
  { event := event5217
    frameStart := 0 },
  { event := event5218
    frameStart := 0 },
  { event := event5219
    frameStart := 0 },
  { event := event5220
    frameStart := 0 },
  { event := event5221
    frameStart := 0 },
  { event := event5222
    frameStart := 0 },
  { event := event5223
    frameStart := 0 },
  { event := event5224
    frameStart := 0 },
  { event := event5225
    frameStart := 0 },
  { event := event5226
    frameStart := 0 },
  { event := event5227
    frameStart := 0 },
  { event := event5228
    frameStart := 0 },
  { event := event5229
    frameStart := 0 },
  { event := event5230
    frameStart := 0 },
  { event := event5231
    frameStart := 0 }
]

def eventLeaf327 : Array AnnotatedEvent := #[
  { event := event5232
    frameStart := 0 },
  { event := event5233
    frameStart := 0 },
  { event := event5234
    frameStart := 0 },
  { event := event5235
    frameStart := 0 },
  { event := event5236
    frameStart := 0 },
  { event := event5237
    frameStart := 0 },
  { event := event5238
    frameStart := 0 },
  { event := event5239
    frameStart := 0 },
  { event := event5240
    frameStart := 0 },
  { event := event5241
    frameStart := 0 },
  { event := event5242
    frameStart := 0 },
  { event := event5243
    frameStart := 0 },
  { event := event5244
    frameStart := 0 },
  { event := event5245
    frameStart := 0 },
  { event := event5246
    frameStart := 0 },
  { event := event5247
    frameStart := 0 }
]

def eventLeaf328 : Array AnnotatedEvent := #[
  { event := event5248
    frameStart := 0 },
  { event := event5249
    frameStart := 0 },
  { event := event5250
    frameStart := 0 },
  { event := event5251
    frameStart := 0 },
  { event := event5252
    frameStart := 0 },
  { event := event5253
    frameStart := 0 },
  { event := event5254
    frameStart := 0 },
  { event := event5255
    frameStart := 0 },
  { event := event5256
    frameStart := 0 },
  { event := event5257
    frameStart := 0 },
  { event := event5258
    frameStart := 0 },
  { event := event5259
    frameStart := 0 },
  { event := event5260
    frameStart := 0 },
  { event := event5261
    frameStart := 0 },
  { event := event5262
    frameStart := 0 },
  { event := event5263
    frameStart := 0 }
]

def eventLeaf329 : Array AnnotatedEvent := #[
  { event := event5264
    frameStart := 0 },
  { event := event5265
    frameStart := 0 },
  { event := event5266
    frameStart := 0 },
  { event := event5267
    frameStart := 0 },
  { event := event5268
    frameStart := 0 },
  { event := event5269
    frameStart := 0 },
  { event := event5270
    frameStart := 0 },
  { event := event5271
    frameStart := 0 },
  { event := event5272
    frameStart := 0 },
  { event := event5273
    frameStart := 0 },
  { event := event5274
    frameStart := 0 },
  { event := event5275
    frameStart := 0 },
  { event := event5276
    frameStart := 0 },
  { event := event5277
    frameStart := 0 },
  { event := event5278
    frameStart := 0 },
  { event := event5279
    frameStart := 0 }
]

def eventLeaf330 : Array AnnotatedEvent := #[
  { event := event5280
    frameStart := 0 },
  { event := event5281
    frameStart := 0 },
  { event := event5282
    frameStart := 0 },
  { event := event5283
    frameStart := 0 },
  { event := event5284
    frameStart := 0 },
  { event := event5285
    frameStart := 0 },
  { event := event5286
    frameStart := 0 },
  { event := event5287
    frameStart := 0 },
  { event := event5288
    frameStart := 0 },
  { event := event5289
    frameStart := 0 },
  { event := event5290
    frameStart := 0 },
  { event := event5291
    frameStart := 0 },
  { event := event5292
    frameStart := 0 },
  { event := event5293
    frameStart := 0 },
  { event := event5294
    frameStart := 0 },
  { event := event5295
    frameStart := 0 }
]

def eventLeaf331 : Array AnnotatedEvent := #[
  { event := event5296
    frameStart := 0 },
  { event := event5297
    frameStart := 0 },
  { event := event5298
    frameStart := 0 },
  { event := event5299
    frameStart := 0 },
  { event := event5300
    frameStart := 0 },
  { event := event5301
    frameStart := 0 },
  { event := event5302
    frameStart := 0 },
  { event := event5303
    frameStart := 0 },
  { event := event5304
    frameStart := 0 },
  { event := event5305
    frameStart := 0 },
  { event := event5306
    frameStart := 0 },
  { event := event5307
    frameStart := 0 },
  { event := event5308
    frameStart := 0 },
  { event := event5309
    frameStart := 0 },
  { event := event5310
    frameStart := 0 },
  { event := event5311
    frameStart := 0 }
]

def eventLeaf332 : Array AnnotatedEvent := #[
  { event := event5312
    frameStart := 0 },
  { event := event5313
    frameStart := 0 },
  { event := event5314
    frameStart := 0 },
  { event := event5315
    frameStart := 0 },
  { event := event5316
    frameStart := 0 },
  { event := event5317
    frameStart := 0 },
  { event := event5318
    frameStart := 0 },
  { event := event5319
    frameStart := 0 },
  { event := event5320
    frameStart := 0 },
  { event := event5321
    frameStart := 0 },
  { event := event5322
    frameStart := 0 },
  { event := event5323
    frameStart := 0 },
  { event := event5324
    frameStart := 0 },
  { event := event5325
    frameStart := 0 },
  { event := event5326
    frameStart := 0 },
  { event := event5327
    frameStart := 0 }
]

def eventLeaf333 : Array AnnotatedEvent := #[
  { event := event5328
    frameStart := 0 },
  { event := event5329
    frameStart := 0 },
  { event := event5330
    frameStart := 0 },
  { event := event5331
    frameStart := 0 },
  { event := event5332
    frameStart := 0 },
  { event := event5333
    frameStart := 0 },
  { event := event5334
    frameStart := 0 },
  { event := event5335
    frameStart := 0 },
  { event := event5336
    frameStart := 0 },
  { event := event5337
    frameStart := 0 },
  { event := event5338
    frameStart := 0 },
  { event := event5339
    frameStart := 0 },
  { event := event5340
    frameStart := 0 },
  { event := event5341
    frameStart := 0 },
  { event := event5342
    frameStart := 0 },
  { event := event5343
    frameStart := 0 }
]

def eventLeaf334 : Array AnnotatedEvent := #[
  { event := event5344
    frameStart := 0 },
  { event := event5345
    frameStart := 0 },
  { event := event5346
    frameStart := 0 },
  { event := event5347
    frameStart := 0 },
  { event := event5348
    frameStart := 0 },
  { event := event5349
    frameStart := 0 },
  { event := event5350
    frameStart := 0 },
  { event := event5351
    frameStart := 0 },
  { event := event5352
    frameStart := 0 },
  { event := event5353
    frameStart := 0 },
  { event := event5354
    frameStart := 0 },
  { event := event5355
    frameStart := 0 },
  { event := event5356
    frameStart := 0 },
  { event := event5357
    frameStart := 0 },
  { event := event5358
    frameStart := 0 },
  { event := event5359
    frameStart := 0 }
]

def eventLeaf335 : Array AnnotatedEvent := #[
  { event := event5360
    frameStart := 0 },
  { event := event5361
    frameStart := 0 },
  { event := event5362
    frameStart := 0 },
  { event := event5363
    frameStart := 0 },
  { event := event5364
    frameStart := 0 },
  { event := event5365
    frameStart := 0 },
  { event := event5366
    frameStart := 0 },
  { event := event5367
    frameStart := 0 },
  { event := event5368
    frameStart := 0 },
  { event := event5369
    frameStart := 0 },
  { event := event5370
    frameStart := 0 },
  { event := event5371
    frameStart := 0 },
  { event := event5372
    frameStart := 0 },
  { event := event5373
    frameStart := 0 },
  { event := event5374
    frameStart := 0 },
  { event := event5375
    frameStart := 0 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events020
