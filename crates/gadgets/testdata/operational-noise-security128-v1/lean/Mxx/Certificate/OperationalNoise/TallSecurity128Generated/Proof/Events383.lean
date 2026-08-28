import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events383

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event98048 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨22776⟩⟩) (.authority (.relationPreimageSource ⟨61⟩))

def exact98049RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨22776⟩⟩]⟩, (1)⟩]

theorem exact98049RawTermsValid :
    exact98049RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event98049 : Event := .resultExact (⟨.program ⟨257⟩, ⟨22776⟩⟩) exact98049RawTerms (.finite 5647228698) 98048 .exactZero (none)

def event98050 : Event := .predecessor (⟨.program ⟨257⟩, ⟨22778⟩⟩) 0 ⟨22776⟩ 98049

def event98051 : Event := .predecessor (⟨.program ⟨257⟩, ⟨22778⟩⟩) 1 ⟨2370⟩ 4

def event98052 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨22778⟩⟩) (.scale (.predecessor 0 98050 .coefficient) (.value (.predecessor 1 98051 .coefficient)))

def exact98053RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨22776⟩⟩]⟩, (1)⟩]

theorem exact98053RawTermsValid :
    exact98053RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event98053 : Event := .resultExact (⟨.program ⟨257⟩, ⟨22778⟩⟩) exact98053RawTerms (.finite 5647228698) 98052 .exactZero (none)

def event98054 : Event := .predecessor (⟨.program ⟨257⟩, ⟨22779⟩⟩) 0 ⟨9944⟩ 90620

def event98055 : Event := .predecessor (⟨.program ⟨257⟩, ⟨22779⟩⟩) 1 ⟨22778⟩ 98053

def event98056 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨22779⟩⟩) (.product (.predecessor 0 98054 .coefficient) (.predecessor 1 98055 .coefficient) (⟨false, false, none, none, none⟩))

def event98057 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨22779⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨22776⟩⟩]⟩) [⟨.result 98049 .coefficient, false, none⟩])

def event98058 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨22779⟩⟩) (.product (.result 90620 .summary) (.transfer 98057) (⟨false, false, none, none, none⟩))

def event98059 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨22779⟩⟩, .operator (⟨90620, 0⟩, ⟨98053, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22776⟩⟩]⟩, (1)⟩)

def event98060 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨22777⟩⟩)

def event98061 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event98062 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event98063 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9843⟩⟩) (.authority (.operator))

def event98064 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨9843⟩⟩) (.finite 14)

def event98065 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event98066 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event98067 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event98068 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event98069 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 98068

def event98070 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 98066

def event98071 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 98069 .coefficient) (.value (.predecessor 1 98070 .coefficient)))

def event98072 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event98073 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9845⟩⟩) 0 ⟨392⟩ 98072

def event98074 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9845⟩⟩) 1 ⟨9843⟩ 98064

def event98075 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9845⟩⟩) (.sum [.predecessor 0 98073 .coefficient, .predecessor 1 98074 .coefficient])

def event98076 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨9845⟩⟩) (.finite 655354)

def event98077 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9901⟩⟩) 0 ⟨9845⟩ 98076

def event98078 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9901⟩⟩) 1 ⟨5426⟩ 98062

def event98079 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9901⟩⟩) (.identity (.predecessor 1 98078 .coefficient))

def event98080 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨9901⟩⟩) (.finite 655360)

def event98081 : Event := .predecessor (⟨.program ⟨257⟩, ⟨21614⟩⟩) 0 ⟨9901⟩ 98080

def event98082 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨21614⟩⟩) (.authority (.programFamilyFact))

def exact98083RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨21614⟩⟩], []⟩, (1)⟩]

theorem exact98083RawTermsValid :
    exact98083RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event98083 : Event := .resultExact (⟨.program ⟨257⟩, ⟨21614⟩⟩) exact98083RawTerms (.finite 4) 98082 .exactZero (none)

def event98084 : Event := .predecessor (⟨.program ⟨257⟩, ⟨21176⟩⟩) 0 ⟨9901⟩ 98080

def event98085 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨21176⟩⟩) (.authority (.programFamilyFact))

def exact98086RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨21176⟩⟩], []⟩, (1)⟩]

theorem exact98086RawTermsValid :
    exact98086RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event98086 : Event := .resultExact (⟨.program ⟨257⟩, ⟨21176⟩⟩) exact98086RawTerms (.finite 4) 98085 .exactZero (none)

def event98087 : Event := .predecessor (⟨.program ⟨257⟩, ⟨21615⟩⟩) 0 ⟨21176⟩ 98086

def event98088 : Event := .predecessor (⟨.program ⟨257⟩, ⟨21615⟩⟩) 1 ⟨21614⟩ 98083

def event98089 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨21615⟩⟩) (.product (.predecessor 0 98087 .coefficient) (.predecessor 1 98088 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event98090 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨21615⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨21176⟩⟩, ⟨.program ⟨257⟩, ⟨21614⟩⟩], []⟩) [⟨.result 98086 .coefficient, true, some 1⟩, ⟨.result 98083 .coefficient, true, some 1⟩])

def event98091 : Event := .survivorFold (1) 98090

def exact98092RawTerms : List Term := []

theorem exact98092RawTermsValid :
    exact98092RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event98092 : Event := .resultExact (⟨.program ⟨257⟩, ⟨21615⟩⟩) exact98092RawTerms (.finite 16) 98089 (.finite 16) (some (98090))

def event98093 : Event := .predecessor (⟨.program ⟨257⟩, ⟨21616⟩⟩) 0 ⟨21615⟩ 98092

def event98094 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨21616⟩⟩) (.identity (.predecessor 0 98093 .coefficient))

def event98095 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨21616⟩⟩) (.finite 16)

def event98096 : Event := .predecessor (⟨.program ⟨257⟩, ⟨21848⟩⟩) 0 ⟨21616⟩ 98095

def event98097 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨21848⟩⟩) (.authority (.programFamilyFact))

def exact98098RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨21848⟩⟩], []⟩, (1)⟩]

theorem exact98098RawTermsValid :
    exact98098RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event98098 : Event := .resultExact (⟨.program ⟨257⟩, ⟨21848⟩⟩) exact98098RawTerms (.finite 4) 98097 .exactZero (none)

def event98099 : Event := .predecessor (⟨.program ⟨257⟩, ⟨21849⟩⟩) 0 ⟨21848⟩ 98098

def event98100 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨21849⟩⟩) (.identity (.predecessor 0 98099 .coefficient))

def event98101 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨21849⟩⟩) (.finite 4)

def event98102 : Event := .predecessor (⟨.program ⟨257⟩, ⟨22776⟩⟩) 0 ⟨21849⟩ 98101

def event98103 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨22776⟩⟩) (.authority (.relationPreimageSource ⟨61⟩))

def exact98104RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨22776⟩⟩]⟩, (1)⟩]

theorem exact98104RawTermsValid :
    exact98104RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event98104 : Event := .resultExact (⟨.program ⟨257⟩, ⟨22776⟩⟩) exact98104RawTerms (.finite 5647228698) 98103 .exactZero (none)

def event98105 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35⟩⟩) (.authority (.operator))

def exact98106RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩]⟩, (1)⟩]

theorem exact98106RawTermsValid :
    exact98106RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event98106 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35⟩⟩) exact98106RawTerms .large 98105 .exactZero (none)

def event98107 : Event := .predecessor (⟨.program ⟨257⟩, ⟨22777⟩⟩) 0 ⟨35⟩ 98106

def event98108 : Event := .predecessor (⟨.program ⟨257⟩, ⟨22777⟩⟩) 1 ⟨22776⟩ 98104

def event98109 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨22777⟩⟩) (.product (.predecessor 0 98107 .coefficient) (.predecessor 1 98108 .coefficient) (⟨false, false, none, none, none⟩))

def event98110 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨22777⟩⟩, .operator (⟨98106, 0⟩, ⟨98104, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22776⟩⟩]⟩, (1)⟩)

def exact98111RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22776⟩⟩]⟩, (1)⟩]

theorem exact98111RawTermsValid :
    exact98111RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event98111 : Event := .resultExact (⟨.program ⟨257⟩, ⟨22777⟩⟩) exact98111RawTerms .large 98109 .exactZero (none)

def event98112 : Event := .preFoldPolynomial 98111 [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22776⟩⟩]⟩, (1)⟩] .exactZero none

def exact98113RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22776⟩⟩]⟩, (1)⟩]

def event98113 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨22777⟩⟩) 98112 exact98113RawTerms .large 98109 .exactZero (none)

def event98114 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨24032⟩⟩)

def event98115 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event98116 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event98117 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9843⟩⟩) (.authority (.operator))

def event98118 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨9843⟩⟩) (.finite 14)

def event98119 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event98120 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event98121 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event98122 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event98123 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 98122

def event98124 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 98120

def event98125 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 98123 .coefficient) (.value (.predecessor 1 98124 .coefficient)))

def event98126 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event98127 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9845⟩⟩) 0 ⟨392⟩ 98126

def event98128 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9845⟩⟩) 1 ⟨9843⟩ 98118

def event98129 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9845⟩⟩) (.sum [.predecessor 0 98127 .coefficient, .predecessor 1 98128 .coefficient])

def event98130 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨9845⟩⟩) (.finite 655354)

def event98131 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9901⟩⟩) 0 ⟨9845⟩ 98130

def event98132 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9901⟩⟩) 1 ⟨5426⟩ 98116

def event98133 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9901⟩⟩) (.identity (.predecessor 1 98132 .coefficient))

def event98134 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨9901⟩⟩) (.finite 655360)

def event98135 : Event := .predecessor (⟨.program ⟨257⟩, ⟨21614⟩⟩) 0 ⟨9901⟩ 98134

def event98136 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨21614⟩⟩) (.authority (.programFamilyFact))

def exact98137RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨21614⟩⟩], []⟩, (1)⟩]

theorem exact98137RawTermsValid :
    exact98137RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event98137 : Event := .resultExact (⟨.program ⟨257⟩, ⟨21614⟩⟩) exact98137RawTerms (.finite 4) 98136 .exactZero (none)

def event98138 : Event := .predecessor (⟨.program ⟨257⟩, ⟨21176⟩⟩) 0 ⟨9901⟩ 98134

def event98139 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨21176⟩⟩) (.authority (.programFamilyFact))

def exact98140RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨21176⟩⟩], []⟩, (1)⟩]

theorem exact98140RawTermsValid :
    exact98140RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event98140 : Event := .resultExact (⟨.program ⟨257⟩, ⟨21176⟩⟩) exact98140RawTerms (.finite 4) 98139 .exactZero (none)

def event98141 : Event := .predecessor (⟨.program ⟨257⟩, ⟨21615⟩⟩) 0 ⟨21176⟩ 98140

def event98142 : Event := .predecessor (⟨.program ⟨257⟩, ⟨21615⟩⟩) 1 ⟨21614⟩ 98137

def event98143 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨21615⟩⟩) (.product (.predecessor 0 98141 .coefficient) (.predecessor 1 98142 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event98144 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨21615⟩⟩, .operator (⟨98140, 0⟩, ⟨98137, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨21176⟩⟩, ⟨.program ⟨257⟩, ⟨21614⟩⟩], []⟩, (1)⟩)

def exact98145RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨21176⟩⟩, ⟨.program ⟨257⟩, ⟨21614⟩⟩], []⟩, (1)⟩]

theorem exact98145RawTermsValid :
    exact98145RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event98145 : Event := .resultExact (⟨.program ⟨257⟩, ⟨21615⟩⟩) exact98145RawTerms (.finite 16) 98143 .exactZero (none)

def event98146 : Event := .predecessor (⟨.program ⟨257⟩, ⟨21616⟩⟩) 0 ⟨21615⟩ 98145

def event98147 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨21616⟩⟩) (.identity (.predecessor 0 98146 .coefficient))

def event98148 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨21616⟩⟩) (.finite 16)

def event98149 : Event := .predecessor (⟨.program ⟨257⟩, ⟨21848⟩⟩) 0 ⟨21616⟩ 98148

def event98150 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨21848⟩⟩) (.authority (.programFamilyFact))

def exact98151RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨21848⟩⟩], []⟩, (1)⟩]

theorem exact98151RawTermsValid :
    exact98151RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event98151 : Event := .resultExact (⟨.program ⟨257⟩, ⟨21848⟩⟩) exact98151RawTerms (.finite 4) 98150 .exactZero (none)

def event98152 : Event := .predecessor (⟨.program ⟨257⟩, ⟨21849⟩⟩) 0 ⟨21848⟩ 98151

def event98153 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨21849⟩⟩) (.identity (.predecessor 0 98152 .coefficient))

def event98154 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨21849⟩⟩) (.finite 4)

def event98155 : Event := .predecessor (⟨.program ⟨257⟩, ⟨23124⟩⟩) 0 ⟨21849⟩ 98154

def event98156 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨23124⟩⟩) (.authority (.programFamilyFact))

def event98157 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨23124⟩⟩) (.finite 3720)

def event98158 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨7177⟩⟩) .missing

def event98159 : Event := .predecessor (⟨.program ⟨257⟩, ⟨23126⟩⟩) 0 ⟨7177⟩ 98158

def event98160 : Event := .predecessor (⟨.program ⟨257⟩, ⟨23126⟩⟩) 1 ⟨23124⟩ 98157

def event98161 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨23126⟩⟩) (.authority (.operator))

def exact98162RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨23126⟩⟩]⟩, (1)⟩]

theorem exact98162RawTermsValid :
    exact98162RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event98162 : Event := .resultExact (⟨.program ⟨257⟩, ⟨23126⟩⟩) exact98162RawTerms .large 98161 .exactZero (none)

def event98163 : Event := .predecessor (⟨.program ⟨257⟩, ⟨24027⟩⟩) 0 ⟨23126⟩ 98162

def event98164 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨24027⟩⟩) (.authority (.operator))

def exact98165RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨24027⟩⟩]⟩, (1)⟩]

theorem exact98165RawTermsValid :
    exact98165RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event98165 : Event := .resultExact (⟨.program ⟨257⟩, ⟨24027⟩⟩) exact98165RawTerms (.finite 8192) 98164 .exactZero (none)

def event98166 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨136⟩⟩) (.authority (.operator))

def event98167 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨136⟩⟩) .exactZero

def event98168 : Event := .predecessor (⟨.program ⟨257⟩, ⟨23306⟩⟩) 0 ⟨21849⟩ 98154

def event98169 : Event := .predecessor (⟨.program ⟨257⟩, ⟨23306⟩⟩) 1 ⟨136⟩ 98167

def event98170 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨23306⟩⟩) (.sum [.predecessor 0 98168 .coefficient, .predecessor 1 98169 .coefficient])

def event98171 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨23306⟩⟩) (.finite 4)

def event98172 : Event := .predecessor (⟨.program ⟨257⟩, ⟨23307⟩⟩) 0 ⟨23306⟩ 98171

def event98173 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨23307⟩⟩) (.identity (.predecessor 0 98172 .coefficient))

def exact98174RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨21848⟩⟩], []⟩, (1)⟩]

theorem exact98174RawTermsValid :
    exact98174RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event98174 : Event := .resultExact (⟨.program ⟨257⟩, ⟨23307⟩⟩) exact98174RawTerms (.finite 4) 98173 .exactZero (none)

def event98175 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6908⟩⟩) (.authority (.factStore))

def exact98176RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact98176RawTermsValid :
    exact98176RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event98176 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6908⟩⟩) exact98176RawTerms .large 98175 .exactZero (none)

def event98177 : Event := .predecessor (⟨.program ⟨257⟩, ⟨23308⟩⟩) 0 ⟨6908⟩ 98176

def event98178 : Event := .predecessor (⟨.program ⟨257⟩, ⟨23308⟩⟩) 1 ⟨23307⟩ 98174

def event98179 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨23308⟩⟩) (.product (.predecessor 0 98177 .coefficient) (.predecessor 1 98178 .coefficient) (⟨false, false, none, none, none⟩))

def event98180 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨23308⟩⟩, .operator (⟨98176, 0⟩, ⟨98174, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨21848⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact98181RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨21848⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact98181RawTermsValid :
    exact98181RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event98181 : Event := .resultExact (⟨.program ⟨257⟩, ⟨23308⟩⟩) exact98181RawTerms .large 98179 .exactZero (none)

def event98182 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7181⟩⟩) 0 ⟨7177⟩ 98158

def event98183 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7181⟩⟩) (.authority (.operator))

def exact98184RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7181⟩⟩]⟩, (1)⟩]

theorem exact98184RawTermsValid :
    exact98184RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event98184 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7181⟩⟩) exact98184RawTerms .large 98183 .exactZero (none)

def event98185 : Event := .predecessor (⟨.program ⟨257⟩, ⟨23309⟩⟩) 0 ⟨7181⟩ 98184

def event98186 : Event := .predecessor (⟨.program ⟨257⟩, ⟨23309⟩⟩) 1 ⟨23308⟩ 98181

def event98187 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨23309⟩⟩) (.sum [.predecessor 0 98185 .coefficient, .predecessor 1 98186 .coefficient])

def exact98188RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7181⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨21848⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact98188RawTermsValid :
    exact98188RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event98188 : Event := .resultExact (⟨.program ⟨257⟩, ⟨23309⟩⟩) exact98188RawTerms .large 98187 .exactZero (none)

def event98189 : Event := .predecessor (⟨.program ⟨257⟩, ⟨24028⟩⟩) 0 ⟨23309⟩ 98188

def event98190 : Event := .predecessor (⟨.program ⟨257⟩, ⟨24028⟩⟩) 1 ⟨24027⟩ 98165

def event98191 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨24028⟩⟩) (.product (.predecessor 0 98189 .coefficient) (.predecessor 1 98190 .coefficient) (⟨false, false, none, none, none⟩))

def event98192 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨24028⟩⟩, .operator (⟨98188, 0⟩, ⟨98165, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7181⟩⟩, ⟨.program ⟨257⟩, ⟨24027⟩⟩]⟩, (1)⟩)

def event98193 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨24028⟩⟩, .operator (⟨98188, 1⟩, ⟨98165, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨21848⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨24027⟩⟩]⟩, (-1)⟩)

def event98194 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨24028⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨21848⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨24027⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨24027⟩⟩) ⟨23126⟩ 98162)

def event98195 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨24028⟩⟩, .relation 98194 0, ⟨[⟨.program ⟨257⟩, ⟨21848⟩⟩], [⟨.program ⟨257⟩, ⟨23126⟩⟩]⟩, (-1)⟩)

def exact98196RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7181⟩⟩, ⟨.program ⟨257⟩, ⟨24027⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨21848⟩⟩], [⟨.program ⟨257⟩, ⟨23126⟩⟩]⟩, (-1)⟩]

theorem exact98196RawTermsValid :
    exact98196RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event98196 : Event := .resultExact (⟨.program ⟨257⟩, ⟨24028⟩⟩) exact98196RawTerms .large 98191 .exactZero (none)

def event98197 : Event := .predecessor (⟨.program ⟨257⟩, ⟨22181⟩⟩) 0 ⟨21849⟩ 98154

def event98198 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨22181⟩⟩) (.authority (.programFamilyFact))

def exact98199RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨22181⟩⟩], []⟩, (1)⟩]

theorem exact98199RawTermsValid :
    exact98199RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event98199 : Event := .resultExact (⟨.program ⟨257⟩, ⟨22181⟩⟩) exact98199RawTerms (.finite 51) 98198 .exactZero (none)

def event98200 : Event := .predecessor (⟨.program ⟨257⟩, ⟨22183⟩⟩) 0 ⟨6908⟩ 98176

def event98201 : Event := .predecessor (⟨.program ⟨257⟩, ⟨22183⟩⟩) 1 ⟨22181⟩ 98199

def event98202 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨22183⟩⟩) (.product (.predecessor 0 98200 .coefficient) (.predecessor 1 98201 .coefficient) (⟨false, true, none, none, some 1⟩))

def event98203 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨22183⟩⟩, .operator (⟨98176, 0⟩, ⟨98199, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨22181⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact98204RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨22181⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact98204RawTermsValid :
    exact98204RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event98204 : Event := .resultExact (⟨.program ⟨257⟩, ⟨22183⟩⟩) exact98204RawTerms .large 98202 .exactZero (none)

def event98205 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7202⟩⟩) 0 ⟨7177⟩ 98158

def event98206 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7202⟩⟩) (.authority (.operator))

def exact98207RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩]

theorem exact98207RawTermsValid :
    exact98207RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event98207 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7202⟩⟩) exact98207RawTerms .large 98206 .exactZero (none)

def event98208 : Event := .predecessor (⟨.program ⟨257⟩, ⟨22184⟩⟩) 0 ⟨7202⟩ 98207

def event98209 : Event := .predecessor (⟨.program ⟨257⟩, ⟨22184⟩⟩) 1 ⟨22183⟩ 98204

def event98210 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨22184⟩⟩) (.sum [.predecessor 0 98208 .coefficient, .predecessor 1 98209 .coefficient])

def exact98211RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨22181⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact98211RawTermsValid :
    exact98211RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event98211 : Event := .resultExact (⟨.program ⟨257⟩, ⟨22184⟩⟩) exact98211RawTerms .large 98210 .exactZero (none)

def event98212 : Event := .predecessor (⟨.program ⟨257⟩, ⟨24032⟩⟩) 0 ⟨22184⟩ 98211

def event98213 : Event := .predecessor (⟨.program ⟨257⟩, ⟨24032⟩⟩) 1 ⟨24028⟩ 98196

def event98214 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨24032⟩⟩) (.sum [.predecessor 0 98212 .coefficient, .predecessor 1 98213 .coefficient])

def exact98215RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7181⟩⟩, ⟨.program ⟨257⟩, ⟨24027⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨21848⟩⟩], [⟨.program ⟨257⟩, ⟨23126⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨22181⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact98215RawTermsValid :
    exact98215RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event98215 : Event := .resultExact (⟨.program ⟨257⟩, ⟨24032⟩⟩) exact98215RawTerms .large 98214 .exactZero (none)

def event98216 : Event := .preFoldPolynomial 98215 [⟨⟨[], [⟨.program ⟨257⟩, ⟨7181⟩⟩, ⟨.program ⟨257⟩, ⟨24027⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨21848⟩⟩], [⟨.program ⟨257⟩, ⟨23126⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨22181⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩] .exactZero none

def exact98217RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7181⟩⟩, ⟨.program ⟨257⟩, ⟨24027⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨21848⟩⟩], [⟨.program ⟨257⟩, ⟨23126⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨22181⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

def event98217 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨24032⟩⟩) 98216 exact98217RawTerms .large 98214 .exactZero (none)

def event98218 : Event := .specializationComputed (⟨.program ⟨257⟩, ⟨21849⟩⟩) ⟨⟨81⟩, ⟨61⟩, ⟨135⟩⟩ ⟨98060, 98218⟩

def event98219 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨22779⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22776⟩⟩]⟩) (1) 0 2 (.universal 98218 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22776⟩⟩]⟩) (none) 98217)

def event98220 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨22779⟩⟩, .relation 98219 1, ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩)

def event98221 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨22779⟩⟩, .relation 98219 0, ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7181⟩⟩, ⟨.program ⟨257⟩, ⟨24027⟩⟩]⟩, (-1)⟩)

def event98222 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨22779⟩⟩, .relation 98219 2, ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨21848⟩⟩], [⟨.program ⟨257⟩, ⟨23126⟩⟩]⟩, (1)⟩)

def event98223 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨22779⟩⟩, .relation 98219 3, ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨22181⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact98224RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7181⟩⟩, ⟨.program ⟨257⟩, ⟨24027⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨21848⟩⟩], [⟨.program ⟨257⟩, ⟨23126⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨22181⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact98224RawTermsValid :
    exact98224RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event98224 : Event := .resultExact (⟨.program ⟨257⟩, ⟨22779⟩⟩) exact98224RawTerms .large 98056 (.finite 202072841853861888) (some (98058))

def event98225 : Event := .predecessor (⟨.program ⟨257⟩, ⟨24030⟩⟩) 0 ⟨22779⟩ 98224

def event98226 : Event := .predecessor (⟨.program ⟨257⟩, ⟨24030⟩⟩) 1 ⟨24029⟩ 98046

def event98227 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨24030⟩⟩) (.sum [.predecessor 0 98225 .coefficient, .predecessor 1 98226 .coefficient])

def event98228 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨24030⟩⟩, .operator (⟨98224, 0⟩, ⟨98046, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7181⟩⟩, ⟨.program ⟨257⟩, ⟨24027⟩⟩]⟩, (1)⟩)

def event98229 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨24030⟩⟩, .operator (⟨98224, 2⟩, ⟨98046, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨21848⟩⟩], [⟨.program ⟨257⟩, ⟨23126⟩⟩]⟩, (-1)⟩)

def event98230 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨24030⟩⟩) (.sum [.result 98224 .summary, .result 98046 .summary])

def exact98231RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨22181⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact98231RawTermsValid :
    exact98231RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event98231 : Event := .resultExact (⟨.program ⟨257⟩, ⟨24030⟩⟩) exact98231RawTerms .large 98227 (.finite 32189003662929394266751515230208) (some (98230))

def event98232 : Event := .predecessor (⟨.program ⟨257⟩, ⟨19904⟩⟩) 0 ⟨18629⟩ 4219

def event98233 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨19904⟩⟩) (.authority (.programFamilyFact))

def event98234 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨19904⟩⟩) (.finite 3720)

def event98235 : Event := .predecessor (⟨.program ⟨257⟩, ⟨19906⟩⟩) 0 ⟨7177⟩ 15500

def event98236 : Event := .predecessor (⟨.program ⟨257⟩, ⟨19906⟩⟩) 1 ⟨19904⟩ 98234

def event98237 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨19906⟩⟩) (.authority (.operator))

def exact98238RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨19906⟩⟩]⟩, (1)⟩]

theorem exact98238RawTermsValid :
    exact98238RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event98238 : Event := .resultExact (⟨.program ⟨257⟩, ⟨19906⟩⟩) exact98238RawTerms .large 98237 .exactZero (none)

def event98239 : Event := .predecessor (⟨.program ⟨257⟩, ⟨20807⟩⟩) 0 ⟨19906⟩ 98238

def event98240 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨20807⟩⟩) (.authority (.operator))

def exact98241RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨20807⟩⟩]⟩, (1)⟩]

theorem exact98241RawTermsValid :
    exact98241RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event98241 : Event := .resultExact (⟨.program ⟨257⟩, ⟨20807⟩⟩) exact98241RawTerms (.finite 8192) 98240 .exactZero (none)

def event98242 : Event := .predecessor (⟨.program ⟨257⟩, ⟨19738⟩⟩) 0 ⟨18396⟩ 4213

def event98243 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨19738⟩⟩) (.authority (.programFamilyFact))

def event98244 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨19738⟩⟩) (.finite 3720)

def event98245 : Event := .predecessor (⟨.program ⟨257⟩, ⟨19739⟩⟩) 0 ⟨7177⟩ 15500

def event98246 : Event := .predecessor (⟨.program ⟨257⟩, ⟨19739⟩⟩) 1 ⟨19738⟩ 98244

def event98247 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨19739⟩⟩) (.authority (.operator))

def exact98248RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨19739⟩⟩]⟩, (1)⟩]

theorem exact98248RawTermsValid :
    exact98248RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event98248 : Event := .resultExact (⟨.program ⟨257⟩, ⟨19739⟩⟩) exact98248RawTerms .large 98247 .exactZero (none)

def event98249 : Event := .predecessor (⟨.program ⟨257⟩, ⟨20274⟩⟩) 0 ⟨19739⟩ 98248

def event98250 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨20274⟩⟩) (.authority (.operator))

def exact98251RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨20274⟩⟩]⟩, (1)⟩]

theorem exact98251RawTermsValid :
    exact98251RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event98251 : Event := .resultExact (⟨.program ⟨257⟩, ⟨20274⟩⟩) exact98251RawTerms (.finite 8192) 98250 .exactZero (none)

def event98252 : Event := .predecessor (⟨.program ⟨257⟩, ⟨18397⟩⟩) 0 ⟨18394⟩ 4202

def event98253 : Event := .predecessor (⟨.program ⟨257⟩, ⟨18397⟩⟩) 1 ⟨9904⟩ 90528

def event98254 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨18397⟩⟩) (.tensor (.predecessor 0 98252 .coefficient) (.predecessor 1 98253 .coefficient) true false)

def event98255 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨18397⟩⟩, .operator (⟨4202, 0⟩, ⟨90528, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨18394⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact98256RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨18394⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact98256RawTermsValid :
    exact98256RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event98256 : Event := .resultExact (⟨.program ⟨257⟩, ⟨18397⟩⟩) exact98256RawTerms .large 98254 .exactZero (none)

def event98257 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9939⟩⟩) 0 ⟨9903⟩ 90398

def event98258 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9939⟩⟩) 1 ⟨7305⟩ 25096

def event98259 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9939⟩⟩) (.product (.predecessor 0 98257 .coefficient) (.predecessor 1 98258 .coefficient) (⟨false, false, none, none, none⟩))

def event98260 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨9939⟩⟩, .operator (⟨90398, 0⟩, ⟨25096, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7305⟩⟩]⟩, (1)⟩)

def exact98261RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7305⟩⟩]⟩, (1)⟩]

theorem exact98261RawTermsValid :
    exact98261RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event98261 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9939⟩⟩) exact98261RawTerms .large 98259 .exactZero (none)

def event98262 : Event := .predecessor (⟨.program ⟨257⟩, ⟨18398⟩⟩) 0 ⟨9939⟩ 98261

def event98263 : Event := .predecessor (⟨.program ⟨257⟩, ⟨18398⟩⟩) 1 ⟨18397⟩ 98256

def event98264 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨18398⟩⟩) (.sum [.predecessor 0 98262 .coefficient, .predecessor 1 98263 .coefficient])

def exact98265RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7305⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨18394⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact98265RawTermsValid :
    exact98265RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event98265 : Event := .resultExact (⟨.program ⟨257⟩, ⟨18398⟩⟩) exact98265RawTerms .large 98264 .exactZero (none)

def event98266 : Event := .predecessor (⟨.program ⟨257⟩, ⟨18399⟩⟩) 0 ⟨18398⟩ 98265

def event98267 : Event := .predecessor (⟨.program ⟨257⟩, ⟨18399⟩⟩) 1 ⟨131⟩ 25088

def event98268 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨18399⟩⟩) (.sum [.predecessor 0 98266 .coefficient, .predecessor 1 98267 .coefficient])

def event98269 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨18399⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨131⟩⟩]⟩) [⟨.result 25088 .coefficient, false, none⟩])

def event98270 : Event := .survivorFold (1) 98269

def exact98271RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7305⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨18394⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact98271RawTermsValid :
    exact98271RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event98271 : Event := .resultExact (⟨.program ⟨257⟩, ⟨18399⟩⟩) exact98271RawTerms .large 98268 (.finite 26) (some (98269))

def event98272 : Event := .predecessor (⟨.program ⟨257⟩, ⟨18400⟩⟩) 0 ⟨18399⟩ 98271

def event98273 : Event := .predecessor (⟨.program ⟨257⟩, ⟨18400⟩⟩) 1 ⟨12756⟩ 4205

def event98274 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨18400⟩⟩) (.product (.predecessor 0 98272 .coefficient) (.predecessor 1 98273 .coefficient) (⟨false, true, none, none, some 1⟩))

def event98275 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨18400⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨12756⟩⟩], []⟩) [⟨.result 4205 .coefficient, true, some 1⟩])

def event98276 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨18400⟩⟩) (.product (.result 98271 .summary) (.transfer 98275) (⟨false, false, none, none, none⟩))

def event98277 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨18400⟩⟩, .operator (⟨98271, 1⟩, ⟨4205, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨12756⟩⟩, ⟨.program ⟨257⟩, ⟨18394⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def event98278 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨18400⟩⟩, .operator (⟨98271, 0⟩, ⟨4205, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨12756⟩⟩], [⟨.program ⟨257⟩, ⟨7305⟩⟩]⟩, (1)⟩)

def exact98279RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨12756⟩⟩], [⟨.program ⟨257⟩, ⟨7305⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨12756⟩⟩, ⟨.program ⟨257⟩, ⟨18394⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact98279RawTermsValid :
    exact98279RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event98279 : Event := .resultExact (⟨.program ⟨257⟩, ⟨18400⟩⟩) exact98279RawTerms .large 98274 (.finite 2555904) (some (98276))

def event98280 : Event := .predecessor (⟨.program ⟨257⟩, ⟨12757⟩⟩) 0 ⟨12756⟩ 4205

def event98281 : Event := .predecessor (⟨.program ⟨257⟩, ⟨12757⟩⟩) 1 ⟨9904⟩ 90528

def event98282 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨12757⟩⟩) (.tensor (.predecessor 0 98280 .coefficient) (.predecessor 1 98281 .coefficient) true false)

def event98283 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨12757⟩⟩, .operator (⟨4205, 0⟩, ⟨90528, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨12756⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact98284RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨12756⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact98284RawTermsValid :
    exact98284RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event98284 : Event := .resultExact (⟨.program ⟨257⟩, ⟨12757⟩⟩) exact98284RawTerms .large 98282 .exactZero (none)

def event98285 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9911⟩⟩) 0 ⟨9903⟩ 90398

def event98286 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9911⟩⟩) 1 ⟨7277⟩ 25137

def event98287 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9911⟩⟩) (.product (.predecessor 0 98285 .coefficient) (.predecessor 1 98286 .coefficient) (⟨false, false, none, none, none⟩))

def event98288 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨9911⟩⟩, .operator (⟨90398, 0⟩, ⟨25137, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7277⟩⟩]⟩, (1)⟩)

def exact98289RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7277⟩⟩]⟩, (1)⟩]

theorem exact98289RawTermsValid :
    exact98289RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event98289 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9911⟩⟩) exact98289RawTerms .large 98287 .exactZero (none)

def event98290 : Event := .predecessor (⟨.program ⟨257⟩, ⟨12758⟩⟩) 0 ⟨9911⟩ 98289

def event98291 : Event := .predecessor (⟨.program ⟨257⟩, ⟨12758⟩⟩) 1 ⟨12757⟩ 98284

def event98292 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨12758⟩⟩) (.sum [.predecessor 0 98290 .coefficient, .predecessor 1 98291 .coefficient])

def exact98293RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7277⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨12756⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact98293RawTermsValid :
    exact98293RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event98293 : Event := .resultExact (⟨.program ⟨257⟩, ⟨12758⟩⟩) exact98293RawTerms .large 98292 .exactZero (none)

def event98294 : Event := .predecessor (⟨.program ⟨257⟩, ⟨12759⟩⟩) 0 ⟨12758⟩ 98293

def event98295 : Event := .predecessor (⟨.program ⟨257⟩, ⟨12759⟩⟩) 1 ⟨103⟩ 25129

def event98296 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨12759⟩⟩) (.sum [.predecessor 0 98294 .coefficient, .predecessor 1 98295 .coefficient])

def event98297 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨12759⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨103⟩⟩]⟩) [⟨.result 25129 .coefficient, false, none⟩])

def event98298 : Event := .survivorFold (1) 98297

def exact98299RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7277⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨12756⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact98299RawTermsValid :
    exact98299RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event98299 : Event := .resultExact (⟨.program ⟨257⟩, ⟨12759⟩⟩) exact98299RawTerms .large 98296 (.finite 26) (some (98297))

def event98300 : Event := .predecessor (⟨.program ⟨257⟩, ⟨12760⟩⟩) 0 ⟨12759⟩ 98299

def event98301 : Event := .predecessor (⟨.program ⟨257⟩, ⟨12760⟩⟩) 1 ⟨9572⟩ 25126

def event98302 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨12760⟩⟩) (.product (.predecessor 0 98300 .coefficient) (.predecessor 1 98301 .coefficient) (⟨false, false, none, none, none⟩))

def event98303 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨12760⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨9571⟩⟩]⟩) [⟨.result 25122 .coefficient, false, none⟩])

def eventLeaf6128 : Array AnnotatedEvent := #[
  { event := event98048
    frameStart := 0 },
  { event := event98049
    frameStart := 0 },
  { event := event98050
    frameStart := 0 },
  { event := event98051
    frameStart := 0 },
  { event := event98052
    frameStart := 0 },
  { event := event98053
    frameStart := 0 },
  { event := event98054
    frameStart := 0 },
  { event := event98055
    frameStart := 0 },
  { event := event98056
    frameStart := 0 },
  { event := event98057
    frameStart := 0 },
  { event := event98058
    frameStart := 0 },
  { event := event98059
    frameStart := 0 },
  { event := event98060
    frameStart := 98060 },
  { event := event98061
    frameStart := 98060 },
  { event := event98062
    frameStart := 98060 },
  { event := event98063
    frameStart := 98060 }
]

def eventLeaf6129 : Array AnnotatedEvent := #[
  { event := event98064
    frameStart := 98060 },
  { event := event98065
    frameStart := 98060 },
  { event := event98066
    frameStart := 98060 },
  { event := event98067
    frameStart := 98060 },
  { event := event98068
    frameStart := 98060 },
  { event := event98069
    frameStart := 98060 },
  { event := event98070
    frameStart := 98060 },
  { event := event98071
    frameStart := 98060 },
  { event := event98072
    frameStart := 98060 },
  { event := event98073
    frameStart := 98060 },
  { event := event98074
    frameStart := 98060 },
  { event := event98075
    frameStart := 98060 },
  { event := event98076
    frameStart := 98060 },
  { event := event98077
    frameStart := 98060 },
  { event := event98078
    frameStart := 98060 },
  { event := event98079
    frameStart := 98060 }
]

def eventLeaf6130 : Array AnnotatedEvent := #[
  { event := event98080
    frameStart := 98060 },
  { event := event98081
    frameStart := 98060 },
  { event := event98082
    frameStart := 98060 },
  { event := event98083
    frameStart := 98060 },
  { event := event98084
    frameStart := 98060 },
  { event := event98085
    frameStart := 98060 },
  { event := event98086
    frameStart := 98060 },
  { event := event98087
    frameStart := 98060 },
  { event := event98088
    frameStart := 98060 },
  { event := event98089
    frameStart := 98060 },
  { event := event98090
    frameStart := 98060 },
  { event := event98091
    frameStart := 98060 },
  { event := event98092
    frameStart := 98060 },
  { event := event98093
    frameStart := 98060 },
  { event := event98094
    frameStart := 98060 },
  { event := event98095
    frameStart := 98060 }
]

def eventLeaf6131 : Array AnnotatedEvent := #[
  { event := event98096
    frameStart := 98060 },
  { event := event98097
    frameStart := 98060 },
  { event := event98098
    frameStart := 98060 },
  { event := event98099
    frameStart := 98060 },
  { event := event98100
    frameStart := 98060 },
  { event := event98101
    frameStart := 98060 },
  { event := event98102
    frameStart := 98060 },
  { event := event98103
    frameStart := 98060 },
  { event := event98104
    frameStart := 98060 },
  { event := event98105
    frameStart := 98060 },
  { event := event98106
    frameStart := 98060 },
  { event := event98107
    frameStart := 98060 },
  { event := event98108
    frameStart := 98060 },
  { event := event98109
    frameStart := 98060 },
  { event := event98110
    frameStart := 98060 },
  { event := event98111
    frameStart := 98060 }
]

def eventLeaf6132 : Array AnnotatedEvent := #[
  { event := event98112
    frameStart := 98060 },
  { event := event98113
    frameStart := 98060 },
  { event := event98114
    frameStart := 98114 },
  { event := event98115
    frameStart := 98114 },
  { event := event98116
    frameStart := 98114 },
  { event := event98117
    frameStart := 98114 },
  { event := event98118
    frameStart := 98114 },
  { event := event98119
    frameStart := 98114 },
  { event := event98120
    frameStart := 98114 },
  { event := event98121
    frameStart := 98114 },
  { event := event98122
    frameStart := 98114 },
  { event := event98123
    frameStart := 98114 },
  { event := event98124
    frameStart := 98114 },
  { event := event98125
    frameStart := 98114 },
  { event := event98126
    frameStart := 98114 },
  { event := event98127
    frameStart := 98114 }
]

def eventLeaf6133 : Array AnnotatedEvent := #[
  { event := event98128
    frameStart := 98114 },
  { event := event98129
    frameStart := 98114 },
  { event := event98130
    frameStart := 98114 },
  { event := event98131
    frameStart := 98114 },
  { event := event98132
    frameStart := 98114 },
  { event := event98133
    frameStart := 98114 },
  { event := event98134
    frameStart := 98114 },
  { event := event98135
    frameStart := 98114 },
  { event := event98136
    frameStart := 98114 },
  { event := event98137
    frameStart := 98114 },
  { event := event98138
    frameStart := 98114 },
  { event := event98139
    frameStart := 98114 },
  { event := event98140
    frameStart := 98114 },
  { event := event98141
    frameStart := 98114 },
  { event := event98142
    frameStart := 98114 },
  { event := event98143
    frameStart := 98114 }
]

def eventLeaf6134 : Array AnnotatedEvent := #[
  { event := event98144
    frameStart := 98114 },
  { event := event98145
    frameStart := 98114 },
  { event := event98146
    frameStart := 98114 },
  { event := event98147
    frameStart := 98114 },
  { event := event98148
    frameStart := 98114 },
  { event := event98149
    frameStart := 98114 },
  { event := event98150
    frameStart := 98114 },
  { event := event98151
    frameStart := 98114 },
  { event := event98152
    frameStart := 98114 },
  { event := event98153
    frameStart := 98114 },
  { event := event98154
    frameStart := 98114 },
  { event := event98155
    frameStart := 98114 },
  { event := event98156
    frameStart := 98114 },
  { event := event98157
    frameStart := 98114 },
  { event := event98158
    frameStart := 98114 },
  { event := event98159
    frameStart := 98114 }
]

def eventLeaf6135 : Array AnnotatedEvent := #[
  { event := event98160
    frameStart := 98114 },
  { event := event98161
    frameStart := 98114 },
  { event := event98162
    frameStart := 98114 },
  { event := event98163
    frameStart := 98114 },
  { event := event98164
    frameStart := 98114 },
  { event := event98165
    frameStart := 98114 },
  { event := event98166
    frameStart := 98114 },
  { event := event98167
    frameStart := 98114 },
  { event := event98168
    frameStart := 98114 },
  { event := event98169
    frameStart := 98114 },
  { event := event98170
    frameStart := 98114 },
  { event := event98171
    frameStart := 98114 },
  { event := event98172
    frameStart := 98114 },
  { event := event98173
    frameStart := 98114 },
  { event := event98174
    frameStart := 98114 },
  { event := event98175
    frameStart := 98114 }
]

def eventLeaf6136 : Array AnnotatedEvent := #[
  { event := event98176
    frameStart := 98114 },
  { event := event98177
    frameStart := 98114 },
  { event := event98178
    frameStart := 98114 },
  { event := event98179
    frameStart := 98114 },
  { event := event98180
    frameStart := 98114 },
  { event := event98181
    frameStart := 98114 },
  { event := event98182
    frameStart := 98114 },
  { event := event98183
    frameStart := 98114 },
  { event := event98184
    frameStart := 98114 },
  { event := event98185
    frameStart := 98114 },
  { event := event98186
    frameStart := 98114 },
  { event := event98187
    frameStart := 98114 },
  { event := event98188
    frameStart := 98114 },
  { event := event98189
    frameStart := 98114 },
  { event := event98190
    frameStart := 98114 },
  { event := event98191
    frameStart := 98114 }
]

def eventLeaf6137 : Array AnnotatedEvent := #[
  { event := event98192
    frameStart := 98114 },
  { event := event98193
    frameStart := 98114 },
  { event := event98194
    frameStart := 98114 },
  { event := event98195
    frameStart := 98114 },
  { event := event98196
    frameStart := 98114 },
  { event := event98197
    frameStart := 98114 },
  { event := event98198
    frameStart := 98114 },
  { event := event98199
    frameStart := 98114 },
  { event := event98200
    frameStart := 98114 },
  { event := event98201
    frameStart := 98114 },
  { event := event98202
    frameStart := 98114 },
  { event := event98203
    frameStart := 98114 },
  { event := event98204
    frameStart := 98114 },
  { event := event98205
    frameStart := 98114 },
  { event := event98206
    frameStart := 98114 },
  { event := event98207
    frameStart := 98114 }
]

def eventLeaf6138 : Array AnnotatedEvent := #[
  { event := event98208
    frameStart := 98114 },
  { event := event98209
    frameStart := 98114 },
  { event := event98210
    frameStart := 98114 },
  { event := event98211
    frameStart := 98114 },
  { event := event98212
    frameStart := 98114 },
  { event := event98213
    frameStart := 98114 },
  { event := event98214
    frameStart := 98114 },
  { event := event98215
    frameStart := 98114 },
  { event := event98216
    frameStart := 98114 },
  { event := event98217
    frameStart := 98114 },
  { event := event98218
    frameStart := 0 },
  { event := event98219
    frameStart := 0 },
  { event := event98220
    frameStart := 0 },
  { event := event98221
    frameStart := 0 },
  { event := event98222
    frameStart := 0 },
  { event := event98223
    frameStart := 0 }
]

def eventLeaf6139 : Array AnnotatedEvent := #[
  { event := event98224
    frameStart := 0 },
  { event := event98225
    frameStart := 0 },
  { event := event98226
    frameStart := 0 },
  { event := event98227
    frameStart := 0 },
  { event := event98228
    frameStart := 0 },
  { event := event98229
    frameStart := 0 },
  { event := event98230
    frameStart := 0 },
  { event := event98231
    frameStart := 0 },
  { event := event98232
    frameStart := 0 },
  { event := event98233
    frameStart := 0 },
  { event := event98234
    frameStart := 0 },
  { event := event98235
    frameStart := 0 },
  { event := event98236
    frameStart := 0 },
  { event := event98237
    frameStart := 0 },
  { event := event98238
    frameStart := 0 },
  { event := event98239
    frameStart := 0 }
]

def eventLeaf6140 : Array AnnotatedEvent := #[
  { event := event98240
    frameStart := 0 },
  { event := event98241
    frameStart := 0 },
  { event := event98242
    frameStart := 0 },
  { event := event98243
    frameStart := 0 },
  { event := event98244
    frameStart := 0 },
  { event := event98245
    frameStart := 0 },
  { event := event98246
    frameStart := 0 },
  { event := event98247
    frameStart := 0 },
  { event := event98248
    frameStart := 0 },
  { event := event98249
    frameStart := 0 },
  { event := event98250
    frameStart := 0 },
  { event := event98251
    frameStart := 0 },
  { event := event98252
    frameStart := 0 },
  { event := event98253
    frameStart := 0 },
  { event := event98254
    frameStart := 0 },
  { event := event98255
    frameStart := 0 }
]

def eventLeaf6141 : Array AnnotatedEvent := #[
  { event := event98256
    frameStart := 0 },
  { event := event98257
    frameStart := 0 },
  { event := event98258
    frameStart := 0 },
  { event := event98259
    frameStart := 0 },
  { event := event98260
    frameStart := 0 },
  { event := event98261
    frameStart := 0 },
  { event := event98262
    frameStart := 0 },
  { event := event98263
    frameStart := 0 },
  { event := event98264
    frameStart := 0 },
  { event := event98265
    frameStart := 0 },
  { event := event98266
    frameStart := 0 },
  { event := event98267
    frameStart := 0 },
  { event := event98268
    frameStart := 0 },
  { event := event98269
    frameStart := 0 },
  { event := event98270
    frameStart := 0 },
  { event := event98271
    frameStart := 0 }
]

def eventLeaf6142 : Array AnnotatedEvent := #[
  { event := event98272
    frameStart := 0 },
  { event := event98273
    frameStart := 0 },
  { event := event98274
    frameStart := 0 },
  { event := event98275
    frameStart := 0 },
  { event := event98276
    frameStart := 0 },
  { event := event98277
    frameStart := 0 },
  { event := event98278
    frameStart := 0 },
  { event := event98279
    frameStart := 0 },
  { event := event98280
    frameStart := 0 },
  { event := event98281
    frameStart := 0 },
  { event := event98282
    frameStart := 0 },
  { event := event98283
    frameStart := 0 },
  { event := event98284
    frameStart := 0 },
  { event := event98285
    frameStart := 0 },
  { event := event98286
    frameStart := 0 },
  { event := event98287
    frameStart := 0 }
]

def eventLeaf6143 : Array AnnotatedEvent := #[
  { event := event98288
    frameStart := 0 },
  { event := event98289
    frameStart := 0 },
  { event := event98290
    frameStart := 0 },
  { event := event98291
    frameStart := 0 },
  { event := event98292
    frameStart := 0 },
  { event := event98293
    frameStart := 0 },
  { event := event98294
    frameStart := 0 },
  { event := event98295
    frameStart := 0 },
  { event := event98296
    frameStart := 0 },
  { event := event98297
    frameStart := 0 },
  { event := event98298
    frameStart := 0 },
  { event := event98299
    frameStart := 0 },
  { event := event98300
    frameStart := 0 },
  { event := event98301
    frameStart := 0 },
  { event := event98302
    frameStart := 0 },
  { event := event98303
    frameStart := 0 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events383
