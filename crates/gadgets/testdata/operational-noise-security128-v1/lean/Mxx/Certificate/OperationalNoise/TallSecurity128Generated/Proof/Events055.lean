import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events055

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event14080 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37562⟩⟩) (.product (.predecessor 0 14078 .coefficient) (.predecessor 1 14079 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event14081 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨37562⟩⟩, .operator (⟨14077, 0⟩, ⟨583, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37561⟩⟩], []⟩, (1)⟩)

def exact14082RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37561⟩⟩], []⟩, (1)⟩]

theorem exact14082RawTermsValid :
    exact14082RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event14082 : Event := .resultExact (⟨.program ⟨257⟩, ⟨37562⟩⟩) exact14082RawTerms (.finite 229121489167213617734760) 14080 .exactZero (none)

def event14083 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34881⟩⟩) 0 ⟨34701⟩ 13684

def event14084 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34881⟩⟩) (.authority (.programFamilyFact))

def exact14085RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨34881⟩⟩], []⟩, (1)⟩]

theorem exact14085RawTermsValid :
    exact14085RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event14085 : Event := .resultExact (⟨.program ⟨257⟩, ⟨34881⟩⟩) exact14085RawTerms (.finite 40) 14084 .exactZero (none)

def event14086 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34882⟩⟩) 0 ⟨34881⟩ 14085

def event14087 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34882⟩⟩) 1 ⟨6842⟩ 593

def event14088 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34882⟩⟩) (.product (.predecessor 0 14086 .coefficient) (.predecessor 1 14087 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event14089 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨34882⟩⟩, .operator (⟨14085, 0⟩, ⟨593, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34881⟩⟩], []⟩, (1)⟩)

def exact14090RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34881⟩⟩], []⟩, (1)⟩]

theorem exact14090RawTermsValid :
    exact14090RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event14090 : Event := .resultExact (⟨.program ⟨257⟩, ⟨34882⟩⟩) exact14090RawTerms (.finite 228855378262257504357600) 14088 .exactZero (none)

def event14091 : Event := .predecessor (⟨.program ⟨257⟩, ⟨29224⟩⟩) 0 ⟨29041⟩ 13707

def event14092 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨29224⟩⟩) (.authority (.programFamilyFact))

def exact14093RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨29224⟩⟩], []⟩, (1)⟩]

theorem exact14093RawTermsValid :
    exact14093RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event14093 : Event := .resultExact (⟨.program ⟨257⟩, ⟨29224⟩⟩) exact14093RawTerms (.finite 36) 14092 .exactZero (none)

def event14094 : Event := .predecessor (⟨.program ⟨257⟩, ⟨29225⟩⟩) 0 ⟨29224⟩ 14093

def event14095 : Event := .predecessor (⟨.program ⟨257⟩, ⟨29225⟩⟩) 1 ⟨6857⟩ 603

def event14096 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨29225⟩⟩) (.product (.predecessor 0 14094 .coefficient) (.predecessor 1 14095 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event14097 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨29225⟩⟩, .operator (⟨14093, 0⟩, ⟨603, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29224⟩⟩], []⟩, (1)⟩)

def exact14098RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29224⟩⟩], []⟩, (1)⟩]

theorem exact14098RawTermsValid :
    exact14098RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event14098 : Event := .resultExact (⟨.program ⟨257⟩, ⟨29225⟩⟩) exact14098RawTerms (.finite 228236850212900051643120) 14096 .exactZero (none)

def event14099 : Event := .predecessor (⟨.program ⟨257⟩, ⟨26544⟩⟩) 0 ⟨26361⟩ 13730

def event14100 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨26544⟩⟩) (.authority (.programFamilyFact))

def exact14101RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨26544⟩⟩], []⟩, (1)⟩]

theorem exact14101RawTermsValid :
    exact14101RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event14101 : Event := .resultExact (⟨.program ⟨257⟩, ⟨26544⟩⟩) exact14101RawTerms (.finite 30) 14100 .exactZero (none)

def event14102 : Event := .predecessor (⟨.program ⟨257⟩, ⟨26545⟩⟩) 0 ⟨26544⟩ 14101

def event14103 : Event := .predecessor (⟨.program ⟨257⟩, ⟨26545⟩⟩) 1 ⟨6860⟩ 613

def event14104 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨26545⟩⟩) (.product (.predecessor 0 14102 .coefficient) (.predecessor 1 14103 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event14105 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨26545⟩⟩, .operator (⟨14101, 0⟩, ⟨613, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26544⟩⟩], []⟩, (1)⟩)

def exact14106RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26544⟩⟩], []⟩, (1)⟩]

theorem exact14106RawTermsValid :
    exact14106RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event14106 : Event := .resultExact (⟨.program ⟨257⟩, ⟨26545⟩⟩) exact14106RawTerms (.finite 227009770373045750290200) 14104 .exactZero (none)

def event14107 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66168⟩⟩) 0 ⟨65741⟩ 13753

def event14108 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨66168⟩⟩) (.authority (.programFamilyFact))

def exact14109RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨66168⟩⟩], []⟩, (1)⟩]

theorem exact14109RawTermsValid :
    exact14109RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event14109 : Event := .resultExact (⟨.program ⟨257⟩, ⟨66168⟩⟩) exact14109RawTerms (.finite 28) 14108 .exactZero (none)

def event14110 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66169⟩⟩) 0 ⟨66168⟩ 14109

def event14111 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66169⟩⟩) 1 ⟨6870⟩ 623

def event14112 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨66169⟩⟩) (.product (.predecessor 0 14110 .coefficient) (.predecessor 1 14111 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event14113 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨66169⟩⟩, .operator (⟨14109, 0⟩, ⟨623, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66168⟩⟩], []⟩, (1)⟩)

def exact14114RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66168⟩⟩], []⟩, (1)⟩]

theorem exact14114RawTermsValid :
    exact14114RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event14114 : Event := .resultExact (⟨.program ⟨257⟩, ⟨66169⟩⟩) exact14114RawTerms (.finite 226487908831958288795280) 14112 .exactZero (none)

def event14115 : Event := .predecessor (⟨.program ⟨257⟩, ⟨62971⟩⟩) 0 ⟨62761⟩ 13776

def event14116 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨62971⟩⟩) (.authority (.programFamilyFact))

def exact14117RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨62971⟩⟩], []⟩, (1)⟩]

theorem exact14117RawTermsValid :
    exact14117RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event14117 : Event := .resultExact (⟨.program ⟨257⟩, ⟨62971⟩⟩) exact14117RawTerms (.finite 22) 14116 .exactZero (none)

def event14118 : Event := .predecessor (⟨.program ⟨257⟩, ⟨62972⟩⟩) 0 ⟨62971⟩ 14117

def event14119 : Event := .predecessor (⟨.program ⟨257⟩, ⟨62972⟩⟩) 1 ⟨6732⟩ 633

def event14120 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨62972⟩⟩) (.product (.predecessor 0 14118 .coefficient) (.predecessor 1 14119 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event14121 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨62972⟩⟩, .operator (⟨14117, 0⟩, ⟨633, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨62971⟩⟩], []⟩, (1)⟩)

def exact14122RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨62971⟩⟩], []⟩, (1)⟩]

theorem exact14122RawTermsValid :
    exact14122RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event14122 : Event := .resultExact (⟨.program ⟨257⟩, ⟨62972⟩⟩) exact14122RawTerms (.finite 224377773035387248837560) 14120 .exactZero (none)

def event14123 : Event := .predecessor (⟨.program ⟨257⟩, ⟨59991⟩⟩) 0 ⟨59781⟩ 13799

def event14124 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨59991⟩⟩) (.authority (.programFamilyFact))

def exact14125RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨59991⟩⟩], []⟩, (1)⟩]

theorem exact14125RawTermsValid :
    exact14125RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event14125 : Event := .resultExact (⟨.program ⟨257⟩, ⟨59991⟩⟩) exact14125RawTerms (.finite 18) 14124 .exactZero (none)

def event14126 : Event := .predecessor (⟨.program ⟨257⟩, ⟨59992⟩⟩) 0 ⟨59991⟩ 14125

def event14127 : Event := .predecessor (⟨.program ⟨257⟩, ⟨59992⟩⟩) 1 ⟨6736⟩ 643

def event14128 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨59992⟩⟩) (.product (.predecessor 0 14126 .coefficient) (.predecessor 1 14127 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event14129 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨59992⟩⟩, .operator (⟨14125, 0⟩, ⟨643, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨59991⟩⟩], []⟩, (1)⟩)

def exact14130RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨59991⟩⟩], []⟩, (1)⟩]

theorem exact14130RawTermsValid :
    exact14130RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event14130 : Event := .resultExact (⟨.program ⟨257⟩, ⟨59992⟩⟩) exact14130RawTerms (.finite 222230617312560576599880) 14128 .exactZero (none)

def event14131 : Event := .predecessor (⟨.program ⟨257⟩, ⟨57011⟩⟩) 0 ⟨56801⟩ 13822

def event14132 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨57011⟩⟩) (.authority (.programFamilyFact))

def exact14133RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨57011⟩⟩], []⟩, (1)⟩]

theorem exact14133RawTermsValid :
    exact14133RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event14133 : Event := .resultExact (⟨.program ⟨257⟩, ⟨57011⟩⟩) exact14133RawTerms (.finite 16) 14132 .exactZero (none)

def event14134 : Event := .predecessor (⟨.program ⟨257⟩, ⟨57012⟩⟩) 0 ⟨57011⟩ 14133

def event14135 : Event := .predecessor (⟨.program ⟨257⟩, ⟨57012⟩⟩) 1 ⟨6741⟩ 653

def event14136 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨57012⟩⟩) (.product (.predecessor 0 14134 .coefficient) (.predecessor 1 14135 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event14137 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨57012⟩⟩, .operator (⟨14133, 0⟩, ⟨653, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57011⟩⟩], []⟩, (1)⟩)

def exact14138RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57011⟩⟩], []⟩, (1)⟩]

theorem exact14138RawTermsValid :
    exact14138RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event14138 : Event := .resultExact (⟨.program ⟨257⟩, ⟨57012⟩⟩) exact14138RawTerms (.finite 220778129617707239497920) 14136 .exactZero (none)

def event14139 : Event := .predecessor (⟨.program ⟨257⟩, ⟨54031⟩⟩) 0 ⟨53821⟩ 13845

def event14140 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨54031⟩⟩) (.authority (.programFamilyFact))

def exact14141RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨54031⟩⟩], []⟩, (1)⟩]

theorem exact14141RawTermsValid :
    exact14141RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event14141 : Event := .resultExact (⟨.program ⟨257⟩, ⟨54031⟩⟩) exact14141RawTerms (.finite 12) 14140 .exactZero (none)

def event14142 : Event := .predecessor (⟨.program ⟨257⟩, ⟨54032⟩⟩) 0 ⟨54031⟩ 14141

def event14143 : Event := .predecessor (⟨.program ⟨257⟩, ⟨54032⟩⟩) 1 ⟨6757⟩ 663

def event14144 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨54032⟩⟩) (.product (.predecessor 0 14142 .coefficient) (.predecessor 1 14143 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event14145 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨54032⟩⟩, .operator (⟨14141, 0⟩, ⟨663, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54031⟩⟩], []⟩, (1)⟩)

def exact14146RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54031⟩⟩], []⟩, (1)⟩]

theorem exact14146RawTermsValid :
    exact14146RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event14146 : Event := .resultExact (⟨.program ⟨257⟩, ⟨54032⟩⟩) exact14146RawTerms (.finite 216532396355828254122960) 14144 .exactZero (none)

def event14147 : Event := .predecessor (⟨.program ⟨257⟩, ⟨51051⟩⟩) 0 ⟨50841⟩ 13868

def event14148 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨51051⟩⟩) (.authority (.programFamilyFact))

def exact14149RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨51051⟩⟩], []⟩, (1)⟩]

theorem exact14149RawTermsValid :
    exact14149RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event14149 : Event := .resultExact (⟨.program ⟨257⟩, ⟨51051⟩⟩) exact14149RawTerms (.finite 10) 14148 .exactZero (none)

def event14150 : Event := .predecessor (⟨.program ⟨257⟩, ⟨51052⟩⟩) 0 ⟨51051⟩ 14149

def event14151 : Event := .predecessor (⟨.program ⟨257⟩, ⟨51052⟩⟩) 1 ⟨6768⟩ 673

def event14152 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨51052⟩⟩) (.product (.predecessor 0 14150 .coefficient) (.predecessor 1 14151 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event14153 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨51052⟩⟩, .operator (⟨14149, 0⟩, ⟨673, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51051⟩⟩], []⟩, (1)⟩)

def exact14154RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51051⟩⟩], []⟩, (1)⟩]

theorem exact14154RawTermsValid :
    exact14154RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event14154 : Event := .resultExact (⟨.program ⟨257⟩, ⟨51052⟩⟩) exact14154RawTerms (.finite 213251602471649038151400) 14152 .exactZero (none)

def event14155 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31987⟩⟩) 0 ⟨31781⟩ 13891

def event14156 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31987⟩⟩) (.authority (.programFamilyFact))

def exact14157RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨31987⟩⟩], []⟩, (1)⟩]

theorem exact14157RawTermsValid :
    exact14157RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event14157 : Event := .resultExact (⟨.program ⟨257⟩, ⟨31987⟩⟩) exact14157RawTerms (.finite 6) 14156 .exactZero (none)

def event14158 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31988⟩⟩) 0 ⟨31987⟩ 14157

def event14159 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31988⟩⟩) 1 ⟨6794⟩ 683

def event14160 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31988⟩⟩) (.product (.predecessor 0 14158 .coefficient) (.predecessor 1 14159 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event14161 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨31988⟩⟩, .operator (⟨14157, 0⟩, ⟨683, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨31987⟩⟩], []⟩, (1)⟩)

def exact14162RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨31987⟩⟩], []⟩, (1)⟩]

theorem exact14162RawTermsValid :
    exact14162RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event14162 : Event := .resultExact (⟨.program ⟨257⟩, ⟨31988⟩⟩) exact14162RawTerms (.finite 201065796616126235971320) 14160 .exactZero (none)

def event14163 : Event := .predecessor (⟨.program ⟨257⟩, ⟨21967⟩⟩) 0 ⟨21761⟩ 13914

def event14164 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨21967⟩⟩) (.authority (.programFamilyFact))

def exact14165RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨21967⟩⟩], []⟩, (1)⟩]

theorem exact14165RawTermsValid :
    exact14165RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event14165 : Event := .resultExact (⟨.program ⟨257⟩, ⟨21967⟩⟩) exact14165RawTerms (.finite 4) 14164 .exactZero (none)

def event14166 : Event := .predecessor (⟨.program ⟨257⟩, ⟨21968⟩⟩) 0 ⟨21967⟩ 14165

def event14167 : Event := .predecessor (⟨.program ⟨257⟩, ⟨21968⟩⟩) 1 ⟨6822⟩ 693

def event14168 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨21968⟩⟩) (.product (.predecessor 0 14166 .coefficient) (.predecessor 1 14167 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event14169 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨21968⟩⟩, .operator (⟨14165, 0⟩, ⟨693, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨21967⟩⟩], []⟩, (1)⟩)

def exact14170RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨21967⟩⟩], []⟩, (1)⟩]

theorem exact14170RawTermsValid :
    exact14170RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event14170 : Event := .resultExact (⟨.program ⟨257⟩, ⟨21968⟩⟩) exact14170RawTerms (.finite 187661410175051153573232) 14168 .exactZero (none)

def event14171 : Event := .predecessor (⟨.program ⟨257⟩, ⟨18747⟩⟩) 0 ⟨18541⟩ 13937

def event14172 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨18747⟩⟩) (.authority (.programFamilyFact))

def exact14173RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨18747⟩⟩], []⟩, (1)⟩]

theorem exact14173RawTermsValid :
    exact14173RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event14173 : Event := .resultExact (⟨.program ⟨257⟩, ⟨18747⟩⟩) exact14173RawTerms (.finite 3) 14172 .exactZero (none)

def event14174 : Event := .predecessor (⟨.program ⟨257⟩, ⟨18748⟩⟩) 0 ⟨18747⟩ 14173

def event14175 : Event := .predecessor (⟨.program ⟨257⟩, ⟨18748⟩⟩) 1 ⟨6846⟩ 703

def event14176 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨18748⟩⟩) (.product (.predecessor 0 14174 .coefficient) (.predecessor 1 14175 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event14177 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨18748⟩⟩, .operator (⟨14173, 0⟩, ⟨703, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18747⟩⟩], []⟩, (1)⟩)

def exact14178RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18747⟩⟩], []⟩, (1)⟩]

theorem exact14178RawTermsValid :
    exact14178RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event14178 : Event := .resultExact (⟨.program ⟨257⟩, ⟨18748⟩⟩) exact14178RawTerms (.finite 175932572039110456474905) 14176 .exactZero (none)

def event14179 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15934⟩⟩) 0 ⟨15741⟩ 13960

def event14180 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15934⟩⟩) (.authority (.programFamilyFact))

def exact14181RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15934⟩⟩], []⟩, (1)⟩]

theorem exact14181RawTermsValid :
    exact14181RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event14181 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15934⟩⟩) exact14181RawTerms (.finite 2) 14180 .exactZero (none)

def event14182 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15935⟩⟩) 0 ⟨15934⟩ 14181

def event14183 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15935⟩⟩) 1 ⟨6863⟩ 713

def event14184 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15935⟩⟩) (.product (.predecessor 0 14182 .coefficient) (.predecessor 1 14183 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event14185 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨15935⟩⟩, .operator (⟨14181, 0⟩, ⟨713, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15934⟩⟩], []⟩, (1)⟩)

def exact14186RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15934⟩⟩], []⟩, (1)⟩]

theorem exact14186RawTermsValid :
    exact14186RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event14186 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15935⟩⟩) exact14186RawTerms (.finite 156384508479209294644360) 14184 .exactZero (none)

def event14187 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15936⟩⟩) 0 ⟨6728⟩ 728

def event14188 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15936⟩⟩) 1 ⟨15935⟩ 14186

def event14189 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15936⟩⟩) (.sum [.predecessor 0 14187 .coefficient, .predecessor 1 14188 .coefficient])

def exact14190RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15934⟩⟩], []⟩, (1)⟩]

theorem exact14190RawTermsValid :
    exact14190RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event14190 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15936⟩⟩) exact14190RawTerms (.finite 156384508479209294644360) 14189 .exactZero (none)

def event14191 : Event := .predecessor (⟨.program ⟨257⟩, ⟨18749⟩⟩) 0 ⟨15936⟩ 14190

def event14192 : Event := .predecessor (⟨.program ⟨257⟩, ⟨18749⟩⟩) 1 ⟨18748⟩ 14178

def event14193 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨18749⟩⟩) (.sum [.predecessor 0 14191 .coefficient, .predecessor 1 14192 .coefficient])

def exact14194RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18747⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15934⟩⟩], []⟩, (1)⟩]

theorem exact14194RawTermsValid :
    exact14194RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event14194 : Event := .resultExact (⟨.program ⟨257⟩, ⟨18749⟩⟩) exact14194RawTerms (.finite 332317080518319751119265) 14193 .exactZero (none)

def event14195 : Event := .predecessor (⟨.program ⟨257⟩, ⟨21969⟩⟩) 0 ⟨18749⟩ 14194

def event14196 : Event := .predecessor (⟨.program ⟨257⟩, ⟨21969⟩⟩) 1 ⟨21968⟩ 14170

def event14197 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨21969⟩⟩) (.sum [.predecessor 0 14195 .coefficient, .predecessor 1 14196 .coefficient])

def exact14198RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨21967⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18747⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15934⟩⟩], []⟩, (1)⟩]

theorem exact14198RawTermsValid :
    exact14198RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event14198 : Event := .resultExact (⟨.program ⟨257⟩, ⟨21969⟩⟩) exact14198RawTerms (.finite 519978490693370904692497) 14197 .exactZero (none)

def event14199 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31989⟩⟩) 0 ⟨21969⟩ 14198

def event14200 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31989⟩⟩) 1 ⟨31988⟩ 14162

def event14201 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31989⟩⟩) (.sum [.predecessor 0 14199 .coefficient, .predecessor 1 14200 .coefficient])

def exact14202RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨31987⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨21967⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18747⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15934⟩⟩], []⟩, (1)⟩]

theorem exact14202RawTermsValid :
    exact14202RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event14202 : Event := .resultExact (⟨.program ⟨257⟩, ⟨31989⟩⟩) exact14202RawTerms (.finite 721044287309497140663817) 14201 .exactZero (none)

def event14203 : Event := .predecessor (⟨.program ⟨257⟩, ⟨51053⟩⟩) 0 ⟨31989⟩ 14202

def event14204 : Event := .predecessor (⟨.program ⟨257⟩, ⟨51053⟩⟩) 1 ⟨51052⟩ 14154

def event14205 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨51053⟩⟩) (.sum [.predecessor 0 14203 .coefficient, .predecessor 1 14204 .coefficient])

def exact14206RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51051⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨31987⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨21967⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18747⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15934⟩⟩], []⟩, (1)⟩]

theorem exact14206RawTermsValid :
    exact14206RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event14206 : Event := .resultExact (⟨.program ⟨257⟩, ⟨51053⟩⟩) exact14206RawTerms (.finite 934295889781146178815217) 14205 .exactZero (none)

def event14207 : Event := .predecessor (⟨.program ⟨257⟩, ⟨54033⟩⟩) 0 ⟨51053⟩ 14206

def event14208 : Event := .predecessor (⟨.program ⟨257⟩, ⟨54033⟩⟩) 1 ⟨54032⟩ 14146

def event14209 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨54033⟩⟩) (.sum [.predecessor 0 14207 .coefficient, .predecessor 1 14208 .coefficient])

def exact14210RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54031⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51051⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨31987⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨21967⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18747⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15934⟩⟩], []⟩, (1)⟩]

theorem exact14210RawTermsValid :
    exact14210RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event14210 : Event := .resultExact (⟨.program ⟨257⟩, ⟨54033⟩⟩) exact14210RawTerms (.finite 1150828286136974432938177) 14209 .exactZero (none)

def event14211 : Event := .predecessor (⟨.program ⟨257⟩, ⟨57013⟩⟩) 0 ⟨54033⟩ 14210

def event14212 : Event := .predecessor (⟨.program ⟨257⟩, ⟨57013⟩⟩) 1 ⟨57012⟩ 14138

def event14213 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨57013⟩⟩) (.sum [.predecessor 0 14211 .coefficient, .predecessor 1 14212 .coefficient])

def exact14214RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57011⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54031⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51051⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨31987⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨21967⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18747⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15934⟩⟩], []⟩, (1)⟩]

theorem exact14214RawTermsValid :
    exact14214RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event14214 : Event := .resultExact (⟨.program ⟨257⟩, ⟨57013⟩⟩) exact14214RawTerms (.finite 1371606415754681672436097) 14213 .exactZero (none)

def event14215 : Event := .predecessor (⟨.program ⟨257⟩, ⟨59993⟩⟩) 0 ⟨57013⟩ 14214

def event14216 : Event := .predecessor (⟨.program ⟨257⟩, ⟨59993⟩⟩) 1 ⟨59992⟩ 14130

def event14217 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨59993⟩⟩) (.sum [.predecessor 0 14215 .coefficient, .predecessor 1 14216 .coefficient])

def exact14218RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨59991⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57011⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54031⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51051⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨31987⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨21967⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18747⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15934⟩⟩], []⟩, (1)⟩]

theorem exact14218RawTermsValid :
    exact14218RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event14218 : Event := .resultExact (⟨.program ⟨257⟩, ⟨59993⟩⟩) exact14218RawTerms (.finite 1593837033067242249035977) 14217 .exactZero (none)

def event14219 : Event := .predecessor (⟨.program ⟨257⟩, ⟨62973⟩⟩) 0 ⟨59993⟩ 14218

def event14220 : Event := .predecessor (⟨.program ⟨257⟩, ⟨62973⟩⟩) 1 ⟨62972⟩ 14122

def event14221 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨62973⟩⟩) (.sum [.predecessor 0 14219 .coefficient, .predecessor 1 14220 .coefficient])

def exact14222RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨62971⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨59991⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57011⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54031⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51051⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨31987⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨21967⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18747⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15934⟩⟩], []⟩, (1)⟩]

theorem exact14222RawTermsValid :
    exact14222RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event14222 : Event := .resultExact (⟨.program ⟨257⟩, ⟨62973⟩⟩) exact14222RawTerms (.finite 1818214806102629497873537) 14221 .exactZero (none)

def event14223 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66170⟩⟩) 0 ⟨62973⟩ 14222

def event14224 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66170⟩⟩) 1 ⟨66169⟩ 14114

def event14225 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨66170⟩⟩) (.sum [.predecessor 0 14223 .coefficient, .predecessor 1 14224 .coefficient])

def exact14226RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨62971⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨59991⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57011⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54031⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51051⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨31987⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨21967⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18747⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15934⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66168⟩⟩], []⟩, (1)⟩]

theorem exact14226RawTermsValid :
    exact14226RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event14226 : Event := .resultExact (⟨.program ⟨257⟩, ⟨66170⟩⟩) exact14226RawTerms (.finite 2044702714934587786668817) 14225 .exactZero (none)

def event14227 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66171⟩⟩) 0 ⟨66170⟩ 14226

def event14228 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66171⟩⟩) 1 ⟨26545⟩ 14106

def event14229 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨66171⟩⟩) (.sum [.predecessor 0 14227 .coefficient, .predecessor 1 14228 .coefficient])

def exact14230RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨62971⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨59991⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57011⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54031⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51051⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨31987⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨21967⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18747⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26544⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15934⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66168⟩⟩], []⟩, (1)⟩]

theorem exact14230RawTermsValid :
    exact14230RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event14230 : Event := .resultExact (⟨.program ⟨257⟩, ⟨66171⟩⟩) exact14230RawTerms (.finite 2271712485307633536959017) 14229 .exactZero (none)

def event14231 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66172⟩⟩) 0 ⟨66171⟩ 14230

def event14232 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66172⟩⟩) 1 ⟨29225⟩ 14098

def event14233 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨66172⟩⟩) (.sum [.predecessor 0 14231 .coefficient, .predecessor 1 14232 .coefficient])

def exact14234RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨62971⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨59991⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57011⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54031⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51051⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨31987⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨21967⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18747⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29224⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26544⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15934⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66168⟩⟩], []⟩, (1)⟩]

theorem exact14234RawTermsValid :
    exact14234RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event14234 : Event := .resultExact (⟨.program ⟨257⟩, ⟨66172⟩⟩) exact14234RawTerms (.finite 2499949335520533588602137) 14233 .exactZero (none)

def event14235 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66173⟩⟩) 0 ⟨66172⟩ 14234

def event14236 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66173⟩⟩) 1 ⟨34882⟩ 14090

def event14237 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨66173⟩⟩) (.sum [.predecessor 0 14235 .coefficient, .predecessor 1 14236 .coefficient])

def exact14238RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨62971⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨59991⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57011⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54031⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51051⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨31987⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨21967⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34881⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18747⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29224⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26544⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15934⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66168⟩⟩], []⟩, (1)⟩]

theorem exact14238RawTermsValid :
    exact14238RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event14238 : Event := .resultExact (⟨.program ⟨257⟩, ⟨66173⟩⟩) exact14238RawTerms (.finite 2728804713782791092959737) 14237 .exactZero (none)

def event14239 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66174⟩⟩) 0 ⟨66173⟩ 14238

def event14240 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66174⟩⟩) 1 ⟨37562⟩ 14082

def event14241 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨66174⟩⟩) (.sum [.predecessor 0 14239 .coefficient, .predecessor 1 14240 .coefficient])

def exact14242RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨62971⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨59991⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57011⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54031⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51051⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨31987⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨21967⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37561⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34881⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18747⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29224⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26544⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15934⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66168⟩⟩], []⟩, (1)⟩]

theorem exact14242RawTermsValid :
    exact14242RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event14242 : Event := .resultExact (⟨.program ⟨257⟩, ⟨66174⟩⟩) exact14242RawTerms (.finite 2957926202950004710694497) 14241 .exactZero (none)

def event14243 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66175⟩⟩) 0 ⟨66174⟩ 14242

def event14244 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66175⟩⟩) 1 ⟨40245⟩ 14074

def event14245 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨66175⟩⟩) (.sum [.predecessor 0 14243 .coefficient, .predecessor 1 14244 .coefficient])

def exact14246RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨62971⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨59991⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57011⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54031⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51051⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨31987⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨21967⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40244⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37561⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34881⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18747⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29224⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26544⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15934⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66168⟩⟩], []⟩, (1)⟩]

theorem exact14246RawTermsValid :
    exact14246RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event14246 : Event := .resultExact (⟨.program ⟨257⟩, ⟨66175⟩⟩) exact14246RawTerms (.finite 3187511970717354526236217) 14245 .exactZero (none)

def event14247 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66176⟩⟩) 0 ⟨66175⟩ 14246

def event14248 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66176⟩⟩) 1 ⟨42925⟩ 14066

def event14249 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨66176⟩⟩) (.sum [.predecessor 0 14247 .coefficient, .predecessor 1 14248 .coefficient])

def exact14250RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨62971⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨59991⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57011⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54031⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51051⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨31987⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨42924⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨21967⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40244⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37561⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34881⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18747⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29224⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26544⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15934⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66168⟩⟩], []⟩, (1)⟩]

theorem exact14250RawTermsValid :
    exact14250RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event14250 : Event := .resultExact (⟨.program ⟨257⟩, ⟨66176⟩⟩) exact14250RawTerms (.finite 3417662756781096507033577) 14249 .exactZero (none)

def event14251 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66177⟩⟩) 0 ⟨66176⟩ 14250

def event14252 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66177⟩⟩) 1 ⟨45602⟩ 14058

def event14253 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨66177⟩⟩) (.sum [.predecessor 0 14251 .coefficient, .predecessor 1 14252 .coefficient])

def exact14254RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨62971⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨59991⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57011⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54031⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51051⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨31987⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45601⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨42924⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨21967⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40244⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37561⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34881⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18747⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29224⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26544⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15934⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66168⟩⟩], []⟩, (1)⟩]

theorem exact14254RawTermsValid :
    exact14254RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event14254 : Event := .resultExact (⟨.program ⟨257⟩, ⟨66177⟩⟩) exact14254RawTerms (.finite 3648263642165693263543057) 14253 .exactZero (none)

def event14255 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66178⟩⟩) 0 ⟨66177⟩ 14254

def event14256 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66178⟩⟩) 1 ⟨48282⟩ 14050

def event14257 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨66178⟩⟩) (.sum [.predecessor 0 14255 .coefficient, .predecessor 1 14256 .coefficient])

def exact14258RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨62971⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨59991⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57011⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54031⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51051⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨31987⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48281⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45601⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨42924⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨21967⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40244⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37561⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34881⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18747⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29224⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26544⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15934⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66168⟩⟩], []⟩, (1)⟩]

theorem exact14258RawTermsValid :
    exact14258RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event14258 : Event := .resultExact (⟨.program ⟨257⟩, ⟨66178⟩⟩) exact14258RawTerms (.finite 3878994884184198780231457) 14257 .exactZero (none)

def event14259 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67344⟩⟩) 0 ⟨66178⟩ 14258

def event14260 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67344⟩⟩) 1 ⟨67342⟩ 14042

def event14261 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67344⟩⟩) (.sum [.predecessor 0 14259 .coefficient, .predecessor 1 14260 .coefficient])

def exact14262RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨62971⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨59991⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57011⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54031⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51051⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨67341⟩⟩], []⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨31987⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48281⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45601⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨42924⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨21967⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40244⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37561⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34881⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18747⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29224⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26544⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15934⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66168⟩⟩], []⟩, (1)⟩]

theorem exact14262RawTermsValid :
    exact14262RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event14262 : Event := .resultExact (⟨.program ⟨257⟩, ⟨67344⟩⟩) exact14262RawTerms (.finite 8101376613122849735629177) 14261 .exactZero (none)

def event14263 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67345⟩⟩) 0 ⟨67344⟩ 14262

def event14264 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67345⟩⟩) 1 ⟨6754⟩ 13545

def event14265 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67345⟩⟩) (.product (.predecessor 0 14263 .coefficient) (.predecessor 1 14264 .coefficient) (⟨false, true, none, none, some 1⟩))

def event14266 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67345⟩⟩, .operator (⟨14262, 5⟩, ⟨13545, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6754⟩⟩, ⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨67341⟩⟩], []⟩, (-1)⟩)

def event14267 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67345⟩⟩, .operator (⟨14262, 7⟩, ⟨13545, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6754⟩⟩, ⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48281⟩⟩], []⟩, (1)⟩)

def event14268 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67345⟩⟩, .operator (⟨14262, 8⟩, ⟨13545, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6754⟩⟩, ⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45601⟩⟩], []⟩, (1)⟩)

def event14269 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67345⟩⟩, .operator (⟨14262, 9⟩, ⟨13545, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6754⟩⟩, ⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨42924⟩⟩], []⟩, (1)⟩)

def event14270 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67345⟩⟩, .operator (⟨14262, 11⟩, ⟨13545, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6754⟩⟩, ⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40244⟩⟩], []⟩, (1)⟩)

def event14271 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67345⟩⟩, .operator (⟨14262, 12⟩, ⟨13545, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6754⟩⟩, ⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37561⟩⟩], []⟩, (1)⟩)

def event14272 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67345⟩⟩, .operator (⟨14262, 13⟩, ⟨13545, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6754⟩⟩, ⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34881⟩⟩], []⟩, (1)⟩)

def event14273 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67345⟩⟩, .operator (⟨14262, 15⟩, ⟨13545, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6754⟩⟩, ⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29224⟩⟩], []⟩, (1)⟩)

def event14274 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67345⟩⟩, .operator (⟨14262, 16⟩, ⟨13545, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6754⟩⟩, ⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26544⟩⟩], []⟩, (1)⟩)

def event14275 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67345⟩⟩, .operator (⟨14262, 18⟩, ⟨13545, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6754⟩⟩, ⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66168⟩⟩], []⟩, (1)⟩)

def event14276 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67345⟩⟩, .operator (⟨14262, 0⟩, ⟨13545, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨6754⟩⟩, ⟨.program ⟨257⟩, ⟨62971⟩⟩], []⟩, (1)⟩)

def event14277 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67345⟩⟩, .operator (⟨14262, 1⟩, ⟨13545, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨6754⟩⟩, ⟨.program ⟨257⟩, ⟨59991⟩⟩], []⟩, (1)⟩)

def event14278 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67345⟩⟩, .operator (⟨14262, 2⟩, ⟨13545, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨6754⟩⟩, ⟨.program ⟨257⟩, ⟨57011⟩⟩], []⟩, (1)⟩)

def event14279 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67345⟩⟩, .operator (⟨14262, 3⟩, ⟨13545, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6754⟩⟩, ⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54031⟩⟩], []⟩, (1)⟩)

def event14280 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67345⟩⟩, .operator (⟨14262, 4⟩, ⟨13545, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6754⟩⟩, ⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51051⟩⟩], []⟩, (1)⟩)

def event14281 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67345⟩⟩, .operator (⟨14262, 6⟩, ⟨13545, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6754⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨31987⟩⟩], []⟩, (1)⟩)

def event14282 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67345⟩⟩, .operator (⟨14262, 10⟩, ⟨13545, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6754⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨21967⟩⟩], []⟩, (1)⟩)

def event14283 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67345⟩⟩, .operator (⟨14262, 14⟩, ⟨13545, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6754⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18747⟩⟩], []⟩, (1)⟩)

def event14284 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67345⟩⟩, .operator (⟨14262, 17⟩, ⟨13545, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6754⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15934⟩⟩], []⟩, (1)⟩)

def exact14285RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨6754⟩⟩, ⟨.program ⟨257⟩, ⟨62971⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨6754⟩⟩, ⟨.program ⟨257⟩, ⟨59991⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨6754⟩⟩, ⟨.program ⟨257⟩, ⟨57011⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6754⟩⟩, ⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54031⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6754⟩⟩, ⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51051⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6754⟩⟩, ⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨67341⟩⟩], []⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6754⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨31987⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6754⟩⟩, ⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48281⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6754⟩⟩, ⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45601⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6754⟩⟩, ⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨42924⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6754⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨21967⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6754⟩⟩, ⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40244⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6754⟩⟩, ⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37561⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6754⟩⟩, ⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34881⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6754⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18747⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6754⟩⟩, ⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29224⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6754⟩⟩, ⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26544⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6754⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15934⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6754⟩⟩, ⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66168⟩⟩], []⟩, (1)⟩]

theorem exact14285RawTermsValid :
    exact14285RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event14285 : Event := .resultExact (⟨.program ⟨257⟩, ⟨67345⟩⟩) exact14285RawTerms (.finite 225096675372362460760862075689387889982459993584239673316998329415117550269590472230546999996634554493211422466559599866176826107954173710137268678722981311660348503815015153869609652895704942247936) 14265 .exactZero (none)

def event14286 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6742⟩⟩) (.authority (.factStore))

def exact14287RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6742⟩⟩], []⟩, (1)⟩]

theorem exact14287RawTermsValid :
    exact14287RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event14287 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6742⟩⟩) exact14287RawTerms (.finite 1197517580502842842035498068085245900805013937664077721615212599234784755857741972220330155661978881636229404821270118044300934465378618104014808564255759897621456021763) 14286 .exactZero (none)

def event14288 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47594⟩⟩) 0 ⟨392⟩ 14

def event14289 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47594⟩⟩) (.authority (.programFamilyFact))

def exact14290RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨47594⟩⟩], []⟩, (1)⟩]

theorem exact14290RawTermsValid :
    exact14290RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event14290 : Event := .resultExact (⟨.program ⟨257⟩, ⟨47594⟩⟩) exact14290RawTerms (.finite 60) 14289 .exactZero (none)

def event14291 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14931⟩⟩) 0 ⟨392⟩ 14

def event14292 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14931⟩⟩) (.authority (.programFamilyFact))

def exact14293RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14931⟩⟩], []⟩, (1)⟩]

theorem exact14293RawTermsValid :
    exact14293RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event14293 : Event := .resultExact (⟨.program ⟨257⟩, ⟨14931⟩⟩) exact14293RawTerms (.finite 60) 14292 .exactZero (none)

def event14294 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47595⟩⟩) 0 ⟨14931⟩ 14293

def event14295 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47595⟩⟩) 1 ⟨47594⟩ 14290

def event14296 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47595⟩⟩) (.product (.predecessor 0 14294 .coefficient) (.predecessor 1 14295 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event14297 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨47595⟩⟩, .operator (⟨14293, 0⟩, ⟨14290, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨14931⟩⟩, ⟨.program ⟨257⟩, ⟨47594⟩⟩], []⟩, (1)⟩)

def exact14298RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14931⟩⟩, ⟨.program ⟨257⟩, ⟨47594⟩⟩], []⟩, (1)⟩]

theorem exact14298RawTermsValid :
    exact14298RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event14298 : Event := .resultExact (⟨.program ⟨257⟩, ⟨47595⟩⟩) exact14298RawTerms (.finite 3600) 14296 .exactZero (none)

def event14299 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47596⟩⟩) 0 ⟨47595⟩ 14298

def event14300 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47596⟩⟩) (.identity (.predecessor 0 14299 .coefficient))

def event14301 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨47596⟩⟩) (.finite 3600)

def event14302 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48068⟩⟩) 0 ⟨47596⟩ 14301

def event14303 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48068⟩⟩) (.authority (.programFamilyFact))

def exact14304RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨48068⟩⟩], []⟩, (1)⟩]

theorem exact14304RawTermsValid :
    exact14304RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event14304 : Event := .resultExact (⟨.program ⟨257⟩, ⟨48068⟩⟩) exact14304RawTerms (.finite 60) 14303 .exactZero (none)

def event14305 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48069⟩⟩) 0 ⟨48068⟩ 14304

def event14306 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48069⟩⟩) (.identity (.predecessor 0 14305 .coefficient))

def event14307 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨48069⟩⟩) (.finite 60)

def event14308 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48233⟩⟩) 0 ⟨48069⟩ 14307

def event14309 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48233⟩⟩) (.authority (.programFamilyFact))

def exact14310RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨48233⟩⟩], []⟩, (1)⟩]

theorem exact14310RawTermsValid :
    exact14310RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event14310 : Event := .resultExact (⟨.program ⟨257⟩, ⟨48233⟩⟩) exact14310RawTerms (.finite 63) 14309 .exactZero (none)

def event14311 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44914⟩⟩) 0 ⟨392⟩ 14

def event14312 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44914⟩⟩) (.authority (.programFamilyFact))

def exact14313RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨44914⟩⟩], []⟩, (1)⟩]

theorem exact14313RawTermsValid :
    exact14313RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event14313 : Event := .resultExact (⟨.program ⟨257⟩, ⟨44914⟩⟩) exact14313RawTerms (.finite 58) 14312 .exactZero (none)

def event14314 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14631⟩⟩) 0 ⟨392⟩ 14

def event14315 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14631⟩⟩) (.authority (.programFamilyFact))

def exact14316RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14631⟩⟩], []⟩, (1)⟩]

theorem exact14316RawTermsValid :
    exact14316RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event14316 : Event := .resultExact (⟨.program ⟨257⟩, ⟨14631⟩⟩) exact14316RawTerms (.finite 58) 14315 .exactZero (none)

def event14317 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44915⟩⟩) 0 ⟨14631⟩ 14316

def event14318 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44915⟩⟩) 1 ⟨44914⟩ 14313

def event14319 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44915⟩⟩) (.product (.predecessor 0 14317 .coefficient) (.predecessor 1 14318 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event14320 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨44915⟩⟩, .operator (⟨14316, 0⟩, ⟨14313, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨14631⟩⟩, ⟨.program ⟨257⟩, ⟨44914⟩⟩], []⟩, (1)⟩)

def exact14321RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14631⟩⟩, ⟨.program ⟨257⟩, ⟨44914⟩⟩], []⟩, (1)⟩]

theorem exact14321RawTermsValid :
    exact14321RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event14321 : Event := .resultExact (⟨.program ⟨257⟩, ⟨44915⟩⟩) exact14321RawTerms (.finite 3364) 14319 .exactZero (none)

def event14322 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44916⟩⟩) 0 ⟨44915⟩ 14321

def event14323 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44916⟩⟩) (.identity (.predecessor 0 14322 .coefficient))

def event14324 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨44916⟩⟩) (.finite 3364)

def event14325 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45388⟩⟩) 0 ⟨44916⟩ 14324

def event14326 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45388⟩⟩) (.authority (.programFamilyFact))

def exact14327RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨45388⟩⟩], []⟩, (1)⟩]

theorem exact14327RawTermsValid :
    exact14327RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event14327 : Event := .resultExact (⟨.program ⟨257⟩, ⟨45388⟩⟩) exact14327RawTerms (.finite 58) 14326 .exactZero (none)

def event14328 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45389⟩⟩) 0 ⟨45388⟩ 14327

def event14329 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45389⟩⟩) (.identity (.predecessor 0 14328 .coefficient))

def event14330 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨45389⟩⟩) (.finite 58)

def event14331 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45553⟩⟩) 0 ⟨45389⟩ 14330

def event14332 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45553⟩⟩) (.authority (.programFamilyFact))

def exact14333RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨45553⟩⟩], []⟩, (1)⟩]

theorem exact14333RawTermsValid :
    exact14333RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event14333 : Event := .resultExact (⟨.program ⟨257⟩, ⟨45553⟩⟩) exact14333RawTerms (.finite 63) 14332 .exactZero (none)

def event14334 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42234⟩⟩) 0 ⟨392⟩ 14

def event14335 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42234⟩⟩) (.authority (.programFamilyFact))

def eventLeaf880 : Array AnnotatedEvent := #[
  { event := event14080
    frameStart := 0 },
  { event := event14081
    frameStart := 0 },
  { event := event14082
    frameStart := 0 },
  { event := event14083
    frameStart := 0 },
  { event := event14084
    frameStart := 0 },
  { event := event14085
    frameStart := 0 },
  { event := event14086
    frameStart := 0 },
  { event := event14087
    frameStart := 0 },
  { event := event14088
    frameStart := 0 },
  { event := event14089
    frameStart := 0 },
  { event := event14090
    frameStart := 0 },
  { event := event14091
    frameStart := 0 },
  { event := event14092
    frameStart := 0 },
  { event := event14093
    frameStart := 0 },
  { event := event14094
    frameStart := 0 },
  { event := event14095
    frameStart := 0 }
]

def eventLeaf881 : Array AnnotatedEvent := #[
  { event := event14096
    frameStart := 0 },
  { event := event14097
    frameStart := 0 },
  { event := event14098
    frameStart := 0 },
  { event := event14099
    frameStart := 0 },
  { event := event14100
    frameStart := 0 },
  { event := event14101
    frameStart := 0 },
  { event := event14102
    frameStart := 0 },
  { event := event14103
    frameStart := 0 },
  { event := event14104
    frameStart := 0 },
  { event := event14105
    frameStart := 0 },
  { event := event14106
    frameStart := 0 },
  { event := event14107
    frameStart := 0 },
  { event := event14108
    frameStart := 0 },
  { event := event14109
    frameStart := 0 },
  { event := event14110
    frameStart := 0 },
  { event := event14111
    frameStart := 0 }
]

def eventLeaf882 : Array AnnotatedEvent := #[
  { event := event14112
    frameStart := 0 },
  { event := event14113
    frameStart := 0 },
  { event := event14114
    frameStart := 0 },
  { event := event14115
    frameStart := 0 },
  { event := event14116
    frameStart := 0 },
  { event := event14117
    frameStart := 0 },
  { event := event14118
    frameStart := 0 },
  { event := event14119
    frameStart := 0 },
  { event := event14120
    frameStart := 0 },
  { event := event14121
    frameStart := 0 },
  { event := event14122
    frameStart := 0 },
  { event := event14123
    frameStart := 0 },
  { event := event14124
    frameStart := 0 },
  { event := event14125
    frameStart := 0 },
  { event := event14126
    frameStart := 0 },
  { event := event14127
    frameStart := 0 }
]

def eventLeaf883 : Array AnnotatedEvent := #[
  { event := event14128
    frameStart := 0 },
  { event := event14129
    frameStart := 0 },
  { event := event14130
    frameStart := 0 },
  { event := event14131
    frameStart := 0 },
  { event := event14132
    frameStart := 0 },
  { event := event14133
    frameStart := 0 },
  { event := event14134
    frameStart := 0 },
  { event := event14135
    frameStart := 0 },
  { event := event14136
    frameStart := 0 },
  { event := event14137
    frameStart := 0 },
  { event := event14138
    frameStart := 0 },
  { event := event14139
    frameStart := 0 },
  { event := event14140
    frameStart := 0 },
  { event := event14141
    frameStart := 0 },
  { event := event14142
    frameStart := 0 },
  { event := event14143
    frameStart := 0 }
]

def eventLeaf884 : Array AnnotatedEvent := #[
  { event := event14144
    frameStart := 0 },
  { event := event14145
    frameStart := 0 },
  { event := event14146
    frameStart := 0 },
  { event := event14147
    frameStart := 0 },
  { event := event14148
    frameStart := 0 },
  { event := event14149
    frameStart := 0 },
  { event := event14150
    frameStart := 0 },
  { event := event14151
    frameStart := 0 },
  { event := event14152
    frameStart := 0 },
  { event := event14153
    frameStart := 0 },
  { event := event14154
    frameStart := 0 },
  { event := event14155
    frameStart := 0 },
  { event := event14156
    frameStart := 0 },
  { event := event14157
    frameStart := 0 },
  { event := event14158
    frameStart := 0 },
  { event := event14159
    frameStart := 0 }
]

def eventLeaf885 : Array AnnotatedEvent := #[
  { event := event14160
    frameStart := 0 },
  { event := event14161
    frameStart := 0 },
  { event := event14162
    frameStart := 0 },
  { event := event14163
    frameStart := 0 },
  { event := event14164
    frameStart := 0 },
  { event := event14165
    frameStart := 0 },
  { event := event14166
    frameStart := 0 },
  { event := event14167
    frameStart := 0 },
  { event := event14168
    frameStart := 0 },
  { event := event14169
    frameStart := 0 },
  { event := event14170
    frameStart := 0 },
  { event := event14171
    frameStart := 0 },
  { event := event14172
    frameStart := 0 },
  { event := event14173
    frameStart := 0 },
  { event := event14174
    frameStart := 0 },
  { event := event14175
    frameStart := 0 }
]

def eventLeaf886 : Array AnnotatedEvent := #[
  { event := event14176
    frameStart := 0 },
  { event := event14177
    frameStart := 0 },
  { event := event14178
    frameStart := 0 },
  { event := event14179
    frameStart := 0 },
  { event := event14180
    frameStart := 0 },
  { event := event14181
    frameStart := 0 },
  { event := event14182
    frameStart := 0 },
  { event := event14183
    frameStart := 0 },
  { event := event14184
    frameStart := 0 },
  { event := event14185
    frameStart := 0 },
  { event := event14186
    frameStart := 0 },
  { event := event14187
    frameStart := 0 },
  { event := event14188
    frameStart := 0 },
  { event := event14189
    frameStart := 0 },
  { event := event14190
    frameStart := 0 },
  { event := event14191
    frameStart := 0 }
]

def eventLeaf887 : Array AnnotatedEvent := #[
  { event := event14192
    frameStart := 0 },
  { event := event14193
    frameStart := 0 },
  { event := event14194
    frameStart := 0 },
  { event := event14195
    frameStart := 0 },
  { event := event14196
    frameStart := 0 },
  { event := event14197
    frameStart := 0 },
  { event := event14198
    frameStart := 0 },
  { event := event14199
    frameStart := 0 },
  { event := event14200
    frameStart := 0 },
  { event := event14201
    frameStart := 0 },
  { event := event14202
    frameStart := 0 },
  { event := event14203
    frameStart := 0 },
  { event := event14204
    frameStart := 0 },
  { event := event14205
    frameStart := 0 },
  { event := event14206
    frameStart := 0 },
  { event := event14207
    frameStart := 0 }
]

def eventLeaf888 : Array AnnotatedEvent := #[
  { event := event14208
    frameStart := 0 },
  { event := event14209
    frameStart := 0 },
  { event := event14210
    frameStart := 0 },
  { event := event14211
    frameStart := 0 },
  { event := event14212
    frameStart := 0 },
  { event := event14213
    frameStart := 0 },
  { event := event14214
    frameStart := 0 },
  { event := event14215
    frameStart := 0 },
  { event := event14216
    frameStart := 0 },
  { event := event14217
    frameStart := 0 },
  { event := event14218
    frameStart := 0 },
  { event := event14219
    frameStart := 0 },
  { event := event14220
    frameStart := 0 },
  { event := event14221
    frameStart := 0 },
  { event := event14222
    frameStart := 0 },
  { event := event14223
    frameStart := 0 }
]

def eventLeaf889 : Array AnnotatedEvent := #[
  { event := event14224
    frameStart := 0 },
  { event := event14225
    frameStart := 0 },
  { event := event14226
    frameStart := 0 },
  { event := event14227
    frameStart := 0 },
  { event := event14228
    frameStart := 0 },
  { event := event14229
    frameStart := 0 },
  { event := event14230
    frameStart := 0 },
  { event := event14231
    frameStart := 0 },
  { event := event14232
    frameStart := 0 },
  { event := event14233
    frameStart := 0 },
  { event := event14234
    frameStart := 0 },
  { event := event14235
    frameStart := 0 },
  { event := event14236
    frameStart := 0 },
  { event := event14237
    frameStart := 0 },
  { event := event14238
    frameStart := 0 },
  { event := event14239
    frameStart := 0 }
]

def eventLeaf890 : Array AnnotatedEvent := #[
  { event := event14240
    frameStart := 0 },
  { event := event14241
    frameStart := 0 },
  { event := event14242
    frameStart := 0 },
  { event := event14243
    frameStart := 0 },
  { event := event14244
    frameStart := 0 },
  { event := event14245
    frameStart := 0 },
  { event := event14246
    frameStart := 0 },
  { event := event14247
    frameStart := 0 },
  { event := event14248
    frameStart := 0 },
  { event := event14249
    frameStart := 0 },
  { event := event14250
    frameStart := 0 },
  { event := event14251
    frameStart := 0 },
  { event := event14252
    frameStart := 0 },
  { event := event14253
    frameStart := 0 },
  { event := event14254
    frameStart := 0 },
  { event := event14255
    frameStart := 0 }
]

def eventLeaf891 : Array AnnotatedEvent := #[
  { event := event14256
    frameStart := 0 },
  { event := event14257
    frameStart := 0 },
  { event := event14258
    frameStart := 0 },
  { event := event14259
    frameStart := 0 },
  { event := event14260
    frameStart := 0 },
  { event := event14261
    frameStart := 0 },
  { event := event14262
    frameStart := 0 },
  { event := event14263
    frameStart := 0 },
  { event := event14264
    frameStart := 0 },
  { event := event14265
    frameStart := 0 },
  { event := event14266
    frameStart := 0 },
  { event := event14267
    frameStart := 0 },
  { event := event14268
    frameStart := 0 },
  { event := event14269
    frameStart := 0 },
  { event := event14270
    frameStart := 0 },
  { event := event14271
    frameStart := 0 }
]

def eventLeaf892 : Array AnnotatedEvent := #[
  { event := event14272
    frameStart := 0 },
  { event := event14273
    frameStart := 0 },
  { event := event14274
    frameStart := 0 },
  { event := event14275
    frameStart := 0 },
  { event := event14276
    frameStart := 0 },
  { event := event14277
    frameStart := 0 },
  { event := event14278
    frameStart := 0 },
  { event := event14279
    frameStart := 0 },
  { event := event14280
    frameStart := 0 },
  { event := event14281
    frameStart := 0 },
  { event := event14282
    frameStart := 0 },
  { event := event14283
    frameStart := 0 },
  { event := event14284
    frameStart := 0 },
  { event := event14285
    frameStart := 0 },
  { event := event14286
    frameStart := 0 },
  { event := event14287
    frameStart := 0 }
]

def eventLeaf893 : Array AnnotatedEvent := #[
  { event := event14288
    frameStart := 0 },
  { event := event14289
    frameStart := 0 },
  { event := event14290
    frameStart := 0 },
  { event := event14291
    frameStart := 0 },
  { event := event14292
    frameStart := 0 },
  { event := event14293
    frameStart := 0 },
  { event := event14294
    frameStart := 0 },
  { event := event14295
    frameStart := 0 },
  { event := event14296
    frameStart := 0 },
  { event := event14297
    frameStart := 0 },
  { event := event14298
    frameStart := 0 },
  { event := event14299
    frameStart := 0 },
  { event := event14300
    frameStart := 0 },
  { event := event14301
    frameStart := 0 },
  { event := event14302
    frameStart := 0 },
  { event := event14303
    frameStart := 0 }
]

def eventLeaf894 : Array AnnotatedEvent := #[
  { event := event14304
    frameStart := 0 },
  { event := event14305
    frameStart := 0 },
  { event := event14306
    frameStart := 0 },
  { event := event14307
    frameStart := 0 },
  { event := event14308
    frameStart := 0 },
  { event := event14309
    frameStart := 0 },
  { event := event14310
    frameStart := 0 },
  { event := event14311
    frameStart := 0 },
  { event := event14312
    frameStart := 0 },
  { event := event14313
    frameStart := 0 },
  { event := event14314
    frameStart := 0 },
  { event := event14315
    frameStart := 0 },
  { event := event14316
    frameStart := 0 },
  { event := event14317
    frameStart := 0 },
  { event := event14318
    frameStart := 0 },
  { event := event14319
    frameStart := 0 }
]

def eventLeaf895 : Array AnnotatedEvent := #[
  { event := event14320
    frameStart := 0 },
  { event := event14321
    frameStart := 0 },
  { event := event14322
    frameStart := 0 },
  { event := event14323
    frameStart := 0 },
  { event := event14324
    frameStart := 0 },
  { event := event14325
    frameStart := 0 },
  { event := event14326
    frameStart := 0 },
  { event := event14327
    frameStart := 0 },
  { event := event14328
    frameStart := 0 },
  { event := event14329
    frameStart := 0 },
  { event := event14330
    frameStart := 0 },
  { event := event14331
    frameStart := 0 },
  { event := event14332
    frameStart := 0 },
  { event := event14333
    frameStart := 0 },
  { event := event14334
    frameStart := 0 },
  { event := event14335
    frameStart := 0 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events055
