import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events328

open Mxx.Certificate.OperationalNoise
open CertificateABI

def exact83968RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11637⟩⟩, ⟨.program ⟨214⟩, ⟨14641⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact83968RawTermsValid :
    exact83968RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83968 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14750⟩⟩) exact83968RawTerms .large 83966 .exactZero (none)

def event83969 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6757⟩⟩) 0 ⟨6689⟩ 83945

def event83970 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6757⟩⟩) (.authority (.operator))

def exact83971RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6757⟩⟩]⟩, (1)⟩]

theorem exact83971RawTermsValid :
    exact83971RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83971 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6757⟩⟩) exact83971RawTerms .large 83970 .exactZero (none)

def event83972 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6781⟩⟩) 0 ⟨6757⟩ 83971

def event83973 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6781⟩⟩) (.identity (.predecessor 0 83972 .coefficient))

def exact83974RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6781⟩⟩]⟩, (1)⟩]

theorem exact83974RawTermsValid :
    exact83974RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83974 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6781⟩⟩) exact83974RawTerms .large 83973 .exactZero (none)

def event83975 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7858⟩⟩) 0 ⟨6781⟩ 83974

def event83976 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7858⟩⟩) (.authority (.operator))

def exact83977RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨7858⟩⟩]⟩, (1)⟩]

theorem exact83977RawTermsValid :
    exact83977RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83977 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7858⟩⟩) exact83977RawTerms (.finite 8192) 83976 .exactZero (none)

def event83978 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7859⟩⟩) 0 ⟨7858⟩ 83977

def event83979 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7859⟩⟩) 1 ⟨2348⟩ 83911

def event83980 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7859⟩⟩) (.scale (.predecessor 0 83978 .coefficient) (.value (.predecessor 1 83979 .coefficient)))

def exact83981RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨7858⟩⟩]⟩, (1)⟩]

theorem exact83981RawTermsValid :
    exact83981RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83981 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7859⟩⟩) exact83981RawTerms (.finite 8192) 83980 .exactZero (none)

def event83982 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6762⟩⟩) 0 ⟨6757⟩ 83971

def event83983 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6762⟩⟩) (.identity (.predecessor 0 83982 .coefficient))

def exact83984RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6762⟩⟩]⟩, (1)⟩]

theorem exact83984RawTermsValid :
    exact83984RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83984 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6762⟩⟩) exact83984RawTerms .large 83983 .exactZero (none)

def event83985 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7860⟩⟩) 0 ⟨6762⟩ 83984

def event83986 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7860⟩⟩) 1 ⟨7859⟩ 83981

def event83987 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7860⟩⟩) (.product (.predecessor 0 83985 .coefficient) (.predecessor 1 83986 .coefficient) (⟨false, false, none, none, none⟩))

def event83988 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7860⟩⟩, .operator (⟨83984, 0⟩, ⟨83981, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6762⟩⟩, ⟨.program ⟨214⟩, ⟨7858⟩⟩]⟩, (1)⟩)

def exact83989RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6762⟩⟩, ⟨.program ⟨214⟩, ⟨7858⟩⟩]⟩, (1)⟩]

theorem exact83989RawTermsValid :
    exact83989RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83989 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7860⟩⟩) exact83989RawTerms .large 83987 .exactZero (none)

def event83990 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14751⟩⟩) 0 ⟨7860⟩ 83989

def event83991 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14751⟩⟩) 1 ⟨14750⟩ 83968

def event83992 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14751⟩⟩) (.sum [.predecessor 0 83990 .coefficient, .predecessor 1 83991 .coefficient])

def exact83993RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6762⟩⟩, ⟨.program ⟨214⟩, ⟨7858⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨11637⟩⟩, ⟨.program ⟨214⟩, ⟨14641⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact83993RawTermsValid :
    exact83993RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83993 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14751⟩⟩) exact83993RawTerms .large 83992 .exactZero (none)

def event83994 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26223⟩⟩) 0 ⟨14751⟩ 83993

def event83995 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26223⟩⟩) 1 ⟨26220⟩ 83952

def event83996 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26223⟩⟩) (.product (.predecessor 0 83994 .coefficient) (.predecessor 1 83995 .coefficient) (⟨false, false, none, none, none⟩))

def event83997 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26223⟩⟩, .operator (⟨83993, 0⟩, ⟨83952, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6762⟩⟩, ⟨.program ⟨214⟩, ⟨7858⟩⟩, ⟨.program ⟨214⟩, ⟨26220⟩⟩]⟩, (1)⟩)

def event83998 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26223⟩⟩, .operator (⟨83993, 1⟩, ⟨83952, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨11637⟩⟩, ⟨.program ⟨214⟩, ⟨14641⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26220⟩⟩]⟩, (-1)⟩)

def event83999 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨26223⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨11637⟩⟩, ⟨.program ⟨214⟩, ⟨14641⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26220⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨26220⟩⟩) ⟨23668⟩ 83949)

def event84000 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26223⟩⟩, .relation 83999 0, ⟨[⟨.program ⟨214⟩, ⟨11637⟩⟩, ⟨.program ⟨214⟩, ⟨14641⟩⟩], [⟨.program ⟨214⟩, ⟨23668⟩⟩]⟩, (-1)⟩)

def exact84001RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6762⟩⟩, ⟨.program ⟨214⟩, ⟨7858⟩⟩, ⟨.program ⟨214⟩, ⟨26220⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨11637⟩⟩, ⟨.program ⟨214⟩, ⟨14641⟩⟩], [⟨.program ⟨214⟩, ⟨23668⟩⟩]⟩, (-1)⟩]

theorem exact84001RawTermsValid :
    exact84001RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84001 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26223⟩⟩) exact84001RawTerms .large 83996 .exactZero (none)

def event84002 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16178⟩⟩) 0 ⟨14643⟩ 83941

def event84003 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16178⟩⟩) (.authority (.programFamilyFact))

def exact84004RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16178⟩⟩], []⟩, (1)⟩]

theorem exact84004RawTermsValid :
    exact84004RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84004 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16178⟩⟩) exact84004RawTerms (.finite 28) 84003 .exactZero (none)

def event84005 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16180⟩⟩) 0 ⟨6544⟩ 83963

def event84006 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16180⟩⟩) 1 ⟨16178⟩ 84004

def event84007 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16180⟩⟩) (.product (.predecessor 0 84005 .coefficient) (.predecessor 1 84006 .coefficient) (⟨false, true, none, none, some 1⟩))

def event84008 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨16180⟩⟩, .operator (⟨83963, 0⟩, ⟨84004, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨16178⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact84009RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16178⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact84009RawTermsValid :
    exact84009RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84009 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16180⟩⟩) exact84009RawTerms .large 84007 .exactZero (none)

def event84010 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6699⟩⟩) 0 ⟨6689⟩ 83945

def event84011 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6699⟩⟩) (.authority (.operator))

def exact84012RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6699⟩⟩]⟩, (1)⟩]

theorem exact84012RawTermsValid :
    exact84012RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84012 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6699⟩⟩) exact84012RawTerms .large 84011 .exactZero (none)

def event84013 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16181⟩⟩) 0 ⟨6699⟩ 84012

def event84014 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16181⟩⟩) 1 ⟨16180⟩ 84009

def event84015 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16181⟩⟩) (.sum [.predecessor 0 84013 .coefficient, .predecessor 1 84014 .coefficient])

def exact84016RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6699⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16178⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact84016RawTermsValid :
    exact84016RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84016 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16181⟩⟩) exact84016RawTerms .large 84015 .exactZero (none)

def event84017 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26224⟩⟩) 0 ⟨16181⟩ 84016

def event84018 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26224⟩⟩) 1 ⟨26223⟩ 84001

def event84019 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26224⟩⟩) (.sum [.predecessor 0 84017 .coefficient, .predecessor 1 84018 .coefficient])

def exact84020RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6699⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6762⟩⟩, ⟨.program ⟨214⟩, ⟨7858⟩⟩, ⟨.program ⟨214⟩, ⟨26220⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨11637⟩⟩, ⟨.program ⟨214⟩, ⟨14641⟩⟩], [⟨.program ⟨214⟩, ⟨23668⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16178⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact84020RawTermsValid :
    exact84020RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84020 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26224⟩⟩) exact84020RawTerms .large 84019 .exactZero (none)

def event84021 : Event := .preFoldPolynomial 84020 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6699⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6762⟩⟩, ⟨.program ⟨214⟩, ⟨7858⟩⟩, ⟨.program ⟨214⟩, ⟨26220⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨11637⟩⟩, ⟨.program ⟨214⟩, ⟨14641⟩⟩], [⟨.program ⟨214⟩, ⟨23668⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16178⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩] .exactZero none

def exact84022RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6699⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6762⟩⟩, ⟨.program ⟨214⟩, ⟨7858⟩⟩, ⟨.program ⟨214⟩, ⟨26220⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨11637⟩⟩, ⟨.program ⟨214⟩, ⟨14641⟩⟩], [⟨.program ⟨214⟩, ⟨23668⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16178⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

def event84022 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨26224⟩⟩) 84021 exact84022RawTerms .large 84019 .exactZero (none)

def event84023 : Event := .specializationComputed (⟨.program ⟨214⟩, ⟨14643⟩⟩) ⟨⟨112⟩, ⟨17⟩, ⟨109⟩⟩ ⟨83859, 84023⟩

def event84024 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨19675⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19672⟩⟩]⟩) (1) 0 2 (.universal 84023 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19672⟩⟩]⟩) (none) 84022)

def event84025 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19675⟩⟩, .relation 84024 0, ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6699⟩⟩]⟩, (1)⟩)

def event84026 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19675⟩⟩, .relation 84024 1, ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6762⟩⟩, ⟨.program ⟨214⟩, ⟨7858⟩⟩, ⟨.program ⟨214⟩, ⟨26220⟩⟩]⟩, (-1)⟩)

def event84027 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19675⟩⟩, .relation 84024 2, ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨11637⟩⟩, ⟨.program ⟨214⟩, ⟨14641⟩⟩], [⟨.program ⟨214⟩, ⟨23668⟩⟩]⟩, (1)⟩)

def event84028 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19675⟩⟩, .relation 84024 3, ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16178⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def exact84029RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6699⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6762⟩⟩, ⟨.program ⟨214⟩, ⟨7858⟩⟩, ⟨.program ⟨214⟩, ⟨26220⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨11637⟩⟩, ⟨.program ⟨214⟩, ⟨14641⟩⟩], [⟨.program ⟨214⟩, ⟨23668⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16178⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact84029RawTermsValid :
    exact84029RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84029 : Event := .resultExact (⟨.program ⟨214⟩, ⟨19675⟩⟩) exact84029RawTerms .large 83855 (.finite 1811303510016) (some (83857))

def event84030 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26222⟩⟩) 0 ⟨19675⟩ 84029

def event84031 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26222⟩⟩) 1 ⟨26221⟩ 83845

def event84032 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26222⟩⟩) (.sum [.predecessor 0 84030 .coefficient, .predecessor 1 84031 .coefficient])

def event84033 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26222⟩⟩, .operator (⟨84029, 2⟩, ⟨83845, 1⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨11637⟩⟩, ⟨.program ⟨214⟩, ⟨14641⟩⟩], [⟨.program ⟨214⟩, ⟨23668⟩⟩]⟩, (-1)⟩)

def event84034 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26222⟩⟩, .operator (⟨84029, 1⟩, ⟨83845, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6762⟩⟩, ⟨.program ⟨214⟩, ⟨7858⟩⟩, ⟨.program ⟨214⟩, ⟨26220⟩⟩]⟩, (1)⟩)

def event84035 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26222⟩⟩) (.sum [.result 84029 .summary, .result 83845 .summary])

def exact84036RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6699⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16178⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact84036RawTermsValid :
    exact84036RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84036 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26222⟩⟩) exact84036RawTerms .large 84032 (.finite 352091253649408) (some (84035))

def event84037 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28302⟩⟩) 0 ⟨26222⟩ 84036

def event84038 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28302⟩⟩) 1 ⟨28300⟩ 83761

def event84039 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28302⟩⟩) (.product (.predecessor 0 84037 .coefficient) (.predecessor 1 84038 .coefficient) (⟨false, false, none, none, none⟩))

def event84040 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28302⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨28300⟩⟩]⟩) [⟨.result 83761 .coefficient, false, none⟩])

def event84041 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28302⟩⟩) (.product (.result 84036 .summary) (.transfer 84040) (⟨false, false, none, none, none⟩))

def event84042 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28302⟩⟩, .operator (⟨84036, 0⟩, ⟨83761, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6699⟩⟩, ⟨.program ⟨214⟩, ⟨28300⟩⟩]⟩, (1)⟩)

def event84043 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28302⟩⟩, .operator (⟨84036, 1⟩, ⟨83761, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16178⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28300⟩⟩]⟩, (-1)⟩)

def event84044 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨28302⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16178⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28300⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨28300⟩⟩) ⟨24288⟩ 83758)

def event84045 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28302⟩⟩, .relation 84044 0, ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16178⟩⟩], [⟨.program ⟨214⟩, ⟨24288⟩⟩]⟩, (-1)⟩)

def exact84046RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6699⟩⟩, ⟨.program ⟨214⟩, ⟨28300⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16178⟩⟩], [⟨.program ⟨214⟩, ⟨24288⟩⟩]⟩, (-1)⟩]

theorem exact84046RawTermsValid :
    exact84046RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84046 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28302⟩⟩) exact84046RawTerms .large 84039 (.finite 1292180534353385750528) (some (84041))

def event84047 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21688⟩⟩) 0 ⟨16179⟩ 4029

def event84048 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21688⟩⟩) (.authority (.relationPreimageSource ⟨48⟩))

def exact84049RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨21688⟩⟩]⟩, (1)⟩]

theorem exact84049RawTermsValid :
    exact84049RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84049 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21688⟩⟩) exact84049RawTerms (.finite 136065468) 84048 .exactZero (none)

def event84050 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21690⟩⟩) 0 ⟨21688⟩ 84049

def event84051 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21690⟩⟩) 1 ⟨2348⟩ 4

def event84052 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21690⟩⟩) (.scale (.predecessor 0 84050 .coefficient) (.value (.predecessor 1 84051 .coefficient)))

def exact84053RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨21688⟩⟩]⟩, (1)⟩]

theorem exact84053RawTermsValid :
    exact84053RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84053 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21690⟩⟩) exact84053RawTerms (.finite 136065468) 84052 .exactZero (none)

def event84054 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21691⟩⟩) 0 ⟨5541⟩ 80012

def event84055 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21691⟩⟩) 1 ⟨21690⟩ 84053

def event84056 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21691⟩⟩) (.product (.predecessor 0 84054 .coefficient) (.predecessor 1 84055 .coefficient) (⟨false, false, none, none, none⟩))

def event84057 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21691⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨21688⟩⟩]⟩) [⟨.result 84049 .coefficient, false, none⟩])

def event84058 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21691⟩⟩) (.product (.result 80012 .summary) (.transfer 84057) (⟨false, false, none, none, none⟩))

def event84059 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21691⟩⟩, .operator (⟨80012, 0⟩, ⟨84053, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21688⟩⟩]⟩, (1)⟩)

def event84060 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨21689⟩⟩)

def event84061 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event84062 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event84063 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨2348⟩⟩) (.authority (.operator))

def event84064 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨2348⟩⟩) (.finite 1)

def event84065 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event84066 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event84067 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event84068 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event84069 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 84068

def event84070 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 84066

def event84071 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 84069 .coefficient) (.value (.predecessor 1 84070 .coefficient)))

def event84072 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event84073 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5505⟩⟩) 0 ⟨5503⟩ 84072

def event84074 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5505⟩⟩) 1 ⟨2348⟩ 84064

def event84075 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5505⟩⟩) (.sum [.predecessor 0 84073 .coefficient, .predecessor 1 84074 .coefficient])

def event84076 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5505⟩⟩) (.finite 218)

def event84077 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5536⟩⟩) 0 ⟨5505⟩ 84076

def event84078 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5536⟩⟩) 1 ⟨961⟩ 84062

def event84079 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5536⟩⟩) (.identity (.predecessor 1 84078 .coefficient))

def event84080 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5536⟩⟩) (.finite 224)

def event84081 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11637⟩⟩) 0 ⟨5536⟩ 84080

def event84082 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11637⟩⟩) (.authority (.programFamilyFact))

def exact84083RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11637⟩⟩], []⟩, (1)⟩]

theorem exact84083RawTermsValid :
    exact84083RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84083 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11637⟩⟩) exact84083RawTerms (.finite 28) 84082 .exactZero (none)

def event84084 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14641⟩⟩) 0 ⟨5536⟩ 84080

def event84085 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14641⟩⟩) (.authority (.programFamilyFact))

def exact84086RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨14641⟩⟩], []⟩, (1)⟩]

theorem exact84086RawTermsValid :
    exact84086RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84086 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14641⟩⟩) exact84086RawTerms (.finite 28) 84085 .exactZero (none)

def event84087 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14642⟩⟩) 0 ⟨14641⟩ 84086

def event84088 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14642⟩⟩) 1 ⟨11637⟩ 84083

def event84089 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14642⟩⟩) (.product (.predecessor 0 84087 .coefficient) (.predecessor 1 84088 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event84090 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14642⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨11637⟩⟩, ⟨.program ⟨214⟩, ⟨14641⟩⟩], []⟩) [⟨.result 84086 .coefficient, true, some 1⟩, ⟨.result 84083 .coefficient, true, some 1⟩])

def event84091 : Event := .survivorFold (1) 84090

def exact84092RawTerms : List Term := []

theorem exact84092RawTermsValid :
    exact84092RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84092 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14642⟩⟩) exact84092RawTerms (.finite 784) 84089 (.finite 784) (some (84090))

def event84093 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14643⟩⟩) 0 ⟨14642⟩ 84092

def event84094 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14643⟩⟩) (.identity (.predecessor 0 84093 .coefficient))

def event84095 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨14643⟩⟩) (.finite 784)

def event84096 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16178⟩⟩) 0 ⟨14643⟩ 84095

def event84097 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16178⟩⟩) (.authority (.programFamilyFact))

def exact84098RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16178⟩⟩], []⟩, (1)⟩]

theorem exact84098RawTermsValid :
    exact84098RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84098 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16178⟩⟩) exact84098RawTerms (.finite 28) 84097 .exactZero (none)

def event84099 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16179⟩⟩) 0 ⟨16178⟩ 84098

def event84100 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16179⟩⟩) (.identity (.predecessor 0 84099 .coefficient))

def event84101 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨16179⟩⟩) (.finite 28)

def event84102 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21688⟩⟩) 0 ⟨16179⟩ 84101

def event84103 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21688⟩⟩) (.authority (.relationPreimageSource ⟨48⟩))

def exact84104RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨21688⟩⟩]⟩, (1)⟩]

theorem exact84104RawTermsValid :
    exact84104RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84104 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21688⟩⟩) exact84104RawTerms (.finite 136065468) 84103 .exactZero (none)

def event84105 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6⟩⟩) (.authority (.operator))

def exact84106RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩]⟩, (1)⟩]

theorem exact84106RawTermsValid :
    exact84106RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84106 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6⟩⟩) exact84106RawTerms .large 84105 .exactZero (none)

def event84107 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21689⟩⟩) 0 ⟨6⟩ 84106

def event84108 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21689⟩⟩) 1 ⟨21688⟩ 84104

def event84109 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21689⟩⟩) (.product (.predecessor 0 84107 .coefficient) (.predecessor 1 84108 .coefficient) (⟨false, false, none, none, none⟩))

def event84110 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21689⟩⟩, .operator (⟨84106, 0⟩, ⟨84104, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21688⟩⟩]⟩, (1)⟩)

def exact84111RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21688⟩⟩]⟩, (1)⟩]

theorem exact84111RawTermsValid :
    exact84111RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84111 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21689⟩⟩) exact84111RawTerms .large 84109 .exactZero (none)

def event84112 : Event := .preFoldPolynomial 84111 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21688⟩⟩]⟩, (1)⟩] .exactZero none

def exact84113RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21688⟩⟩]⟩, (1)⟩]

def event84113 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨21689⟩⟩) 84112 exact84113RawTerms .large 84109 .exactZero (none)

def event84114 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨28305⟩⟩)

def event84115 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event84116 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event84117 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨2348⟩⟩) (.authority (.operator))

def event84118 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨2348⟩⟩) (.finite 1)

def event84119 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event84120 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event84121 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event84122 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event84123 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 84122

def event84124 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 84120

def event84125 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 84123 .coefficient) (.value (.predecessor 1 84124 .coefficient)))

def event84126 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event84127 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5505⟩⟩) 0 ⟨5503⟩ 84126

def event84128 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5505⟩⟩) 1 ⟨2348⟩ 84118

def event84129 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5505⟩⟩) (.sum [.predecessor 0 84127 .coefficient, .predecessor 1 84128 .coefficient])

def event84130 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5505⟩⟩) (.finite 218)

def event84131 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5536⟩⟩) 0 ⟨5505⟩ 84130

def event84132 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5536⟩⟩) 1 ⟨961⟩ 84116

def event84133 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5536⟩⟩) (.identity (.predecessor 1 84132 .coefficient))

def event84134 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5536⟩⟩) (.finite 224)

def event84135 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11637⟩⟩) 0 ⟨5536⟩ 84134

def event84136 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11637⟩⟩) (.authority (.programFamilyFact))

def exact84137RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11637⟩⟩], []⟩, (1)⟩]

theorem exact84137RawTermsValid :
    exact84137RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84137 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11637⟩⟩) exact84137RawTerms (.finite 28) 84136 .exactZero (none)

def event84138 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14641⟩⟩) 0 ⟨5536⟩ 84134

def event84139 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14641⟩⟩) (.authority (.programFamilyFact))

def exact84140RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨14641⟩⟩], []⟩, (1)⟩]

theorem exact84140RawTermsValid :
    exact84140RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84140 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14641⟩⟩) exact84140RawTerms (.finite 28) 84139 .exactZero (none)

def event84141 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14642⟩⟩) 0 ⟨14641⟩ 84140

def event84142 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14642⟩⟩) 1 ⟨11637⟩ 84137

def event84143 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14642⟩⟩) (.product (.predecessor 0 84141 .coefficient) (.predecessor 1 84142 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event84144 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨14642⟩⟩, .operator (⟨84140, 0⟩, ⟨84137, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨11637⟩⟩, ⟨.program ⟨214⟩, ⟨14641⟩⟩], []⟩, (1)⟩)

def exact84145RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11637⟩⟩, ⟨.program ⟨214⟩, ⟨14641⟩⟩], []⟩, (1)⟩]

theorem exact84145RawTermsValid :
    exact84145RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84145 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14642⟩⟩) exact84145RawTerms (.finite 784) 84143 .exactZero (none)

def event84146 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14643⟩⟩) 0 ⟨14642⟩ 84145

def event84147 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14643⟩⟩) (.identity (.predecessor 0 84146 .coefficient))

def event84148 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨14643⟩⟩) (.finite 784)

def event84149 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16178⟩⟩) 0 ⟨14643⟩ 84148

def event84150 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16178⟩⟩) (.authority (.programFamilyFact))

def exact84151RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16178⟩⟩], []⟩, (1)⟩]

theorem exact84151RawTermsValid :
    exact84151RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84151 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16178⟩⟩) exact84151RawTerms (.finite 28) 84150 .exactZero (none)

def event84152 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16179⟩⟩) 0 ⟨16178⟩ 84151

def event84153 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16179⟩⟩) (.identity (.predecessor 0 84152 .coefficient))

def event84154 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨16179⟩⟩) (.finite 28)

def event84155 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24286⟩⟩) 0 ⟨16179⟩ 84154

def event84156 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24286⟩⟩) (.authority (.programFamilyFact))

def event84157 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨24286⟩⟩) (.finite 3720)

def event84158 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨6689⟩⟩) .missing

def event84159 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24288⟩⟩) 0 ⟨6689⟩ 84158

def event84160 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24288⟩⟩) 1 ⟨24286⟩ 84157

def event84161 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24288⟩⟩) (.authority (.operator))

def exact84162RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨24288⟩⟩]⟩, (1)⟩]

theorem exact84162RawTermsValid :
    exact84162RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84162 : Event := .resultExact (⟨.program ⟨214⟩, ⟨24288⟩⟩) exact84162RawTerms .large 84161 .exactZero (none)

def event84163 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28300⟩⟩) 0 ⟨24288⟩ 84162

def event84164 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28300⟩⟩) (.authority (.operator))

def exact84165RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨28300⟩⟩]⟩, (1)⟩]

theorem exact84165RawTermsValid :
    exact84165RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84165 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28300⟩⟩) exact84165RawTerms (.finite 8192) 84164 .exactZero (none)

def event84166 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨110⟩⟩) (.authority (.operator))

def event84167 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨110⟩⟩) .exactZero

def event84168 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16218⟩⟩) 0 ⟨16179⟩ 84154

def event84169 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16218⟩⟩) 1 ⟨110⟩ 84167

def event84170 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16218⟩⟩) (.sum [.predecessor 0 84168 .coefficient, .predecessor 1 84169 .coefficient])

def event84171 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨16218⟩⟩) (.finite 28)

def event84172 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16219⟩⟩) 0 ⟨16218⟩ 84171

def event84173 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16219⟩⟩) (.identity (.predecessor 0 84172 .coefficient))

def exact84174RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16178⟩⟩], []⟩, (1)⟩]

theorem exact84174RawTermsValid :
    exact84174RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84174 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16219⟩⟩) exact84174RawTerms (.finite 28) 84173 .exactZero (none)

def event84175 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6544⟩⟩) (.authority (.factStore))

def exact84176RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact84176RawTermsValid :
    exact84176RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84176 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6544⟩⟩) exact84176RawTerms .large 84175 .exactZero (none)

def event84177 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16220⟩⟩) 0 ⟨6544⟩ 84176

def event84178 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16220⟩⟩) 1 ⟨16219⟩ 84174

def event84179 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16220⟩⟩) (.product (.predecessor 0 84177 .coefficient) (.predecessor 1 84178 .coefficient) (⟨false, false, none, none, none⟩))

def event84180 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨16220⟩⟩, .operator (⟨84176, 0⟩, ⟨84174, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨16178⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact84181RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16178⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact84181RawTermsValid :
    exact84181RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84181 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16220⟩⟩) exact84181RawTerms .large 84179 .exactZero (none)

def event84182 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6699⟩⟩) 0 ⟨6689⟩ 84158

def event84183 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6699⟩⟩) (.authority (.operator))

def exact84184RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6699⟩⟩]⟩, (1)⟩]

theorem exact84184RawTermsValid :
    exact84184RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84184 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6699⟩⟩) exact84184RawTerms .large 84183 .exactZero (none)

def event84185 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16221⟩⟩) 0 ⟨6699⟩ 84184

def event84186 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16221⟩⟩) 1 ⟨16220⟩ 84181

def event84187 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16221⟩⟩) (.sum [.predecessor 0 84185 .coefficient, .predecessor 1 84186 .coefficient])

def exact84188RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6699⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16178⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact84188RawTermsValid :
    exact84188RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84188 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16221⟩⟩) exact84188RawTerms .large 84187 .exactZero (none)

def event84189 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28301⟩⟩) 0 ⟨16221⟩ 84188

def event84190 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28301⟩⟩) 1 ⟨28300⟩ 84165

def event84191 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28301⟩⟩) (.product (.predecessor 0 84189 .coefficient) (.predecessor 1 84190 .coefficient) (⟨false, false, none, none, none⟩))

def event84192 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28301⟩⟩, .operator (⟨84188, 0⟩, ⟨84165, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6699⟩⟩, ⟨.program ⟨214⟩, ⟨28300⟩⟩]⟩, (1)⟩)

def event84193 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28301⟩⟩, .operator (⟨84188, 1⟩, ⟨84165, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨16178⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28300⟩⟩]⟩, (-1)⟩)

def event84194 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨28301⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨16178⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28300⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨28300⟩⟩) ⟨24288⟩ 84162)

def event84195 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28301⟩⟩, .relation 84194 0, ⟨[⟨.program ⟨214⟩, ⟨16178⟩⟩], [⟨.program ⟨214⟩, ⟨24288⟩⟩]⟩, (-1)⟩)

def exact84196RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6699⟩⟩, ⟨.program ⟨214⟩, ⟨28300⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16178⟩⟩], [⟨.program ⟨214⟩, ⟨24288⟩⟩]⟩, (-1)⟩]

theorem exact84196RawTermsValid :
    exact84196RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84196 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28301⟩⟩) exact84196RawTerms .large 84191 .exactZero (none)

def event84197 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18340⟩⟩) 0 ⟨16179⟩ 84154

def event84198 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨18340⟩⟩) (.authority (.programFamilyFact))

def exact84199RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨18340⟩⟩], []⟩, (1)⟩]

theorem exact84199RawTermsValid :
    exact84199RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84199 : Event := .resultExact (⟨.program ⟨214⟩, ⟨18340⟩⟩) exact84199RawTerms (.finite 62) 84198 .exactZero (none)

def event84200 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18351⟩⟩) 0 ⟨6544⟩ 84176

def event84201 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18351⟩⟩) 1 ⟨18340⟩ 84199

def event84202 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨18351⟩⟩) (.product (.predecessor 0 84200 .coefficient) (.predecessor 1 84201 .coefficient) (⟨false, true, none, none, some 1⟩))

def event84203 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18351⟩⟩, .operator (⟨84176, 0⟩, ⟨84199, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨18340⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact84204RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨18340⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact84204RawTermsValid :
    exact84204RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84204 : Event := .resultExact (⟨.program ⟨214⟩, ⟨18351⟩⟩) exact84204RawTerms .large 84202 .exactZero (none)

def event84205 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6727⟩⟩) 0 ⟨6689⟩ 84158

def event84206 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6727⟩⟩) (.authority (.operator))

def exact84207RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6727⟩⟩]⟩, (1)⟩]

theorem exact84207RawTermsValid :
    exact84207RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84207 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6727⟩⟩) exact84207RawTerms .large 84206 .exactZero (none)

def event84208 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18352⟩⟩) 0 ⟨6727⟩ 84207

def event84209 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18352⟩⟩) 1 ⟨18351⟩ 84204

def event84210 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨18352⟩⟩) (.sum [.predecessor 0 84208 .coefficient, .predecessor 1 84209 .coefficient])

def exact84211RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6727⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨18340⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact84211RawTermsValid :
    exact84211RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84211 : Event := .resultExact (⟨.program ⟨214⟩, ⟨18352⟩⟩) exact84211RawTerms .large 84210 .exactZero (none)

def event84212 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28305⟩⟩) 0 ⟨18352⟩ 84211

def event84213 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28305⟩⟩) 1 ⟨28301⟩ 84196

def event84214 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28305⟩⟩) (.sum [.predecessor 0 84212 .coefficient, .predecessor 1 84213 .coefficient])

def exact84215RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6699⟩⟩, ⟨.program ⟨214⟩, ⟨28300⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6727⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16178⟩⟩], [⟨.program ⟨214⟩, ⟨24288⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨18340⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact84215RawTermsValid :
    exact84215RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84215 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28305⟩⟩) exact84215RawTerms .large 84214 .exactZero (none)

def event84216 : Event := .preFoldPolynomial 84215 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6699⟩⟩, ⟨.program ⟨214⟩, ⟨28300⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6727⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16178⟩⟩], [⟨.program ⟨214⟩, ⟨24288⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨18340⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩] .exactZero none

def exact84217RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6699⟩⟩, ⟨.program ⟨214⟩, ⟨28300⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6727⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16178⟩⟩], [⟨.program ⟨214⟩, ⟨24288⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨18340⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

def event84217 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨28305⟩⟩) 84216 exact84217RawTerms .large 84214 .exactZero (none)

def event84218 : Event := .specializationComputed (⟨.program ⟨214⟩, ⟨16179⟩⟩) ⟨⟨140⟩, ⟨48⟩, ⟨109⟩⟩ ⟨84060, 84218⟩

def event84219 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨21691⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21688⟩⟩]⟩) (1) 0 2 (.universal 84218 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21688⟩⟩]⟩) (none) 84217)

def event84220 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21691⟩⟩, .relation 84219 1, ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6727⟩⟩]⟩, (1)⟩)

def event84221 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21691⟩⟩, .relation 84219 0, ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6699⟩⟩, ⟨.program ⟨214⟩, ⟨28300⟩⟩]⟩, (-1)⟩)

def event84222 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21691⟩⟩, .relation 84219 2, ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16178⟩⟩], [⟨.program ⟨214⟩, ⟨24288⟩⟩]⟩, (1)⟩)

def event84223 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21691⟩⟩, .relation 84219 3, ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨18340⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def eventLeaf5248 : Array AnnotatedEvent := #[
  { event := event83968
    frameStart := 83907 },
  { event := event83969
    frameStart := 83907 },
  { event := event83970
    frameStart := 83907 },
  { event := event83971
    frameStart := 83907 },
  { event := event83972
    frameStart := 83907 },
  { event := event83973
    frameStart := 83907 },
  { event := event83974
    frameStart := 83907 },
  { event := event83975
    frameStart := 83907 },
  { event := event83976
    frameStart := 83907 },
  { event := event83977
    frameStart := 83907 },
  { event := event83978
    frameStart := 83907 },
  { event := event83979
    frameStart := 83907 },
  { event := event83980
    frameStart := 83907 },
  { event := event83981
    frameStart := 83907 },
  { event := event83982
    frameStart := 83907 },
  { event := event83983
    frameStart := 83907 }
]

def eventLeaf5249 : Array AnnotatedEvent := #[
  { event := event83984
    frameStart := 83907 },
  { event := event83985
    frameStart := 83907 },
  { event := event83986
    frameStart := 83907 },
  { event := event83987
    frameStart := 83907 },
  { event := event83988
    frameStart := 83907 },
  { event := event83989
    frameStart := 83907 },
  { event := event83990
    frameStart := 83907 },
  { event := event83991
    frameStart := 83907 },
  { event := event83992
    frameStart := 83907 },
  { event := event83993
    frameStart := 83907 },
  { event := event83994
    frameStart := 83907 },
  { event := event83995
    frameStart := 83907 },
  { event := event83996
    frameStart := 83907 },
  { event := event83997
    frameStart := 83907 },
  { event := event83998
    frameStart := 83907 },
  { event := event83999
    frameStart := 83907 }
]

def eventLeaf5250 : Array AnnotatedEvent := #[
  { event := event84000
    frameStart := 83907 },
  { event := event84001
    frameStart := 83907 },
  { event := event84002
    frameStart := 83907 },
  { event := event84003
    frameStart := 83907 },
  { event := event84004
    frameStart := 83907 },
  { event := event84005
    frameStart := 83907 },
  { event := event84006
    frameStart := 83907 },
  { event := event84007
    frameStart := 83907 },
  { event := event84008
    frameStart := 83907 },
  { event := event84009
    frameStart := 83907 },
  { event := event84010
    frameStart := 83907 },
  { event := event84011
    frameStart := 83907 },
  { event := event84012
    frameStart := 83907 },
  { event := event84013
    frameStart := 83907 },
  { event := event84014
    frameStart := 83907 },
  { event := event84015
    frameStart := 83907 }
]

def eventLeaf5251 : Array AnnotatedEvent := #[
  { event := event84016
    frameStart := 83907 },
  { event := event84017
    frameStart := 83907 },
  { event := event84018
    frameStart := 83907 },
  { event := event84019
    frameStart := 83907 },
  { event := event84020
    frameStart := 83907 },
  { event := event84021
    frameStart := 83907 },
  { event := event84022
    frameStart := 83907 },
  { event := event84023
    frameStart := 0 },
  { event := event84024
    frameStart := 0 },
  { event := event84025
    frameStart := 0 },
  { event := event84026
    frameStart := 0 },
  { event := event84027
    frameStart := 0 },
  { event := event84028
    frameStart := 0 },
  { event := event84029
    frameStart := 0 },
  { event := event84030
    frameStart := 0 },
  { event := event84031
    frameStart := 0 }
]

def eventLeaf5252 : Array AnnotatedEvent := #[
  { event := event84032
    frameStart := 0 },
  { event := event84033
    frameStart := 0 },
  { event := event84034
    frameStart := 0 },
  { event := event84035
    frameStart := 0 },
  { event := event84036
    frameStart := 0 },
  { event := event84037
    frameStart := 0 },
  { event := event84038
    frameStart := 0 },
  { event := event84039
    frameStart := 0 },
  { event := event84040
    frameStart := 0 },
  { event := event84041
    frameStart := 0 },
  { event := event84042
    frameStart := 0 },
  { event := event84043
    frameStart := 0 },
  { event := event84044
    frameStart := 0 },
  { event := event84045
    frameStart := 0 },
  { event := event84046
    frameStart := 0 },
  { event := event84047
    frameStart := 0 }
]

def eventLeaf5253 : Array AnnotatedEvent := #[
  { event := event84048
    frameStart := 0 },
  { event := event84049
    frameStart := 0 },
  { event := event84050
    frameStart := 0 },
  { event := event84051
    frameStart := 0 },
  { event := event84052
    frameStart := 0 },
  { event := event84053
    frameStart := 0 },
  { event := event84054
    frameStart := 0 },
  { event := event84055
    frameStart := 0 },
  { event := event84056
    frameStart := 0 },
  { event := event84057
    frameStart := 0 },
  { event := event84058
    frameStart := 0 },
  { event := event84059
    frameStart := 0 },
  { event := event84060
    frameStart := 84060 },
  { event := event84061
    frameStart := 84060 },
  { event := event84062
    frameStart := 84060 },
  { event := event84063
    frameStart := 84060 }
]

def eventLeaf5254 : Array AnnotatedEvent := #[
  { event := event84064
    frameStart := 84060 },
  { event := event84065
    frameStart := 84060 },
  { event := event84066
    frameStart := 84060 },
  { event := event84067
    frameStart := 84060 },
  { event := event84068
    frameStart := 84060 },
  { event := event84069
    frameStart := 84060 },
  { event := event84070
    frameStart := 84060 },
  { event := event84071
    frameStart := 84060 },
  { event := event84072
    frameStart := 84060 },
  { event := event84073
    frameStart := 84060 },
  { event := event84074
    frameStart := 84060 },
  { event := event84075
    frameStart := 84060 },
  { event := event84076
    frameStart := 84060 },
  { event := event84077
    frameStart := 84060 },
  { event := event84078
    frameStart := 84060 },
  { event := event84079
    frameStart := 84060 }
]

def eventLeaf5255 : Array AnnotatedEvent := #[
  { event := event84080
    frameStart := 84060 },
  { event := event84081
    frameStart := 84060 },
  { event := event84082
    frameStart := 84060 },
  { event := event84083
    frameStart := 84060 },
  { event := event84084
    frameStart := 84060 },
  { event := event84085
    frameStart := 84060 },
  { event := event84086
    frameStart := 84060 },
  { event := event84087
    frameStart := 84060 },
  { event := event84088
    frameStart := 84060 },
  { event := event84089
    frameStart := 84060 },
  { event := event84090
    frameStart := 84060 },
  { event := event84091
    frameStart := 84060 },
  { event := event84092
    frameStart := 84060 },
  { event := event84093
    frameStart := 84060 },
  { event := event84094
    frameStart := 84060 },
  { event := event84095
    frameStart := 84060 }
]

def eventLeaf5256 : Array AnnotatedEvent := #[
  { event := event84096
    frameStart := 84060 },
  { event := event84097
    frameStart := 84060 },
  { event := event84098
    frameStart := 84060 },
  { event := event84099
    frameStart := 84060 },
  { event := event84100
    frameStart := 84060 },
  { event := event84101
    frameStart := 84060 },
  { event := event84102
    frameStart := 84060 },
  { event := event84103
    frameStart := 84060 },
  { event := event84104
    frameStart := 84060 },
  { event := event84105
    frameStart := 84060 },
  { event := event84106
    frameStart := 84060 },
  { event := event84107
    frameStart := 84060 },
  { event := event84108
    frameStart := 84060 },
  { event := event84109
    frameStart := 84060 },
  { event := event84110
    frameStart := 84060 },
  { event := event84111
    frameStart := 84060 }
]

def eventLeaf5257 : Array AnnotatedEvent := #[
  { event := event84112
    frameStart := 84060 },
  { event := event84113
    frameStart := 84060 },
  { event := event84114
    frameStart := 84114 },
  { event := event84115
    frameStart := 84114 },
  { event := event84116
    frameStart := 84114 },
  { event := event84117
    frameStart := 84114 },
  { event := event84118
    frameStart := 84114 },
  { event := event84119
    frameStart := 84114 },
  { event := event84120
    frameStart := 84114 },
  { event := event84121
    frameStart := 84114 },
  { event := event84122
    frameStart := 84114 },
  { event := event84123
    frameStart := 84114 },
  { event := event84124
    frameStart := 84114 },
  { event := event84125
    frameStart := 84114 },
  { event := event84126
    frameStart := 84114 },
  { event := event84127
    frameStart := 84114 }
]

def eventLeaf5258 : Array AnnotatedEvent := #[
  { event := event84128
    frameStart := 84114 },
  { event := event84129
    frameStart := 84114 },
  { event := event84130
    frameStart := 84114 },
  { event := event84131
    frameStart := 84114 },
  { event := event84132
    frameStart := 84114 },
  { event := event84133
    frameStart := 84114 },
  { event := event84134
    frameStart := 84114 },
  { event := event84135
    frameStart := 84114 },
  { event := event84136
    frameStart := 84114 },
  { event := event84137
    frameStart := 84114 },
  { event := event84138
    frameStart := 84114 },
  { event := event84139
    frameStart := 84114 },
  { event := event84140
    frameStart := 84114 },
  { event := event84141
    frameStart := 84114 },
  { event := event84142
    frameStart := 84114 },
  { event := event84143
    frameStart := 84114 }
]

def eventLeaf5259 : Array AnnotatedEvent := #[
  { event := event84144
    frameStart := 84114 },
  { event := event84145
    frameStart := 84114 },
  { event := event84146
    frameStart := 84114 },
  { event := event84147
    frameStart := 84114 },
  { event := event84148
    frameStart := 84114 },
  { event := event84149
    frameStart := 84114 },
  { event := event84150
    frameStart := 84114 },
  { event := event84151
    frameStart := 84114 },
  { event := event84152
    frameStart := 84114 },
  { event := event84153
    frameStart := 84114 },
  { event := event84154
    frameStart := 84114 },
  { event := event84155
    frameStart := 84114 },
  { event := event84156
    frameStart := 84114 },
  { event := event84157
    frameStart := 84114 },
  { event := event84158
    frameStart := 84114 },
  { event := event84159
    frameStart := 84114 }
]

def eventLeaf5260 : Array AnnotatedEvent := #[
  { event := event84160
    frameStart := 84114 },
  { event := event84161
    frameStart := 84114 },
  { event := event84162
    frameStart := 84114 },
  { event := event84163
    frameStart := 84114 },
  { event := event84164
    frameStart := 84114 },
  { event := event84165
    frameStart := 84114 },
  { event := event84166
    frameStart := 84114 },
  { event := event84167
    frameStart := 84114 },
  { event := event84168
    frameStart := 84114 },
  { event := event84169
    frameStart := 84114 },
  { event := event84170
    frameStart := 84114 },
  { event := event84171
    frameStart := 84114 },
  { event := event84172
    frameStart := 84114 },
  { event := event84173
    frameStart := 84114 },
  { event := event84174
    frameStart := 84114 },
  { event := event84175
    frameStart := 84114 }
]

def eventLeaf5261 : Array AnnotatedEvent := #[
  { event := event84176
    frameStart := 84114 },
  { event := event84177
    frameStart := 84114 },
  { event := event84178
    frameStart := 84114 },
  { event := event84179
    frameStart := 84114 },
  { event := event84180
    frameStart := 84114 },
  { event := event84181
    frameStart := 84114 },
  { event := event84182
    frameStart := 84114 },
  { event := event84183
    frameStart := 84114 },
  { event := event84184
    frameStart := 84114 },
  { event := event84185
    frameStart := 84114 },
  { event := event84186
    frameStart := 84114 },
  { event := event84187
    frameStart := 84114 },
  { event := event84188
    frameStart := 84114 },
  { event := event84189
    frameStart := 84114 },
  { event := event84190
    frameStart := 84114 },
  { event := event84191
    frameStart := 84114 }
]

def eventLeaf5262 : Array AnnotatedEvent := #[
  { event := event84192
    frameStart := 84114 },
  { event := event84193
    frameStart := 84114 },
  { event := event84194
    frameStart := 84114 },
  { event := event84195
    frameStart := 84114 },
  { event := event84196
    frameStart := 84114 },
  { event := event84197
    frameStart := 84114 },
  { event := event84198
    frameStart := 84114 },
  { event := event84199
    frameStart := 84114 },
  { event := event84200
    frameStart := 84114 },
  { event := event84201
    frameStart := 84114 },
  { event := event84202
    frameStart := 84114 },
  { event := event84203
    frameStart := 84114 },
  { event := event84204
    frameStart := 84114 },
  { event := event84205
    frameStart := 84114 },
  { event := event84206
    frameStart := 84114 },
  { event := event84207
    frameStart := 84114 }
]

def eventLeaf5263 : Array AnnotatedEvent := #[
  { event := event84208
    frameStart := 84114 },
  { event := event84209
    frameStart := 84114 },
  { event := event84210
    frameStart := 84114 },
  { event := event84211
    frameStart := 84114 },
  { event := event84212
    frameStart := 84114 },
  { event := event84213
    frameStart := 84114 },
  { event := event84214
    frameStart := 84114 },
  { event := event84215
    frameStart := 84114 },
  { event := event84216
    frameStart := 84114 },
  { event := event84217
    frameStart := 84114 },
  { event := event84218
    frameStart := 0 },
  { event := event84219
    frameStart := 0 },
  { event := event84220
    frameStart := 0 },
  { event := event84221
    frameStart := 0 },
  { event := event84222
    frameStart := 0 },
  { event := event84223
    frameStart := 0 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events328
