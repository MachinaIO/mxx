import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events381

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event97536 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19734⟩⟩) 0 ⟨6⟩ 97535

def event97537 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19734⟩⟩) 1 ⟨19733⟩ 97533

def event97538 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19734⟩⟩) (.product (.predecessor 0 97536 .coefficient) (.predecessor 1 97537 .coefficient) (⟨false, false, none, none, none⟩))

def event97539 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19734⟩⟩, .operator (⟨97535, 0⟩, ⟨97533, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19733⟩⟩]⟩, (1)⟩)

def exact97540RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19733⟩⟩]⟩, (1)⟩]

theorem exact97540RawTermsValid :
    exact97540RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97540 : Event := .resultExact (⟨.program ⟨214⟩, ⟨19734⟩⟩) exact97540RawTerms .large 97538 .exactZero (none)

def event97541 : Event := .preFoldPolynomial 97540 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19733⟩⟩]⟩, (1)⟩] .exactZero none

def exact97542RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19733⟩⟩]⟩, (1)⟩]

def event97542 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨19734⟩⟩) 97541 exact97542RawTerms .large 97538 .exactZero (none)

def event97543 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨25133⟩⟩)

def event97544 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event97545 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event97546 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event97547 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event97548 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 97547

def event97549 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 97545

def event97550 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 97548 .coefficient) (.value (.predecessor 1 97549 .coefficient)))

def event97551 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event97552 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11737⟩⟩) 0 ⟨5503⟩ 97551

def event97553 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11737⟩⟩) (.authority (.programFamilyFact))

def exact97554RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11737⟩⟩], []⟩, (1)⟩]

theorem exact97554RawTermsValid :
    exact97554RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97554 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11737⟩⟩) exact97554RawTerms (.finite 30) 97553 .exactZero (none)

def event97555 : Event := .predecessor (⟨.program ⟨214⟩, ⟨9595⟩⟩) 0 ⟨5503⟩ 97551

def event97556 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨9595⟩⟩) (.authority (.programFamilyFact))

def exact97557RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨9595⟩⟩], []⟩, (1)⟩]

theorem exact97557RawTermsValid :
    exact97557RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97557 : Event := .resultExact (⟨.program ⟨214⟩, ⟨9595⟩⟩) exact97557RawTerms (.finite 30) 97556 .exactZero (none)

def event97558 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11738⟩⟩) 0 ⟨9595⟩ 97557

def event97559 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11738⟩⟩) 1 ⟨11737⟩ 97554

def event97560 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11738⟩⟩) (.product (.predecessor 0 97558 .coefficient) (.predecessor 1 97559 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event97561 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨11738⟩⟩, .operator (⟨97557, 0⟩, ⟨97554, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨9595⟩⟩, ⟨.program ⟨214⟩, ⟨11737⟩⟩], []⟩, (1)⟩)

def exact97562RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨9595⟩⟩, ⟨.program ⟨214⟩, ⟨11737⟩⟩], []⟩, (1)⟩]

theorem exact97562RawTermsValid :
    exact97562RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97562 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11738⟩⟩) exact97562RawTerms (.finite 900) 97560 .exactZero (none)

def event97563 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11739⟩⟩) 0 ⟨11738⟩ 97562

def event97564 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11739⟩⟩) (.identity (.predecessor 0 97563 .coefficient))

def event97565 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨11739⟩⟩) (.finite 900)

def event97566 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23073⟩⟩) 0 ⟨11739⟩ 97565

def event97567 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23073⟩⟩) (.authority (.programFamilyFact))

def event97568 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨23073⟩⟩) (.finite 3720)

def event97569 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨6689⟩⟩) .missing

def event97570 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23074⟩⟩) 0 ⟨6689⟩ 97569

def event97571 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23074⟩⟩) 1 ⟨23073⟩ 97568

def event97572 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23074⟩⟩) (.authority (.operator))

def exact97573RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨23074⟩⟩]⟩, (1)⟩]

theorem exact97573RawTermsValid :
    exact97573RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97573 : Event := .resultExact (⟨.program ⟨214⟩, ⟨23074⟩⟩) exact97573RawTerms .large 97572 .exactZero (none)

def event97574 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25129⟩⟩) 0 ⟨23074⟩ 97573

def event97575 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25129⟩⟩) (.authority (.operator))

def exact97576RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨25129⟩⟩]⟩, (1)⟩]

theorem exact97576RawTermsValid :
    exact97576RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97576 : Event := .resultExact (⟨.program ⟨214⟩, ⟨25129⟩⟩) exact97576RawTerms (.finite 8192) 97575 .exactZero (none)

def event97577 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨110⟩⟩) (.authority (.operator))

def event97578 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨110⟩⟩) .exactZero

def event97579 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11849⟩⟩) 0 ⟨11739⟩ 97565

def event97580 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11849⟩⟩) 1 ⟨110⟩ 97578

def event97581 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11849⟩⟩) (.sum [.predecessor 0 97579 .coefficient, .predecessor 1 97580 .coefficient])

def event97582 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨11849⟩⟩) (.finite 900)

def event97583 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11850⟩⟩) 0 ⟨11849⟩ 97582

def event97584 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11850⟩⟩) (.identity (.predecessor 0 97583 .coefficient))

def exact97585RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨9595⟩⟩, ⟨.program ⟨214⟩, ⟨11737⟩⟩], []⟩, (1)⟩]

theorem exact97585RawTermsValid :
    exact97585RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97585 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11850⟩⟩) exact97585RawTerms (.finite 900) 97584 .exactZero (none)

def event97586 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6544⟩⟩) (.authority (.factStore))

def exact97587RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact97587RawTermsValid :
    exact97587RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97587 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6544⟩⟩) exact97587RawTerms .large 97586 .exactZero (none)

def event97588 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11851⟩⟩) 0 ⟨6544⟩ 97587

def event97589 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11851⟩⟩) 1 ⟨11850⟩ 97585

def event97590 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11851⟩⟩) (.product (.predecessor 0 97588 .coefficient) (.predecessor 1 97589 .coefficient) (⟨false, false, none, none, none⟩))

def event97591 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨11851⟩⟩, .operator (⟨97587, 0⟩, ⟨97585, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨9595⟩⟩, ⟨.program ⟨214⟩, ⟨11737⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact97592RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨9595⟩⟩, ⟨.program ⟨214⟩, ⟨11737⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact97592RawTermsValid :
    exact97592RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97592 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11851⟩⟩) exact97592RawTerms .large 97590 .exactZero (none)

def event97593 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨2348⟩⟩) (.authority (.operator))

def event97594 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨2348⟩⟩) (.finite 1)

def event97595 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6757⟩⟩) 0 ⟨6689⟩ 97569

def event97596 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6757⟩⟩) (.authority (.operator))

def exact97597RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6757⟩⟩]⟩, (1)⟩]

theorem exact97597RawTermsValid :
    exact97597RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97597 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6757⟩⟩) exact97597RawTerms .large 97596 .exactZero (none)

def event97598 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6783⟩⟩) 0 ⟨6757⟩ 97597

def event97599 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6783⟩⟩) (.identity (.predecessor 0 97598 .coefficient))

def exact97600RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6783⟩⟩]⟩, (1)⟩]

theorem exact97600RawTermsValid :
    exact97600RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97600 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6783⟩⟩) exact97600RawTerms .large 97599 .exactZero (none)

def event97601 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7861⟩⟩) 0 ⟨6783⟩ 97600

def event97602 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7861⟩⟩) (.authority (.operator))

def exact97603RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨7861⟩⟩]⟩, (1)⟩]

theorem exact97603RawTermsValid :
    exact97603RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97603 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7861⟩⟩) exact97603RawTerms (.finite 8192) 97602 .exactZero (none)

def event97604 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7862⟩⟩) 0 ⟨7861⟩ 97603

def event97605 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7862⟩⟩) 1 ⟨2348⟩ 97594

def event97606 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7862⟩⟩) (.scale (.predecessor 0 97604 .coefficient) (.value (.predecessor 1 97605 .coefficient)))

def exact97607RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨7861⟩⟩]⟩, (1)⟩]

theorem exact97607RawTermsValid :
    exact97607RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97607 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7862⟩⟩) exact97607RawTerms (.finite 8192) 97606 .exactZero (none)

def event97608 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6763⟩⟩) 0 ⟨6757⟩ 97597

def event97609 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6763⟩⟩) (.identity (.predecessor 0 97608 .coefficient))

def exact97610RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6763⟩⟩]⟩, (1)⟩]

theorem exact97610RawTermsValid :
    exact97610RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97610 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6763⟩⟩) exact97610RawTerms .large 97609 .exactZero (none)

def event97611 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7863⟩⟩) 0 ⟨6763⟩ 97610

def event97612 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7863⟩⟩) 1 ⟨7862⟩ 97607

def event97613 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7863⟩⟩) (.product (.predecessor 0 97611 .coefficient) (.predecessor 1 97612 .coefficient) (⟨false, false, none, none, none⟩))

def event97614 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7863⟩⟩, .operator (⟨97610, 0⟩, ⟨97607, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6763⟩⟩, ⟨.program ⟨214⟩, ⟨7861⟩⟩]⟩, (1)⟩)

def exact97615RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6763⟩⟩, ⟨.program ⟨214⟩, ⟨7861⟩⟩]⟩, (1)⟩]

theorem exact97615RawTermsValid :
    exact97615RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97615 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7863⟩⟩) exact97615RawTerms .large 97613 .exactZero (none)

def event97616 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11852⟩⟩) 0 ⟨7863⟩ 97615

def event97617 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11852⟩⟩) 1 ⟨11851⟩ 97592

def event97618 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11852⟩⟩) (.sum [.predecessor 0 97616 .coefficient, .predecessor 1 97617 .coefficient])

def exact97619RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6763⟩⟩, ⟨.program ⟨214⟩, ⟨7861⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨9595⟩⟩, ⟨.program ⟨214⟩, ⟨11737⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact97619RawTermsValid :
    exact97619RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97619 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11852⟩⟩) exact97619RawTerms .large 97618 .exactZero (none)

def event97620 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25132⟩⟩) 0 ⟨11852⟩ 97619

def event97621 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25132⟩⟩) 1 ⟨25129⟩ 97576

def event97622 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25132⟩⟩) (.product (.predecessor 0 97620 .coefficient) (.predecessor 1 97621 .coefficient) (⟨false, false, none, none, none⟩))

def event97623 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25132⟩⟩, .operator (⟨97619, 0⟩, ⟨97576, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6763⟩⟩, ⟨.program ⟨214⟩, ⟨7861⟩⟩, ⟨.program ⟨214⟩, ⟨25129⟩⟩]⟩, (1)⟩)

def event97624 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25132⟩⟩, .operator (⟨97619, 1⟩, ⟨97576, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨9595⟩⟩, ⟨.program ⟨214⟩, ⟨11737⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25129⟩⟩]⟩, (-1)⟩)

def event97625 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨25132⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨9595⟩⟩, ⟨.program ⟨214⟩, ⟨11737⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25129⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨25129⟩⟩) ⟨23074⟩ 97573)

def event97626 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25132⟩⟩, .relation 97625 0, ⟨[⟨.program ⟨214⟩, ⟨9595⟩⟩, ⟨.program ⟨214⟩, ⟨11737⟩⟩], [⟨.program ⟨214⟩, ⟨23074⟩⟩]⟩, (-1)⟩)

def exact97627RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6763⟩⟩, ⟨.program ⟨214⟩, ⟨7861⟩⟩, ⟨.program ⟨214⟩, ⟨25129⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨9595⟩⟩, ⟨.program ⟨214⟩, ⟨11737⟩⟩], [⟨.program ⟨214⟩, ⟨23074⟩⟩]⟩, (-1)⟩]

theorem exact97627RawTermsValid :
    exact97627RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97627 : Event := .resultExact (⟨.program ⟨214⟩, ⟨25132⟩⟩) exact97627RawTerms .large 97622 .exactZero (none)

def event97628 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16252⟩⟩) 0 ⟨11739⟩ 97565

def event97629 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16252⟩⟩) (.authority (.programFamilyFact))

def exact97630RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16252⟩⟩], []⟩, (1)⟩]

theorem exact97630RawTermsValid :
    exact97630RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97630 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16252⟩⟩) exact97630RawTerms (.finite 30) 97629 .exactZero (none)

def event97631 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16254⟩⟩) 0 ⟨6544⟩ 97587

def event97632 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16254⟩⟩) 1 ⟨16252⟩ 97630

def event97633 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16254⟩⟩) (.product (.predecessor 0 97631 .coefficient) (.predecessor 1 97632 .coefficient) (⟨false, true, none, none, some 1⟩))

def event97634 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨16254⟩⟩, .operator (⟨97587, 0⟩, ⟨97630, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨16252⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact97635RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16252⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact97635RawTermsValid :
    exact97635RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97635 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16254⟩⟩) exact97635RawTerms .large 97633 .exactZero (none)

def event97636 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6700⟩⟩) 0 ⟨6689⟩ 97569

def event97637 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6700⟩⟩) (.authority (.operator))

def exact97638RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6700⟩⟩]⟩, (1)⟩]

theorem exact97638RawTermsValid :
    exact97638RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97638 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6700⟩⟩) exact97638RawTerms .large 97637 .exactZero (none)

def event97639 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16255⟩⟩) 0 ⟨6700⟩ 97638

def event97640 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16255⟩⟩) 1 ⟨16254⟩ 97635

def event97641 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16255⟩⟩) (.sum [.predecessor 0 97639 .coefficient, .predecessor 1 97640 .coefficient])

def exact97642RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6700⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16252⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact97642RawTermsValid :
    exact97642RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97642 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16255⟩⟩) exact97642RawTerms .large 97641 .exactZero (none)

def event97643 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25133⟩⟩) 0 ⟨16255⟩ 97642

def event97644 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25133⟩⟩) 1 ⟨25132⟩ 97627

def event97645 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25133⟩⟩) (.sum [.predecessor 0 97643 .coefficient, .predecessor 1 97644 .coefficient])

def exact97646RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6700⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6763⟩⟩, ⟨.program ⟨214⟩, ⟨7861⟩⟩, ⟨.program ⟨214⟩, ⟨25129⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨9595⟩⟩, ⟨.program ⟨214⟩, ⟨11737⟩⟩], [⟨.program ⟨214⟩, ⟨23074⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16252⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact97646RawTermsValid :
    exact97646RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97646 : Event := .resultExact (⟨.program ⟨214⟩, ⟨25133⟩⟩) exact97646RawTerms .large 97645 .exactZero (none)

def event97647 : Event := .preFoldPolynomial 97646 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6700⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6763⟩⟩, ⟨.program ⟨214⟩, ⟨7861⟩⟩, ⟨.program ⟨214⟩, ⟨25129⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨9595⟩⟩, ⟨.program ⟨214⟩, ⟨11737⟩⟩], [⟨.program ⟨214⟩, ⟨23074⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16252⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩] .exactZero none

def exact97648RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6700⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6763⟩⟩, ⟨.program ⟨214⟩, ⟨7861⟩⟩, ⟨.program ⟨214⟩, ⟨25129⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨9595⟩⟩, ⟨.program ⟨214⟩, ⟨11737⟩⟩], [⟨.program ⟨214⟩, ⟨23074⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16252⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

def event97648 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨25133⟩⟩) 97647 exact97648RawTerms .large 97645 .exactZero (none)

def event97649 : Event := .specializationComputed (⟨.program ⟨214⟩, ⟨11739⟩⟩) ⟨⟨113⟩, ⟨18⟩, ⟨109⟩⟩ ⟨97507, 97649⟩

def event97650 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨19736⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19733⟩⟩]⟩) (1) 0 2 (.universal 97649 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19733⟩⟩]⟩) (none) 97648)

def event97651 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19736⟩⟩, .relation 97650 0, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6700⟩⟩]⟩, (1)⟩)

def event97652 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19736⟩⟩, .relation 97650 1, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6763⟩⟩, ⟨.program ⟨214⟩, ⟨7861⟩⟩, ⟨.program ⟨214⟩, ⟨25129⟩⟩]⟩, (-1)⟩)

def event97653 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19736⟩⟩, .relation 97650 2, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨9595⟩⟩, ⟨.program ⟨214⟩, ⟨11737⟩⟩], [⟨.program ⟨214⟩, ⟨23074⟩⟩]⟩, (1)⟩)

def event97654 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19736⟩⟩, .relation 97650 3, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16252⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def exact97655RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6700⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6763⟩⟩, ⟨.program ⟨214⟩, ⟨7861⟩⟩, ⟨.program ⟨214⟩, ⟨25129⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨9595⟩⟩, ⟨.program ⟨214⟩, ⟨11737⟩⟩], [⟨.program ⟨214⟩, ⟨23074⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16252⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact97655RawTermsValid :
    exact97655RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97655 : Event := .resultExact (⟨.program ⟨214⟩, ⟨19736⟩⟩) exact97655RawTerms .large 97503 (.finite 1811303510016) (some (97505))

def event97656 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25131⟩⟩) 0 ⟨19736⟩ 97655

def event97657 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25131⟩⟩) 1 ⟨25130⟩ 97493

def event97658 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25131⟩⟩) (.sum [.predecessor 0 97656 .coefficient, .predecessor 1 97657 .coefficient])

def event97659 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25131⟩⟩, .operator (⟨97655, 2⟩, ⟨97493, 1⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨9595⟩⟩, ⟨.program ⟨214⟩, ⟨11737⟩⟩], [⟨.program ⟨214⟩, ⟨23074⟩⟩]⟩, (-1)⟩)

def event97660 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25131⟩⟩, .operator (⟨97655, 1⟩, ⟨97493, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6763⟩⟩, ⟨.program ⟨214⟩, ⟨7861⟩⟩, ⟨.program ⟨214⟩, ⟨25129⟩⟩]⟩, (1)⟩)

def event97661 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25131⟩⟩) (.sum [.result 97655 .summary, .result 97493 .summary])

def exact97662RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6700⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16252⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact97662RawTermsValid :
    exact97662RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97662 : Event := .resultExact (⟨.program ⟨214⟩, ⟨25131⟩⟩) exact97662RawTerms .large 97658 (.finite 352097360556032) (some (97661))

def event97663 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28484⟩⟩) 0 ⟨25131⟩ 97662

def event97664 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28484⟩⟩) 1 ⟨28482⟩ 97409

def event97665 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28484⟩⟩) (.product (.predecessor 0 97663 .coefficient) (.predecessor 1 97664 .coefficient) (⟨false, false, none, none, none⟩))

def event97666 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28484⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨28482⟩⟩]⟩) [⟨.result 97409 .coefficient, false, none⟩])

def event97667 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28484⟩⟩) (.product (.result 97662 .summary) (.transfer 97666) (⟨false, false, none, none, none⟩))

def event97668 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28484⟩⟩, .operator (⟨97662, 0⟩, ⟨97409, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6700⟩⟩, ⟨.program ⟨214⟩, ⟨28482⟩⟩]⟩, (1)⟩)

def event97669 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28484⟩⟩, .operator (⟨97662, 1⟩, ⟨97409, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16252⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28482⟩⟩]⟩, (-1)⟩)

def event97670 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨28484⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16252⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28482⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨28482⟩⟩) ⟨24342⟩ 97406)

def event97671 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28484⟩⟩, .relation 97670 0, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16252⟩⟩], [⟨.program ⟨214⟩, ⟨24342⟩⟩]⟩, (-1)⟩)

def exact97672RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6700⟩⟩, ⟨.program ⟨214⟩, ⟨28482⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16252⟩⟩], [⟨.program ⟨214⟩, ⟨24342⟩⟩]⟩, (-1)⟩]

theorem exact97672RawTermsValid :
    exact97672RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97672 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28484⟩⟩) exact97672RawTerms .large 97665 (.finite 1292202946798406336512) (some (97667))

def event97673 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21821⟩⟩) 0 ⟨16253⟩ 4744

def event97674 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21821⟩⟩) (.authority (.relationPreimageSource ⟨50⟩))

def exact97675RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨21821⟩⟩]⟩, (1)⟩]

theorem exact97675RawTermsValid :
    exact97675RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97675 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21821⟩⟩) exact97675RawTerms (.finite 136065468) 97674 .exactZero (none)

def event97676 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21823⟩⟩) 0 ⟨21821⟩ 97675

def event97677 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21823⟩⟩) 1 ⟨2348⟩ 4

def event97678 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21823⟩⟩) (.scale (.predecessor 0 97676 .coefficient) (.value (.predecessor 1 97677 .coefficient)))

def exact97679RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨21821⟩⟩]⟩, (1)⟩]

theorem exact97679RawTermsValid :
    exact97679RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97679 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21823⟩⟩) exact97679RawTerms (.finite 136065468) 97678 .exactZero (none)

def event97680 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21824⟩⟩) 0 ⟨5509⟩ 94462

def event97681 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21824⟩⟩) 1 ⟨21823⟩ 97679

def event97682 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21824⟩⟩) (.product (.predecessor 0 97680 .coefficient) (.predecessor 1 97681 .coefficient) (⟨false, false, none, none, none⟩))

def event97683 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21824⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨21821⟩⟩]⟩) [⟨.result 97675 .coefficient, false, none⟩])

def event97684 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21824⟩⟩) (.product (.result 94462 .summary) (.transfer 97683) (⟨false, false, none, none, none⟩))

def event97685 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21824⟩⟩, .operator (⟨94462, 0⟩, ⟨97679, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21821⟩⟩]⟩, (1)⟩)

def event97686 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨21822⟩⟩)

def event97687 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event97688 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event97689 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event97690 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event97691 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 97690

def event97692 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 97688

def event97693 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 97691 .coefficient) (.value (.predecessor 1 97692 .coefficient)))

def event97694 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event97695 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11737⟩⟩) 0 ⟨5503⟩ 97694

def event97696 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11737⟩⟩) (.authority (.programFamilyFact))

def exact97697RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11737⟩⟩], []⟩, (1)⟩]

theorem exact97697RawTermsValid :
    exact97697RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97697 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11737⟩⟩) exact97697RawTerms (.finite 30) 97696 .exactZero (none)

def event97698 : Event := .predecessor (⟨.program ⟨214⟩, ⟨9595⟩⟩) 0 ⟨5503⟩ 97694

def event97699 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨9595⟩⟩) (.authority (.programFamilyFact))

def exact97700RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨9595⟩⟩], []⟩, (1)⟩]

theorem exact97700RawTermsValid :
    exact97700RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97700 : Event := .resultExact (⟨.program ⟨214⟩, ⟨9595⟩⟩) exact97700RawTerms (.finite 30) 97699 .exactZero (none)

def event97701 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11738⟩⟩) 0 ⟨9595⟩ 97700

def event97702 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11738⟩⟩) 1 ⟨11737⟩ 97697

def event97703 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11738⟩⟩) (.product (.predecessor 0 97701 .coefficient) (.predecessor 1 97702 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event97704 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11738⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨9595⟩⟩, ⟨.program ⟨214⟩, ⟨11737⟩⟩], []⟩) [⟨.result 97700 .coefficient, true, some 1⟩, ⟨.result 97697 .coefficient, true, some 1⟩])

def event97705 : Event := .survivorFold (1) 97704

def exact97706RawTerms : List Term := []

theorem exact97706RawTermsValid :
    exact97706RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97706 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11738⟩⟩) exact97706RawTerms (.finite 900) 97703 (.finite 900) (some (97704))

def event97707 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11739⟩⟩) 0 ⟨11738⟩ 97706

def event97708 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11739⟩⟩) (.identity (.predecessor 0 97707 .coefficient))

def event97709 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨11739⟩⟩) (.finite 900)

def event97710 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16252⟩⟩) 0 ⟨11739⟩ 97709

def event97711 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16252⟩⟩) (.authority (.programFamilyFact))

def exact97712RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16252⟩⟩], []⟩, (1)⟩]

theorem exact97712RawTermsValid :
    exact97712RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97712 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16252⟩⟩) exact97712RawTerms (.finite 30) 97711 .exactZero (none)

def event97713 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16253⟩⟩) 0 ⟨16252⟩ 97712

def event97714 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16253⟩⟩) (.identity (.predecessor 0 97713 .coefficient))

def event97715 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨16253⟩⟩) (.finite 30)

def event97716 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21821⟩⟩) 0 ⟨16253⟩ 97715

def event97717 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21821⟩⟩) (.authority (.relationPreimageSource ⟨50⟩))

def exact97718RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨21821⟩⟩]⟩, (1)⟩]

theorem exact97718RawTermsValid :
    exact97718RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97718 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21821⟩⟩) exact97718RawTerms (.finite 136065468) 97717 .exactZero (none)

def event97719 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6⟩⟩) (.authority (.operator))

def exact97720RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩]⟩, (1)⟩]

theorem exact97720RawTermsValid :
    exact97720RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97720 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6⟩⟩) exact97720RawTerms .large 97719 .exactZero (none)

def event97721 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21822⟩⟩) 0 ⟨6⟩ 97720

def event97722 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21822⟩⟩) 1 ⟨21821⟩ 97718

def event97723 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21822⟩⟩) (.product (.predecessor 0 97721 .coefficient) (.predecessor 1 97722 .coefficient) (⟨false, false, none, none, none⟩))

def event97724 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21822⟩⟩, .operator (⟨97720, 0⟩, ⟨97718, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21821⟩⟩]⟩, (1)⟩)

def exact97725RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21821⟩⟩]⟩, (1)⟩]

theorem exact97725RawTermsValid :
    exact97725RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97725 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21822⟩⟩) exact97725RawTerms .large 97723 .exactZero (none)

def event97726 : Event := .preFoldPolynomial 97725 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21821⟩⟩]⟩, (1)⟩] .exactZero none

def exact97727RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21821⟩⟩]⟩, (1)⟩]

def event97727 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨21822⟩⟩) 97726 exact97727RawTerms .large 97723 .exactZero (none)

def event97728 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨28487⟩⟩)

def event97729 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event97730 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event97731 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event97732 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event97733 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 97732

def event97734 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 97730

def event97735 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 97733 .coefficient) (.value (.predecessor 1 97734 .coefficient)))

def event97736 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event97737 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11737⟩⟩) 0 ⟨5503⟩ 97736

def event97738 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11737⟩⟩) (.authority (.programFamilyFact))

def exact97739RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11737⟩⟩], []⟩, (1)⟩]

theorem exact97739RawTermsValid :
    exact97739RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97739 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11737⟩⟩) exact97739RawTerms (.finite 30) 97738 .exactZero (none)

def event97740 : Event := .predecessor (⟨.program ⟨214⟩, ⟨9595⟩⟩) 0 ⟨5503⟩ 97736

def event97741 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨9595⟩⟩) (.authority (.programFamilyFact))

def exact97742RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨9595⟩⟩], []⟩, (1)⟩]

theorem exact97742RawTermsValid :
    exact97742RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97742 : Event := .resultExact (⟨.program ⟨214⟩, ⟨9595⟩⟩) exact97742RawTerms (.finite 30) 97741 .exactZero (none)

def event97743 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11738⟩⟩) 0 ⟨9595⟩ 97742

def event97744 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11738⟩⟩) 1 ⟨11737⟩ 97739

def event97745 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11738⟩⟩) (.product (.predecessor 0 97743 .coefficient) (.predecessor 1 97744 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event97746 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨11738⟩⟩, .operator (⟨97742, 0⟩, ⟨97739, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨9595⟩⟩, ⟨.program ⟨214⟩, ⟨11737⟩⟩], []⟩, (1)⟩)

def exact97747RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨9595⟩⟩, ⟨.program ⟨214⟩, ⟨11737⟩⟩], []⟩, (1)⟩]

theorem exact97747RawTermsValid :
    exact97747RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97747 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11738⟩⟩) exact97747RawTerms (.finite 900) 97745 .exactZero (none)

def event97748 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11739⟩⟩) 0 ⟨11738⟩ 97747

def event97749 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11739⟩⟩) (.identity (.predecessor 0 97748 .coefficient))

def event97750 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨11739⟩⟩) (.finite 900)

def event97751 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16252⟩⟩) 0 ⟨11739⟩ 97750

def event97752 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16252⟩⟩) (.authority (.programFamilyFact))

def exact97753RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16252⟩⟩], []⟩, (1)⟩]

theorem exact97753RawTermsValid :
    exact97753RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97753 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16252⟩⟩) exact97753RawTerms (.finite 30) 97752 .exactZero (none)

def event97754 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16253⟩⟩) 0 ⟨16252⟩ 97753

def event97755 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16253⟩⟩) (.identity (.predecessor 0 97754 .coefficient))

def event97756 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨16253⟩⟩) (.finite 30)

def event97757 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24340⟩⟩) 0 ⟨16253⟩ 97756

def event97758 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24340⟩⟩) (.authority (.programFamilyFact))

def event97759 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨24340⟩⟩) (.finite 3720)

def event97760 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨6689⟩⟩) .missing

def event97761 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24342⟩⟩) 0 ⟨6689⟩ 97760

def event97762 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24342⟩⟩) 1 ⟨24340⟩ 97759

def event97763 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24342⟩⟩) (.authority (.operator))

def exact97764RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨24342⟩⟩]⟩, (1)⟩]

theorem exact97764RawTermsValid :
    exact97764RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97764 : Event := .resultExact (⟨.program ⟨214⟩, ⟨24342⟩⟩) exact97764RawTerms .large 97763 .exactZero (none)

def event97765 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28482⟩⟩) 0 ⟨24342⟩ 97764

def event97766 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28482⟩⟩) (.authority (.operator))

def exact97767RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨28482⟩⟩]⟩, (1)⟩]

theorem exact97767RawTermsValid :
    exact97767RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97767 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28482⟩⟩) exact97767RawTerms (.finite 8192) 97766 .exactZero (none)

def event97768 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨110⟩⟩) (.authority (.operator))

def event97769 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨110⟩⟩) .exactZero

def event97770 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16329⟩⟩) 0 ⟨16253⟩ 97756

def event97771 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16329⟩⟩) 1 ⟨110⟩ 97769

def event97772 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16329⟩⟩) (.sum [.predecessor 0 97770 .coefficient, .predecessor 1 97771 .coefficient])

def event97773 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨16329⟩⟩) (.finite 30)

def event97774 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16330⟩⟩) 0 ⟨16329⟩ 97773

def event97775 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16330⟩⟩) (.identity (.predecessor 0 97774 .coefficient))

def exact97776RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16252⟩⟩], []⟩, (1)⟩]

theorem exact97776RawTermsValid :
    exact97776RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97776 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16330⟩⟩) exact97776RawTerms (.finite 30) 97775 .exactZero (none)

def event97777 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6544⟩⟩) (.authority (.factStore))

def exact97778RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact97778RawTermsValid :
    exact97778RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97778 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6544⟩⟩) exact97778RawTerms .large 97777 .exactZero (none)

def event97779 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16331⟩⟩) 0 ⟨6544⟩ 97778

def event97780 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16331⟩⟩) 1 ⟨16330⟩ 97776

def event97781 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16331⟩⟩) (.product (.predecessor 0 97779 .coefficient) (.predecessor 1 97780 .coefficient) (⟨false, false, none, none, none⟩))

def event97782 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨16331⟩⟩, .operator (⟨97778, 0⟩, ⟨97776, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨16252⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact97783RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16252⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact97783RawTermsValid :
    exact97783RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97783 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16331⟩⟩) exact97783RawTerms .large 97781 .exactZero (none)

def event97784 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6700⟩⟩) 0 ⟨6689⟩ 97760

def event97785 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6700⟩⟩) (.authority (.operator))

def exact97786RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6700⟩⟩]⟩, (1)⟩]

theorem exact97786RawTermsValid :
    exact97786RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97786 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6700⟩⟩) exact97786RawTerms .large 97785 .exactZero (none)

def event97787 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16332⟩⟩) 0 ⟨6700⟩ 97786

def event97788 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16332⟩⟩) 1 ⟨16331⟩ 97783

def event97789 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16332⟩⟩) (.sum [.predecessor 0 97787 .coefficient, .predecessor 1 97788 .coefficient])

def exact97790RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6700⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16252⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact97790RawTermsValid :
    exact97790RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97790 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16332⟩⟩) exact97790RawTerms .large 97789 .exactZero (none)

def event97791 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28483⟩⟩) 0 ⟨16332⟩ 97790

def eventLeaf6096 : Array AnnotatedEvent := #[
  { event := event97536
    frameStart := 97507 },
  { event := event97537
    frameStart := 97507 },
  { event := event97538
    frameStart := 97507 },
  { event := event97539
    frameStart := 97507 },
  { event := event97540
    frameStart := 97507 },
  { event := event97541
    frameStart := 97507 },
  { event := event97542
    frameStart := 97507 },
  { event := event97543
    frameStart := 97543 },
  { event := event97544
    frameStart := 97543 },
  { event := event97545
    frameStart := 97543 },
  { event := event97546
    frameStart := 97543 },
  { event := event97547
    frameStart := 97543 },
  { event := event97548
    frameStart := 97543 },
  { event := event97549
    frameStart := 97543 },
  { event := event97550
    frameStart := 97543 },
  { event := event97551
    frameStart := 97543 }
]

def eventLeaf6097 : Array AnnotatedEvent := #[
  { event := event97552
    frameStart := 97543 },
  { event := event97553
    frameStart := 97543 },
  { event := event97554
    frameStart := 97543 },
  { event := event97555
    frameStart := 97543 },
  { event := event97556
    frameStart := 97543 },
  { event := event97557
    frameStart := 97543 },
  { event := event97558
    frameStart := 97543 },
  { event := event97559
    frameStart := 97543 },
  { event := event97560
    frameStart := 97543 },
  { event := event97561
    frameStart := 97543 },
  { event := event97562
    frameStart := 97543 },
  { event := event97563
    frameStart := 97543 },
  { event := event97564
    frameStart := 97543 },
  { event := event97565
    frameStart := 97543 },
  { event := event97566
    frameStart := 97543 },
  { event := event97567
    frameStart := 97543 }
]

def eventLeaf6098 : Array AnnotatedEvent := #[
  { event := event97568
    frameStart := 97543 },
  { event := event97569
    frameStart := 97543 },
  { event := event97570
    frameStart := 97543 },
  { event := event97571
    frameStart := 97543 },
  { event := event97572
    frameStart := 97543 },
  { event := event97573
    frameStart := 97543 },
  { event := event97574
    frameStart := 97543 },
  { event := event97575
    frameStart := 97543 },
  { event := event97576
    frameStart := 97543 },
  { event := event97577
    frameStart := 97543 },
  { event := event97578
    frameStart := 97543 },
  { event := event97579
    frameStart := 97543 },
  { event := event97580
    frameStart := 97543 },
  { event := event97581
    frameStart := 97543 },
  { event := event97582
    frameStart := 97543 },
  { event := event97583
    frameStart := 97543 }
]

def eventLeaf6099 : Array AnnotatedEvent := #[
  { event := event97584
    frameStart := 97543 },
  { event := event97585
    frameStart := 97543 },
  { event := event97586
    frameStart := 97543 },
  { event := event97587
    frameStart := 97543 },
  { event := event97588
    frameStart := 97543 },
  { event := event97589
    frameStart := 97543 },
  { event := event97590
    frameStart := 97543 },
  { event := event97591
    frameStart := 97543 },
  { event := event97592
    frameStart := 97543 },
  { event := event97593
    frameStart := 97543 },
  { event := event97594
    frameStart := 97543 },
  { event := event97595
    frameStart := 97543 },
  { event := event97596
    frameStart := 97543 },
  { event := event97597
    frameStart := 97543 },
  { event := event97598
    frameStart := 97543 },
  { event := event97599
    frameStart := 97543 }
]

def eventLeaf6100 : Array AnnotatedEvent := #[
  { event := event97600
    frameStart := 97543 },
  { event := event97601
    frameStart := 97543 },
  { event := event97602
    frameStart := 97543 },
  { event := event97603
    frameStart := 97543 },
  { event := event97604
    frameStart := 97543 },
  { event := event97605
    frameStart := 97543 },
  { event := event97606
    frameStart := 97543 },
  { event := event97607
    frameStart := 97543 },
  { event := event97608
    frameStart := 97543 },
  { event := event97609
    frameStart := 97543 },
  { event := event97610
    frameStart := 97543 },
  { event := event97611
    frameStart := 97543 },
  { event := event97612
    frameStart := 97543 },
  { event := event97613
    frameStart := 97543 },
  { event := event97614
    frameStart := 97543 },
  { event := event97615
    frameStart := 97543 }
]

def eventLeaf6101 : Array AnnotatedEvent := #[
  { event := event97616
    frameStart := 97543 },
  { event := event97617
    frameStart := 97543 },
  { event := event97618
    frameStart := 97543 },
  { event := event97619
    frameStart := 97543 },
  { event := event97620
    frameStart := 97543 },
  { event := event97621
    frameStart := 97543 },
  { event := event97622
    frameStart := 97543 },
  { event := event97623
    frameStart := 97543 },
  { event := event97624
    frameStart := 97543 },
  { event := event97625
    frameStart := 97543 },
  { event := event97626
    frameStart := 97543 },
  { event := event97627
    frameStart := 97543 },
  { event := event97628
    frameStart := 97543 },
  { event := event97629
    frameStart := 97543 },
  { event := event97630
    frameStart := 97543 },
  { event := event97631
    frameStart := 97543 }
]

def eventLeaf6102 : Array AnnotatedEvent := #[
  { event := event97632
    frameStart := 97543 },
  { event := event97633
    frameStart := 97543 },
  { event := event97634
    frameStart := 97543 },
  { event := event97635
    frameStart := 97543 },
  { event := event97636
    frameStart := 97543 },
  { event := event97637
    frameStart := 97543 },
  { event := event97638
    frameStart := 97543 },
  { event := event97639
    frameStart := 97543 },
  { event := event97640
    frameStart := 97543 },
  { event := event97641
    frameStart := 97543 },
  { event := event97642
    frameStart := 97543 },
  { event := event97643
    frameStart := 97543 },
  { event := event97644
    frameStart := 97543 },
  { event := event97645
    frameStart := 97543 },
  { event := event97646
    frameStart := 97543 },
  { event := event97647
    frameStart := 97543 }
]

def eventLeaf6103 : Array AnnotatedEvent := #[
  { event := event97648
    frameStart := 97543 },
  { event := event97649
    frameStart := 0 },
  { event := event97650
    frameStart := 0 },
  { event := event97651
    frameStart := 0 },
  { event := event97652
    frameStart := 0 },
  { event := event97653
    frameStart := 0 },
  { event := event97654
    frameStart := 0 },
  { event := event97655
    frameStart := 0 },
  { event := event97656
    frameStart := 0 },
  { event := event97657
    frameStart := 0 },
  { event := event97658
    frameStart := 0 },
  { event := event97659
    frameStart := 0 },
  { event := event97660
    frameStart := 0 },
  { event := event97661
    frameStart := 0 },
  { event := event97662
    frameStart := 0 },
  { event := event97663
    frameStart := 0 }
]

def eventLeaf6104 : Array AnnotatedEvent := #[
  { event := event97664
    frameStart := 0 },
  { event := event97665
    frameStart := 0 },
  { event := event97666
    frameStart := 0 },
  { event := event97667
    frameStart := 0 },
  { event := event97668
    frameStart := 0 },
  { event := event97669
    frameStart := 0 },
  { event := event97670
    frameStart := 0 },
  { event := event97671
    frameStart := 0 },
  { event := event97672
    frameStart := 0 },
  { event := event97673
    frameStart := 0 },
  { event := event97674
    frameStart := 0 },
  { event := event97675
    frameStart := 0 },
  { event := event97676
    frameStart := 0 },
  { event := event97677
    frameStart := 0 },
  { event := event97678
    frameStart := 0 },
  { event := event97679
    frameStart := 0 }
]

def eventLeaf6105 : Array AnnotatedEvent := #[
  { event := event97680
    frameStart := 0 },
  { event := event97681
    frameStart := 0 },
  { event := event97682
    frameStart := 0 },
  { event := event97683
    frameStart := 0 },
  { event := event97684
    frameStart := 0 },
  { event := event97685
    frameStart := 0 },
  { event := event97686
    frameStart := 97686 },
  { event := event97687
    frameStart := 97686 },
  { event := event97688
    frameStart := 97686 },
  { event := event97689
    frameStart := 97686 },
  { event := event97690
    frameStart := 97686 },
  { event := event97691
    frameStart := 97686 },
  { event := event97692
    frameStart := 97686 },
  { event := event97693
    frameStart := 97686 },
  { event := event97694
    frameStart := 97686 },
  { event := event97695
    frameStart := 97686 }
]

def eventLeaf6106 : Array AnnotatedEvent := #[
  { event := event97696
    frameStart := 97686 },
  { event := event97697
    frameStart := 97686 },
  { event := event97698
    frameStart := 97686 },
  { event := event97699
    frameStart := 97686 },
  { event := event97700
    frameStart := 97686 },
  { event := event97701
    frameStart := 97686 },
  { event := event97702
    frameStart := 97686 },
  { event := event97703
    frameStart := 97686 },
  { event := event97704
    frameStart := 97686 },
  { event := event97705
    frameStart := 97686 },
  { event := event97706
    frameStart := 97686 },
  { event := event97707
    frameStart := 97686 },
  { event := event97708
    frameStart := 97686 },
  { event := event97709
    frameStart := 97686 },
  { event := event97710
    frameStart := 97686 },
  { event := event97711
    frameStart := 97686 }
]

def eventLeaf6107 : Array AnnotatedEvent := #[
  { event := event97712
    frameStart := 97686 },
  { event := event97713
    frameStart := 97686 },
  { event := event97714
    frameStart := 97686 },
  { event := event97715
    frameStart := 97686 },
  { event := event97716
    frameStart := 97686 },
  { event := event97717
    frameStart := 97686 },
  { event := event97718
    frameStart := 97686 },
  { event := event97719
    frameStart := 97686 },
  { event := event97720
    frameStart := 97686 },
  { event := event97721
    frameStart := 97686 },
  { event := event97722
    frameStart := 97686 },
  { event := event97723
    frameStart := 97686 },
  { event := event97724
    frameStart := 97686 },
  { event := event97725
    frameStart := 97686 },
  { event := event97726
    frameStart := 97686 },
  { event := event97727
    frameStart := 97686 }
]

def eventLeaf6108 : Array AnnotatedEvent := #[
  { event := event97728
    frameStart := 97728 },
  { event := event97729
    frameStart := 97728 },
  { event := event97730
    frameStart := 97728 },
  { event := event97731
    frameStart := 97728 },
  { event := event97732
    frameStart := 97728 },
  { event := event97733
    frameStart := 97728 },
  { event := event97734
    frameStart := 97728 },
  { event := event97735
    frameStart := 97728 },
  { event := event97736
    frameStart := 97728 },
  { event := event97737
    frameStart := 97728 },
  { event := event97738
    frameStart := 97728 },
  { event := event97739
    frameStart := 97728 },
  { event := event97740
    frameStart := 97728 },
  { event := event97741
    frameStart := 97728 },
  { event := event97742
    frameStart := 97728 },
  { event := event97743
    frameStart := 97728 }
]

def eventLeaf6109 : Array AnnotatedEvent := #[
  { event := event97744
    frameStart := 97728 },
  { event := event97745
    frameStart := 97728 },
  { event := event97746
    frameStart := 97728 },
  { event := event97747
    frameStart := 97728 },
  { event := event97748
    frameStart := 97728 },
  { event := event97749
    frameStart := 97728 },
  { event := event97750
    frameStart := 97728 },
  { event := event97751
    frameStart := 97728 },
  { event := event97752
    frameStart := 97728 },
  { event := event97753
    frameStart := 97728 },
  { event := event97754
    frameStart := 97728 },
  { event := event97755
    frameStart := 97728 },
  { event := event97756
    frameStart := 97728 },
  { event := event97757
    frameStart := 97728 },
  { event := event97758
    frameStart := 97728 },
  { event := event97759
    frameStart := 97728 }
]

def eventLeaf6110 : Array AnnotatedEvent := #[
  { event := event97760
    frameStart := 97728 },
  { event := event97761
    frameStart := 97728 },
  { event := event97762
    frameStart := 97728 },
  { event := event97763
    frameStart := 97728 },
  { event := event97764
    frameStart := 97728 },
  { event := event97765
    frameStart := 97728 },
  { event := event97766
    frameStart := 97728 },
  { event := event97767
    frameStart := 97728 },
  { event := event97768
    frameStart := 97728 },
  { event := event97769
    frameStart := 97728 },
  { event := event97770
    frameStart := 97728 },
  { event := event97771
    frameStart := 97728 },
  { event := event97772
    frameStart := 97728 },
  { event := event97773
    frameStart := 97728 },
  { event := event97774
    frameStart := 97728 },
  { event := event97775
    frameStart := 97728 }
]

def eventLeaf6111 : Array AnnotatedEvent := #[
  { event := event97776
    frameStart := 97728 },
  { event := event97777
    frameStart := 97728 },
  { event := event97778
    frameStart := 97728 },
  { event := event97779
    frameStart := 97728 },
  { event := event97780
    frameStart := 97728 },
  { event := event97781
    frameStart := 97728 },
  { event := event97782
    frameStart := 97728 },
  { event := event97783
    frameStart := 97728 },
  { event := event97784
    frameStart := 97728 },
  { event := event97785
    frameStart := 97728 },
  { event := event97786
    frameStart := 97728 },
  { event := event97787
    frameStart := 97728 },
  { event := event97788
    frameStart := 97728 },
  { event := event97789
    frameStart := 97728 },
  { event := event97790
    frameStart := 97728 },
  { event := event97791
    frameStart := 97728 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events381
