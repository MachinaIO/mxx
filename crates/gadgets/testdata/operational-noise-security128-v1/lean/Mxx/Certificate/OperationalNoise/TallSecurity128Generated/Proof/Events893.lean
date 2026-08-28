import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events893

open Mxx.Certificate.OperationalNoise
open CertificateABI

def exact228608RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨52003⟩⟩]⟩, (1)⟩]

theorem exact228608RawTermsValid :
    exact228608RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event228608 : Event := .resultExact (⟨.program ⟨257⟩, ⟨52003⟩⟩) exact228608RawTerms .large 228607 .exactZero (none)

def event228609 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52508⟩⟩) 0 ⟨52003⟩ 228608

def event228610 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨52508⟩⟩) (.authority (.operator))

def exact228611RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨52508⟩⟩]⟩, (1)⟩]

theorem exact228611RawTermsValid :
    exact228611RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event228611 : Event := .resultExact (⟨.program ⟨257⟩, ⟨52508⟩⟩) exact228611RawTerms (.finite 8192) 228610 .exactZero (none)

def event228612 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨136⟩⟩) (.authority (.operator))

def event228613 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨136⟩⟩) .exactZero

def event228614 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52282⟩⟩) 0 ⟨50520⟩ 228600

def event228615 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52282⟩⟩) 1 ⟨136⟩ 228613

def event228616 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨52282⟩⟩) (.sum [.predecessor 0 228614 .coefficient, .predecessor 1 228615 .coefficient])

def event228617 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨52282⟩⟩) (.finite 100)

def event228618 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52283⟩⟩) 0 ⟨52282⟩ 228617

def event228619 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨52283⟩⟩) (.identity (.predecessor 0 228618 .coefficient))

def exact228620RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨24518⟩⟩, ⟨.program ⟨257⟩, ⟨50518⟩⟩], []⟩, (1)⟩]

theorem exact228620RawTermsValid :
    exact228620RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event228620 : Event := .resultExact (⟨.program ⟨257⟩, ⟨52283⟩⟩) exact228620RawTerms (.finite 100) 228619 .exactZero (none)

def event228621 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6908⟩⟩) (.authority (.factStore))

def exact228622RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact228622RawTermsValid :
    exact228622RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event228622 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6908⟩⟩) exact228622RawTerms .large 228621 .exactZero (none)

def event228623 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52284⟩⟩) 0 ⟨6908⟩ 228622

def event228624 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52284⟩⟩) 1 ⟨52283⟩ 228620

def event228625 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨52284⟩⟩) (.product (.predecessor 0 228623 .coefficient) (.predecessor 1 228624 .coefficient) (⟨false, false, none, none, none⟩))

def event228626 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨52284⟩⟩, .operator (⟨228622, 0⟩, ⟨228620, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨24518⟩⟩, ⟨.program ⟨257⟩, ⟨50518⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact228627RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨24518⟩⟩, ⟨.program ⟨257⟩, ⟨50518⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact228627RawTermsValid :
    exact228627RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event228627 : Event := .resultExact (⟨.program ⟨257⟩, ⟨52284⟩⟩) exact228627RawTerms .large 228625 .exactZero (none)

def event228628 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨2370⟩⟩) (.authority (.operator))

def event228629 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨2370⟩⟩) (.finite 1)

def event228630 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7178⟩⟩) 0 ⟨7177⟩ 228604

def event228631 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7178⟩⟩) (.authority (.operator))

def exact228632RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7178⟩⟩]⟩, (1)⟩]

theorem exact228632RawTermsValid :
    exact228632RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event228632 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7178⟩⟩) exact228632RawTerms .large 228631 .exactZero (none)

def event228633 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7308⟩⟩) 0 ⟨7178⟩ 228632

def event228634 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7308⟩⟩) (.identity (.predecessor 0 228633 .coefficient))

def exact228635RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7308⟩⟩]⟩, (1)⟩]

theorem exact228635RawTermsValid :
    exact228635RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event228635 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7308⟩⟩) exact228635RawTerms .large 228634 .exactZero (none)

def event228636 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9580⟩⟩) 0 ⟨7308⟩ 228635

def event228637 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9580⟩⟩) (.authority (.operator))

def exact228638RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨9580⟩⟩]⟩, (1)⟩]

theorem exact228638RawTermsValid :
    exact228638RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event228638 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9580⟩⟩) exact228638RawTerms (.finite 8192) 228637 .exactZero (none)

def event228639 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9581⟩⟩) 0 ⟨9580⟩ 228638

def event228640 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9581⟩⟩) 1 ⟨2370⟩ 228629

def event228641 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9581⟩⟩) (.scale (.predecessor 0 228639 .coefficient) (.value (.predecessor 1 228640 .coefficient)))

def exact228642RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨9580⟩⟩]⟩, (1)⟩]

theorem exact228642RawTermsValid :
    exact228642RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event228642 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9581⟩⟩) exact228642RawTerms (.finite 8192) 228641 .exactZero (none)

def event228643 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7288⟩⟩) 0 ⟨7178⟩ 228632

def event228644 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7288⟩⟩) (.identity (.predecessor 0 228643 .coefficient))

def exact228645RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7288⟩⟩]⟩, (1)⟩]

theorem exact228645RawTermsValid :
    exact228645RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event228645 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7288⟩⟩) exact228645RawTerms .large 228644 .exactZero (none)

def event228646 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9582⟩⟩) 0 ⟨7288⟩ 228645

def event228647 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9582⟩⟩) 1 ⟨9581⟩ 228642

def event228648 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9582⟩⟩) (.product (.predecessor 0 228646 .coefficient) (.predecessor 1 228647 .coefficient) (⟨false, false, none, none, none⟩))

def event228649 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨9582⟩⟩, .operator (⟨228645, 0⟩, ⟨228642, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩]⟩, (1)⟩)

def exact228650RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩]⟩, (1)⟩]

theorem exact228650RawTermsValid :
    exact228650RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event228650 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9582⟩⟩) exact228650RawTerms .large 228648 .exactZero (none)

def event228651 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52285⟩⟩) 0 ⟨9582⟩ 228650

def event228652 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52285⟩⟩) 1 ⟨52284⟩ 228627

def event228653 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨52285⟩⟩) (.sum [.predecessor 0 228651 .coefficient, .predecessor 1 228652 .coefficient])

def exact228654RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨24518⟩⟩, ⟨.program ⟨257⟩, ⟨50518⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact228654RawTermsValid :
    exact228654RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event228654 : Event := .resultExact (⟨.program ⟨257⟩, ⟨52285⟩⟩) exact228654RawTerms .large 228653 .exactZero (none)

def event228655 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52511⟩⟩) 0 ⟨52285⟩ 228654

def event228656 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52511⟩⟩) 1 ⟨52508⟩ 228611

def event228657 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨52511⟩⟩) (.product (.predecessor 0 228655 .coefficient) (.predecessor 1 228656 .coefficient) (⟨false, false, none, none, none⟩))

def event228658 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨52511⟩⟩, .operator (⟨228654, 0⟩, ⟨228611, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52508⟩⟩]⟩, (1)⟩)

def event228659 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨52511⟩⟩, .operator (⟨228654, 1⟩, ⟨228611, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨24518⟩⟩, ⟨.program ⟨257⟩, ⟨50518⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨52508⟩⟩]⟩, (-1)⟩)

def event228660 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨52511⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨24518⟩⟩, ⟨.program ⟨257⟩, ⟨50518⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨52508⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨52508⟩⟩) ⟨52003⟩ 228608)

def event228661 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨52511⟩⟩, .relation 228660 0, ⟨[⟨.program ⟨257⟩, ⟨24518⟩⟩, ⟨.program ⟨257⟩, ⟨50518⟩⟩], [⟨.program ⟨257⟩, ⟨52003⟩⟩]⟩, (-1)⟩)

def exact228662RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52508⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨24518⟩⟩, ⟨.program ⟨257⟩, ⟨50518⟩⟩], [⟨.program ⟨257⟩, ⟨52003⟩⟩]⟩, (-1)⟩]

theorem exact228662RawTermsValid :
    exact228662RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event228662 : Event := .resultExact (⟨.program ⟨257⟩, ⟨52511⟩⟩) exact228662RawTerms .large 228657 .exactZero (none)

def event228663 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50880⟩⟩) 0 ⟨50520⟩ 228600

def event228664 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50880⟩⟩) (.authority (.programFamilyFact))

def exact228665RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨50880⟩⟩], []⟩, (1)⟩]

theorem exact228665RawTermsValid :
    exact228665RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event228665 : Event := .resultExact (⟨.program ⟨257⟩, ⟨50880⟩⟩) exact228665RawTerms (.finite 10) 228664 .exactZero (none)

def event228666 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50882⟩⟩) 0 ⟨6908⟩ 228622

def event228667 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50882⟩⟩) 1 ⟨50880⟩ 228665

def event228668 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50882⟩⟩) (.product (.predecessor 0 228666 .coefficient) (.predecessor 1 228667 .coefficient) (⟨false, true, none, none, some 1⟩))

def event228669 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨50882⟩⟩, .operator (⟨228622, 0⟩, ⟨228665, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨50880⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact228670RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨50880⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact228670RawTermsValid :
    exact228670RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event228670 : Event := .resultExact (⟨.program ⟨257⟩, ⟨50882⟩⟩) exact228670RawTerms .large 228668 .exactZero (none)

def event228671 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7183⟩⟩) 0 ⟨7177⟩ 228604

def event228672 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7183⟩⟩) (.authority (.operator))

def exact228673RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7183⟩⟩]⟩, (1)⟩]

theorem exact228673RawTermsValid :
    exact228673RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event228673 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7183⟩⟩) exact228673RawTerms .large 228672 .exactZero (none)

def event228674 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50883⟩⟩) 0 ⟨7183⟩ 228673

def event228675 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50883⟩⟩) 1 ⟨50882⟩ 228670

def event228676 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50883⟩⟩) (.sum [.predecessor 0 228674 .coefficient, .predecessor 1 228675 .coefficient])

def exact228677RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7183⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨50880⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact228677RawTermsValid :
    exact228677RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event228677 : Event := .resultExact (⟨.program ⟨257⟩, ⟨50883⟩⟩) exact228677RawTerms .large 228676 .exactZero (none)

def event228678 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52512⟩⟩) 0 ⟨50883⟩ 228677

def event228679 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52512⟩⟩) 1 ⟨52511⟩ 228662

def event228680 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨52512⟩⟩) (.sum [.predecessor 0 228678 .coefficient, .predecessor 1 228679 .coefficient])

def exact228681RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7183⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52508⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨24518⟩⟩, ⟨.program ⟨257⟩, ⟨50518⟩⟩], [⟨.program ⟨257⟩, ⟨52003⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨50880⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact228681RawTermsValid :
    exact228681RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event228681 : Event := .resultExact (⟨.program ⟨257⟩, ⟨52512⟩⟩) exact228681RawTerms .large 228680 .exactZero (none)

def event228682 : Event := .preFoldPolynomial 228681 [⟨⟨[], [⟨.program ⟨257⟩, ⟨7183⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52508⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨24518⟩⟩, ⟨.program ⟨257⟩, ⟨50518⟩⟩], [⟨.program ⟨257⟩, ⟨52003⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨50880⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩] .exactZero none

def exact228683RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7183⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52508⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨24518⟩⟩, ⟨.program ⟨257⟩, ⟨50518⟩⟩], [⟨.program ⟨257⟩, ⟨52003⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨50880⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

def event228683 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨52512⟩⟩) 228682 exact228683RawTerms .large 228680 .exactZero (none)

def event228684 : Event := .specializationComputed (⟨.program ⟨257⟩, ⟨50520⟩⟩) ⟨⟨62⟩, ⟨40⟩, ⟨135⟩⟩ ⟨228518, 228684⟩

def event228685 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨51442⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51439⟩⟩]⟩) (1) 0 2 (.universal 228684 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51439⟩⟩]⟩) (none) 228683)

def event228686 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨51442⟩⟩, .relation 228685 0, ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7183⟩⟩]⟩, (1)⟩)

def event228687 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨51442⟩⟩, .relation 228685 1, ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52508⟩⟩]⟩, (-1)⟩)

def event228688 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨51442⟩⟩, .relation 228685 2, ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨24518⟩⟩, ⟨.program ⟨257⟩, ⟨50518⟩⟩], [⟨.program ⟨257⟩, ⟨52003⟩⟩]⟩, (1)⟩)

def event228689 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨51442⟩⟩, .relation 228685 3, ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨50880⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact228690RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7183⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52508⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨24518⟩⟩, ⟨.program ⟨257⟩, ⟨50518⟩⟩], [⟨.program ⟨257⟩, ⟨52003⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨50880⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact228690RawTermsValid :
    exact228690RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event228690 : Event := .resultExact (⟨.program ⟨257⟩, ⟨51442⟩⟩) exact228690RawTerms .large 228514 (.finite 202072841853861888) (some (228516))

def event228691 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52510⟩⟩) 0 ⟨51442⟩ 228690

def event228692 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52510⟩⟩) 1 ⟨52509⟩ 228504

def event228693 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨52510⟩⟩) (.sum [.predecessor 0 228691 .coefficient, .predecessor 1 228692 .coefficient])

def event228694 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨52510⟩⟩, .operator (⟨228690, 2⟩, ⟨228504, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨24518⟩⟩, ⟨.program ⟨257⟩, ⟨50518⟩⟩], [⟨.program ⟨257⟩, ⟨52003⟩⟩]⟩, (-1)⟩)

def event228695 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨52510⟩⟩, .operator (⟨228690, 1⟩, ⟨228504, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52508⟩⟩]⟩, (1)⟩)

def event228696 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨52510⟩⟩) (.sum [.result 228690 .summary, .result 228504 .summary])

def exact228697RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7183⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨50880⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact228697RawTermsValid :
    exact228697RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event228697 : Event := .resultExact (⟨.program ⟨257⟩, ⟨52510⟩⟩) exact228697RawTerms .large 228693 (.finite 2997889464187086962688) (some (228696))

def event228698 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52923⟩⟩) 0 ⟨52510⟩ 228697

def event228699 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52923⟩⟩) 1 ⟨52921⟩ 228420

def event228700 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨52923⟩⟩) (.product (.predecessor 0 228698 .coefficient) (.predecessor 1 228699 .coefficient) (⟨false, false, none, none, none⟩))

def event228701 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨52923⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨52921⟩⟩]⟩) [⟨.result 228420 .coefficient, false, none⟩])

def event228702 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨52923⟩⟩) (.product (.result 228697 .summary) (.transfer 228701) (⟨false, false, none, none, none⟩))

def event228703 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨52923⟩⟩, .operator (⟨228697, 0⟩, ⟨228420, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7183⟩⟩, ⟨.program ⟨257⟩, ⟨52921⟩⟩]⟩, (1)⟩)

def event228704 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨52923⟩⟩, .operator (⟨228697, 1⟩, ⟨228420, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨50880⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨52921⟩⟩]⟩, (-1)⟩)

def event228705 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨52923⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨50880⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨52921⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨52921⟩⟩) ⟨52152⟩ 228417)

def event228706 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨52923⟩⟩, .relation 228705 0, ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨50880⟩⟩], [⟨.program ⟨257⟩, ⟨52152⟩⟩]⟩, (-1)⟩)

def exact228707RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7183⟩⟩, ⟨.program ⟨257⟩, ⟨52921⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨50880⟩⟩], [⟨.program ⟨257⟩, ⟨52152⟩⟩]⟩, (-1)⟩]

theorem exact228707RawTermsValid :
    exact228707RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event228707 : Event := .resultExact (⟨.program ⟨257⟩, ⟨52923⟩⟩) exact228707RawTerms .large 228700 (.finite 32189593014266254325632330629120) (some (228702))

def event228708 : Event := .predecessor (⟨.program ⟨257⟩, ⟨51736⟩⟩) 0 ⟨50881⟩ 10882

def event228709 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨51736⟩⟩) (.authority (.relationPreimageSource ⟨65⟩))

def exact228710RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨51736⟩⟩]⟩, (1)⟩]

theorem exact228710RawTermsValid :
    exact228710RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event228710 : Event := .resultExact (⟨.program ⟨257⟩, ⟨51736⟩⟩) exact228710RawTerms (.finite 5647228698) 228709 .exactZero (none)

def event228711 : Event := .predecessor (⟨.program ⟨257⟩, ⟨51738⟩⟩) 0 ⟨51736⟩ 228710

def event228712 : Event := .predecessor (⟨.program ⟨257⟩, ⟨51738⟩⟩) 1 ⟨2370⟩ 4

def event228713 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨51738⟩⟩) (.scale (.predecessor 0 228711 .coefficient) (.value (.predecessor 1 228712 .coefficient)))

def exact228714RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨51736⟩⟩]⟩, (1)⟩]

theorem exact228714RawTermsValid :
    exact228714RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event228714 : Event := .resultExact (⟨.program ⟨257⟩, ⟨51738⟩⟩) exact228714RawTerms (.finite 5647228698) 228713 .exactZero (none)

def event228715 : Event := .predecessor (⟨.program ⟨257⟩, ⟨51739⟩⟩) 0 ⟨5581⟩ 222245

def event228716 : Event := .predecessor (⟨.program ⟨257⟩, ⟨51739⟩⟩) 1 ⟨51738⟩ 228714

def event228717 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨51739⟩⟩) (.product (.predecessor 0 228715 .coefficient) (.predecessor 1 228716 .coefficient) (⟨false, false, none, none, none⟩))

def event228718 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨51739⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨51736⟩⟩]⟩) [⟨.result 228710 .coefficient, false, none⟩])

def event228719 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨51739⟩⟩) (.product (.result 222245 .summary) (.transfer 228718) (⟨false, false, none, none, none⟩))

def event228720 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨51739⟩⟩, .operator (⟨222245, 0⟩, ⟨228714, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51736⟩⟩]⟩, (1)⟩)

def event228721 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨51737⟩⟩)

def event228722 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event228723 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event228724 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨4990⟩⟩) (.authority (.operator))

def event228725 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨4990⟩⟩) (.finite 5)

def event228726 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event228727 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event228728 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event228729 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event228730 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 228729

def event228731 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 228727

def event228732 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 228730 .coefficient) (.value (.predecessor 1 228731 .coefficient)))

def event228733 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event228734 : Event := .predecessor (⟨.program ⟨257⟩, ⟨4992⟩⟩) 0 ⟨392⟩ 228733

def event228735 : Event := .predecessor (⟨.program ⟨257⟩, ⟨4992⟩⟩) 1 ⟨4990⟩ 228725

def event228736 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨4992⟩⟩) (.sum [.predecessor 0 228734 .coefficient, .predecessor 1 228735 .coefficient])

def event228737 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨4992⟩⟩) (.finite 655345)

def event228738 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5577⟩⟩) 0 ⟨4992⟩ 228737

def event228739 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5577⟩⟩) 1 ⟨5426⟩ 228723

def event228740 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5577⟩⟩) (.identity (.predecessor 1 228739 .coefficient))

def event228741 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5577⟩⟩) (.finite 655360)

def event228742 : Event := .predecessor (⟨.program ⟨257⟩, ⟨24518⟩⟩) 0 ⟨5577⟩ 228741

def event228743 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨24518⟩⟩) (.authority (.programFamilyFact))

def exact228744RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨24518⟩⟩], []⟩, (1)⟩]

theorem exact228744RawTermsValid :
    exact228744RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event228744 : Event := .resultExact (⟨.program ⟨257⟩, ⟨24518⟩⟩) exact228744RawTerms (.finite 10) 228743 .exactZero (none)

def event228745 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50518⟩⟩) 0 ⟨5577⟩ 228741

def event228746 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50518⟩⟩) (.authority (.programFamilyFact))

def exact228747RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨50518⟩⟩], []⟩, (1)⟩]

theorem exact228747RawTermsValid :
    exact228747RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event228747 : Event := .resultExact (⟨.program ⟨257⟩, ⟨50518⟩⟩) exact228747RawTerms (.finite 10) 228746 .exactZero (none)

def event228748 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50519⟩⟩) 0 ⟨50518⟩ 228747

def event228749 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50519⟩⟩) 1 ⟨24518⟩ 228744

def event228750 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50519⟩⟩) (.product (.predecessor 0 228748 .coefficient) (.predecessor 1 228749 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event228751 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50519⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨24518⟩⟩, ⟨.program ⟨257⟩, ⟨50518⟩⟩], []⟩) [⟨.result 228747 .coefficient, true, some 1⟩, ⟨.result 228744 .coefficient, true, some 1⟩])

def event228752 : Event := .survivorFold (1) 228751

def exact228753RawTerms : List Term := []

theorem exact228753RawTermsValid :
    exact228753RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event228753 : Event := .resultExact (⟨.program ⟨257⟩, ⟨50519⟩⟩) exact228753RawTerms (.finite 100) 228750 (.finite 100) (some (228751))

def event228754 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50520⟩⟩) 0 ⟨50519⟩ 228753

def event228755 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50520⟩⟩) (.identity (.predecessor 0 228754 .coefficient))

def event228756 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨50520⟩⟩) (.finite 100)

def event228757 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50880⟩⟩) 0 ⟨50520⟩ 228756

def event228758 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50880⟩⟩) (.authority (.programFamilyFact))

def exact228759RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨50880⟩⟩], []⟩, (1)⟩]

theorem exact228759RawTermsValid :
    exact228759RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event228759 : Event := .resultExact (⟨.program ⟨257⟩, ⟨50880⟩⟩) exact228759RawTerms (.finite 10) 228758 .exactZero (none)

def event228760 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50881⟩⟩) 0 ⟨50880⟩ 228759

def event228761 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50881⟩⟩) (.identity (.predecessor 0 228760 .coefficient))

def event228762 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨50881⟩⟩) (.finite 10)

def event228763 : Event := .predecessor (⟨.program ⟨257⟩, ⟨51736⟩⟩) 0 ⟨50881⟩ 228762

def event228764 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨51736⟩⟩) (.authority (.relationPreimageSource ⟨65⟩))

def exact228765RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨51736⟩⟩]⟩, (1)⟩]

theorem exact228765RawTermsValid :
    exact228765RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event228765 : Event := .resultExact (⟨.program ⟨257⟩, ⟨51736⟩⟩) exact228765RawTerms (.finite 5647228698) 228764 .exactZero (none)

def event228766 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35⟩⟩) (.authority (.operator))

def exact228767RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩]⟩, (1)⟩]

theorem exact228767RawTermsValid :
    exact228767RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event228767 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35⟩⟩) exact228767RawTerms .large 228766 .exactZero (none)

def event228768 : Event := .predecessor (⟨.program ⟨257⟩, ⟨51737⟩⟩) 0 ⟨35⟩ 228767

def event228769 : Event := .predecessor (⟨.program ⟨257⟩, ⟨51737⟩⟩) 1 ⟨51736⟩ 228765

def event228770 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨51737⟩⟩) (.product (.predecessor 0 228768 .coefficient) (.predecessor 1 228769 .coefficient) (⟨false, false, none, none, none⟩))

def event228771 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨51737⟩⟩, .operator (⟨228767, 0⟩, ⟨228765, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51736⟩⟩]⟩, (1)⟩)

def exact228772RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51736⟩⟩]⟩, (1)⟩]

theorem exact228772RawTermsValid :
    exact228772RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event228772 : Event := .resultExact (⟨.program ⟨257⟩, ⟨51737⟩⟩) exact228772RawTerms .large 228770 .exactZero (none)

def event228773 : Event := .preFoldPolynomial 228772 [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51736⟩⟩]⟩, (1)⟩] .exactZero none

def exact228774RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51736⟩⟩]⟩, (1)⟩]

def event228774 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨51737⟩⟩) 228773 exact228774RawTerms .large 228770 .exactZero (none)

def event228775 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨52926⟩⟩)

def event228776 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event228777 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event228778 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨4990⟩⟩) (.authority (.operator))

def event228779 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨4990⟩⟩) (.finite 5)

def event228780 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event228781 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event228782 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event228783 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event228784 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 228783

def event228785 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 228781

def event228786 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 228784 .coefficient) (.value (.predecessor 1 228785 .coefficient)))

def event228787 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event228788 : Event := .predecessor (⟨.program ⟨257⟩, ⟨4992⟩⟩) 0 ⟨392⟩ 228787

def event228789 : Event := .predecessor (⟨.program ⟨257⟩, ⟨4992⟩⟩) 1 ⟨4990⟩ 228779

def event228790 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨4992⟩⟩) (.sum [.predecessor 0 228788 .coefficient, .predecessor 1 228789 .coefficient])

def event228791 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨4992⟩⟩) (.finite 655345)

def event228792 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5577⟩⟩) 0 ⟨4992⟩ 228791

def event228793 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5577⟩⟩) 1 ⟨5426⟩ 228777

def event228794 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5577⟩⟩) (.identity (.predecessor 1 228793 .coefficient))

def event228795 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5577⟩⟩) (.finite 655360)

def event228796 : Event := .predecessor (⟨.program ⟨257⟩, ⟨24518⟩⟩) 0 ⟨5577⟩ 228795

def event228797 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨24518⟩⟩) (.authority (.programFamilyFact))

def exact228798RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨24518⟩⟩], []⟩, (1)⟩]

theorem exact228798RawTermsValid :
    exact228798RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event228798 : Event := .resultExact (⟨.program ⟨257⟩, ⟨24518⟩⟩) exact228798RawTerms (.finite 10) 228797 .exactZero (none)

def event228799 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50518⟩⟩) 0 ⟨5577⟩ 228795

def event228800 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50518⟩⟩) (.authority (.programFamilyFact))

def exact228801RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨50518⟩⟩], []⟩, (1)⟩]

theorem exact228801RawTermsValid :
    exact228801RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event228801 : Event := .resultExact (⟨.program ⟨257⟩, ⟨50518⟩⟩) exact228801RawTerms (.finite 10) 228800 .exactZero (none)

def event228802 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50519⟩⟩) 0 ⟨50518⟩ 228801

def event228803 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50519⟩⟩) 1 ⟨24518⟩ 228798

def event228804 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50519⟩⟩) (.product (.predecessor 0 228802 .coefficient) (.predecessor 1 228803 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event228805 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨50519⟩⟩, .operator (⟨228801, 0⟩, ⟨228798, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨24518⟩⟩, ⟨.program ⟨257⟩, ⟨50518⟩⟩], []⟩, (1)⟩)

def exact228806RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨24518⟩⟩, ⟨.program ⟨257⟩, ⟨50518⟩⟩], []⟩, (1)⟩]

theorem exact228806RawTermsValid :
    exact228806RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event228806 : Event := .resultExact (⟨.program ⟨257⟩, ⟨50519⟩⟩) exact228806RawTerms (.finite 100) 228804 .exactZero (none)

def event228807 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50520⟩⟩) 0 ⟨50519⟩ 228806

def event228808 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50520⟩⟩) (.identity (.predecessor 0 228807 .coefficient))

def event228809 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨50520⟩⟩) (.finite 100)

def event228810 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50880⟩⟩) 0 ⟨50520⟩ 228809

def event228811 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50880⟩⟩) (.authority (.programFamilyFact))

def exact228812RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨50880⟩⟩], []⟩, (1)⟩]

theorem exact228812RawTermsValid :
    exact228812RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event228812 : Event := .resultExact (⟨.program ⟨257⟩, ⟨50880⟩⟩) exact228812RawTerms (.finite 10) 228811 .exactZero (none)

def event228813 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50881⟩⟩) 0 ⟨50880⟩ 228812

def event228814 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50881⟩⟩) (.identity (.predecessor 0 228813 .coefficient))

def event228815 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨50881⟩⟩) (.finite 10)

def event228816 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52150⟩⟩) 0 ⟨50881⟩ 228815

def event228817 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨52150⟩⟩) (.authority (.programFamilyFact))

def event228818 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨52150⟩⟩) (.finite 3720)

def event228819 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨7177⟩⟩) .missing

def event228820 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52152⟩⟩) 0 ⟨7177⟩ 228819

def event228821 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52152⟩⟩) 1 ⟨52150⟩ 228818

def event228822 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨52152⟩⟩) (.authority (.operator))

def exact228823RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨52152⟩⟩]⟩, (1)⟩]

theorem exact228823RawTermsValid :
    exact228823RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event228823 : Event := .resultExact (⟨.program ⟨257⟩, ⟨52152⟩⟩) exact228823RawTerms .large 228822 .exactZero (none)

def event228824 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52921⟩⟩) 0 ⟨52152⟩ 228823

def event228825 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨52921⟩⟩) (.authority (.operator))

def exact228826RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨52921⟩⟩]⟩, (1)⟩]

theorem exact228826RawTermsValid :
    exact228826RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event228826 : Event := .resultExact (⟨.program ⟨257⟩, ⟨52921⟩⟩) exact228826RawTerms (.finite 8192) 228825 .exactZero (none)

def event228827 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨136⟩⟩) (.authority (.operator))

def event228828 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨136⟩⟩) .exactZero

def event228829 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52362⟩⟩) 0 ⟨50881⟩ 228815

def event228830 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52362⟩⟩) 1 ⟨136⟩ 228828

def event228831 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨52362⟩⟩) (.sum [.predecessor 0 228829 .coefficient, .predecessor 1 228830 .coefficient])

def event228832 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨52362⟩⟩) (.finite 10)

def event228833 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52363⟩⟩) 0 ⟨52362⟩ 228832

def event228834 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨52363⟩⟩) (.identity (.predecessor 0 228833 .coefficient))

def exact228835RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨50880⟩⟩], []⟩, (1)⟩]

theorem exact228835RawTermsValid :
    exact228835RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event228835 : Event := .resultExact (⟨.program ⟨257⟩, ⟨52363⟩⟩) exact228835RawTerms (.finite 10) 228834 .exactZero (none)

def event228836 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6908⟩⟩) (.authority (.factStore))

def exact228837RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact228837RawTermsValid :
    exact228837RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event228837 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6908⟩⟩) exact228837RawTerms .large 228836 .exactZero (none)

def event228838 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52364⟩⟩) 0 ⟨6908⟩ 228837

def event228839 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52364⟩⟩) 1 ⟨52363⟩ 228835

def event228840 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨52364⟩⟩) (.product (.predecessor 0 228838 .coefficient) (.predecessor 1 228839 .coefficient) (⟨false, false, none, none, none⟩))

def event228841 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨52364⟩⟩, .operator (⟨228837, 0⟩, ⟨228835, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨50880⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact228842RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨50880⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact228842RawTermsValid :
    exact228842RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event228842 : Event := .resultExact (⟨.program ⟨257⟩, ⟨52364⟩⟩) exact228842RawTerms .large 228840 .exactZero (none)

def event228843 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7183⟩⟩) 0 ⟨7177⟩ 228819

def event228844 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7183⟩⟩) (.authority (.operator))

def exact228845RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7183⟩⟩]⟩, (1)⟩]

theorem exact228845RawTermsValid :
    exact228845RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event228845 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7183⟩⟩) exact228845RawTerms .large 228844 .exactZero (none)

def event228846 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52365⟩⟩) 0 ⟨7183⟩ 228845

def event228847 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52365⟩⟩) 1 ⟨52364⟩ 228842

def event228848 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨52365⟩⟩) (.sum [.predecessor 0 228846 .coefficient, .predecessor 1 228847 .coefficient])

def exact228849RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7183⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨50880⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact228849RawTermsValid :
    exact228849RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event228849 : Event := .resultExact (⟨.program ⟨257⟩, ⟨52365⟩⟩) exact228849RawTerms .large 228848 .exactZero (none)

def event228850 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52922⟩⟩) 0 ⟨52365⟩ 228849

def event228851 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52922⟩⟩) 1 ⟨52921⟩ 228826

def event228852 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨52922⟩⟩) (.product (.predecessor 0 228850 .coefficient) (.predecessor 1 228851 .coefficient) (⟨false, false, none, none, none⟩))

def event228853 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨52922⟩⟩, .operator (⟨228849, 0⟩, ⟨228826, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7183⟩⟩, ⟨.program ⟨257⟩, ⟨52921⟩⟩]⟩, (1)⟩)

def event228854 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨52922⟩⟩, .operator (⟨228849, 1⟩, ⟨228826, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨50880⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨52921⟩⟩]⟩, (-1)⟩)

def event228855 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨52922⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨50880⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨52921⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨52921⟩⟩) ⟨52152⟩ 228823)

def event228856 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨52922⟩⟩, .relation 228855 0, ⟨[⟨.program ⟨257⟩, ⟨50880⟩⟩], [⟨.program ⟨257⟩, ⟨52152⟩⟩]⟩, (-1)⟩)

def exact228857RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7183⟩⟩, ⟨.program ⟨257⟩, ⟨52921⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨50880⟩⟩], [⟨.program ⟨257⟩, ⟨52152⟩⟩]⟩, (-1)⟩]

theorem exact228857RawTermsValid :
    exact228857RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event228857 : Event := .resultExact (⟨.program ⟨257⟩, ⟨52922⟩⟩) exact228857RawTerms .large 228852 .exactZero (none)

def event228858 : Event := .predecessor (⟨.program ⟨257⟩, ⟨51142⟩⟩) 0 ⟨50881⟩ 228815

def event228859 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨51142⟩⟩) (.authority (.programFamilyFact))

def exact228860RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨51142⟩⟩], []⟩, (1)⟩]

theorem exact228860RawTermsValid :
    exact228860RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event228860 : Event := .resultExact (⟨.program ⟨257⟩, ⟨51142⟩⟩) exact228860RawTerms (.finite 58) 228859 .exactZero (none)

def event228861 : Event := .predecessor (⟨.program ⟨257⟩, ⟨51144⟩⟩) 0 ⟨6908⟩ 228837

def event228862 : Event := .predecessor (⟨.program ⟨257⟩, ⟨51144⟩⟩) 1 ⟨51142⟩ 228860

def event228863 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨51144⟩⟩) (.product (.predecessor 0 228861 .coefficient) (.predecessor 1 228862 .coefficient) (⟨false, true, none, none, some 1⟩))

def eventLeaf14288 : Array AnnotatedEvent := #[
  { event := event228608
    frameStart := 228566 },
  { event := event228609
    frameStart := 228566 },
  { event := event228610
    frameStart := 228566 },
  { event := event228611
    frameStart := 228566 },
  { event := event228612
    frameStart := 228566 },
  { event := event228613
    frameStart := 228566 },
  { event := event228614
    frameStart := 228566 },
  { event := event228615
    frameStart := 228566 },
  { event := event228616
    frameStart := 228566 },
  { event := event228617
    frameStart := 228566 },
  { event := event228618
    frameStart := 228566 },
  { event := event228619
    frameStart := 228566 },
  { event := event228620
    frameStart := 228566 },
  { event := event228621
    frameStart := 228566 },
  { event := event228622
    frameStart := 228566 },
  { event := event228623
    frameStart := 228566 }
]

def eventLeaf14289 : Array AnnotatedEvent := #[
  { event := event228624
    frameStart := 228566 },
  { event := event228625
    frameStart := 228566 },
  { event := event228626
    frameStart := 228566 },
  { event := event228627
    frameStart := 228566 },
  { event := event228628
    frameStart := 228566 },
  { event := event228629
    frameStart := 228566 },
  { event := event228630
    frameStart := 228566 },
  { event := event228631
    frameStart := 228566 },
  { event := event228632
    frameStart := 228566 },
  { event := event228633
    frameStart := 228566 },
  { event := event228634
    frameStart := 228566 },
  { event := event228635
    frameStart := 228566 },
  { event := event228636
    frameStart := 228566 },
  { event := event228637
    frameStart := 228566 },
  { event := event228638
    frameStart := 228566 },
  { event := event228639
    frameStart := 228566 }
]

def eventLeaf14290 : Array AnnotatedEvent := #[
  { event := event228640
    frameStart := 228566 },
  { event := event228641
    frameStart := 228566 },
  { event := event228642
    frameStart := 228566 },
  { event := event228643
    frameStart := 228566 },
  { event := event228644
    frameStart := 228566 },
  { event := event228645
    frameStart := 228566 },
  { event := event228646
    frameStart := 228566 },
  { event := event228647
    frameStart := 228566 },
  { event := event228648
    frameStart := 228566 },
  { event := event228649
    frameStart := 228566 },
  { event := event228650
    frameStart := 228566 },
  { event := event228651
    frameStart := 228566 },
  { event := event228652
    frameStart := 228566 },
  { event := event228653
    frameStart := 228566 },
  { event := event228654
    frameStart := 228566 },
  { event := event228655
    frameStart := 228566 }
]

def eventLeaf14291 : Array AnnotatedEvent := #[
  { event := event228656
    frameStart := 228566 },
  { event := event228657
    frameStart := 228566 },
  { event := event228658
    frameStart := 228566 },
  { event := event228659
    frameStart := 228566 },
  { event := event228660
    frameStart := 228566 },
  { event := event228661
    frameStart := 228566 },
  { event := event228662
    frameStart := 228566 },
  { event := event228663
    frameStart := 228566 },
  { event := event228664
    frameStart := 228566 },
  { event := event228665
    frameStart := 228566 },
  { event := event228666
    frameStart := 228566 },
  { event := event228667
    frameStart := 228566 },
  { event := event228668
    frameStart := 228566 },
  { event := event228669
    frameStart := 228566 },
  { event := event228670
    frameStart := 228566 },
  { event := event228671
    frameStart := 228566 }
]

def eventLeaf14292 : Array AnnotatedEvent := #[
  { event := event228672
    frameStart := 228566 },
  { event := event228673
    frameStart := 228566 },
  { event := event228674
    frameStart := 228566 },
  { event := event228675
    frameStart := 228566 },
  { event := event228676
    frameStart := 228566 },
  { event := event228677
    frameStart := 228566 },
  { event := event228678
    frameStart := 228566 },
  { event := event228679
    frameStart := 228566 },
  { event := event228680
    frameStart := 228566 },
  { event := event228681
    frameStart := 228566 },
  { event := event228682
    frameStart := 228566 },
  { event := event228683
    frameStart := 228566 },
  { event := event228684
    frameStart := 0 },
  { event := event228685
    frameStart := 0 },
  { event := event228686
    frameStart := 0 },
  { event := event228687
    frameStart := 0 }
]

def eventLeaf14293 : Array AnnotatedEvent := #[
  { event := event228688
    frameStart := 0 },
  { event := event228689
    frameStart := 0 },
  { event := event228690
    frameStart := 0 },
  { event := event228691
    frameStart := 0 },
  { event := event228692
    frameStart := 0 },
  { event := event228693
    frameStart := 0 },
  { event := event228694
    frameStart := 0 },
  { event := event228695
    frameStart := 0 },
  { event := event228696
    frameStart := 0 },
  { event := event228697
    frameStart := 0 },
  { event := event228698
    frameStart := 0 },
  { event := event228699
    frameStart := 0 },
  { event := event228700
    frameStart := 0 },
  { event := event228701
    frameStart := 0 },
  { event := event228702
    frameStart := 0 },
  { event := event228703
    frameStart := 0 }
]

def eventLeaf14294 : Array AnnotatedEvent := #[
  { event := event228704
    frameStart := 0 },
  { event := event228705
    frameStart := 0 },
  { event := event228706
    frameStart := 0 },
  { event := event228707
    frameStart := 0 },
  { event := event228708
    frameStart := 0 },
  { event := event228709
    frameStart := 0 },
  { event := event228710
    frameStart := 0 },
  { event := event228711
    frameStart := 0 },
  { event := event228712
    frameStart := 0 },
  { event := event228713
    frameStart := 0 },
  { event := event228714
    frameStart := 0 },
  { event := event228715
    frameStart := 0 },
  { event := event228716
    frameStart := 0 },
  { event := event228717
    frameStart := 0 },
  { event := event228718
    frameStart := 0 },
  { event := event228719
    frameStart := 0 }
]

def eventLeaf14295 : Array AnnotatedEvent := #[
  { event := event228720
    frameStart := 0 },
  { event := event228721
    frameStart := 228721 },
  { event := event228722
    frameStart := 228721 },
  { event := event228723
    frameStart := 228721 },
  { event := event228724
    frameStart := 228721 },
  { event := event228725
    frameStart := 228721 },
  { event := event228726
    frameStart := 228721 },
  { event := event228727
    frameStart := 228721 },
  { event := event228728
    frameStart := 228721 },
  { event := event228729
    frameStart := 228721 },
  { event := event228730
    frameStart := 228721 },
  { event := event228731
    frameStart := 228721 },
  { event := event228732
    frameStart := 228721 },
  { event := event228733
    frameStart := 228721 },
  { event := event228734
    frameStart := 228721 },
  { event := event228735
    frameStart := 228721 }
]

def eventLeaf14296 : Array AnnotatedEvent := #[
  { event := event228736
    frameStart := 228721 },
  { event := event228737
    frameStart := 228721 },
  { event := event228738
    frameStart := 228721 },
  { event := event228739
    frameStart := 228721 },
  { event := event228740
    frameStart := 228721 },
  { event := event228741
    frameStart := 228721 },
  { event := event228742
    frameStart := 228721 },
  { event := event228743
    frameStart := 228721 },
  { event := event228744
    frameStart := 228721 },
  { event := event228745
    frameStart := 228721 },
  { event := event228746
    frameStart := 228721 },
  { event := event228747
    frameStart := 228721 },
  { event := event228748
    frameStart := 228721 },
  { event := event228749
    frameStart := 228721 },
  { event := event228750
    frameStart := 228721 },
  { event := event228751
    frameStart := 228721 }
]

def eventLeaf14297 : Array AnnotatedEvent := #[
  { event := event228752
    frameStart := 228721 },
  { event := event228753
    frameStart := 228721 },
  { event := event228754
    frameStart := 228721 },
  { event := event228755
    frameStart := 228721 },
  { event := event228756
    frameStart := 228721 },
  { event := event228757
    frameStart := 228721 },
  { event := event228758
    frameStart := 228721 },
  { event := event228759
    frameStart := 228721 },
  { event := event228760
    frameStart := 228721 },
  { event := event228761
    frameStart := 228721 },
  { event := event228762
    frameStart := 228721 },
  { event := event228763
    frameStart := 228721 },
  { event := event228764
    frameStart := 228721 },
  { event := event228765
    frameStart := 228721 },
  { event := event228766
    frameStart := 228721 },
  { event := event228767
    frameStart := 228721 }
]

def eventLeaf14298 : Array AnnotatedEvent := #[
  { event := event228768
    frameStart := 228721 },
  { event := event228769
    frameStart := 228721 },
  { event := event228770
    frameStart := 228721 },
  { event := event228771
    frameStart := 228721 },
  { event := event228772
    frameStart := 228721 },
  { event := event228773
    frameStart := 228721 },
  { event := event228774
    frameStart := 228721 },
  { event := event228775
    frameStart := 228775 },
  { event := event228776
    frameStart := 228775 },
  { event := event228777
    frameStart := 228775 },
  { event := event228778
    frameStart := 228775 },
  { event := event228779
    frameStart := 228775 },
  { event := event228780
    frameStart := 228775 },
  { event := event228781
    frameStart := 228775 },
  { event := event228782
    frameStart := 228775 },
  { event := event228783
    frameStart := 228775 }
]

def eventLeaf14299 : Array AnnotatedEvent := #[
  { event := event228784
    frameStart := 228775 },
  { event := event228785
    frameStart := 228775 },
  { event := event228786
    frameStart := 228775 },
  { event := event228787
    frameStart := 228775 },
  { event := event228788
    frameStart := 228775 },
  { event := event228789
    frameStart := 228775 },
  { event := event228790
    frameStart := 228775 },
  { event := event228791
    frameStart := 228775 },
  { event := event228792
    frameStart := 228775 },
  { event := event228793
    frameStart := 228775 },
  { event := event228794
    frameStart := 228775 },
  { event := event228795
    frameStart := 228775 },
  { event := event228796
    frameStart := 228775 },
  { event := event228797
    frameStart := 228775 },
  { event := event228798
    frameStart := 228775 },
  { event := event228799
    frameStart := 228775 }
]

def eventLeaf14300 : Array AnnotatedEvent := #[
  { event := event228800
    frameStart := 228775 },
  { event := event228801
    frameStart := 228775 },
  { event := event228802
    frameStart := 228775 },
  { event := event228803
    frameStart := 228775 },
  { event := event228804
    frameStart := 228775 },
  { event := event228805
    frameStart := 228775 },
  { event := event228806
    frameStart := 228775 },
  { event := event228807
    frameStart := 228775 },
  { event := event228808
    frameStart := 228775 },
  { event := event228809
    frameStart := 228775 },
  { event := event228810
    frameStart := 228775 },
  { event := event228811
    frameStart := 228775 },
  { event := event228812
    frameStart := 228775 },
  { event := event228813
    frameStart := 228775 },
  { event := event228814
    frameStart := 228775 },
  { event := event228815
    frameStart := 228775 }
]

def eventLeaf14301 : Array AnnotatedEvent := #[
  { event := event228816
    frameStart := 228775 },
  { event := event228817
    frameStart := 228775 },
  { event := event228818
    frameStart := 228775 },
  { event := event228819
    frameStart := 228775 },
  { event := event228820
    frameStart := 228775 },
  { event := event228821
    frameStart := 228775 },
  { event := event228822
    frameStart := 228775 },
  { event := event228823
    frameStart := 228775 },
  { event := event228824
    frameStart := 228775 },
  { event := event228825
    frameStart := 228775 },
  { event := event228826
    frameStart := 228775 },
  { event := event228827
    frameStart := 228775 },
  { event := event228828
    frameStart := 228775 },
  { event := event228829
    frameStart := 228775 },
  { event := event228830
    frameStart := 228775 },
  { event := event228831
    frameStart := 228775 }
]

def eventLeaf14302 : Array AnnotatedEvent := #[
  { event := event228832
    frameStart := 228775 },
  { event := event228833
    frameStart := 228775 },
  { event := event228834
    frameStart := 228775 },
  { event := event228835
    frameStart := 228775 },
  { event := event228836
    frameStart := 228775 },
  { event := event228837
    frameStart := 228775 },
  { event := event228838
    frameStart := 228775 },
  { event := event228839
    frameStart := 228775 },
  { event := event228840
    frameStart := 228775 },
  { event := event228841
    frameStart := 228775 },
  { event := event228842
    frameStart := 228775 },
  { event := event228843
    frameStart := 228775 },
  { event := event228844
    frameStart := 228775 },
  { event := event228845
    frameStart := 228775 },
  { event := event228846
    frameStart := 228775 },
  { event := event228847
    frameStart := 228775 }
]

def eventLeaf14303 : Array AnnotatedEvent := #[
  { event := event228848
    frameStart := 228775 },
  { event := event228849
    frameStart := 228775 },
  { event := event228850
    frameStart := 228775 },
  { event := event228851
    frameStart := 228775 },
  { event := event228852
    frameStart := 228775 },
  { event := event228853
    frameStart := 228775 },
  { event := event228854
    frameStart := 228775 },
  { event := event228855
    frameStart := 228775 },
  { event := event228856
    frameStart := 228775 },
  { event := event228857
    frameStart := 228775 },
  { event := event228858
    frameStart := 228775 },
  { event := event228859
    frameStart := 228775 },
  { event := event228860
    frameStart := 228775 },
  { event := event228861
    frameStart := 228775 },
  { event := event228862
    frameStart := 228775 },
  { event := event228863
    frameStart := 228775 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events893
