import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events557

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event142592 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16938⟩⟩) 0 ⟨7177⟩ 15500

def event142593 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16938⟩⟩) 1 ⟨16936⟩ 142591

def event142594 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨16938⟩⟩) (.authority (.operator))

def exact142595RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨16938⟩⟩]⟩, (1)⟩]

theorem exact142595RawTermsValid :
    exact142595RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event142595 : Event := .resultExact (⟨.program ⟨257⟩, ⟨16938⟩⟩) exact142595RawTerms .large 142594 .exactZero (none)

def event142596 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17565⟩⟩) 0 ⟨16938⟩ 142595

def event142597 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17565⟩⟩) (.authority (.operator))

def exact142598RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨17565⟩⟩]⟩, (1)⟩]

theorem exact142598RawTermsValid :
    exact142598RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event142598 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17565⟩⟩) exact142598RawTerms (.finite 8192) 142597 .exactZero (none)

def event142599 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16806⟩⟩) 0 ⟨15308⟩ 6480

def event142600 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨16806⟩⟩) (.authority (.programFamilyFact))

def event142601 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨16806⟩⟩) (.finite 3720)

def event142602 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16807⟩⟩) 0 ⟨7177⟩ 15500

def event142603 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16807⟩⟩) 1 ⟨16806⟩ 142601

def event142604 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨16807⟩⟩) (.authority (.operator))

def exact142605RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨16807⟩⟩]⟩, (1)⟩]

theorem exact142605RawTermsValid :
    exact142605RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event142605 : Event := .resultExact (⟨.program ⟨257⟩, ⟨16807⟩⟩) exact142605RawTerms .large 142604 .exactZero (none)

def event142606 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17282⟩⟩) 0 ⟨16807⟩ 142605

def event142607 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17282⟩⟩) (.authority (.operator))

def exact142608RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨17282⟩⟩]⟩, (1)⟩]

theorem exact142608RawTermsValid :
    exact142608RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event142608 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17282⟩⟩) exact142608RawTerms (.finite 8192) 142607 .exactZero (none)

def event142609 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15309⟩⟩) 0 ⟨15306⟩ 6469

def event142610 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15309⟩⟩) 1 ⟨6919⟩ 134403

def event142611 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15309⟩⟩) (.tensor (.predecessor 0 142609 .coefficient) (.predecessor 1 142610 .coefficient) true false)

def event142612 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨15309⟩⟩, .operator (⟨6469, 0⟩, ⟨134403, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨15306⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact142613RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨15306⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact142613RawTermsValid :
    exact142613RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event142613 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15309⟩⟩) exact142613RawTerms .large 142611 .exactZero (none)

def event142614 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7812⟩⟩) 0 ⟨5471⟩ 134273

def event142615 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7812⟩⟩) 1 ⟨7304⟩ 25597

def event142616 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7812⟩⟩) (.product (.predecessor 0 142614 .coefficient) (.predecessor 1 142615 .coefficient) (⟨false, false, none, none, none⟩))

def event142617 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨7812⟩⟩, .operator (⟨134273, 0⟩, ⟨25597, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7304⟩⟩]⟩, (1)⟩)

def exact142618RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7304⟩⟩]⟩, (1)⟩]

theorem exact142618RawTermsValid :
    exact142618RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event142618 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7812⟩⟩) exact142618RawTerms .large 142616 .exactZero (none)

def event142619 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15310⟩⟩) 0 ⟨7812⟩ 142618

def event142620 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15310⟩⟩) 1 ⟨15309⟩ 142613

def event142621 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15310⟩⟩) (.sum [.predecessor 0 142619 .coefficient, .predecessor 1 142620 .coefficient])

def exact142622RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7304⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨15306⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact142622RawTermsValid :
    exact142622RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event142622 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15310⟩⟩) exact142622RawTerms .large 142621 .exactZero (none)

def event142623 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15311⟩⟩) 0 ⟨15310⟩ 142622

def event142624 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15311⟩⟩) 1 ⟨130⟩ 25589

def event142625 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15311⟩⟩) (.sum [.predecessor 0 142623 .coefficient, .predecessor 1 142624 .coefficient])

def event142626 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15311⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨130⟩⟩]⟩) [⟨.result 25589 .coefficient, false, none⟩])

def event142627 : Event := .survivorFold (1) 142626

def exact142628RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7304⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨15306⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact142628RawTermsValid :
    exact142628RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event142628 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15311⟩⟩) exact142628RawTerms .large 142625 (.finite 26) (some (142626))

def event142629 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15312⟩⟩) 0 ⟨15311⟩ 142628

def event142630 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15312⟩⟩) 1 ⟨12276⟩ 6472

def event142631 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15312⟩⟩) (.product (.predecessor 0 142629 .coefficient) (.predecessor 1 142630 .coefficient) (⟨false, true, none, none, some 1⟩))

def event142632 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15312⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨12276⟩⟩], []⟩) [⟨.result 6472 .coefficient, true, some 1⟩])

def event142633 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15312⟩⟩) (.product (.result 142628 .summary) (.transfer 142632) (⟨false, false, none, none, none⟩))

def event142634 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨15312⟩⟩, .operator (⟨142628, 1⟩, ⟨6472, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨12276⟩⟩, ⟨.program ⟨257⟩, ⟨15306⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def event142635 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨15312⟩⟩, .operator (⟨142628, 0⟩, ⟨6472, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨12276⟩⟩], [⟨.program ⟨257⟩, ⟨7304⟩⟩]⟩, (1)⟩)

def exact142636RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨12276⟩⟩], [⟨.program ⟨257⟩, ⟨7304⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨12276⟩⟩, ⟨.program ⟨257⟩, ⟨15306⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact142636RawTermsValid :
    exact142636RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event142636 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15312⟩⟩) exact142636RawTerms .large 142631 (.finite 1703936) (some (142633))

def event142637 : Event := .predecessor (⟨.program ⟨257⟩, ⟨12277⟩⟩) 0 ⟨12276⟩ 6472

def event142638 : Event := .predecessor (⟨.program ⟨257⟩, ⟨12277⟩⟩) 1 ⟨6919⟩ 134403

def event142639 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨12277⟩⟩) (.tensor (.predecessor 0 142637 .coefficient) (.predecessor 1 142638 .coefficient) true false)

def event142640 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨12277⟩⟩, .operator (⟨6472, 0⟩, ⟨134403, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨12276⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact142641RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨12276⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact142641RawTermsValid :
    exact142641RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event142641 : Event := .resultExact (⟨.program ⟨257⟩, ⟨12277⟩⟩) exact142641RawTerms .large 142639 .exactZero (none)

def event142642 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7811⟩⟩) 0 ⟨5471⟩ 134273

def event142643 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7811⟩⟩) 1 ⟨7303⟩ 25638

def event142644 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7811⟩⟩) (.product (.predecessor 0 142642 .coefficient) (.predecessor 1 142643 .coefficient) (⟨false, false, none, none, none⟩))

def event142645 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨7811⟩⟩, .operator (⟨134273, 0⟩, ⟨25638, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7303⟩⟩]⟩, (1)⟩)

def exact142646RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7303⟩⟩]⟩, (1)⟩]

theorem exact142646RawTermsValid :
    exact142646RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event142646 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7811⟩⟩) exact142646RawTerms .large 142644 .exactZero (none)

def event142647 : Event := .predecessor (⟨.program ⟨257⟩, ⟨12278⟩⟩) 0 ⟨7811⟩ 142646

def event142648 : Event := .predecessor (⟨.program ⟨257⟩, ⟨12278⟩⟩) 1 ⟨12277⟩ 142641

def event142649 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨12278⟩⟩) (.sum [.predecessor 0 142647 .coefficient, .predecessor 1 142648 .coefficient])

def exact142650RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7303⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨12276⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact142650RawTermsValid :
    exact142650RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event142650 : Event := .resultExact (⟨.program ⟨257⟩, ⟨12278⟩⟩) exact142650RawTerms .large 142649 .exactZero (none)

def event142651 : Event := .predecessor (⟨.program ⟨257⟩, ⟨12279⟩⟩) 0 ⟨12278⟩ 142650

def event142652 : Event := .predecessor (⟨.program ⟨257⟩, ⟨12279⟩⟩) 1 ⟨129⟩ 25630

def event142653 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨12279⟩⟩) (.sum [.predecessor 0 142651 .coefficient, .predecessor 1 142652 .coefficient])

def event142654 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨12279⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨129⟩⟩]⟩) [⟨.result 25630 .coefficient, false, none⟩])

def event142655 : Event := .survivorFold (1) 142654

def exact142656RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7303⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨12276⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact142656RawTermsValid :
    exact142656RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event142656 : Event := .resultExact (⟨.program ⟨257⟩, ⟨12279⟩⟩) exact142656RawTerms .large 142653 (.finite 26) (some (142654))

def event142657 : Event := .predecessor (⟨.program ⟨257⟩, ⟨12280⟩⟩) 0 ⟨12279⟩ 142656

def event142658 : Event := .predecessor (⟨.program ⟨257⟩, ⟨12280⟩⟩) 1 ⟨9569⟩ 25627

def event142659 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨12280⟩⟩) (.product (.predecessor 0 142657 .coefficient) (.predecessor 1 142658 .coefficient) (⟨false, false, none, none, none⟩))

def event142660 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨12280⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨9568⟩⟩]⟩) [⟨.result 25623 .coefficient, false, none⟩])

def event142661 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨12280⟩⟩) (.product (.result 142656 .summary) (.transfer 142660) (⟨false, false, none, none, none⟩))

def event142662 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨12280⟩⟩, .operator (⟨142656, 1⟩, ⟨25627, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨12276⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩]⟩, (-1)⟩)

def event142663 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨12280⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨12276⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨9568⟩⟩) ⟨7304⟩ 25597)

def event142664 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨12280⟩⟩, .relation 142663 0, ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨12276⟩⟩], [⟨.program ⟨257⟩, ⟨7304⟩⟩]⟩, (-1)⟩)

def event142665 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨12280⟩⟩, .operator (⟨142656, 0⟩, ⟨25627, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩]⟩, (1)⟩)

def exact142666RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨12276⟩⟩], [⟨.program ⟨257⟩, ⟨7304⟩⟩]⟩, (-1)⟩]

theorem exact142666RawTermsValid :
    exact142666RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event142666 : Event := .resultExact (⟨.program ⟨257⟩, ⟨12280⟩⟩) exact142666RawTerms .large 142659 (.finite 279172874240) (some (142661))

def event142667 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15313⟩⟩) 0 ⟨12280⟩ 142666

def event142668 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15313⟩⟩) 1 ⟨15312⟩ 142636

def event142669 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15313⟩⟩) (.sum [.predecessor 0 142667 .coefficient, .predecessor 1 142668 .coefficient])

def event142670 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨15313⟩⟩, .operator (⟨142666, 1⟩, ⟨142636, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨12276⟩⟩], [⟨.program ⟨257⟩, ⟨7304⟩⟩]⟩, (1)⟩)

def event142671 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15313⟩⟩) (.sum [.result 142666 .summary, .result 142636 .summary])

def exact142672RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨12276⟩⟩, ⟨.program ⟨257⟩, ⟨15306⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact142672RawTermsValid :
    exact142672RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event142672 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15313⟩⟩) exact142672RawTerms .large 142669 (.finite 279174578176) (some (142671))

def event142673 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17283⟩⟩) 0 ⟨15313⟩ 142672

def event142674 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17283⟩⟩) 1 ⟨17282⟩ 142608

def event142675 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17283⟩⟩) (.product (.predecessor 0 142673 .coefficient) (.predecessor 1 142674 .coefficient) (⟨false, false, none, none, none⟩))

def event142676 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17283⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨17282⟩⟩]⟩) [⟨.result 142608 .coefficient, false, none⟩])

def event142677 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17283⟩⟩) (.product (.result 142672 .summary) (.transfer 142676) (⟨false, false, none, none, none⟩))

def event142678 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17283⟩⟩, .operator (⟨142672, 1⟩, ⟨142608, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨12276⟩⟩, ⟨.program ⟨257⟩, ⟨15306⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17282⟩⟩]⟩, (-1)⟩)

def event142679 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨17283⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨12276⟩⟩, ⟨.program ⟨257⟩, ⟨15306⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17282⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨17282⟩⟩) ⟨16807⟩ 142605)

def event142680 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17283⟩⟩, .relation 142679 0, ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨12276⟩⟩, ⟨.program ⟨257⟩, ⟨15306⟩⟩], [⟨.program ⟨257⟩, ⟨16807⟩⟩]⟩, (-1)⟩)

def event142681 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17283⟩⟩, .operator (⟨142672, 0⟩, ⟨142608, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩, ⟨.program ⟨257⟩, ⟨17282⟩⟩]⟩, (1)⟩)

def exact142682RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩, ⟨.program ⟨257⟩, ⟨17282⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨12276⟩⟩, ⟨.program ⟨257⟩, ⟨15306⟩⟩], [⟨.program ⟨257⟩, ⟨16807⟩⟩]⟩, (-1)⟩]

theorem exact142682RawTermsValid :
    exact142682RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event142682 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17283⟩⟩) exact142682RawTerms .large 142675 (.finite 2997614207851288330240) (some (142677))

def event142683 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16219⟩⟩) 0 ⟨15308⟩ 6480

def event142684 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨16219⟩⟩) (.authority (.relationPreimageSource ⟨36⟩))

def exact142685RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨16219⟩⟩]⟩, (1)⟩]

theorem exact142685RawTermsValid :
    exact142685RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event142685 : Event := .resultExact (⟨.program ⟨257⟩, ⟨16219⟩⟩) exact142685RawTerms (.finite 5647228698) 142684 .exactZero (none)

def event142686 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16221⟩⟩) 0 ⟨16219⟩ 142685

def event142687 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16221⟩⟩) 1 ⟨2370⟩ 4

def event142688 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨16221⟩⟩) (.scale (.predecessor 0 142686 .coefficient) (.value (.predecessor 1 142687 .coefficient)))

def exact142689RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨16219⟩⟩]⟩, (1)⟩]

theorem exact142689RawTermsValid :
    exact142689RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event142689 : Event := .resultExact (⟨.program ⟨257⟩, ⟨16221⟩⟩) exact142689RawTerms (.finite 5647228698) 142688 .exactZero (none)

def event142690 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16222⟩⟩) 0 ⟨5473⟩ 134495

def event142691 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16222⟩⟩) 1 ⟨16221⟩ 142689

def event142692 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨16222⟩⟩) (.product (.predecessor 0 142690 .coefficient) (.predecessor 1 142691 .coefficient) (⟨false, false, none, none, none⟩))

def event142693 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨16222⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨16219⟩⟩]⟩) [⟨.result 142685 .coefficient, false, none⟩])

def event142694 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨16222⟩⟩) (.product (.result 134495 .summary) (.transfer 142693) (⟨false, false, none, none, none⟩))

def event142695 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨16222⟩⟩, .operator (⟨134495, 0⟩, ⟨142689, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16219⟩⟩]⟩, (1)⟩)

def event142696 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨16220⟩⟩)

def event142697 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event142698 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event142699 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨455⟩⟩) (.authority (.operator))

def event142700 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨455⟩⟩) (.finite 11)

def event142701 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event142702 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event142703 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event142704 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event142705 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 142704

def event142706 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 142702

def event142707 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 142705 .coefficient) (.value (.predecessor 1 142706 .coefficient)))

def event142708 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event142709 : Event := .predecessor (⟨.program ⟨257⟩, ⟨457⟩⟩) 0 ⟨392⟩ 142708

def event142710 : Event := .predecessor (⟨.program ⟨257⟩, ⟨457⟩⟩) 1 ⟨455⟩ 142700

def event142711 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨457⟩⟩) (.sum [.predecessor 0 142709 .coefficient, .predecessor 1 142710 .coefficient])

def event142712 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨457⟩⟩) (.finite 655351)

def event142713 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5469⟩⟩) 0 ⟨457⟩ 142712

def event142714 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5469⟩⟩) 1 ⟨5426⟩ 142698

def event142715 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5469⟩⟩) (.identity (.predecessor 1 142714 .coefficient))

def event142716 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5469⟩⟩) (.finite 655360)

def event142717 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15306⟩⟩) 0 ⟨5469⟩ 142716

def event142718 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15306⟩⟩) (.authority (.programFamilyFact))

def exact142719RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15306⟩⟩], []⟩, (1)⟩]

theorem exact142719RawTermsValid :
    exact142719RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event142719 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15306⟩⟩) exact142719RawTerms (.finite 2) 142718 .exactZero (none)

def event142720 : Event := .predecessor (⟨.program ⟨257⟩, ⟨12276⟩⟩) 0 ⟨5469⟩ 142716

def event142721 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨12276⟩⟩) (.authority (.programFamilyFact))

def exact142722RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨12276⟩⟩], []⟩, (1)⟩]

theorem exact142722RawTermsValid :
    exact142722RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event142722 : Event := .resultExact (⟨.program ⟨257⟩, ⟨12276⟩⟩) exact142722RawTerms (.finite 2) 142721 .exactZero (none)

def event142723 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15307⟩⟩) 0 ⟨12276⟩ 142722

def event142724 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15307⟩⟩) 1 ⟨15306⟩ 142719

def event142725 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15307⟩⟩) (.product (.predecessor 0 142723 .coefficient) (.predecessor 1 142724 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event142726 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15307⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨12276⟩⟩, ⟨.program ⟨257⟩, ⟨15306⟩⟩], []⟩) [⟨.result 142722 .coefficient, true, some 1⟩, ⟨.result 142719 .coefficient, true, some 1⟩])

def event142727 : Event := .survivorFold (1) 142726

def exact142728RawTerms : List Term := []

theorem exact142728RawTermsValid :
    exact142728RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event142728 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15307⟩⟩) exact142728RawTerms (.finite 4) 142725 (.finite 4) (some (142726))

def event142729 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15308⟩⟩) 0 ⟨15307⟩ 142728

def event142730 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15308⟩⟩) (.identity (.predecessor 0 142729 .coefficient))

def event142731 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨15308⟩⟩) (.finite 4)

def event142732 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16219⟩⟩) 0 ⟨15308⟩ 142731

def event142733 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨16219⟩⟩) (.authority (.relationPreimageSource ⟨36⟩))

def exact142734RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨16219⟩⟩]⟩, (1)⟩]

theorem exact142734RawTermsValid :
    exact142734RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event142734 : Event := .resultExact (⟨.program ⟨257⟩, ⟨16219⟩⟩) exact142734RawTerms (.finite 5647228698) 142733 .exactZero (none)

def event142735 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35⟩⟩) (.authority (.operator))

def exact142736RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩]⟩, (1)⟩]

theorem exact142736RawTermsValid :
    exact142736RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event142736 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35⟩⟩) exact142736RawTerms .large 142735 .exactZero (none)

def event142737 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16220⟩⟩) 0 ⟨35⟩ 142736

def event142738 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16220⟩⟩) 1 ⟨16219⟩ 142734

def event142739 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨16220⟩⟩) (.product (.predecessor 0 142737 .coefficient) (.predecessor 1 142738 .coefficient) (⟨false, false, none, none, none⟩))

def event142740 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨16220⟩⟩, .operator (⟨142736, 0⟩, ⟨142734, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16219⟩⟩]⟩, (1)⟩)

def exact142741RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16219⟩⟩]⟩, (1)⟩]

theorem exact142741RawTermsValid :
    exact142741RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event142741 : Event := .resultExact (⟨.program ⟨257⟩, ⟨16220⟩⟩) exact142741RawTerms .large 142739 .exactZero (none)

def event142742 : Event := .preFoldPolynomial 142741 [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16219⟩⟩]⟩, (1)⟩] .exactZero none

def exact142743RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16219⟩⟩]⟩, (1)⟩]

def event142743 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨16220⟩⟩) 142742 exact142743RawTerms .large 142739 .exactZero (none)

def event142744 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨17286⟩⟩)

def event142745 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event142746 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event142747 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨455⟩⟩) (.authority (.operator))

def event142748 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨455⟩⟩) (.finite 11)

def event142749 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event142750 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event142751 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event142752 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event142753 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 142752

def event142754 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 142750

def event142755 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 142753 .coefficient) (.value (.predecessor 1 142754 .coefficient)))

def event142756 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event142757 : Event := .predecessor (⟨.program ⟨257⟩, ⟨457⟩⟩) 0 ⟨392⟩ 142756

def event142758 : Event := .predecessor (⟨.program ⟨257⟩, ⟨457⟩⟩) 1 ⟨455⟩ 142748

def event142759 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨457⟩⟩) (.sum [.predecessor 0 142757 .coefficient, .predecessor 1 142758 .coefficient])

def event142760 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨457⟩⟩) (.finite 655351)

def event142761 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5469⟩⟩) 0 ⟨457⟩ 142760

def event142762 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5469⟩⟩) 1 ⟨5426⟩ 142746

def event142763 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5469⟩⟩) (.identity (.predecessor 1 142762 .coefficient))

def event142764 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5469⟩⟩) (.finite 655360)

def event142765 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15306⟩⟩) 0 ⟨5469⟩ 142764

def event142766 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15306⟩⟩) (.authority (.programFamilyFact))

def exact142767RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15306⟩⟩], []⟩, (1)⟩]

theorem exact142767RawTermsValid :
    exact142767RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event142767 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15306⟩⟩) exact142767RawTerms (.finite 2) 142766 .exactZero (none)

def event142768 : Event := .predecessor (⟨.program ⟨257⟩, ⟨12276⟩⟩) 0 ⟨5469⟩ 142764

def event142769 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨12276⟩⟩) (.authority (.programFamilyFact))

def exact142770RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨12276⟩⟩], []⟩, (1)⟩]

theorem exact142770RawTermsValid :
    exact142770RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event142770 : Event := .resultExact (⟨.program ⟨257⟩, ⟨12276⟩⟩) exact142770RawTerms (.finite 2) 142769 .exactZero (none)

def event142771 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15307⟩⟩) 0 ⟨12276⟩ 142770

def event142772 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15307⟩⟩) 1 ⟨15306⟩ 142767

def event142773 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15307⟩⟩) (.product (.predecessor 0 142771 .coefficient) (.predecessor 1 142772 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event142774 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨15307⟩⟩, .operator (⟨142770, 0⟩, ⟨142767, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨12276⟩⟩, ⟨.program ⟨257⟩, ⟨15306⟩⟩], []⟩, (1)⟩)

def exact142775RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨12276⟩⟩, ⟨.program ⟨257⟩, ⟨15306⟩⟩], []⟩, (1)⟩]

theorem exact142775RawTermsValid :
    exact142775RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event142775 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15307⟩⟩) exact142775RawTerms (.finite 4) 142773 .exactZero (none)

def event142776 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15308⟩⟩) 0 ⟨15307⟩ 142775

def event142777 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15308⟩⟩) (.identity (.predecessor 0 142776 .coefficient))

def event142778 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨15308⟩⟩) (.finite 4)

def event142779 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16806⟩⟩) 0 ⟨15308⟩ 142778

def event142780 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨16806⟩⟩) (.authority (.programFamilyFact))

def event142781 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨16806⟩⟩) (.finite 3720)

def event142782 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨7177⟩⟩) .missing

def event142783 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16807⟩⟩) 0 ⟨7177⟩ 142782

def event142784 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16807⟩⟩) 1 ⟨16806⟩ 142781

def event142785 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨16807⟩⟩) (.authority (.operator))

def exact142786RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨16807⟩⟩]⟩, (1)⟩]

theorem exact142786RawTermsValid :
    exact142786RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event142786 : Event := .resultExact (⟨.program ⟨257⟩, ⟨16807⟩⟩) exact142786RawTerms .large 142785 .exactZero (none)

def event142787 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17282⟩⟩) 0 ⟨16807⟩ 142786

def event142788 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17282⟩⟩) (.authority (.operator))

def exact142789RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨17282⟩⟩]⟩, (1)⟩]

theorem exact142789RawTermsValid :
    exact142789RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event142789 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17282⟩⟩) exact142789RawTerms (.finite 8192) 142788 .exactZero (none)

def event142790 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨136⟩⟩) (.authority (.operator))

def event142791 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨136⟩⟩) .exactZero

def event142792 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17098⟩⟩) 0 ⟨15308⟩ 142778

def event142793 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17098⟩⟩) 1 ⟨136⟩ 142791

def event142794 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17098⟩⟩) (.sum [.predecessor 0 142792 .coefficient, .predecessor 1 142793 .coefficient])

def event142795 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨17098⟩⟩) (.finite 4)

def event142796 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17099⟩⟩) 0 ⟨17098⟩ 142795

def event142797 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17099⟩⟩) (.identity (.predecessor 0 142796 .coefficient))

def exact142798RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨12276⟩⟩, ⟨.program ⟨257⟩, ⟨15306⟩⟩], []⟩, (1)⟩]

theorem exact142798RawTermsValid :
    exact142798RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event142798 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17099⟩⟩) exact142798RawTerms (.finite 4) 142797 .exactZero (none)

def event142799 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6908⟩⟩) (.authority (.factStore))

def exact142800RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact142800RawTermsValid :
    exact142800RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event142800 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6908⟩⟩) exact142800RawTerms .large 142799 .exactZero (none)

def event142801 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17100⟩⟩) 0 ⟨6908⟩ 142800

def event142802 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17100⟩⟩) 1 ⟨17099⟩ 142798

def event142803 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17100⟩⟩) (.product (.predecessor 0 142801 .coefficient) (.predecessor 1 142802 .coefficient) (⟨false, false, none, none, none⟩))

def event142804 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17100⟩⟩, .operator (⟨142800, 0⟩, ⟨142798, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨12276⟩⟩, ⟨.program ⟨257⟩, ⟨15306⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact142805RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨12276⟩⟩, ⟨.program ⟨257⟩, ⟨15306⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact142805RawTermsValid :
    exact142805RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event142805 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17100⟩⟩) exact142805RawTerms .large 142803 .exactZero (none)

def event142806 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨2370⟩⟩) (.authority (.operator))

def event142807 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨2370⟩⟩) (.finite 1)

def event142808 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7178⟩⟩) 0 ⟨7177⟩ 142782

def event142809 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7178⟩⟩) (.authority (.operator))

def exact142810RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7178⟩⟩]⟩, (1)⟩]

theorem exact142810RawTermsValid :
    exact142810RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event142810 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7178⟩⟩) exact142810RawTerms .large 142809 .exactZero (none)

def event142811 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7304⟩⟩) 0 ⟨7178⟩ 142810

def event142812 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7304⟩⟩) (.identity (.predecessor 0 142811 .coefficient))

def exact142813RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7304⟩⟩]⟩, (1)⟩]

theorem exact142813RawTermsValid :
    exact142813RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event142813 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7304⟩⟩) exact142813RawTerms .large 142812 .exactZero (none)

def event142814 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9568⟩⟩) 0 ⟨7304⟩ 142813

def event142815 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9568⟩⟩) (.authority (.operator))

def exact142816RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨9568⟩⟩]⟩, (1)⟩]

theorem exact142816RawTermsValid :
    exact142816RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event142816 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9568⟩⟩) exact142816RawTerms (.finite 8192) 142815 .exactZero (none)

def event142817 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9569⟩⟩) 0 ⟨9568⟩ 142816

def event142818 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9569⟩⟩) 1 ⟨2370⟩ 142807

def event142819 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9569⟩⟩) (.scale (.predecessor 0 142817 .coefficient) (.value (.predecessor 1 142818 .coefficient)))

def exact142820RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨9568⟩⟩]⟩, (1)⟩]

theorem exact142820RawTermsValid :
    exact142820RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event142820 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9569⟩⟩) exact142820RawTerms (.finite 8192) 142819 .exactZero (none)

def event142821 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7303⟩⟩) 0 ⟨7178⟩ 142810

def event142822 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7303⟩⟩) (.identity (.predecessor 0 142821 .coefficient))

def exact142823RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7303⟩⟩]⟩, (1)⟩]

theorem exact142823RawTermsValid :
    exact142823RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event142823 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7303⟩⟩) exact142823RawTerms .large 142822 .exactZero (none)

def event142824 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9570⟩⟩) 0 ⟨7303⟩ 142823

def event142825 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9570⟩⟩) 1 ⟨9569⟩ 142820

def event142826 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9570⟩⟩) (.product (.predecessor 0 142824 .coefficient) (.predecessor 1 142825 .coefficient) (⟨false, false, none, none, none⟩))

def event142827 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨9570⟩⟩, .operator (⟨142823, 0⟩, ⟨142820, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩]⟩, (1)⟩)

def exact142828RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩]⟩, (1)⟩]

theorem exact142828RawTermsValid :
    exact142828RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event142828 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9570⟩⟩) exact142828RawTerms .large 142826 .exactZero (none)

def event142829 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17101⟩⟩) 0 ⟨9570⟩ 142828

def event142830 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17101⟩⟩) 1 ⟨17100⟩ 142805

def event142831 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17101⟩⟩) (.sum [.predecessor 0 142829 .coefficient, .predecessor 1 142830 .coefficient])

def exact142832RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨12276⟩⟩, ⟨.program ⟨257⟩, ⟨15306⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact142832RawTermsValid :
    exact142832RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event142832 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17101⟩⟩) exact142832RawTerms .large 142831 .exactZero (none)

def event142833 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17285⟩⟩) 0 ⟨17101⟩ 142832

def event142834 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17285⟩⟩) 1 ⟨17282⟩ 142789

def event142835 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17285⟩⟩) (.product (.predecessor 0 142833 .coefficient) (.predecessor 1 142834 .coefficient) (⟨false, false, none, none, none⟩))

def event142836 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17285⟩⟩, .operator (⟨142832, 0⟩, ⟨142789, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩, ⟨.program ⟨257⟩, ⟨17282⟩⟩]⟩, (1)⟩)

def event142837 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17285⟩⟩, .operator (⟨142832, 1⟩, ⟨142789, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨12276⟩⟩, ⟨.program ⟨257⟩, ⟨15306⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17282⟩⟩]⟩, (-1)⟩)

def event142838 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨17285⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨12276⟩⟩, ⟨.program ⟨257⟩, ⟨15306⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17282⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨17282⟩⟩) ⟨16807⟩ 142786)

def event142839 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17285⟩⟩, .relation 142838 0, ⟨[⟨.program ⟨257⟩, ⟨12276⟩⟩, ⟨.program ⟨257⟩, ⟨15306⟩⟩], [⟨.program ⟨257⟩, ⟨16807⟩⟩]⟩, (-1)⟩)

def exact142840RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩, ⟨.program ⟨257⟩, ⟨17282⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨12276⟩⟩, ⟨.program ⟨257⟩, ⟨15306⟩⟩], [⟨.program ⟨257⟩, ⟨16807⟩⟩]⟩, (-1)⟩]

theorem exact142840RawTermsValid :
    exact142840RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event142840 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17285⟩⟩) exact142840RawTerms .large 142835 .exactZero (none)

def event142841 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15732⟩⟩) 0 ⟨15308⟩ 142778

def event142842 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15732⟩⟩) (.authority (.programFamilyFact))

def exact142843RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15732⟩⟩], []⟩, (1)⟩]

theorem exact142843RawTermsValid :
    exact142843RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event142843 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15732⟩⟩) exact142843RawTerms (.finite 2) 142842 .exactZero (none)

def event142844 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15734⟩⟩) 0 ⟨6908⟩ 142800

def event142845 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15734⟩⟩) 1 ⟨15732⟩ 142843

def event142846 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15734⟩⟩) (.product (.predecessor 0 142844 .coefficient) (.predecessor 1 142845 .coefficient) (⟨false, true, none, none, some 1⟩))

def event142847 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨15734⟩⟩, .operator (⟨142800, 0⟩, ⟨142843, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨15732⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def eventLeaf8912 : Array AnnotatedEvent := #[
  { event := event142592
    frameStart := 0 },
  { event := event142593
    frameStart := 0 },
  { event := event142594
    frameStart := 0 },
  { event := event142595
    frameStart := 0 },
  { event := event142596
    frameStart := 0 },
  { event := event142597
    frameStart := 0 },
  { event := event142598
    frameStart := 0 },
  { event := event142599
    frameStart := 0 },
  { event := event142600
    frameStart := 0 },
  { event := event142601
    frameStart := 0 },
  { event := event142602
    frameStart := 0 },
  { event := event142603
    frameStart := 0 },
  { event := event142604
    frameStart := 0 },
  { event := event142605
    frameStart := 0 },
  { event := event142606
    frameStart := 0 },
  { event := event142607
    frameStart := 0 }
]

def eventLeaf8913 : Array AnnotatedEvent := #[
  { event := event142608
    frameStart := 0 },
  { event := event142609
    frameStart := 0 },
  { event := event142610
    frameStart := 0 },
  { event := event142611
    frameStart := 0 },
  { event := event142612
    frameStart := 0 },
  { event := event142613
    frameStart := 0 },
  { event := event142614
    frameStart := 0 },
  { event := event142615
    frameStart := 0 },
  { event := event142616
    frameStart := 0 },
  { event := event142617
    frameStart := 0 },
  { event := event142618
    frameStart := 0 },
  { event := event142619
    frameStart := 0 },
  { event := event142620
    frameStart := 0 },
  { event := event142621
    frameStart := 0 },
  { event := event142622
    frameStart := 0 },
  { event := event142623
    frameStart := 0 }
]

def eventLeaf8914 : Array AnnotatedEvent := #[
  { event := event142624
    frameStart := 0 },
  { event := event142625
    frameStart := 0 },
  { event := event142626
    frameStart := 0 },
  { event := event142627
    frameStart := 0 },
  { event := event142628
    frameStart := 0 },
  { event := event142629
    frameStart := 0 },
  { event := event142630
    frameStart := 0 },
  { event := event142631
    frameStart := 0 },
  { event := event142632
    frameStart := 0 },
  { event := event142633
    frameStart := 0 },
  { event := event142634
    frameStart := 0 },
  { event := event142635
    frameStart := 0 },
  { event := event142636
    frameStart := 0 },
  { event := event142637
    frameStart := 0 },
  { event := event142638
    frameStart := 0 },
  { event := event142639
    frameStart := 0 }
]

def eventLeaf8915 : Array AnnotatedEvent := #[
  { event := event142640
    frameStart := 0 },
  { event := event142641
    frameStart := 0 },
  { event := event142642
    frameStart := 0 },
  { event := event142643
    frameStart := 0 },
  { event := event142644
    frameStart := 0 },
  { event := event142645
    frameStart := 0 },
  { event := event142646
    frameStart := 0 },
  { event := event142647
    frameStart := 0 },
  { event := event142648
    frameStart := 0 },
  { event := event142649
    frameStart := 0 },
  { event := event142650
    frameStart := 0 },
  { event := event142651
    frameStart := 0 },
  { event := event142652
    frameStart := 0 },
  { event := event142653
    frameStart := 0 },
  { event := event142654
    frameStart := 0 },
  { event := event142655
    frameStart := 0 }
]

def eventLeaf8916 : Array AnnotatedEvent := #[
  { event := event142656
    frameStart := 0 },
  { event := event142657
    frameStart := 0 },
  { event := event142658
    frameStart := 0 },
  { event := event142659
    frameStart := 0 },
  { event := event142660
    frameStart := 0 },
  { event := event142661
    frameStart := 0 },
  { event := event142662
    frameStart := 0 },
  { event := event142663
    frameStart := 0 },
  { event := event142664
    frameStart := 0 },
  { event := event142665
    frameStart := 0 },
  { event := event142666
    frameStart := 0 },
  { event := event142667
    frameStart := 0 },
  { event := event142668
    frameStart := 0 },
  { event := event142669
    frameStart := 0 },
  { event := event142670
    frameStart := 0 },
  { event := event142671
    frameStart := 0 }
]

def eventLeaf8917 : Array AnnotatedEvent := #[
  { event := event142672
    frameStart := 0 },
  { event := event142673
    frameStart := 0 },
  { event := event142674
    frameStart := 0 },
  { event := event142675
    frameStart := 0 },
  { event := event142676
    frameStart := 0 },
  { event := event142677
    frameStart := 0 },
  { event := event142678
    frameStart := 0 },
  { event := event142679
    frameStart := 0 },
  { event := event142680
    frameStart := 0 },
  { event := event142681
    frameStart := 0 },
  { event := event142682
    frameStart := 0 },
  { event := event142683
    frameStart := 0 },
  { event := event142684
    frameStart := 0 },
  { event := event142685
    frameStart := 0 },
  { event := event142686
    frameStart := 0 },
  { event := event142687
    frameStart := 0 }
]

def eventLeaf8918 : Array AnnotatedEvent := #[
  { event := event142688
    frameStart := 0 },
  { event := event142689
    frameStart := 0 },
  { event := event142690
    frameStart := 0 },
  { event := event142691
    frameStart := 0 },
  { event := event142692
    frameStart := 0 },
  { event := event142693
    frameStart := 0 },
  { event := event142694
    frameStart := 0 },
  { event := event142695
    frameStart := 0 },
  { event := event142696
    frameStart := 142696 },
  { event := event142697
    frameStart := 142696 },
  { event := event142698
    frameStart := 142696 },
  { event := event142699
    frameStart := 142696 },
  { event := event142700
    frameStart := 142696 },
  { event := event142701
    frameStart := 142696 },
  { event := event142702
    frameStart := 142696 },
  { event := event142703
    frameStart := 142696 }
]

def eventLeaf8919 : Array AnnotatedEvent := #[
  { event := event142704
    frameStart := 142696 },
  { event := event142705
    frameStart := 142696 },
  { event := event142706
    frameStart := 142696 },
  { event := event142707
    frameStart := 142696 },
  { event := event142708
    frameStart := 142696 },
  { event := event142709
    frameStart := 142696 },
  { event := event142710
    frameStart := 142696 },
  { event := event142711
    frameStart := 142696 },
  { event := event142712
    frameStart := 142696 },
  { event := event142713
    frameStart := 142696 },
  { event := event142714
    frameStart := 142696 },
  { event := event142715
    frameStart := 142696 },
  { event := event142716
    frameStart := 142696 },
  { event := event142717
    frameStart := 142696 },
  { event := event142718
    frameStart := 142696 },
  { event := event142719
    frameStart := 142696 }
]

def eventLeaf8920 : Array AnnotatedEvent := #[
  { event := event142720
    frameStart := 142696 },
  { event := event142721
    frameStart := 142696 },
  { event := event142722
    frameStart := 142696 },
  { event := event142723
    frameStart := 142696 },
  { event := event142724
    frameStart := 142696 },
  { event := event142725
    frameStart := 142696 },
  { event := event142726
    frameStart := 142696 },
  { event := event142727
    frameStart := 142696 },
  { event := event142728
    frameStart := 142696 },
  { event := event142729
    frameStart := 142696 },
  { event := event142730
    frameStart := 142696 },
  { event := event142731
    frameStart := 142696 },
  { event := event142732
    frameStart := 142696 },
  { event := event142733
    frameStart := 142696 },
  { event := event142734
    frameStart := 142696 },
  { event := event142735
    frameStart := 142696 }
]

def eventLeaf8921 : Array AnnotatedEvent := #[
  { event := event142736
    frameStart := 142696 },
  { event := event142737
    frameStart := 142696 },
  { event := event142738
    frameStart := 142696 },
  { event := event142739
    frameStart := 142696 },
  { event := event142740
    frameStart := 142696 },
  { event := event142741
    frameStart := 142696 },
  { event := event142742
    frameStart := 142696 },
  { event := event142743
    frameStart := 142696 },
  { event := event142744
    frameStart := 142744 },
  { event := event142745
    frameStart := 142744 },
  { event := event142746
    frameStart := 142744 },
  { event := event142747
    frameStart := 142744 },
  { event := event142748
    frameStart := 142744 },
  { event := event142749
    frameStart := 142744 },
  { event := event142750
    frameStart := 142744 },
  { event := event142751
    frameStart := 142744 }
]

def eventLeaf8922 : Array AnnotatedEvent := #[
  { event := event142752
    frameStart := 142744 },
  { event := event142753
    frameStart := 142744 },
  { event := event142754
    frameStart := 142744 },
  { event := event142755
    frameStart := 142744 },
  { event := event142756
    frameStart := 142744 },
  { event := event142757
    frameStart := 142744 },
  { event := event142758
    frameStart := 142744 },
  { event := event142759
    frameStart := 142744 },
  { event := event142760
    frameStart := 142744 },
  { event := event142761
    frameStart := 142744 },
  { event := event142762
    frameStart := 142744 },
  { event := event142763
    frameStart := 142744 },
  { event := event142764
    frameStart := 142744 },
  { event := event142765
    frameStart := 142744 },
  { event := event142766
    frameStart := 142744 },
  { event := event142767
    frameStart := 142744 }
]

def eventLeaf8923 : Array AnnotatedEvent := #[
  { event := event142768
    frameStart := 142744 },
  { event := event142769
    frameStart := 142744 },
  { event := event142770
    frameStart := 142744 },
  { event := event142771
    frameStart := 142744 },
  { event := event142772
    frameStart := 142744 },
  { event := event142773
    frameStart := 142744 },
  { event := event142774
    frameStart := 142744 },
  { event := event142775
    frameStart := 142744 },
  { event := event142776
    frameStart := 142744 },
  { event := event142777
    frameStart := 142744 },
  { event := event142778
    frameStart := 142744 },
  { event := event142779
    frameStart := 142744 },
  { event := event142780
    frameStart := 142744 },
  { event := event142781
    frameStart := 142744 },
  { event := event142782
    frameStart := 142744 },
  { event := event142783
    frameStart := 142744 }
]

def eventLeaf8924 : Array AnnotatedEvent := #[
  { event := event142784
    frameStart := 142744 },
  { event := event142785
    frameStart := 142744 },
  { event := event142786
    frameStart := 142744 },
  { event := event142787
    frameStart := 142744 },
  { event := event142788
    frameStart := 142744 },
  { event := event142789
    frameStart := 142744 },
  { event := event142790
    frameStart := 142744 },
  { event := event142791
    frameStart := 142744 },
  { event := event142792
    frameStart := 142744 },
  { event := event142793
    frameStart := 142744 },
  { event := event142794
    frameStart := 142744 },
  { event := event142795
    frameStart := 142744 },
  { event := event142796
    frameStart := 142744 },
  { event := event142797
    frameStart := 142744 },
  { event := event142798
    frameStart := 142744 },
  { event := event142799
    frameStart := 142744 }
]

def eventLeaf8925 : Array AnnotatedEvent := #[
  { event := event142800
    frameStart := 142744 },
  { event := event142801
    frameStart := 142744 },
  { event := event142802
    frameStart := 142744 },
  { event := event142803
    frameStart := 142744 },
  { event := event142804
    frameStart := 142744 },
  { event := event142805
    frameStart := 142744 },
  { event := event142806
    frameStart := 142744 },
  { event := event142807
    frameStart := 142744 },
  { event := event142808
    frameStart := 142744 },
  { event := event142809
    frameStart := 142744 },
  { event := event142810
    frameStart := 142744 },
  { event := event142811
    frameStart := 142744 },
  { event := event142812
    frameStart := 142744 },
  { event := event142813
    frameStart := 142744 },
  { event := event142814
    frameStart := 142744 },
  { event := event142815
    frameStart := 142744 }
]

def eventLeaf8926 : Array AnnotatedEvent := #[
  { event := event142816
    frameStart := 142744 },
  { event := event142817
    frameStart := 142744 },
  { event := event142818
    frameStart := 142744 },
  { event := event142819
    frameStart := 142744 },
  { event := event142820
    frameStart := 142744 },
  { event := event142821
    frameStart := 142744 },
  { event := event142822
    frameStart := 142744 },
  { event := event142823
    frameStart := 142744 },
  { event := event142824
    frameStart := 142744 },
  { event := event142825
    frameStart := 142744 },
  { event := event142826
    frameStart := 142744 },
  { event := event142827
    frameStart := 142744 },
  { event := event142828
    frameStart := 142744 },
  { event := event142829
    frameStart := 142744 },
  { event := event142830
    frameStart := 142744 },
  { event := event142831
    frameStart := 142744 }
]

def eventLeaf8927 : Array AnnotatedEvent := #[
  { event := event142832
    frameStart := 142744 },
  { event := event142833
    frameStart := 142744 },
  { event := event142834
    frameStart := 142744 },
  { event := event142835
    frameStart := 142744 },
  { event := event142836
    frameStart := 142744 },
  { event := event142837
    frameStart := 142744 },
  { event := event142838
    frameStart := 142744 },
  { event := event142839
    frameStart := 142744 },
  { event := event142840
    frameStart := 142744 },
  { event := event142841
    frameStart := 142744 },
  { event := event142842
    frameStart := 142744 },
  { event := event142843
    frameStart := 142744 },
  { event := event142844
    frameStart := 142744 },
  { event := event142845
    frameStart := 142744 },
  { event := event142846
    frameStart := 142744 },
  { event := event142847
    frameStart := 142744 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events557
