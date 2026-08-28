import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events335

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event85760 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15620⟩⟩) (.identity (.predecessor 0 85759 .coefficient))

def event85761 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨15620⟩⟩) (.finite 4)

def event85762 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15836⟩⟩) 0 ⟨15620⟩ 85761

def event85763 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15836⟩⟩) (.authority (.programFamilyFact))

def exact85764RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15836⟩⟩], []⟩, (1)⟩]

theorem exact85764RawTermsValid :
    exact85764RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event85764 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15836⟩⟩) exact85764RawTerms (.finite 2) 85763 .exactZero (none)

def event85765 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15837⟩⟩) 0 ⟨15836⟩ 85764

def event85766 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15837⟩⟩) (.identity (.predecessor 0 85765 .coefficient))

def event85767 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨15837⟩⟩) (.finite 2)

def event85768 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16131⟩⟩) 0 ⟨15837⟩ 85767

def event85769 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨16131⟩⟩) (.authority (.programFamilyFact))

def exact85770RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨16131⟩⟩], []⟩, (1)⟩]

theorem exact85770RawTermsValid :
    exact85770RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event85770 : Event := .resultExact (⟨.program ⟨257⟩, ⟨16131⟩⟩) exact85770RawTerms (.finite 43) 85769 .exactZero (none)

def event85771 : Event := .predecessor (⟨.program ⟨257⟩, ⟨18981⟩⟩) 0 ⟨16131⟩ 85770

def event85772 : Event := .predecessor (⟨.program ⟨257⟩, ⟨18981⟩⟩) 1 ⟨18980⟩ 85747

def event85773 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨18981⟩⟩) (.sum [.predecessor 0 85771 .coefficient, .predecessor 1 85772 .coefficient])

def exact85774RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨16131⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨18980⟩⟩], []⟩, (1)⟩]

theorem exact85774RawTermsValid :
    exact85774RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event85774 : Event := .resultExact (⟨.program ⟨257⟩, ⟨18981⟩⟩) exact85774RawTerms (.finite 91) 85773 .exactZero (none)

def event85775 : Event := .predecessor (⟨.program ⟨257⟩, ⟨22201⟩⟩) 0 ⟨18981⟩ 85774

def event85776 : Event := .predecessor (⟨.program ⟨257⟩, ⟨22201⟩⟩) 1 ⟨22200⟩ 85724

def event85777 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨22201⟩⟩) (.sum [.predecessor 0 85775 .coefficient, .predecessor 1 85776 .coefficient])

def exact85778RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨16131⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨18980⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨22200⟩⟩], []⟩, (1)⟩]

theorem exact85778RawTermsValid :
    exact85778RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event85778 : Event := .resultExact (⟨.program ⟨257⟩, ⟨22201⟩⟩) exact85778RawTerms (.finite 142) 85777 .exactZero (none)

def event85779 : Event := .predecessor (⟨.program ⟨257⟩, ⟨32221⟩⟩) 0 ⟨22201⟩ 85778

def event85780 : Event := .predecessor (⟨.program ⟨257⟩, ⟨32221⟩⟩) 1 ⟨32220⟩ 85701

def event85781 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨32221⟩⟩) (.sum [.predecessor 0 85779 .coefficient, .predecessor 1 85780 .coefficient])

def exact85782RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨16131⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨18980⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨22200⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨32220⟩⟩], []⟩, (1)⟩]

theorem exact85782RawTermsValid :
    exact85782RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event85782 : Event := .resultExact (⟨.program ⟨257⟩, ⟨32221⟩⟩) exact85782RawTerms (.finite 197) 85781 .exactZero (none)

def event85783 : Event := .predecessor (⟨.program ⟨257⟩, ⟨51276⟩⟩) 0 ⟨32221⟩ 85782

def event85784 : Event := .predecessor (⟨.program ⟨257⟩, ⟨51276⟩⟩) 1 ⟨51275⟩ 85678

def event85785 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨51276⟩⟩) (.sum [.predecessor 0 85783 .coefficient, .predecessor 1 85784 .coefficient])

def exact85786RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨16131⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨18980⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨22200⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨32220⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨51275⟩⟩], []⟩, (1)⟩]

theorem exact85786RawTermsValid :
    exact85786RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event85786 : Event := .resultExact (⟨.program ⟨257⟩, ⟨51276⟩⟩) exact85786RawTerms (.finite 255) 85785 .exactZero (none)

def event85787 : Event := .predecessor (⟨.program ⟨257⟩, ⟨54256⟩⟩) 0 ⟨51276⟩ 85786

def event85788 : Event := .predecessor (⟨.program ⟨257⟩, ⟨54256⟩⟩) 1 ⟨54255⟩ 85655

def event85789 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨54256⟩⟩) (.sum [.predecessor 0 85787 .coefficient, .predecessor 1 85788 .coefficient])

def exact85790RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨16131⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨18980⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨22200⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨32220⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨51275⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨54255⟩⟩], []⟩, (1)⟩]

theorem exact85790RawTermsValid :
    exact85790RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event85790 : Event := .resultExact (⟨.program ⟨257⟩, ⟨54256⟩⟩) exact85790RawTerms (.finite 314) 85789 .exactZero (none)

def event85791 : Event := .predecessor (⟨.program ⟨257⟩, ⟨57236⟩⟩) 0 ⟨54256⟩ 85790

def event85792 : Event := .predecessor (⟨.program ⟨257⟩, ⟨57236⟩⟩) 1 ⟨57235⟩ 85632

def event85793 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨57236⟩⟩) (.sum [.predecessor 0 85791 .coefficient, .predecessor 1 85792 .coefficient])

def exact85794RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨16131⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨18980⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨22200⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨32220⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨51275⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨54255⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨57235⟩⟩], []⟩, (1)⟩]

theorem exact85794RawTermsValid :
    exact85794RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event85794 : Event := .resultExact (⟨.program ⟨257⟩, ⟨57236⟩⟩) exact85794RawTerms (.finite 374) 85793 .exactZero (none)

def event85795 : Event := .predecessor (⟨.program ⟨257⟩, ⟨60216⟩⟩) 0 ⟨57236⟩ 85794

def event85796 : Event := .predecessor (⟨.program ⟨257⟩, ⟨60216⟩⟩) 1 ⟨60215⟩ 85609

def event85797 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨60216⟩⟩) (.sum [.predecessor 0 85795 .coefficient, .predecessor 1 85796 .coefficient])

def exact85798RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨16131⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨18980⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨22200⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨32220⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨51275⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨54255⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨57235⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨60215⟩⟩], []⟩, (1)⟩]

theorem exact85798RawTermsValid :
    exact85798RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event85798 : Event := .resultExact (⟨.program ⟨257⟩, ⟨60216⟩⟩) exact85798RawTerms (.finite 435) 85797 .exactZero (none)

def event85799 : Event := .predecessor (⟨.program ⟨257⟩, ⟨63196⟩⟩) 0 ⟨60216⟩ 85798

def event85800 : Event := .predecessor (⟨.program ⟨257⟩, ⟨63196⟩⟩) 1 ⟨63195⟩ 85586

def event85801 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨63196⟩⟩) (.sum [.predecessor 0 85799 .coefficient, .predecessor 1 85800 .coefficient])

def exact85802RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨16131⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨18980⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨22200⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨32220⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨51275⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨54255⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨57235⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨60215⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨63195⟩⟩], []⟩, (1)⟩]

theorem exact85802RawTermsValid :
    exact85802RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event85802 : Event := .resultExact (⟨.program ⟨257⟩, ⟨63196⟩⟩) exact85802RawTerms (.finite 496) 85801 .exactZero (none)

def event85803 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67022⟩⟩) 0 ⟨63196⟩ 85802

def event85804 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67022⟩⟩) 1 ⟨67021⟩ 85563

def event85805 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67022⟩⟩) (.sum [.predecessor 0 85803 .coefficient, .predecessor 1 85804 .coefficient])

def exact85806RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨16131⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨18980⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨22200⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨32220⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨51275⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨54255⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨57235⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨60215⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨63195⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨67021⟩⟩], []⟩, (1)⟩]

theorem exact85806RawTermsValid :
    exact85806RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event85806 : Event := .resultExact (⟨.program ⟨257⟩, ⟨67022⟩⟩) exact85806RawTerms (.finite 558) 85805 .exactZero (none)

def event85807 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67023⟩⟩) 0 ⟨67022⟩ 85806

def event85808 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67023⟩⟩) 1 ⟨26697⟩ 85540

def event85809 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67023⟩⟩) (.sum [.predecessor 0 85807 .coefficient, .predecessor 1 85808 .coefficient])

def exact85810RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨16131⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨18980⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨22200⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨26697⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨32220⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨51275⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨54255⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨57235⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨60215⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨63195⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨67021⟩⟩], []⟩, (1)⟩]

theorem exact85810RawTermsValid :
    exact85810RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event85810 : Event := .resultExact (⟨.program ⟨257⟩, ⟨67023⟩⟩) exact85810RawTerms (.finite 620) 85809 .exactZero (none)

def event85811 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67024⟩⟩) 0 ⟨67023⟩ 85810

def event85812 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67024⟩⟩) 1 ⟨29377⟩ 85517

def event85813 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67024⟩⟩) (.sum [.predecessor 0 85811 .coefficient, .predecessor 1 85812 .coefficient])

def exact85814RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨16131⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨18980⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨22200⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨26697⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨29377⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨32220⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨51275⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨54255⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨57235⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨60215⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨63195⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨67021⟩⟩], []⟩, (1)⟩]

theorem exact85814RawTermsValid :
    exact85814RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event85814 : Event := .resultExact (⟨.program ⟨257⟩, ⟨67024⟩⟩) exact85814RawTerms (.finite 682) 85813 .exactZero (none)

def event85815 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67025⟩⟩) 0 ⟨67024⟩ 85814

def event85816 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67025⟩⟩) 1 ⟨35041⟩ 85494

def event85817 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67025⟩⟩) (.sum [.predecessor 0 85815 .coefficient, .predecessor 1 85816 .coefficient])

def exact85818RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨16131⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨18980⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨22200⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨26697⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨29377⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨32220⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨35041⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨51275⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨54255⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨57235⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨60215⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨63195⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨67021⟩⟩], []⟩, (1)⟩]

theorem exact85818RawTermsValid :
    exact85818RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event85818 : Event := .resultExact (⟨.program ⟨257⟩, ⟨67025⟩⟩) exact85818RawTerms (.finite 744) 85817 .exactZero (none)

def event85819 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67026⟩⟩) 0 ⟨67025⟩ 85818

def event85820 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67026⟩⟩) 1 ⟨37721⟩ 85471

def event85821 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67026⟩⟩) (.sum [.predecessor 0 85819 .coefficient, .predecessor 1 85820 .coefficient])

def exact85822RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨16131⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨18980⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨22200⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨26697⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨29377⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨32220⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨35041⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨37721⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨51275⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨54255⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨57235⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨60215⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨63195⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨67021⟩⟩], []⟩, (1)⟩]

theorem exact85822RawTermsValid :
    exact85822RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event85822 : Event := .resultExact (⟨.program ⟨257⟩, ⟨67026⟩⟩) exact85822RawTerms (.finite 807) 85821 .exactZero (none)

def event85823 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67027⟩⟩) 0 ⟨67026⟩ 85822

def event85824 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67027⟩⟩) 1 ⟨40397⟩ 85448

def event85825 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67027⟩⟩) (.sum [.predecessor 0 85823 .coefficient, .predecessor 1 85824 .coefficient])

def exact85826RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨16131⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨18980⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨22200⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨26697⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨29377⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨32220⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨35041⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨37721⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨40397⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨51275⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨54255⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨57235⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨60215⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨63195⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨67021⟩⟩], []⟩, (1)⟩]

theorem exact85826RawTermsValid :
    exact85826RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event85826 : Event := .resultExact (⟨.program ⟨257⟩, ⟨67027⟩⟩) exact85826RawTerms (.finite 870) 85825 .exactZero (none)

def event85827 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67028⟩⟩) 0 ⟨67027⟩ 85826

def event85828 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67028⟩⟩) 1 ⟨43077⟩ 85425

def event85829 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67028⟩⟩) (.sum [.predecessor 0 85827 .coefficient, .predecessor 1 85828 .coefficient])

def exact85830RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨16131⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨18980⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨22200⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨26697⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨29377⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨32220⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨35041⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨37721⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨40397⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨43077⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨51275⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨54255⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨57235⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨60215⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨63195⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨67021⟩⟩], []⟩, (1)⟩]

theorem exact85830RawTermsValid :
    exact85830RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event85830 : Event := .resultExact (⟨.program ⟨257⟩, ⟨67028⟩⟩) exact85830RawTerms (.finite 933) 85829 .exactZero (none)

def event85831 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67029⟩⟩) 0 ⟨67028⟩ 85830

def event85832 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67029⟩⟩) 1 ⟨45761⟩ 85402

def event85833 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67029⟩⟩) (.sum [.predecessor 0 85831 .coefficient, .predecessor 1 85832 .coefficient])

def exact85834RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨16131⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨18980⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨22200⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨26697⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨29377⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨32220⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨35041⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨37721⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨40397⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨43077⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨45761⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨51275⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨54255⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨57235⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨60215⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨63195⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨67021⟩⟩], []⟩, (1)⟩]

theorem exact85834RawTermsValid :
    exact85834RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event85834 : Event := .resultExact (⟨.program ⟨257⟩, ⟨67029⟩⟩) exact85834RawTerms (.finite 996) 85833 .exactZero (none)

def event85835 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67030⟩⟩) 0 ⟨67029⟩ 85834

def event85836 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67030⟩⟩) 1 ⟨48441⟩ 85379

def event85837 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67030⟩⟩) (.sum [.predecessor 0 85835 .coefficient, .predecessor 1 85836 .coefficient])

def exact85838RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨16131⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨18980⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨22200⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨26697⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨29377⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨32220⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨35041⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨37721⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨40397⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨43077⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨45761⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨48441⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨51275⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨54255⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨57235⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨60215⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨63195⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨67021⟩⟩], []⟩, (1)⟩]

theorem exact85838RawTermsValid :
    exact85838RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event85838 : Event := .resultExact (⟨.program ⟨257⟩, ⟨67030⟩⟩) exact85838RawTerms (.finite 1059) 85837 .exactZero (none)

def event85839 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67031⟩⟩) 0 ⟨67030⟩ 85838

def event85840 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67031⟩⟩) (.identity (.predecessor 0 85839 .coefficient))

def event85841 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨67031⟩⟩) (.finite 1059)

def event85842 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68865⟩⟩) 0 ⟨67031⟩ 85841

def event85843 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨68865⟩⟩) (.authority (.programFamilyFact))

def event85844 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨68865⟩⟩) (.finite 1152)

def event85845 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨7177⟩⟩) .missing

def event85846 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68866⟩⟩) 0 ⟨7177⟩ 85845

def event85847 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68866⟩⟩) 1 ⟨68865⟩ 85844

def event85848 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨68866⟩⟩) (.authority (.operator))

def exact85849RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨68866⟩⟩]⟩, (1)⟩]

theorem exact85849RawTermsValid :
    exact85849RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event85849 : Event := .resultExact (⟨.program ⟨257⟩, ⟨68866⟩⟩) exact85849RawTerms .large 85848 .exactZero (none)

def event85850 : Event := .predecessor (⟨.program ⟨257⟩, ⟨71437⟩⟩) 0 ⟨68866⟩ 85849

def event85851 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨71437⟩⟩) (.authority (.operator))

def exact85852RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨71437⟩⟩]⟩, (1)⟩]

theorem exact85852RawTermsValid :
    exact85852RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event85852 : Event := .resultExact (⟨.program ⟨257⟩, ⟨71437⟩⟩) exact85852RawTerms (.finite 8192) 85851 .exactZero (none)

def event85853 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨136⟩⟩) (.authority (.operator))

def event85854 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨136⟩⟩) .exactZero

def event85855 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69111⟩⟩) 0 ⟨67031⟩ 85841

def event85856 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69111⟩⟩) 1 ⟨136⟩ 85854

def event85857 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69111⟩⟩) (.sum [.predecessor 0 85855 .coefficient, .predecessor 1 85856 .coefficient])

def event85858 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨69111⟩⟩) (.finite 1059)

def event85859 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69112⟩⟩) 0 ⟨69111⟩ 85858

def event85860 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69112⟩⟩) (.identity (.predecessor 0 85859 .coefficient))

def exact85861RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨16131⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨18980⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨22200⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨26697⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨29377⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨32220⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨35041⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨37721⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨40397⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨43077⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨45761⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨48441⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨51275⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨54255⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨57235⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨60215⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨63195⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨67021⟩⟩], []⟩, (1)⟩]

theorem exact85861RawTermsValid :
    exact85861RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event85861 : Event := .resultExact (⟨.program ⟨257⟩, ⟨69112⟩⟩) exact85861RawTerms (.finite 1059) 85860 .exactZero (none)

def event85862 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6908⟩⟩) (.authority (.factStore))

def exact85863RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact85863RawTermsValid :
    exact85863RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event85863 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6908⟩⟩) exact85863RawTerms .large 85862 .exactZero (none)

def event85864 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69113⟩⟩) 0 ⟨6908⟩ 85863

def event85865 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69113⟩⟩) 1 ⟨69112⟩ 85861

def event85866 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69113⟩⟩) (.product (.predecessor 0 85864 .coefficient) (.predecessor 1 85865 .coefficient) (⟨false, false, none, none, none⟩))

def event85867 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨69113⟩⟩, .operator (⟨85863, 0⟩, ⟨85861, 11⟩), ⟨[⟨.program ⟨257⟩, ⟨48441⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def event85868 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨69113⟩⟩, .operator (⟨85863, 0⟩, ⟨85861, 10⟩), ⟨[⟨.program ⟨257⟩, ⟨45761⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def event85869 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨69113⟩⟩, .operator (⟨85863, 0⟩, ⟨85861, 9⟩), ⟨[⟨.program ⟨257⟩, ⟨43077⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def event85870 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨69113⟩⟩, .operator (⟨85863, 0⟩, ⟨85861, 8⟩), ⟨[⟨.program ⟨257⟩, ⟨40397⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def event85871 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨69113⟩⟩, .operator (⟨85863, 0⟩, ⟨85861, 7⟩), ⟨[⟨.program ⟨257⟩, ⟨37721⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def event85872 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨69113⟩⟩, .operator (⟨85863, 0⟩, ⟨85861, 6⟩), ⟨[⟨.program ⟨257⟩, ⟨35041⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def event85873 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨69113⟩⟩, .operator (⟨85863, 0⟩, ⟨85861, 4⟩), ⟨[⟨.program ⟨257⟩, ⟨29377⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def event85874 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨69113⟩⟩, .operator (⟨85863, 0⟩, ⟨85861, 3⟩), ⟨[⟨.program ⟨257⟩, ⟨26697⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def event85875 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨69113⟩⟩, .operator (⟨85863, 0⟩, ⟨85861, 17⟩), ⟨[⟨.program ⟨257⟩, ⟨67021⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def event85876 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨69113⟩⟩, .operator (⟨85863, 0⟩, ⟨85861, 16⟩), ⟨[⟨.program ⟨257⟩, ⟨63195⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def event85877 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨69113⟩⟩, .operator (⟨85863, 0⟩, ⟨85861, 15⟩), ⟨[⟨.program ⟨257⟩, ⟨60215⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def event85878 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨69113⟩⟩, .operator (⟨85863, 0⟩, ⟨85861, 14⟩), ⟨[⟨.program ⟨257⟩, ⟨57235⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def event85879 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨69113⟩⟩, .operator (⟨85863, 0⟩, ⟨85861, 13⟩), ⟨[⟨.program ⟨257⟩, ⟨54255⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def event85880 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨69113⟩⟩, .operator (⟨85863, 0⟩, ⟨85861, 12⟩), ⟨[⟨.program ⟨257⟩, ⟨51275⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def event85881 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨69113⟩⟩, .operator (⟨85863, 0⟩, ⟨85861, 5⟩), ⟨[⟨.program ⟨257⟩, ⟨32220⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def event85882 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨69113⟩⟩, .operator (⟨85863, 0⟩, ⟨85861, 2⟩), ⟨[⟨.program ⟨257⟩, ⟨22200⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def event85883 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨69113⟩⟩, .operator (⟨85863, 0⟩, ⟨85861, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨18980⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def event85884 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨69113⟩⟩, .operator (⟨85863, 0⟩, ⟨85861, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨16131⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact85885RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨16131⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨18980⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨22200⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨26697⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨29377⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨32220⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨35041⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨37721⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨40397⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨43077⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨45761⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨48441⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨51275⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨54255⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨57235⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨60215⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨63195⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨67021⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact85885RawTermsValid :
    exact85885RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event85885 : Event := .resultExact (⟨.program ⟨257⟩, ⟨69113⟩⟩) exact85885RawTerms .large 85866 .exactZero (none)

def event85886 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7232⟩⟩) 0 ⟨7177⟩ 85845

def event85887 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7232⟩⟩) (.authority (.operator))

def exact85888RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7232⟩⟩]⟩, (1)⟩]

theorem exact85888RawTermsValid :
    exact85888RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event85888 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7232⟩⟩) exact85888RawTerms .large 85887 .exactZero (none)

def event85889 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7230⟩⟩) 0 ⟨7177⟩ 85845

def event85890 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7230⟩⟩) (.authority (.operator))

def exact85891RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7230⟩⟩]⟩, (1)⟩]

theorem exact85891RawTermsValid :
    exact85891RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event85891 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7230⟩⟩) exact85891RawTerms .large 85890 .exactZero (none)

def event85892 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7228⟩⟩) 0 ⟨7177⟩ 85845

def event85893 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7228⟩⟩) (.authority (.operator))

def exact85894RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7228⟩⟩]⟩, (1)⟩]

theorem exact85894RawTermsValid :
    exact85894RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event85894 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7228⟩⟩) exact85894RawTerms .large 85893 .exactZero (none)

def event85895 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7226⟩⟩) 0 ⟨7177⟩ 85845

def event85896 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7226⟩⟩) (.authority (.operator))

def exact85897RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7226⟩⟩]⟩, (1)⟩]

theorem exact85897RawTermsValid :
    exact85897RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event85897 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7226⟩⟩) exact85897RawTerms .large 85896 .exactZero (none)

def event85898 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7224⟩⟩) 0 ⟨7177⟩ 85845

def event85899 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7224⟩⟩) (.authority (.operator))

def exact85900RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7224⟩⟩]⟩, (1)⟩]

theorem exact85900RawTermsValid :
    exact85900RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event85900 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7224⟩⟩) exact85900RawTerms .large 85899 .exactZero (none)

def event85901 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7222⟩⟩) 0 ⟨7177⟩ 85845

def event85902 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7222⟩⟩) (.authority (.operator))

def exact85903RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7222⟩⟩]⟩, (1)⟩]

theorem exact85903RawTermsValid :
    exact85903RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event85903 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7222⟩⟩) exact85903RawTerms .large 85902 .exactZero (none)

def event85904 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7220⟩⟩) 0 ⟨7177⟩ 85845

def event85905 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7220⟩⟩) (.authority (.operator))

def exact85906RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩]

theorem exact85906RawTermsValid :
    exact85906RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event85906 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7220⟩⟩) exact85906RawTerms .large 85905 .exactZero (none)

def event85907 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7218⟩⟩) 0 ⟨7177⟩ 85845

def event85908 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7218⟩⟩) (.authority (.operator))

def exact85909RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩]

theorem exact85909RawTermsValid :
    exact85909RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event85909 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7218⟩⟩) exact85909RawTerms .large 85908 .exactZero (none)

def event85910 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7216⟩⟩) 0 ⟨7177⟩ 85845

def event85911 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7216⟩⟩) (.authority (.operator))

def exact85912RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩]

theorem exact85912RawTermsValid :
    exact85912RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event85912 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7216⟩⟩) exact85912RawTerms .large 85911 .exactZero (none)

def event85913 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7214⟩⟩) 0 ⟨7177⟩ 85845

def event85914 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7214⟩⟩) (.authority (.operator))

def exact85915RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩]

theorem exact85915RawTermsValid :
    exact85915RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event85915 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7214⟩⟩) exact85915RawTerms .large 85914 .exactZero (none)

def event85916 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7212⟩⟩) 0 ⟨7177⟩ 85845

def event85917 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7212⟩⟩) (.authority (.operator))

def exact85918RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩]

theorem exact85918RawTermsValid :
    exact85918RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event85918 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7212⟩⟩) exact85918RawTerms .large 85917 .exactZero (none)

def event85919 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7210⟩⟩) 0 ⟨7177⟩ 85845

def event85920 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7210⟩⟩) (.authority (.operator))

def exact85921RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩]

theorem exact85921RawTermsValid :
    exact85921RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event85921 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7210⟩⟩) exact85921RawTerms .large 85920 .exactZero (none)

def event85922 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7208⟩⟩) 0 ⟨7177⟩ 85845

def event85923 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7208⟩⟩) (.authority (.operator))

def exact85924RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩]

theorem exact85924RawTermsValid :
    exact85924RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event85924 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7208⟩⟩) exact85924RawTerms .large 85923 .exactZero (none)

def event85925 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7206⟩⟩) 0 ⟨7177⟩ 85845

def event85926 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7206⟩⟩) (.authority (.operator))

def exact85927RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩]

theorem exact85927RawTermsValid :
    exact85927RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event85927 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7206⟩⟩) exact85927RawTerms .large 85926 .exactZero (none)

def event85928 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7204⟩⟩) 0 ⟨7177⟩ 85845

def event85929 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7204⟩⟩) (.authority (.operator))

def exact85930RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩]

theorem exact85930RawTermsValid :
    exact85930RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event85930 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7204⟩⟩) exact85930RawTerms .large 85929 .exactZero (none)

def event85931 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7202⟩⟩) 0 ⟨7177⟩ 85845

def event85932 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7202⟩⟩) (.authority (.operator))

def exact85933RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩]

theorem exact85933RawTermsValid :
    exact85933RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event85933 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7202⟩⟩) exact85933RawTerms .large 85932 .exactZero (none)

def event85934 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7200⟩⟩) 0 ⟨7177⟩ 85845

def event85935 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7200⟩⟩) (.authority (.operator))

def exact85936RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩]

theorem exact85936RawTermsValid :
    exact85936RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event85936 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7200⟩⟩) exact85936RawTerms .large 85935 .exactZero (none)

def event85937 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7198⟩⟩) 0 ⟨7177⟩ 85845

def event85938 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7198⟩⟩) (.authority (.operator))

def exact85939RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩]

theorem exact85939RawTermsValid :
    exact85939RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event85939 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7198⟩⟩) exact85939RawTerms .large 85938 .exactZero (none)

def event85940 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7309⟩⟩) 0 ⟨7198⟩ 85939

def event85941 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7309⟩⟩) 1 ⟨7200⟩ 85936

def event85942 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7309⟩⟩) (.sum [.predecessor 0 85940 .coefficient, .predecessor 1 85941 .coefficient])

def exact85943RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩]

theorem exact85943RawTermsValid :
    exact85943RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event85943 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7309⟩⟩) exact85943RawTerms .large 85942 .exactZero (none)

def event85944 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7310⟩⟩) 0 ⟨7309⟩ 85943

def event85945 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7310⟩⟩) 1 ⟨7202⟩ 85933

def event85946 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7310⟩⟩) (.sum [.predecessor 0 85944 .coefficient, .predecessor 1 85945 .coefficient])

def exact85947RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩]

theorem exact85947RawTermsValid :
    exact85947RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event85947 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7310⟩⟩) exact85947RawTerms .large 85946 .exactZero (none)

def event85948 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7311⟩⟩) 0 ⟨7310⟩ 85947

def event85949 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7311⟩⟩) 1 ⟨7204⟩ 85930

def event85950 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7311⟩⟩) (.sum [.predecessor 0 85948 .coefficient, .predecessor 1 85949 .coefficient])

def exact85951RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩]

theorem exact85951RawTermsValid :
    exact85951RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event85951 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7311⟩⟩) exact85951RawTerms .large 85950 .exactZero (none)

def event85952 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7312⟩⟩) 0 ⟨7311⟩ 85951

def event85953 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7312⟩⟩) 1 ⟨7206⟩ 85927

def event85954 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7312⟩⟩) (.sum [.predecessor 0 85952 .coefficient, .predecessor 1 85953 .coefficient])

def exact85955RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩]

theorem exact85955RawTermsValid :
    exact85955RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event85955 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7312⟩⟩) exact85955RawTerms .large 85954 .exactZero (none)

def event85956 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7313⟩⟩) 0 ⟨7312⟩ 85955

def event85957 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7313⟩⟩) 1 ⟨7208⟩ 85924

def event85958 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7313⟩⟩) (.sum [.predecessor 0 85956 .coefficient, .predecessor 1 85957 .coefficient])

def exact85959RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩]

theorem exact85959RawTermsValid :
    exact85959RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event85959 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7313⟩⟩) exact85959RawTerms .large 85958 .exactZero (none)

def event85960 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7314⟩⟩) 0 ⟨7313⟩ 85959

def event85961 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7314⟩⟩) 1 ⟨7210⟩ 85921

def event85962 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7314⟩⟩) (.sum [.predecessor 0 85960 .coefficient, .predecessor 1 85961 .coefficient])

def exact85963RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩]

theorem exact85963RawTermsValid :
    exact85963RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event85963 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7314⟩⟩) exact85963RawTerms .large 85962 .exactZero (none)

def event85964 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7315⟩⟩) 0 ⟨7314⟩ 85963

def event85965 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7315⟩⟩) 1 ⟨7212⟩ 85918

def event85966 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7315⟩⟩) (.sum [.predecessor 0 85964 .coefficient, .predecessor 1 85965 .coefficient])

def exact85967RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩]

theorem exact85967RawTermsValid :
    exact85967RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event85967 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7315⟩⟩) exact85967RawTerms .large 85966 .exactZero (none)

def event85968 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7316⟩⟩) 0 ⟨7315⟩ 85967

def event85969 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7316⟩⟩) 1 ⟨7214⟩ 85915

def event85970 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7316⟩⟩) (.sum [.predecessor 0 85968 .coefficient, .predecessor 1 85969 .coefficient])

def exact85971RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩]

theorem exact85971RawTermsValid :
    exact85971RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event85971 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7316⟩⟩) exact85971RawTerms .large 85970 .exactZero (none)

def event85972 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7317⟩⟩) 0 ⟨7316⟩ 85971

def event85973 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7317⟩⟩) 1 ⟨7216⟩ 85912

def event85974 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7317⟩⟩) (.sum [.predecessor 0 85972 .coefficient, .predecessor 1 85973 .coefficient])

def exact85975RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩]

theorem exact85975RawTermsValid :
    exact85975RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event85975 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7317⟩⟩) exact85975RawTerms .large 85974 .exactZero (none)

def event85976 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7318⟩⟩) 0 ⟨7317⟩ 85975

def event85977 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7318⟩⟩) 1 ⟨7218⟩ 85909

def event85978 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7318⟩⟩) (.sum [.predecessor 0 85976 .coefficient, .predecessor 1 85977 .coefficient])

def exact85979RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩]

theorem exact85979RawTermsValid :
    exact85979RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event85979 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7318⟩⟩) exact85979RawTerms .large 85978 .exactZero (none)

def event85980 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7319⟩⟩) 0 ⟨7318⟩ 85979

def event85981 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7319⟩⟩) 1 ⟨7220⟩ 85906

def event85982 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7319⟩⟩) (.sum [.predecessor 0 85980 .coefficient, .predecessor 1 85981 .coefficient])

def exact85983RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩]

theorem exact85983RawTermsValid :
    exact85983RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event85983 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7319⟩⟩) exact85983RawTerms .large 85982 .exactZero (none)

def event85984 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7320⟩⟩) 0 ⟨7319⟩ 85983

def event85985 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7320⟩⟩) 1 ⟨7222⟩ 85903

def event85986 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7320⟩⟩) (.sum [.predecessor 0 85984 .coefficient, .predecessor 1 85985 .coefficient])

def exact85987RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7222⟩⟩]⟩, (1)⟩]

theorem exact85987RawTermsValid :
    exact85987RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event85987 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7320⟩⟩) exact85987RawTerms .large 85986 .exactZero (none)

def event85988 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7321⟩⟩) 0 ⟨7320⟩ 85987

def event85989 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7321⟩⟩) 1 ⟨7224⟩ 85900

def event85990 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7321⟩⟩) (.sum [.predecessor 0 85988 .coefficient, .predecessor 1 85989 .coefficient])

def exact85991RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7222⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7224⟩⟩]⟩, (1)⟩]

theorem exact85991RawTermsValid :
    exact85991RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event85991 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7321⟩⟩) exact85991RawTerms .large 85990 .exactZero (none)

def event85992 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7322⟩⟩) 0 ⟨7321⟩ 85991

def event85993 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7322⟩⟩) 1 ⟨7226⟩ 85897

def event85994 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7322⟩⟩) (.sum [.predecessor 0 85992 .coefficient, .predecessor 1 85993 .coefficient])

def exact85995RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7222⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7224⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7226⟩⟩]⟩, (1)⟩]

theorem exact85995RawTermsValid :
    exact85995RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event85995 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7322⟩⟩) exact85995RawTerms .large 85994 .exactZero (none)

def event85996 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7323⟩⟩) 0 ⟨7322⟩ 85995

def event85997 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7323⟩⟩) 1 ⟨7228⟩ 85894

def event85998 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7323⟩⟩) (.sum [.predecessor 0 85996 .coefficient, .predecessor 1 85997 .coefficient])

def exact85999RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7222⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7224⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7226⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7228⟩⟩]⟩, (1)⟩]

theorem exact85999RawTermsValid :
    exact85999RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event85999 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7323⟩⟩) exact85999RawTerms .large 85998 .exactZero (none)

def event86000 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7324⟩⟩) 0 ⟨7323⟩ 85999

def event86001 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7324⟩⟩) 1 ⟨7230⟩ 85891

def event86002 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7324⟩⟩) (.sum [.predecessor 0 86000 .coefficient, .predecessor 1 86001 .coefficient])

def exact86003RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7222⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7224⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7226⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7228⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7230⟩⟩]⟩, (1)⟩]

theorem exact86003RawTermsValid :
    exact86003RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86003 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7324⟩⟩) exact86003RawTerms .large 86002 .exactZero (none)

def event86004 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7325⟩⟩) 0 ⟨7324⟩ 86003

def event86005 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7325⟩⟩) 1 ⟨7232⟩ 85888

def event86006 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7325⟩⟩) (.sum [.predecessor 0 86004 .coefficient, .predecessor 1 86005 .coefficient])

def exact86007RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7222⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7224⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7226⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7228⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7230⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7232⟩⟩]⟩, (1)⟩]

theorem exact86007RawTermsValid :
    exact86007RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86007 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7325⟩⟩) exact86007RawTerms .large 86006 .exactZero (none)

def event86008 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69114⟩⟩) 0 ⟨7325⟩ 86007

def event86009 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69114⟩⟩) 1 ⟨69113⟩ 85885

def event86010 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69114⟩⟩) (.sum [.predecessor 0 86008 .coefficient, .predecessor 1 86009 .coefficient])

def exact86011RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7222⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7224⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7226⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7228⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7230⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7232⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨16131⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨18980⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨22200⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨26697⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨29377⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨32220⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨35041⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨37721⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨40397⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨43077⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨45761⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨48441⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨51275⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨54255⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨57235⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨60215⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨63195⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨67021⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact86011RawTermsValid :
    exact86011RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86011 : Event := .resultExact (⟨.program ⟨257⟩, ⟨69114⟩⟩) exact86011RawTerms .large 86010 .exactZero (none)

def event86012 : Event := .predecessor (⟨.program ⟨257⟩, ⟨71438⟩⟩) 0 ⟨69114⟩ 86011

def event86013 : Event := .predecessor (⟨.program ⟨257⟩, ⟨71438⟩⟩) 1 ⟨71437⟩ 85852

def event86014 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨71438⟩⟩) (.product (.predecessor 0 86012 .coefficient) (.predecessor 1 86013 .coefficient) (⟨false, false, none, none, none⟩))

def event86015 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71438⟩⟩, .operator (⟨86011, 17⟩, ⟨85852, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7232⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩]⟩, (1)⟩)

def eventLeaf5360 : Array AnnotatedEvent := #[
  { event := event85760
    frameStart := 85336 },
  { event := event85761
    frameStart := 85336 },
  { event := event85762
    frameStart := 85336 },
  { event := event85763
    frameStart := 85336 },
  { event := event85764
    frameStart := 85336 },
  { event := event85765
    frameStart := 85336 },
  { event := event85766
    frameStart := 85336 },
  { event := event85767
    frameStart := 85336 },
  { event := event85768
    frameStart := 85336 },
  { event := event85769
    frameStart := 85336 },
  { event := event85770
    frameStart := 85336 },
  { event := event85771
    frameStart := 85336 },
  { event := event85772
    frameStart := 85336 },
  { event := event85773
    frameStart := 85336 },
  { event := event85774
    frameStart := 85336 },
  { event := event85775
    frameStart := 85336 }
]

def eventLeaf5361 : Array AnnotatedEvent := #[
  { event := event85776
    frameStart := 85336 },
  { event := event85777
    frameStart := 85336 },
  { event := event85778
    frameStart := 85336 },
  { event := event85779
    frameStart := 85336 },
  { event := event85780
    frameStart := 85336 },
  { event := event85781
    frameStart := 85336 },
  { event := event85782
    frameStart := 85336 },
  { event := event85783
    frameStart := 85336 },
  { event := event85784
    frameStart := 85336 },
  { event := event85785
    frameStart := 85336 },
  { event := event85786
    frameStart := 85336 },
  { event := event85787
    frameStart := 85336 },
  { event := event85788
    frameStart := 85336 },
  { event := event85789
    frameStart := 85336 },
  { event := event85790
    frameStart := 85336 },
  { event := event85791
    frameStart := 85336 }
]

def eventLeaf5362 : Array AnnotatedEvent := #[
  { event := event85792
    frameStart := 85336 },
  { event := event85793
    frameStart := 85336 },
  { event := event85794
    frameStart := 85336 },
  { event := event85795
    frameStart := 85336 },
  { event := event85796
    frameStart := 85336 },
  { event := event85797
    frameStart := 85336 },
  { event := event85798
    frameStart := 85336 },
  { event := event85799
    frameStart := 85336 },
  { event := event85800
    frameStart := 85336 },
  { event := event85801
    frameStart := 85336 },
  { event := event85802
    frameStart := 85336 },
  { event := event85803
    frameStart := 85336 },
  { event := event85804
    frameStart := 85336 },
  { event := event85805
    frameStart := 85336 },
  { event := event85806
    frameStart := 85336 },
  { event := event85807
    frameStart := 85336 }
]

def eventLeaf5363 : Array AnnotatedEvent := #[
  { event := event85808
    frameStart := 85336 },
  { event := event85809
    frameStart := 85336 },
  { event := event85810
    frameStart := 85336 },
  { event := event85811
    frameStart := 85336 },
  { event := event85812
    frameStart := 85336 },
  { event := event85813
    frameStart := 85336 },
  { event := event85814
    frameStart := 85336 },
  { event := event85815
    frameStart := 85336 },
  { event := event85816
    frameStart := 85336 },
  { event := event85817
    frameStart := 85336 },
  { event := event85818
    frameStart := 85336 },
  { event := event85819
    frameStart := 85336 },
  { event := event85820
    frameStart := 85336 },
  { event := event85821
    frameStart := 85336 },
  { event := event85822
    frameStart := 85336 },
  { event := event85823
    frameStart := 85336 }
]

def eventLeaf5364 : Array AnnotatedEvent := #[
  { event := event85824
    frameStart := 85336 },
  { event := event85825
    frameStart := 85336 },
  { event := event85826
    frameStart := 85336 },
  { event := event85827
    frameStart := 85336 },
  { event := event85828
    frameStart := 85336 },
  { event := event85829
    frameStart := 85336 },
  { event := event85830
    frameStart := 85336 },
  { event := event85831
    frameStart := 85336 },
  { event := event85832
    frameStart := 85336 },
  { event := event85833
    frameStart := 85336 },
  { event := event85834
    frameStart := 85336 },
  { event := event85835
    frameStart := 85336 },
  { event := event85836
    frameStart := 85336 },
  { event := event85837
    frameStart := 85336 },
  { event := event85838
    frameStart := 85336 },
  { event := event85839
    frameStart := 85336 }
]

def eventLeaf5365 : Array AnnotatedEvent := #[
  { event := event85840
    frameStart := 85336 },
  { event := event85841
    frameStart := 85336 },
  { event := event85842
    frameStart := 85336 },
  { event := event85843
    frameStart := 85336 },
  { event := event85844
    frameStart := 85336 },
  { event := event85845
    frameStart := 85336 },
  { event := event85846
    frameStart := 85336 },
  { event := event85847
    frameStart := 85336 },
  { event := event85848
    frameStart := 85336 },
  { event := event85849
    frameStart := 85336 },
  { event := event85850
    frameStart := 85336 },
  { event := event85851
    frameStart := 85336 },
  { event := event85852
    frameStart := 85336 },
  { event := event85853
    frameStart := 85336 },
  { event := event85854
    frameStart := 85336 },
  { event := event85855
    frameStart := 85336 }
]

def eventLeaf5366 : Array AnnotatedEvent := #[
  { event := event85856
    frameStart := 85336 },
  { event := event85857
    frameStart := 85336 },
  { event := event85858
    frameStart := 85336 },
  { event := event85859
    frameStart := 85336 },
  { event := event85860
    frameStart := 85336 },
  { event := event85861
    frameStart := 85336 },
  { event := event85862
    frameStart := 85336 },
  { event := event85863
    frameStart := 85336 },
  { event := event85864
    frameStart := 85336 },
  { event := event85865
    frameStart := 85336 },
  { event := event85866
    frameStart := 85336 },
  { event := event85867
    frameStart := 85336 },
  { event := event85868
    frameStart := 85336 },
  { event := event85869
    frameStart := 85336 },
  { event := event85870
    frameStart := 85336 },
  { event := event85871
    frameStart := 85336 }
]

def eventLeaf5367 : Array AnnotatedEvent := #[
  { event := event85872
    frameStart := 85336 },
  { event := event85873
    frameStart := 85336 },
  { event := event85874
    frameStart := 85336 },
  { event := event85875
    frameStart := 85336 },
  { event := event85876
    frameStart := 85336 },
  { event := event85877
    frameStart := 85336 },
  { event := event85878
    frameStart := 85336 },
  { event := event85879
    frameStart := 85336 },
  { event := event85880
    frameStart := 85336 },
  { event := event85881
    frameStart := 85336 },
  { event := event85882
    frameStart := 85336 },
  { event := event85883
    frameStart := 85336 },
  { event := event85884
    frameStart := 85336 },
  { event := event85885
    frameStart := 85336 },
  { event := event85886
    frameStart := 85336 },
  { event := event85887
    frameStart := 85336 }
]

def eventLeaf5368 : Array AnnotatedEvent := #[
  { event := event85888
    frameStart := 85336 },
  { event := event85889
    frameStart := 85336 },
  { event := event85890
    frameStart := 85336 },
  { event := event85891
    frameStart := 85336 },
  { event := event85892
    frameStart := 85336 },
  { event := event85893
    frameStart := 85336 },
  { event := event85894
    frameStart := 85336 },
  { event := event85895
    frameStart := 85336 },
  { event := event85896
    frameStart := 85336 },
  { event := event85897
    frameStart := 85336 },
  { event := event85898
    frameStart := 85336 },
  { event := event85899
    frameStart := 85336 },
  { event := event85900
    frameStart := 85336 },
  { event := event85901
    frameStart := 85336 },
  { event := event85902
    frameStart := 85336 },
  { event := event85903
    frameStart := 85336 }
]

def eventLeaf5369 : Array AnnotatedEvent := #[
  { event := event85904
    frameStart := 85336 },
  { event := event85905
    frameStart := 85336 },
  { event := event85906
    frameStart := 85336 },
  { event := event85907
    frameStart := 85336 },
  { event := event85908
    frameStart := 85336 },
  { event := event85909
    frameStart := 85336 },
  { event := event85910
    frameStart := 85336 },
  { event := event85911
    frameStart := 85336 },
  { event := event85912
    frameStart := 85336 },
  { event := event85913
    frameStart := 85336 },
  { event := event85914
    frameStart := 85336 },
  { event := event85915
    frameStart := 85336 },
  { event := event85916
    frameStart := 85336 },
  { event := event85917
    frameStart := 85336 },
  { event := event85918
    frameStart := 85336 },
  { event := event85919
    frameStart := 85336 }
]

def eventLeaf5370 : Array AnnotatedEvent := #[
  { event := event85920
    frameStart := 85336 },
  { event := event85921
    frameStart := 85336 },
  { event := event85922
    frameStart := 85336 },
  { event := event85923
    frameStart := 85336 },
  { event := event85924
    frameStart := 85336 },
  { event := event85925
    frameStart := 85336 },
  { event := event85926
    frameStart := 85336 },
  { event := event85927
    frameStart := 85336 },
  { event := event85928
    frameStart := 85336 },
  { event := event85929
    frameStart := 85336 },
  { event := event85930
    frameStart := 85336 },
  { event := event85931
    frameStart := 85336 },
  { event := event85932
    frameStart := 85336 },
  { event := event85933
    frameStart := 85336 },
  { event := event85934
    frameStart := 85336 },
  { event := event85935
    frameStart := 85336 }
]

def eventLeaf5371 : Array AnnotatedEvent := #[
  { event := event85936
    frameStart := 85336 },
  { event := event85937
    frameStart := 85336 },
  { event := event85938
    frameStart := 85336 },
  { event := event85939
    frameStart := 85336 },
  { event := event85940
    frameStart := 85336 },
  { event := event85941
    frameStart := 85336 },
  { event := event85942
    frameStart := 85336 },
  { event := event85943
    frameStart := 85336 },
  { event := event85944
    frameStart := 85336 },
  { event := event85945
    frameStart := 85336 },
  { event := event85946
    frameStart := 85336 },
  { event := event85947
    frameStart := 85336 },
  { event := event85948
    frameStart := 85336 },
  { event := event85949
    frameStart := 85336 },
  { event := event85950
    frameStart := 85336 },
  { event := event85951
    frameStart := 85336 }
]

def eventLeaf5372 : Array AnnotatedEvent := #[
  { event := event85952
    frameStart := 85336 },
  { event := event85953
    frameStart := 85336 },
  { event := event85954
    frameStart := 85336 },
  { event := event85955
    frameStart := 85336 },
  { event := event85956
    frameStart := 85336 },
  { event := event85957
    frameStart := 85336 },
  { event := event85958
    frameStart := 85336 },
  { event := event85959
    frameStart := 85336 },
  { event := event85960
    frameStart := 85336 },
  { event := event85961
    frameStart := 85336 },
  { event := event85962
    frameStart := 85336 },
  { event := event85963
    frameStart := 85336 },
  { event := event85964
    frameStart := 85336 },
  { event := event85965
    frameStart := 85336 },
  { event := event85966
    frameStart := 85336 },
  { event := event85967
    frameStart := 85336 }
]

def eventLeaf5373 : Array AnnotatedEvent := #[
  { event := event85968
    frameStart := 85336 },
  { event := event85969
    frameStart := 85336 },
  { event := event85970
    frameStart := 85336 },
  { event := event85971
    frameStart := 85336 },
  { event := event85972
    frameStart := 85336 },
  { event := event85973
    frameStart := 85336 },
  { event := event85974
    frameStart := 85336 },
  { event := event85975
    frameStart := 85336 },
  { event := event85976
    frameStart := 85336 },
  { event := event85977
    frameStart := 85336 },
  { event := event85978
    frameStart := 85336 },
  { event := event85979
    frameStart := 85336 },
  { event := event85980
    frameStart := 85336 },
  { event := event85981
    frameStart := 85336 },
  { event := event85982
    frameStart := 85336 },
  { event := event85983
    frameStart := 85336 }
]

def eventLeaf5374 : Array AnnotatedEvent := #[
  { event := event85984
    frameStart := 85336 },
  { event := event85985
    frameStart := 85336 },
  { event := event85986
    frameStart := 85336 },
  { event := event85987
    frameStart := 85336 },
  { event := event85988
    frameStart := 85336 },
  { event := event85989
    frameStart := 85336 },
  { event := event85990
    frameStart := 85336 },
  { event := event85991
    frameStart := 85336 },
  { event := event85992
    frameStart := 85336 },
  { event := event85993
    frameStart := 85336 },
  { event := event85994
    frameStart := 85336 },
  { event := event85995
    frameStart := 85336 },
  { event := event85996
    frameStart := 85336 },
  { event := event85997
    frameStart := 85336 },
  { event := event85998
    frameStart := 85336 },
  { event := event85999
    frameStart := 85336 }
]

def eventLeaf5375 : Array AnnotatedEvent := #[
  { event := event86000
    frameStart := 85336 },
  { event := event86001
    frameStart := 85336 },
  { event := event86002
    frameStart := 85336 },
  { event := event86003
    frameStart := 85336 },
  { event := event86004
    frameStart := 85336 },
  { event := event86005
    frameStart := 85336 },
  { event := event86006
    frameStart := 85336 },
  { event := event86007
    frameStart := 85336 },
  { event := event86008
    frameStart := 85336 },
  { event := event86009
    frameStart := 85336 },
  { event := event86010
    frameStart := 85336 },
  { event := event86011
    frameStart := 85336 },
  { event := event86012
    frameStart := 85336 },
  { event := event86013
    frameStart := 85336 },
  { event := event86014
    frameStart := 85336 },
  { event := event86015
    frameStart := 85336 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events335
