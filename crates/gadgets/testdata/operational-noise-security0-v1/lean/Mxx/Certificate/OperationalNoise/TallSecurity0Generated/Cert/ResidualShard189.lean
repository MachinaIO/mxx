import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard165
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard188

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace ResidualResult24728
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 24728
end ResidualResult24728

namespace ResidualResult24730
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 24730
end ResidualResult24730

namespace ResidualResult24735
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult24730.actual selector witness *
    ResidualResult24728.actual selector witness
end ResidualResult24735

namespace ResidualResult24738
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 24738
end ResidualResult24738

namespace ResidualResult24742
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult24738.actual selector witness -
    ResidualResult24735.actual selector witness
end ResidualResult24742

namespace ResidualResult24750
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult24742.actual selector witness *
    ResidualResult24719.actual selector witness
end ResidualResult24750

namespace ResidualResult24753
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 24753
end ResidualResult24753

namespace ResidualResult24758
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult24730.actual selector witness *
    ResidualResult24753.actual selector witness
end ResidualResult24758

namespace ResidualResult24761
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 24761
end ResidualResult24761

namespace ResidualResult24765
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult24761.actual selector witness -
    ResidualResult24758.actual selector witness
end ResidualResult24765

namespace ResidualResult24769
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult24765.actual selector witness -
    ResidualResult24750.actual selector witness
end ResidualResult24769

namespace ResidualResult24778
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult21512.actual selector witness *
    ResidualResult24607.actual selector witness
end ResidualResult24778

namespace ResidualResult24785
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult24778.actual selector witness +
    ResidualResult24600.actual selector witness
end ResidualResult24785

namespace ResidualResult24792
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 24792
end ResidualResult24792

namespace ResidualResult24795
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 24795
end ResidualResult24795

namespace ResidualResult24802
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 24802
end ResidualResult24802

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert
