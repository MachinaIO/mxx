import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard027
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard117
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard118
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard465

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace ResidualResult72523
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 72523
end ResidualResult72523

namespace ResidualResult72526
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 72526
end ResidualResult72526

namespace ResidualResult72533
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 72533
end ResidualResult72533

namespace ResidualResult72536
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 72536
end ResidualResult72536

namespace ResidualResult72541
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult3431.actual selector witness *
    ResidualResult65295.actual selector witness
end ResidualResult72541

namespace ResidualResult72546
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult65165.actual selector witness *
    ResidualResult13987.actual selector witness
end ResidualResult72546

namespace ResidualResult72550
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult72546.actual selector witness -
    ResidualResult72541.actual selector witness
end ResidualResult72550

namespace ResidualResult72556
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult72550.actual selector witness +
    ResidualResult13979.actual selector witness
end ResidualResult72556

namespace ResidualResult72564
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult72556.actual selector witness *
    ResidualResult3434.actual selector witness
end ResidualResult72564

namespace ResidualResult72569
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult3434.actual selector witness *
    ResidualResult65295.actual selector witness
end ResidualResult72569

namespace ResidualResult72574
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult65165.actual selector witness *
    ResidualResult14028.actual selector witness
end ResidualResult72574

namespace ResidualResult72578
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult72574.actual selector witness -
    ResidualResult72569.actual selector witness
end ResidualResult72578

namespace ResidualResult72584
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult72578.actual selector witness +
    ResidualResult14020.actual selector witness
end ResidualResult72584

namespace ResidualResult72594
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult72584.actual selector witness *
    ResidualResult14017.actual selector witness
end ResidualResult72594

namespace ResidualResult72600
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult72594.actual selector witness +
    ResidualResult72564.actual selector witness
end ResidualResult72600

namespace ResidualResult72610
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult72600.actual selector witness *
    ResidualResult72536.actual selector witness
end ResidualResult72610

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert
