import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard027
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard125
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard126
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard465
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard466
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard527

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace ResidualResult73453
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult73425.actual selector witness *
    ResidualResult73448.actual selector witness
end ResidualResult73453

namespace ResidualResult73456
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 73456
end ResidualResult73456

namespace ResidualResult73460
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult73456.actual selector witness -
    ResidualResult73453.actual selector witness
end ResidualResult73460

namespace ResidualResult73464
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult73460.actual selector witness -
    ResidualResult73445.actual selector witness
end ResidualResult73464

namespace ResidualResult73473
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult65387.actual selector witness *
    ResidualResult73302.actual selector witness
end ResidualResult73473

namespace ResidualResult73480
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult73473.actual selector witness +
    ResidualResult73295.actual selector witness
end ResidualResult73480

namespace ResidualResult73487
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 73487
end ResidualResult73487

namespace ResidualResult73490
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 73490
end ResidualResult73490

namespace ResidualResult73497
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 73497
end ResidualResult73497

namespace ResidualResult73500
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 73500
end ResidualResult73500

namespace ResidualResult73505
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult3477.actual selector witness *
    ResidualResult65295.actual selector witness
end ResidualResult73505

namespace ResidualResult73510
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult65165.actual selector witness *
    ResidualResult14989.actual selector witness
end ResidualResult73510

namespace ResidualResult73514
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult73510.actual selector witness -
    ResidualResult73505.actual selector witness
end ResidualResult73514

namespace ResidualResult73520
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult73514.actual selector witness +
    ResidualResult14981.actual selector witness
end ResidualResult73520

namespace ResidualResult73528
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult73520.actual selector witness *
    ResidualResult3480.actual selector witness
end ResidualResult73528

namespace ResidualResult73533
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult3480.actual selector witness *
    ResidualResult65295.actual selector witness
end ResidualResult73533

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert
