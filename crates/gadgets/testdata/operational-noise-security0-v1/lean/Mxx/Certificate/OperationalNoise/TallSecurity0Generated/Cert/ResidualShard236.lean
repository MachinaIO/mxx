import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard044
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard165
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard231
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard233
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard234
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard235

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace ResidualResult31508
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult31504.actual selector witness +
    ResidualResult31417.actual selector witness
end ResidualResult31508

namespace ResidualResult31512
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult31508.actual selector witness +
    ResidualResult31414.actual selector witness
end ResidualResult31512

namespace ResidualResult31516
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult31512.actual selector witness +
    ResidualResult31411.actual selector witness
end ResidualResult31516

namespace ResidualResult31520
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult31516.actual selector witness +
    ResidualResult31408.actual selector witness
end ResidualResult31520

namespace ResidualResult31524
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult31520.actual selector witness +
    ResidualResult31405.actual selector witness
end ResidualResult31524

namespace ResidualResult31528
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult31524.actual selector witness -
    ResidualResult31402.actual selector witness
end ResidualResult31528

namespace ResidualResult31604
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult31528.actual selector witness *
    ResidualResult31369.actual selector witness
end ResidualResult31604

namespace ResidualResult31607
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 31607
end ResidualResult31607

namespace ResidualResult31612
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult31380.actual selector witness *
    ResidualResult31607.actual selector witness
end ResidualResult31612

namespace ResidualResult31615
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 31615
end ResidualResult31615

namespace ResidualResult31619
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult31615.actual selector witness -
    ResidualResult31612.actual selector witness
end ResidualResult31619

namespace ResidualResult31623
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult31619.actual selector witness -
    ResidualResult31604.actual selector witness
end ResidualResult31623

namespace ResidualResult31666
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult21512.actual selector witness *
    ResidualResult30257.actual selector witness
end ResidualResult31666

namespace ResidualResult31707
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult31666.actual selector witness +
    ResidualResult30250.actual selector witness
end ResidualResult31707

namespace ResidualResult31717
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult31707.actual selector witness *
    ResidualResult5499.actual selector witness
end ResidualResult31717

namespace ResidualResult31721
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 31721
end ResidualResult31721

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert
