import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard000
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard051
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard052
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard163
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard164
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard236
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard238
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard239
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard262

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace ResidualResult35645
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult35640.actual selector witness +
    ResidualResult32141.actual selector witness
end ResidualResult35645

namespace ResidualResult35650
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult35645.actual selector witness +
    ResidualResult31929.actual selector witness
end ResidualResult35650

namespace ResidualResult35655
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult35650.actual selector witness -
    ResidualResult31717.actual selector witness
end ResidualResult35655

namespace ResidualResult35657
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 35657
end ResidualResult35657

namespace ResidualResult35662
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult27.actual selector witness *
    ResidualResult6004.actual selector witness
end ResidualResult35662

namespace ResidualResult35666
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult35662.actual selector witness -
    ResidualResult21420.actual selector witness
end ResidualResult35666

namespace ResidualResult35672
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult35666.actual selector witness +
    ResidualResult35657.actual selector witness
end ResidualResult35672

namespace ResidualResult35700
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult35672.actual selector witness *
    ResidualResult5961.actual selector witness
end ResidualResult35700

namespace ResidualResult35724
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult35700.actual selector witness +
    ResidualResult35655.actual selector witness
end ResidualResult35724

namespace ResidualResult35788
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult35724.actual selector witness *
    ResidualResult6001.actual selector witness
end ResidualResult35788

namespace ResidualResult35812
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult35788.actual selector witness +
    ResidualResult21385.actual selector witness
end ResidualResult35812

namespace ResidualResult35876
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult35812.actual selector witness *
    ResidualResult5991.actual selector witness
end ResidualResult35876

namespace ResidualResult35878
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 35878
end ResidualResult35878

namespace ResidualResult35899
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 35899
end ResidualResult35899

namespace ResidualResult35904
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult35899.actual selector witness *
    ResidualResult2.actual selector witness
end ResidualResult35904

namespace ResidualResult35915
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 35915
end ResidualResult35915

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert
