import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard165
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard182
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard183

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace ResidualResult24038
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult24033.actual selector witness *
    ResidualResult24031.actual selector witness
end ResidualResult24038

namespace ResidualResult24043
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 24043
end ResidualResult24043

namespace ResidualResult24049
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 24049
end ResidualResult24049

namespace ResidualResult24053
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 24053
end ResidualResult24053

namespace ResidualResult24056
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 24056
end ResidualResult24056

namespace ResidualResult24061
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult24056.actual selector witness *
    ResidualResult24053.actual selector witness
end ResidualResult24061

namespace ResidualResult24065
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult24061.actual selector witness -
    ResidualResult24038.actual selector witness
end ResidualResult24065

namespace ResidualResult24073
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult24065.actual selector witness *
    ResidualResult24022.actual selector witness
end ResidualResult24073

namespace ResidualResult24076
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 24076
end ResidualResult24076

namespace ResidualResult24081
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult24033.actual selector witness *
    ResidualResult24076.actual selector witness
end ResidualResult24081

namespace ResidualResult24084
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 24084
end ResidualResult24084

namespace ResidualResult24088
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult24084.actual selector witness -
    ResidualResult24081.actual selector witness
end ResidualResult24088

namespace ResidualResult24092
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult24088.actual selector witness -
    ResidualResult24073.actual selector witness
end ResidualResult24092

namespace ResidualResult24101
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult21512.actual selector witness *
    ResidualResult23922.actual selector witness
end ResidualResult24101

namespace ResidualResult24108
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult24101.actual selector witness +
    ResidualResult23915.actual selector witness
end ResidualResult24108

namespace ResidualResult24118
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult24108.actual selector witness *
    ResidualResult23831.actual selector witness
end ResidualResult24118

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert
