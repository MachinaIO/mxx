import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard667
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard670
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard671

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace ResidualResult95026
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 95026
end ResidualResult95026

namespace ResidualResult95031
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult94983.actual selector witness *
    ResidualResult95026.actual selector witness
end ResidualResult95031

namespace ResidualResult95034
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 95034
end ResidualResult95034

namespace ResidualResult95038
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult95034.actual selector witness -
    ResidualResult95031.actual selector witness
end ResidualResult95038

namespace ResidualResult95042
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult95038.actual selector witness -
    ResidualResult95023.actual selector witness
end ResidualResult95042

namespace ResidualResult95051
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult94462.actual selector witness *
    ResidualResult94896.actual selector witness
end ResidualResult95051

namespace ResidualResult95058
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult95051.actual selector witness +
    ResidualResult94889.actual selector witness
end ResidualResult95058

namespace ResidualResult95068
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult95058.actual selector witness *
    ResidualResult94805.actual selector witness
end ResidualResult95068

namespace ResidualResult95071
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 95071
end ResidualResult95071

namespace ResidualResult95075
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 95075
end ResidualResult95075

namespace ResidualResult95149
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 95149
end ResidualResult95149

namespace ResidualResult95160
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 95160
end ResidualResult95160

namespace ResidualResult95163
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 95163
end ResidualResult95163

namespace ResidualResult95172
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 95172
end ResidualResult95172

namespace ResidualResult95174
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 95174
end ResidualResult95174

namespace ResidualResult95179
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult95174.actual selector witness *
    ResidualResult95172.actual selector witness
end ResidualResult95179

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert
