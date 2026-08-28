import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard366
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard420
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard421

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace ResidualResult58108
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult58103.actual selector witness *
    ResidualResult58101.actual selector witness
end ResidualResult58108

namespace ResidualResult58113
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 58113
end ResidualResult58113

namespace ResidualResult58119
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 58119
end ResidualResult58119

namespace ResidualResult58123
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 58123
end ResidualResult58123

namespace ResidualResult58126
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 58126
end ResidualResult58126

namespace ResidualResult58131
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult58126.actual selector witness *
    ResidualResult58123.actual selector witness
end ResidualResult58131

namespace ResidualResult58135
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult58131.actual selector witness -
    ResidualResult58108.actual selector witness
end ResidualResult58135

namespace ResidualResult58143
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult58135.actual selector witness *
    ResidualResult58092.actual selector witness
end ResidualResult58143

namespace ResidualResult58146
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 58146
end ResidualResult58146

namespace ResidualResult58151
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult58103.actual selector witness *
    ResidualResult58146.actual selector witness
end ResidualResult58151

namespace ResidualResult58154
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 58154
end ResidualResult58154

namespace ResidualResult58158
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult58154.actual selector witness -
    ResidualResult58151.actual selector witness
end ResidualResult58158

namespace ResidualResult58162
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult58158.actual selector witness -
    ResidualResult58143.actual selector witness
end ResidualResult58162

namespace ResidualResult58171
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult50762.actual selector witness *
    ResidualResult57992.actual selector witness
end ResidualResult58171

namespace ResidualResult58178
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult58171.actual selector witness +
    ResidualResult57985.actual selector witness
end ResidualResult58178

namespace ResidualResult58188
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult58178.actual selector witness *
    ResidualResult57901.actual selector witness
end ResidualResult58188

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert
