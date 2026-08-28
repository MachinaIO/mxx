import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard004
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard005
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard006

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace ResidualResult5147
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 5147
end ResidualResult5147

namespace ResidualResult5152
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult5147.actual selector witness *
    ResidualResult653.actual selector witness
end ResidualResult5152

namespace ResidualResult5155
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 5155
end ResidualResult5155

namespace ResidualResult5160
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult5155.actual selector witness *
    ResidualResult663.actual selector witness
end ResidualResult5160

namespace ResidualResult5163
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 5163
end ResidualResult5163

namespace ResidualResult5168
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult5163.actual selector witness *
    ResidualResult673.actual selector witness
end ResidualResult5168

namespace ResidualResult5171
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 5171
end ResidualResult5171

namespace ResidualResult5176
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult5171.actual selector witness *
    ResidualResult683.actual selector witness
end ResidualResult5176

namespace ResidualResult5179
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 5179
end ResidualResult5179

namespace ResidualResult5184
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult5179.actual selector witness *
    ResidualResult693.actual selector witness
end ResidualResult5184

namespace ResidualResult5187
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 5187
end ResidualResult5187

namespace ResidualResult5192
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult5187.actual selector witness *
    ResidualResult703.actual selector witness
end ResidualResult5192

namespace ResidualResult5195
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 5195
end ResidualResult5195

namespace ResidualResult5200
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult5195.actual selector witness *
    ResidualResult713.actual selector witness
end ResidualResult5200

namespace ResidualResult5204
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult728.actual selector witness +
    ResidualResult5200.actual selector witness
end ResidualResult5204

namespace ResidualResult5208
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult5204.actual selector witness +
    ResidualResult5192.actual selector witness
end ResidualResult5208

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert
