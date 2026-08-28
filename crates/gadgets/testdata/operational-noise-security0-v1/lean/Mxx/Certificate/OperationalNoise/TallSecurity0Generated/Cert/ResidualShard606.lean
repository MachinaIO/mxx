import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard567
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard605

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace ResidualResult85134
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 85134
end ResidualResult85134

namespace ResidualResult85136
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 85136
end ResidualResult85136

namespace ResidualResult85141
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult85136.actual selector witness *
    ResidualResult85134.actual selector witness
end ResidualResult85141

namespace ResidualResult85144
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 85144
end ResidualResult85144

namespace ResidualResult85148
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult85144.actual selector witness -
    ResidualResult85141.actual selector witness
end ResidualResult85148

namespace ResidualResult85156
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult85148.actual selector witness *
    ResidualResult85125.actual selector witness
end ResidualResult85156

namespace ResidualResult85159
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 85159
end ResidualResult85159

namespace ResidualResult85164
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult85136.actual selector witness *
    ResidualResult85159.actual selector witness
end ResidualResult85164

namespace ResidualResult85167
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 85167
end ResidualResult85167

namespace ResidualResult85171
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult85167.actual selector witness -
    ResidualResult85164.actual selector witness
end ResidualResult85171

namespace ResidualResult85175
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult85171.actual selector witness -
    ResidualResult85156.actual selector witness
end ResidualResult85175

namespace ResidualResult85184
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult80012.actual selector witness *
    ResidualResult85013.actual selector witness
end ResidualResult85184

namespace ResidualResult85191
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult85184.actual selector witness +
    ResidualResult85006.actual selector witness
end ResidualResult85191

namespace ResidualResult85198
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 85198
end ResidualResult85198

namespace ResidualResult85201
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 85201
end ResidualResult85201

namespace ResidualResult85208
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 85208
end ResidualResult85208

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert
