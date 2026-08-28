import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard058
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard094

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace ResidualResult11179
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 11179
end ResidualResult11179

namespace ResidualResult11181
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 11181
end ResidualResult11181

namespace ResidualResult11186
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult11181.actual selector witness *
    ResidualResult11179.actual selector witness
end ResidualResult11186

namespace ResidualResult11191
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 11191
end ResidualResult11191

namespace ResidualResult11197
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 11197
end ResidualResult11197

namespace ResidualResult11201
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 11201
end ResidualResult11201

namespace ResidualResult11204
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 11204
end ResidualResult11204

namespace ResidualResult11209
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult11204.actual selector witness *
    ResidualResult11201.actual selector witness
end ResidualResult11209

namespace ResidualResult11213
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult11209.actual selector witness -
    ResidualResult11186.actual selector witness
end ResidualResult11213

namespace ResidualResult11221
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult11213.actual selector witness *
    ResidualResult11170.actual selector witness
end ResidualResult11221

namespace ResidualResult11224
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 11224
end ResidualResult11224

namespace ResidualResult11229
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult11181.actual selector witness *
    ResidualResult11224.actual selector witness
end ResidualResult11229

namespace ResidualResult11232
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 11232
end ResidualResult11232

namespace ResidualResult11236
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult11232.actual selector witness -
    ResidualResult11229.actual selector witness
end ResidualResult11236

namespace ResidualResult11240
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult11236.actual selector witness -
    ResidualResult11221.actual selector witness
end ResidualResult11240

namespace ResidualResult11249
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult6561.actual selector witness *
    ResidualResult11070.actual selector witness
end ResidualResult11249

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert
