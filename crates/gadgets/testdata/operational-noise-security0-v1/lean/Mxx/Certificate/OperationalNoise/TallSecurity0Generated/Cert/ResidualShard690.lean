import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard667
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard688
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard689

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace ResidualResult97163
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 97163
end ResidualResult97163

namespace ResidualResult97169
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 97169
end ResidualResult97169

namespace ResidualResult97173
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 97173
end ResidualResult97173

namespace ResidualResult97176
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 97176
end ResidualResult97176

namespace ResidualResult97181
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult97176.actual selector witness *
    ResidualResult97173.actual selector witness
end ResidualResult97181

namespace ResidualResult97185
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult97181.actual selector witness -
    ResidualResult97158.actual selector witness
end ResidualResult97185

namespace ResidualResult97193
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult97185.actual selector witness *
    ResidualResult97142.actual selector witness
end ResidualResult97193

namespace ResidualResult97196
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 97196
end ResidualResult97196

namespace ResidualResult97201
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult97153.actual selector witness *
    ResidualResult97196.actual selector witness
end ResidualResult97201

namespace ResidualResult97204
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 97204
end ResidualResult97204

namespace ResidualResult97208
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult97204.actual selector witness -
    ResidualResult97201.actual selector witness
end ResidualResult97208

namespace ResidualResult97212
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult97208.actual selector witness -
    ResidualResult97193.actual selector witness
end ResidualResult97212

namespace ResidualResult97221
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult94462.actual selector witness *
    ResidualResult97066.actual selector witness
end ResidualResult97221

namespace ResidualResult97228
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult97221.actual selector witness +
    ResidualResult97059.actual selector witness
end ResidualResult97228

namespace ResidualResult97238
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult97228.actual selector witness *
    ResidualResult96975.actual selector witness
end ResidualResult97238

namespace ResidualResult97241
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 97241
end ResidualResult97241

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert
