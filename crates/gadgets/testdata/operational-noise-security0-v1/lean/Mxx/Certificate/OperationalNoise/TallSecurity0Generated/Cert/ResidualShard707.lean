import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard000
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard038
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard101
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard102

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace ResidualResult99155
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 99155
end ResidualResult99155

namespace ResidualResult99160
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult4819.actual selector witness *
    ResidualResult32.actual selector witness
end ResidualResult99160

namespace ResidualResult99165
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult27.actual selector witness *
    ResidualResult11983.actual selector witness
end ResidualResult99165

namespace ResidualResult99169
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult99165.actual selector witness -
    ResidualResult99160.actual selector witness
end ResidualResult99169

namespace ResidualResult99175
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult99169.actual selector witness +
    ResidualResult11975.actual selector witness
end ResidualResult99175

namespace ResidualResult99183
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult99175.actual selector witness *
    ResidualResult4822.actual selector witness
end ResidualResult99183

namespace ResidualResult99188
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult4822.actual selector witness *
    ResidualResult32.actual selector witness
end ResidualResult99188

namespace ResidualResult99193
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult27.actual selector witness *
    ResidualResult12024.actual selector witness
end ResidualResult99193

namespace ResidualResult99197
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult99193.actual selector witness -
    ResidualResult99188.actual selector witness
end ResidualResult99197

namespace ResidualResult99203
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult99197.actual selector witness +
    ResidualResult12016.actual selector witness
end ResidualResult99203

namespace ResidualResult99213
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult99203.actual selector witness *
    ResidualResult12013.actual selector witness
end ResidualResult99213

namespace ResidualResult99219
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult99213.actual selector witness +
    ResidualResult99183.actual selector witness
end ResidualResult99219

namespace ResidualResult99229
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult99219.actual selector witness *
    ResidualResult99155.actual selector witness
end ResidualResult99229

namespace ResidualResult99232
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 99232
end ResidualResult99232

namespace ResidualResult99236
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 99236
end ResidualResult99236

namespace ResidualResult99290
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 99290
end ResidualResult99290

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert
