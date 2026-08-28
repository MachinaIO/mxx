import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard019
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard060
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard061
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard364
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard365

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace ResidualResult51163
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 51163
end ResidualResult51163

namespace ResidualResult51168
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult2361.actual selector witness *
    ResidualResult50670.actual selector witness
end ResidualResult51168

namespace ResidualResult51173
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult50540.actual selector witness *
    ResidualResult6973.actual selector witness
end ResidualResult51173

namespace ResidualResult51177
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult51173.actual selector witness -
    ResidualResult51168.actual selector witness
end ResidualResult51177

namespace ResidualResult51183
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult51177.actual selector witness +
    ResidualResult6965.actual selector witness
end ResidualResult51183

namespace ResidualResult51191
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult51183.actual selector witness *
    ResidualResult2364.actual selector witness
end ResidualResult51191

namespace ResidualResult51196
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult2364.actual selector witness *
    ResidualResult50670.actual selector witness
end ResidualResult51196

namespace ResidualResult51201
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult50540.actual selector witness *
    ResidualResult7014.actual selector witness
end ResidualResult51201

namespace ResidualResult51205
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult51201.actual selector witness -
    ResidualResult51196.actual selector witness
end ResidualResult51205

namespace ResidualResult51211
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult51205.actual selector witness +
    ResidualResult7006.actual selector witness
end ResidualResult51211

namespace ResidualResult51221
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult51211.actual selector witness *
    ResidualResult7003.actual selector witness
end ResidualResult51221

namespace ResidualResult51227
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult51221.actual selector witness +
    ResidualResult51191.actual selector witness
end ResidualResult51227

namespace ResidualResult51237
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult51227.actual selector witness *
    ResidualResult51163.actual selector witness
end ResidualResult51237

namespace ResidualResult51240
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 51240
end ResidualResult51240

namespace ResidualResult51244
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 51244
end ResidualResult51244

namespace ResidualResult51322
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 51322
end ResidualResult51322

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert
