import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard025
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard073
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard465
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard466
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard479

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace ResidualResult67182
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 67182
end ResidualResult67182

namespace ResidualResult67187
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult67159.actual selector witness *
    ResidualResult67182.actual selector witness
end ResidualResult67187

namespace ResidualResult67190
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 67190
end ResidualResult67190

namespace ResidualResult67194
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult67190.actual selector witness -
    ResidualResult67187.actual selector witness
end ResidualResult67194

namespace ResidualResult67198
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult67194.actual selector witness -
    ResidualResult67179.actual selector witness
end ResidualResult67198

namespace ResidualResult67207
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult65387.actual selector witness *
    ResidualResult67036.actual selector witness
end ResidualResult67207

namespace ResidualResult67214
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult67207.actual selector witness +
    ResidualResult67029.actual selector witness
end ResidualResult67214

namespace ResidualResult67221
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 67221
end ResidualResult67221

namespace ResidualResult67224
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 67224
end ResidualResult67224

namespace ResidualResult67231
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 67231
end ResidualResult67231

namespace ResidualResult67234
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 67234
end ResidualResult67234

namespace ResidualResult67239
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult3178.actual selector witness *
    ResidualResult65295.actual selector witness
end ResidualResult67239

namespace ResidualResult67244
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult65165.actual selector witness *
    ResidualResult8476.actual selector witness
end ResidualResult67244

namespace ResidualResult67248
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult67244.actual selector witness -
    ResidualResult67239.actual selector witness
end ResidualResult67248

namespace ResidualResult67254
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult67248.actual selector witness +
    ResidualResult8468.actual selector witness
end ResidualResult67254

namespace ResidualResult67262
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult67254.actual selector witness *
    ResidualResult3181.actual selector witness
end ResidualResult67262

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert
