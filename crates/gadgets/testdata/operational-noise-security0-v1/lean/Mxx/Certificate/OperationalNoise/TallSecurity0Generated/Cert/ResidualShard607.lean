import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard032
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard101
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard102
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard565
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard566

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace ResidualResult85211
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 85211
end ResidualResult85211

namespace ResidualResult85216
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult4081.actual selector witness *
    ResidualResult79920.actual selector witness
end ResidualResult85216

namespace ResidualResult85221
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult79790.actual selector witness *
    ResidualResult11983.actual selector witness
end ResidualResult85221

namespace ResidualResult85225
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult85221.actual selector witness -
    ResidualResult85216.actual selector witness
end ResidualResult85225

namespace ResidualResult85231
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult85225.actual selector witness +
    ResidualResult11975.actual selector witness
end ResidualResult85231

namespace ResidualResult85239
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult85231.actual selector witness *
    ResidualResult4084.actual selector witness
end ResidualResult85239

namespace ResidualResult85244
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult4084.actual selector witness *
    ResidualResult79920.actual selector witness
end ResidualResult85244

namespace ResidualResult85249
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult79790.actual selector witness *
    ResidualResult12024.actual selector witness
end ResidualResult85249

namespace ResidualResult85253
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult85249.actual selector witness -
    ResidualResult85244.actual selector witness
end ResidualResult85253

namespace ResidualResult85259
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult85253.actual selector witness +
    ResidualResult12016.actual selector witness
end ResidualResult85259

namespace ResidualResult85269
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult85259.actual selector witness *
    ResidualResult12013.actual selector witness
end ResidualResult85269

namespace ResidualResult85275
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult85269.actual selector witness +
    ResidualResult85239.actual selector witness
end ResidualResult85275

namespace ResidualResult85285
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult85275.actual selector witness *
    ResidualResult85211.actual selector witness
end ResidualResult85285

namespace ResidualResult85288
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 85288
end ResidualResult85288

namespace ResidualResult85292
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 85292
end ResidualResult85292

namespace ResidualResult85370
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 85370
end ResidualResult85370

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert
