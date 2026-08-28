import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard366
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard384

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace ResidualResult53281
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 53281
end ResidualResult53281

namespace ResidualResult53283
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 53283
end ResidualResult53283

namespace ResidualResult53288
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult53283.actual selector witness *
    ResidualResult53281.actual selector witness
end ResidualResult53288

namespace ResidualResult53293
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 53293
end ResidualResult53293

namespace ResidualResult53299
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 53299
end ResidualResult53299

namespace ResidualResult53303
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 53303
end ResidualResult53303

namespace ResidualResult53306
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 53306
end ResidualResult53306

namespace ResidualResult53311
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult53306.actual selector witness *
    ResidualResult53303.actual selector witness
end ResidualResult53311

namespace ResidualResult53315
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult53311.actual selector witness -
    ResidualResult53288.actual selector witness
end ResidualResult53315

namespace ResidualResult53323
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult53315.actual selector witness *
    ResidualResult53272.actual selector witness
end ResidualResult53323

namespace ResidualResult53326
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 53326
end ResidualResult53326

namespace ResidualResult53331
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult53283.actual selector witness *
    ResidualResult53326.actual selector witness
end ResidualResult53331

namespace ResidualResult53334
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 53334
end ResidualResult53334

namespace ResidualResult53338
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult53334.actual selector witness -
    ResidualResult53331.actual selector witness
end ResidualResult53338

namespace ResidualResult53342
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult53338.actual selector witness -
    ResidualResult53323.actual selector witness
end ResidualResult53342

namespace ResidualResult53351
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult50762.actual selector witness *
    ResidualResult53172.actual selector witness
end ResidualResult53351

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert
