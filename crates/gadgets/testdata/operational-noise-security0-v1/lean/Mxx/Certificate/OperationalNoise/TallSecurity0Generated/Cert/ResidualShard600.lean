import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard032
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard093
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard094
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard565
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard566
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard599

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace ResidualResult84284
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult4038.actual selector witness *
    ResidualResult79920.actual selector witness
end ResidualResult84284

namespace ResidualResult84289
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult79790.actual selector witness *
    ResidualResult11022.actual selector witness
end ResidualResult84289

namespace ResidualResult84293
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult84289.actual selector witness -
    ResidualResult84284.actual selector witness
end ResidualResult84293

namespace ResidualResult84299
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult84293.actual selector witness +
    ResidualResult11014.actual selector witness
end ResidualResult84299

namespace ResidualResult84309
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult84299.actual selector witness *
    ResidualResult11011.actual selector witness
end ResidualResult84309

namespace ResidualResult84315
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult84309.actual selector witness +
    ResidualResult84279.actual selector witness
end ResidualResult84315

namespace ResidualResult84325
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult84315.actual selector witness *
    ResidualResult84251.actual selector witness
end ResidualResult84325

namespace ResidualResult84328
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 84328
end ResidualResult84328

namespace ResidualResult84332
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 84332
end ResidualResult84332

namespace ResidualResult84410
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 84410
end ResidualResult84410

namespace ResidualResult84413
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 84413
end ResidualResult84413

namespace ResidualResult84418
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult84413.actual selector witness *
    ResidualResult84410.actual selector witness
end ResidualResult84418

namespace ResidualResult84429
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 84429
end ResidualResult84429

namespace ResidualResult84432
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 84432
end ResidualResult84432

namespace ResidualResult84441
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 84441
end ResidualResult84441

namespace ResidualResult84443
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 84443
end ResidualResult84443

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert
