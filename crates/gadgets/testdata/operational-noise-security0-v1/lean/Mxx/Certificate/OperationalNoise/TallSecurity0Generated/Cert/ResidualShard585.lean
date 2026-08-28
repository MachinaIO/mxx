import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard032
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard077
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard565
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard566
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard584

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace ResidualResult82341
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult79790.actual selector witness *
    ResidualResult8977.actual selector witness
end ResidualResult82341

namespace ResidualResult82345
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult82341.actual selector witness -
    ResidualResult82336.actual selector witness
end ResidualResult82345

namespace ResidualResult82351
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult82345.actual selector witness +
    ResidualResult8969.actual selector witness
end ResidualResult82351

namespace ResidualResult82359
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult82351.actual selector witness *
    ResidualResult3946.actual selector witness
end ResidualResult82359

namespace ResidualResult82364
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult3946.actual selector witness *
    ResidualResult79920.actual selector witness
end ResidualResult82364

namespace ResidualResult82369
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult79790.actual selector witness *
    ResidualResult9018.actual selector witness
end ResidualResult82369

namespace ResidualResult82373
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult82369.actual selector witness -
    ResidualResult82364.actual selector witness
end ResidualResult82373

namespace ResidualResult82379
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult82373.actual selector witness +
    ResidualResult9010.actual selector witness
end ResidualResult82379

namespace ResidualResult82389
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult82379.actual selector witness *
    ResidualResult9007.actual selector witness
end ResidualResult82389

namespace ResidualResult82395
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult82389.actual selector witness +
    ResidualResult82359.actual selector witness
end ResidualResult82395

namespace ResidualResult82405
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult82395.actual selector witness *
    ResidualResult82331.actual selector witness
end ResidualResult82405

namespace ResidualResult82408
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 82408
end ResidualResult82408

namespace ResidualResult82412
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 82412
end ResidualResult82412

namespace ResidualResult82490
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 82490
end ResidualResult82490

namespace ResidualResult82493
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 82493
end ResidualResult82493

namespace ResidualResult82498
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult82493.actual selector witness *
    ResidualResult82490.actual selector witness
end ResidualResult82498

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert
