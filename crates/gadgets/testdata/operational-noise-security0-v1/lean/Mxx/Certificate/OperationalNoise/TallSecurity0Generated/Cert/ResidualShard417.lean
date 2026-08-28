import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard021
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard113
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard114
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard364
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard365
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard416

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace ResidualResult57434
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult2660.actual selector witness *
    ResidualResult50670.actual selector witness
end ResidualResult57434

namespace ResidualResult57439
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult50540.actual selector witness *
    ResidualResult13486.actual selector witness
end ResidualResult57439

namespace ResidualResult57443
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult57439.actual selector witness -
    ResidualResult57434.actual selector witness
end ResidualResult57443

namespace ResidualResult57449
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult57443.actual selector witness +
    ResidualResult13478.actual selector witness
end ResidualResult57449

namespace ResidualResult57457
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult57449.actual selector witness *
    ResidualResult2663.actual selector witness
end ResidualResult57457

namespace ResidualResult57462
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult2663.actual selector witness *
    ResidualResult50670.actual selector witness
end ResidualResult57462

namespace ResidualResult57467
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult50540.actual selector witness *
    ResidualResult13527.actual selector witness
end ResidualResult57467

namespace ResidualResult57471
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult57467.actual selector witness -
    ResidualResult57462.actual selector witness
end ResidualResult57471

namespace ResidualResult57477
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult57471.actual selector witness +
    ResidualResult13519.actual selector witness
end ResidualResult57477

namespace ResidualResult57487
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult57477.actual selector witness *
    ResidualResult13516.actual selector witness
end ResidualResult57487

namespace ResidualResult57493
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult57487.actual selector witness +
    ResidualResult57457.actual selector witness
end ResidualResult57493

namespace ResidualResult57503
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult57493.actual selector witness *
    ResidualResult57429.actual selector witness
end ResidualResult57503

namespace ResidualResult57506
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 57506
end ResidualResult57506

namespace ResidualResult57510
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 57510
end ResidualResult57510

namespace ResidualResult57588
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 57588
end ResidualResult57588

namespace ResidualResult57591
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 57591
end ResidualResult57591

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert
