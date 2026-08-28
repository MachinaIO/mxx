import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard058
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard113
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard117
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard121
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard125
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard128

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace ResidualResult15404
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 15404
end ResidualResult15404

namespace ResidualResult15409
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult15404.actual selector witness *
    ResidualResult15402.actual selector witness
end ResidualResult15409

namespace ResidualResult15412
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 15412
end ResidualResult15412

namespace ResidualResult15416
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult15412.actual selector witness -
    ResidualResult15409.actual selector witness
end ResidualResult15416

namespace ResidualResult15424
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult15416.actual selector witness *
    ResidualResult15393.actual selector witness
end ResidualResult15424

namespace ResidualResult15427
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 15427
end ResidualResult15427

namespace ResidualResult15432
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult15404.actual selector witness *
    ResidualResult15427.actual selector witness
end ResidualResult15432

namespace ResidualResult15435
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 15435
end ResidualResult15435

namespace ResidualResult15439
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult15435.actual selector witness -
    ResidualResult15432.actual selector witness
end ResidualResult15439

namespace ResidualResult15443
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult15439.actual selector witness -
    ResidualResult15424.actual selector witness
end ResidualResult15443

namespace ResidualResult15452
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult6561.actual selector witness *
    ResidualResult15281.actual selector witness
end ResidualResult15452

namespace ResidualResult15459
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult15452.actual selector witness +
    ResidualResult15274.actual selector witness
end ResidualResult15459

namespace ResidualResult15464
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult15459.actual selector witness +
    ResidualResult14958.actual selector witness
end ResidualResult15464

namespace ResidualResult15469
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult15464.actual selector witness +
    ResidualResult14457.actual selector witness
end ResidualResult15469

namespace ResidualResult15474
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult15469.actual selector witness +
    ResidualResult13956.actual selector witness
end ResidualResult15474

namespace ResidualResult15479
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult15474.actual selector witness +
    ResidualResult13455.actual selector witness
end ResidualResult15479

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert
