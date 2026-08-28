import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard044
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard265
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard337

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace ResidualResult46487
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 46487
end ResidualResult46487

namespace ResidualResult46489
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 46489
end ResidualResult46489

namespace ResidualResult46494
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult46489.actual selector witness *
    ResidualResult46487.actual selector witness
end ResidualResult46494

namespace ResidualResult46497
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 46497
end ResidualResult46497

namespace ResidualResult46501
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult46497.actual selector witness -
    ResidualResult46494.actual selector witness
end ResidualResult46501

namespace ResidualResult46509
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult46501.actual selector witness *
    ResidualResult46478.actual selector witness
end ResidualResult46509

namespace ResidualResult46512
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 46512
end ResidualResult46512

namespace ResidualResult46517
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult46489.actual selector witness *
    ResidualResult46512.actual selector witness
end ResidualResult46517

namespace ResidualResult46520
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 46520
end ResidualResult46520

namespace ResidualResult46524
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult46520.actual selector witness -
    ResidualResult46517.actual selector witness
end ResidualResult46524

namespace ResidualResult46528
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult46524.actual selector witness -
    ResidualResult46509.actual selector witness
end ResidualResult46528

namespace ResidualResult46537
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult36137.actual selector witness *
    ResidualResult46366.actual selector witness
end ResidualResult46537

namespace ResidualResult46544
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult46537.actual selector witness +
    ResidualResult46359.actual selector witness
end ResidualResult46544

namespace ResidualResult46554
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult46544.actual selector witness *
    ResidualResult5519.actual selector witness
end ResidualResult46554

namespace ResidualResult46558
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 46558
end ResidualResult46558

namespace ResidualResult46561
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 46561
end ResidualResult46561

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert
