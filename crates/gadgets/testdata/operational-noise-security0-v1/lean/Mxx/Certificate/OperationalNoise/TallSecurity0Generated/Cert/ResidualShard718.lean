import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard000
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard039
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard113
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard114

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace ResidualResult100454
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 100454
end ResidualResult100454

namespace ResidualResult100457
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 100457
end ResidualResult100457

namespace ResidualResult100462
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult4888.actual selector witness *
    ResidualResult32.actual selector witness
end ResidualResult100462

namespace ResidualResult100467
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult27.actual selector witness *
    ResidualResult13486.actual selector witness
end ResidualResult100467

namespace ResidualResult100471
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult100467.actual selector witness -
    ResidualResult100462.actual selector witness
end ResidualResult100471

namespace ResidualResult100477
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult100471.actual selector witness +
    ResidualResult13478.actual selector witness
end ResidualResult100477

namespace ResidualResult100485
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult100477.actual selector witness *
    ResidualResult4891.actual selector witness
end ResidualResult100485

namespace ResidualResult100490
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult4891.actual selector witness *
    ResidualResult32.actual selector witness
end ResidualResult100490

namespace ResidualResult100495
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult27.actual selector witness *
    ResidualResult13527.actual selector witness
end ResidualResult100495

namespace ResidualResult100499
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult100495.actual selector witness -
    ResidualResult100490.actual selector witness
end ResidualResult100499

namespace ResidualResult100505
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult100499.actual selector witness +
    ResidualResult13519.actual selector witness
end ResidualResult100505

namespace ResidualResult100515
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult100505.actual selector witness *
    ResidualResult13516.actual selector witness
end ResidualResult100515

namespace ResidualResult100521
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult100515.actual selector witness +
    ResidualResult100485.actual selector witness
end ResidualResult100521

namespace ResidualResult100531
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult100521.actual selector witness *
    ResidualResult100457.actual selector witness
end ResidualResult100531

namespace ResidualResult100534
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 100534
end ResidualResult100534

namespace ResidualResult100538
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 100538
end ResidualResult100538

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert
