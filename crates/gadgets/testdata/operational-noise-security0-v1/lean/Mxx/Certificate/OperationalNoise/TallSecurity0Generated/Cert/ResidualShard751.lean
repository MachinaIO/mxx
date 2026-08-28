import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard046
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard667
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard701
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard750

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace ResidualResult105426
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult105418.actual selector witness *
    ResidualResult105395.actual selector witness
end ResidualResult105426

namespace ResidualResult105429
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 105429
end ResidualResult105429

namespace ResidualResult105434
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult105406.actual selector witness *
    ResidualResult105429.actual selector witness
end ResidualResult105434

namespace ResidualResult105437
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 105437
end ResidualResult105437

namespace ResidualResult105441
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult105437.actual selector witness -
    ResidualResult105434.actual selector witness
end ResidualResult105441

namespace ResidualResult105445
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult105441.actual selector witness -
    ResidualResult105426.actual selector witness
end ResidualResult105445

namespace ResidualResult105454
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult94462.actual selector witness *
    ResidualResult105307.actual selector witness
end ResidualResult105454

namespace ResidualResult105461
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult105454.actual selector witness +
    ResidualResult105300.actual selector witness
end ResidualResult105461

namespace ResidualResult105471
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult105461.actual selector witness *
    ResidualResult5679.actual selector witness
end ResidualResult105471

namespace ResidualResult105475
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 105475
end ResidualResult105475

namespace ResidualResult105478
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 105478
end ResidualResult105478

namespace ResidualResult105488
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult98530.actual selector witness *
    ResidualResult105478.actual selector witness
end ResidualResult105488

namespace ResidualResult105491
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 105491
end ResidualResult105491

namespace ResidualResult105495
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 105495
end ResidualResult105495

namespace ResidualResult105569
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 105569
end ResidualResult105569

namespace ResidualResult105580
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 105580
end ResidualResult105580

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert
