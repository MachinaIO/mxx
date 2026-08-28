import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard466
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard503

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace ResidualResult70316
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 70316
end ResidualResult70316

namespace ResidualResult70318
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 70318
end ResidualResult70318

namespace ResidualResult70323
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult70318.actual selector witness *
    ResidualResult70316.actual selector witness
end ResidualResult70323

namespace ResidualResult70328
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 70328
end ResidualResult70328

namespace ResidualResult70334
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 70334
end ResidualResult70334

namespace ResidualResult70338
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 70338
end ResidualResult70338

namespace ResidualResult70341
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 70341
end ResidualResult70341

namespace ResidualResult70346
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult70341.actual selector witness *
    ResidualResult70338.actual selector witness
end ResidualResult70346

namespace ResidualResult70350
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult70346.actual selector witness -
    ResidualResult70323.actual selector witness
end ResidualResult70350

namespace ResidualResult70358
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult70350.actual selector witness *
    ResidualResult70307.actual selector witness
end ResidualResult70358

namespace ResidualResult70361
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 70361
end ResidualResult70361

namespace ResidualResult70366
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult70318.actual selector witness *
    ResidualResult70361.actual selector witness
end ResidualResult70366

namespace ResidualResult70369
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 70369
end ResidualResult70369

namespace ResidualResult70373
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult70369.actual selector witness -
    ResidualResult70366.actual selector witness
end ResidualResult70373

namespace ResidualResult70377
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult70373.actual selector witness -
    ResidualResult70358.actual selector witness
end ResidualResult70377

namespace ResidualResult70386
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult65387.actual selector witness *
    ResidualResult70207.actual selector witness
end ResidualResult70386

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert
