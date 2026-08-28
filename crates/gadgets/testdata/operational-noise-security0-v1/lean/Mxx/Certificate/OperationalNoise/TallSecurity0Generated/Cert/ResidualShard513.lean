import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard027
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard109
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard465
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard466
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard512

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace ResidualResult71505
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 71505
end ResidualResult71505

namespace ResidualResult71509
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult71505.actual selector witness -
    ResidualResult71502.actual selector witness
end ResidualResult71509

namespace ResidualResult71517
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult71509.actual selector witness *
    ResidualResult71486.actual selector witness
end ResidualResult71517

namespace ResidualResult71520
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 71520
end ResidualResult71520

namespace ResidualResult71525
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult71497.actual selector witness *
    ResidualResult71520.actual selector witness
end ResidualResult71525

namespace ResidualResult71528
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 71528
end ResidualResult71528

namespace ResidualResult71532
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult71528.actual selector witness -
    ResidualResult71525.actual selector witness
end ResidualResult71532

namespace ResidualResult71536
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult71532.actual selector witness -
    ResidualResult71517.actual selector witness
end ResidualResult71536

namespace ResidualResult71545
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult65387.actual selector witness *
    ResidualResult71374.actual selector witness
end ResidualResult71545

namespace ResidualResult71552
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult71545.actual selector witness +
    ResidualResult71367.actual selector witness
end ResidualResult71552

namespace ResidualResult71559
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 71559
end ResidualResult71559

namespace ResidualResult71562
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 71562
end ResidualResult71562

namespace ResidualResult71569
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 71569
end ResidualResult71569

namespace ResidualResult71572
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 71572
end ResidualResult71572

namespace ResidualResult71577
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult3385.actual selector witness *
    ResidualResult65295.actual selector witness
end ResidualResult71577

namespace ResidualResult71582
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult65165.actual selector witness *
    ResidualResult12985.actual selector witness
end ResidualResult71582

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert
