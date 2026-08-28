import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard000
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard038
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard106
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard710

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace ResidualResult99617
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult99609.actual selector witness *
    ResidualResult4845.actual selector witness
end ResidualResult99617

namespace ResidualResult99622
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult4845.actual selector witness *
    ResidualResult32.actual selector witness
end ResidualResult99622

namespace ResidualResult99627
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult27.actual selector witness *
    ResidualResult12525.actual selector witness
end ResidualResult99627

namespace ResidualResult99631
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult99627.actual selector witness -
    ResidualResult99622.actual selector witness
end ResidualResult99631

namespace ResidualResult99637
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult99631.actual selector witness +
    ResidualResult12517.actual selector witness
end ResidualResult99637

namespace ResidualResult99647
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult99637.actual selector witness *
    ResidualResult12514.actual selector witness
end ResidualResult99647

namespace ResidualResult99653
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult99647.actual selector witness +
    ResidualResult99617.actual selector witness
end ResidualResult99653

namespace ResidualResult99663
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult99653.actual selector witness *
    ResidualResult99589.actual selector witness
end ResidualResult99663

namespace ResidualResult99666
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 99666
end ResidualResult99666

namespace ResidualResult99670
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 99670
end ResidualResult99670

namespace ResidualResult99724
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 99724
end ResidualResult99724

namespace ResidualResult99727
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 99727
end ResidualResult99727

namespace ResidualResult99732
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult99727.actual selector witness *
    ResidualResult99724.actual selector witness
end ResidualResult99732

namespace ResidualResult99743
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 99743
end ResidualResult99743

namespace ResidualResult99746
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 99746
end ResidualResult99746

namespace ResidualResult99755
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 99755
end ResidualResult99755

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert
