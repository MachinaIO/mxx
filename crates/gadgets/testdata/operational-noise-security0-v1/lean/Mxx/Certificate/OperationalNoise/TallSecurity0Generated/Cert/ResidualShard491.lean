import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard026
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard085
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard465
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard466
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard490

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace ResidualResult68625
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult68617.actual selector witness *
    ResidualResult68594.actual selector witness
end ResidualResult68625

namespace ResidualResult68628
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 68628
end ResidualResult68628

namespace ResidualResult68633
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult68605.actual selector witness *
    ResidualResult68628.actual selector witness
end ResidualResult68633

namespace ResidualResult68636
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 68636
end ResidualResult68636

namespace ResidualResult68640
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult68636.actual selector witness -
    ResidualResult68633.actual selector witness
end ResidualResult68640

namespace ResidualResult68644
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult68640.actual selector witness -
    ResidualResult68625.actual selector witness
end ResidualResult68644

namespace ResidualResult68653
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult65387.actual selector witness *
    ResidualResult68482.actual selector witness
end ResidualResult68653

namespace ResidualResult68660
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult68653.actual selector witness +
    ResidualResult68475.actual selector witness
end ResidualResult68660

namespace ResidualResult68667
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 68667
end ResidualResult68667

namespace ResidualResult68670
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 68670
end ResidualResult68670

namespace ResidualResult68677
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 68677
end ResidualResult68677

namespace ResidualResult68680
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 68680
end ResidualResult68680

namespace ResidualResult68685
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult3247.actual selector witness *
    ResidualResult65295.actual selector witness
end ResidualResult68685

namespace ResidualResult68690
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult65165.actual selector witness *
    ResidualResult9979.actual selector witness
end ResidualResult68690

namespace ResidualResult68694
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult68690.actual selector witness -
    ResidualResult68685.actual selector witness
end ResidualResult68694

namespace ResidualResult68700
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult68694.actual selector witness +
    ResidualResult9971.actual selector witness
end ResidualResult68700

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert
