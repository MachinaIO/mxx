import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard045
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard165
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard181
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard241

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace ResidualResult32528
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult32500.actual selector witness *
    ResidualResult32523.actual selector witness
end ResidualResult32528

namespace ResidualResult32531
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 32531
end ResidualResult32531

namespace ResidualResult32535
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult32531.actual selector witness -
    ResidualResult32528.actual selector witness
end ResidualResult32535

namespace ResidualResult32539
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult32535.actual selector witness -
    ResidualResult32520.actual selector witness
end ResidualResult32539

namespace ResidualResult32548
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult21512.actual selector witness *
    ResidualResult32377.actual selector witness
end ResidualResult32548

namespace ResidualResult32555
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult32548.actual selector witness +
    ResidualResult32370.actual selector witness
end ResidualResult32555

namespace ResidualResult32565
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult32555.actual selector witness *
    ResidualResult5579.actual selector witness
end ResidualResult32565

namespace ResidualResult32569
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 32569
end ResidualResult32569

namespace ResidualResult32572
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 32572
end ResidualResult32572

namespace ResidualResult32582
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult23626.actual selector witness *
    ResidualResult32572.actual selector witness
end ResidualResult32582

namespace ResidualResult32585
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 32585
end ResidualResult32585

namespace ResidualResult32589
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 32589
end ResidualResult32589

namespace ResidualResult32687
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 32687
end ResidualResult32687

namespace ResidualResult32698
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 32698
end ResidualResult32698

namespace ResidualResult32701
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 32701
end ResidualResult32701

namespace ResidualResult32710
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 32710
end ResidualResult32710

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert
