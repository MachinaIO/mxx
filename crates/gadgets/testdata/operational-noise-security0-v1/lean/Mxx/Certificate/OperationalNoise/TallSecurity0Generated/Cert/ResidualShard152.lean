import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard047
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard058
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard108
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard151

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace ResidualResult19568
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 19568
end ResidualResult19568

namespace ResidualResult19573
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult19568.actual selector witness *
    ResidualResult19566.actual selector witness
end ResidualResult19573

namespace ResidualResult19576
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 19576
end ResidualResult19576

namespace ResidualResult19580
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult19576.actual selector witness -
    ResidualResult19573.actual selector witness
end ResidualResult19580

namespace ResidualResult19588
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult19580.actual selector witness *
    ResidualResult19557.actual selector witness
end ResidualResult19588

namespace ResidualResult19591
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 19591
end ResidualResult19591

namespace ResidualResult19596
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult19568.actual selector witness *
    ResidualResult19591.actual selector witness
end ResidualResult19596

namespace ResidualResult19599
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 19599
end ResidualResult19599

namespace ResidualResult19603
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult19599.actual selector witness -
    ResidualResult19596.actual selector witness
end ResidualResult19603

namespace ResidualResult19607
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult19603.actual selector witness -
    ResidualResult19588.actual selector witness
end ResidualResult19607

namespace ResidualResult19616
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult6561.actual selector witness *
    ResidualResult19445.actual selector witness
end ResidualResult19616

namespace ResidualResult19623
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult19616.actual selector witness +
    ResidualResult19438.actual selector witness
end ResidualResult19623

namespace ResidualResult19633
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult19623.actual selector witness *
    ResidualResult5739.actual selector witness
end ResidualResult19633

namespace ResidualResult19637
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 19637
end ResidualResult19637

namespace ResidualResult19640
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 19640
end ResidualResult19640

namespace ResidualResult19650
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult12759.actual selector witness *
    ResidualResult19640.actual selector witness
end ResidualResult19650

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert
