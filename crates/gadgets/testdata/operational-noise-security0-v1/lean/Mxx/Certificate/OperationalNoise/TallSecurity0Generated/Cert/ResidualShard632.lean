import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard567
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard610
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard614
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard617
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard621
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard625
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard628
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard631

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace ResidualResult88504
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 88504
end ResidualResult88504

namespace ResidualResult88508
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult88504.actual selector witness -
    ResidualResult88501.actual selector witness
end ResidualResult88508

namespace ResidualResult88516
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult88508.actual selector witness *
    ResidualResult88485.actual selector witness
end ResidualResult88516

namespace ResidualResult88519
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 88519
end ResidualResult88519

namespace ResidualResult88524
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult88496.actual selector witness *
    ResidualResult88519.actual selector witness
end ResidualResult88524

namespace ResidualResult88527
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 88527
end ResidualResult88527

namespace ResidualResult88531
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult88527.actual selector witness -
    ResidualResult88524.actual selector witness
end ResidualResult88531

namespace ResidualResult88535
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult88531.actual selector witness -
    ResidualResult88516.actual selector witness
end ResidualResult88535

namespace ResidualResult88544
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult80012.actual selector witness *
    ResidualResult88373.actual selector witness
end ResidualResult88544

namespace ResidualResult88551
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult88544.actual selector witness +
    ResidualResult88366.actual selector witness
end ResidualResult88551

namespace ResidualResult88556
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult88551.actual selector witness +
    ResidualResult88071.actual selector witness
end ResidualResult88556

namespace ResidualResult88561
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult88556.actual selector witness +
    ResidualResult87591.actual selector witness
end ResidualResult88561

namespace ResidualResult88566
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult88561.actual selector witness +
    ResidualResult87111.actual selector witness
end ResidualResult88566

namespace ResidualResult88571
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult88566.actual selector witness +
    ResidualResult86631.actual selector witness
end ResidualResult88571

namespace ResidualResult88576
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult88571.actual selector witness +
    ResidualResult86151.actual selector witness
end ResidualResult88576

namespace ResidualResult88581
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult88576.actual selector witness +
    ResidualResult85671.actual selector witness
end ResidualResult88581

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert
